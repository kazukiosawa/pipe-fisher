import argparse
import os
import random
import math
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import BertTokenizer, BertConfig

from pipeline import PipelineStage, PIPELINE_1F1B
from utils import init_dist_process_group
from bert_optim import BertAdam
from bert_dataset import BERTDataset
from bert_model import get_stage_bert_for_pretraining


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser()
# Dataset & BERT
parser.add_argument("--corpus_path", default=None, type=str, required=True,
                    help="The input train corpus.")
parser.add_argument('--corpus_lines', default=None, type=int)
parser.add_argument("--vocab_path", type=str, required=True)
parser.add_argument("--on_memory", action='store_true',
                    help="Whether to load train samples into memory or use disk")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--bert_config_path", type=str, required=True,
                    help="config to use.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
# Training
parser.add_argument("--micro_batch_size", default=32, type=int,
                    help="Micro-batch size for training.")
parser.add_argument('--num_optimization_steps', default=None, type=int,
                    help="Total number of optimization steps to perform.")
parser.add_argument("--num_epochs", default=None, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for.")
# Pipeline
parser.add_argument('--pipeline_method', choices=[PIPELINE_1F1B], default=PIPELINE_1F1B)
parser.add_argument('--num_stages', type=int, default=4,
                    help='number of stages in configurable BERT model')
# Others
parser.add_argument('--checkpoint_dir', default=None, type=str,
                    help='path to directory to save checkpoints')
parser.add_argument('--seed', type=int, default=1,
                    help="random seed for initialization")
parser.add_argument('--dist_backend', default='nccl', type=str)
parser.add_argument('--num_workers', default=4, type=int)


def main():
    total_steps = 0
    for epoch in range(num_epochs):
        dist.barrier()
        if num_replicas > 1:
            # deterministically shuffle based on epoch
            train_loader.sampler.set_epoch(epoch)

        steps_for_this_epoch = min(num_steps - total_steps, max_steps_per_epoch)
        train_one_epoch(epoch, total_steps, steps_for_this_epoch)
        total_steps += steps_for_this_epoch

        if args.checkpoint_dir is not None and is_stage_master:
            state = {
                'epoch': epoch + 1,
                'state_dict': stage_module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            assert os.path.isdir(args.checkpoint_dir)
            ckpt_file_path = os.path.join(args.checkpoint_dir, f'epoch{epoch+1}_stage{stage_id}.pt')
            torch.save(state, ckpt_file_path)
            print(f'Saved checkpoint to {ckpt_file_path}')

    if is_master:
        print('Finished.')


def train_one_epoch(epoch, step, num_steps_for_this_epoch):
    stage.stage_module.train()
    train_iterator = iter(train_loader)

    for i in range(num_steps_for_this_epoch):
        dist.barrier()
        optimizer.zero_grad()

        loss = stage.call_pipeline(train_iterator, num_micro_batches_per_step)

        optimizer.step()

        tensor = torch.tensor(loss, device=stage.device)
        dist.reduce(tensor, dst=0)
        tensor /= (num_replicas * num_micro_batches_per_step)
        if is_master:
            print(f'epoch{epoch+1} step{step+i+1} loss = {float(tensor)}')


if __name__ == "__main__":
    args = parser.parse_args()

    # Setup rank and device
    local_rank, local_size, rank, world_size = init_dist_process_group(backend=args.dist_backend)
    assert local_size <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    assert world_size > 1
    is_master = rank == 0
    logging.info(f'world_rank: {rank}/{world_size} local_rank: {local_rank}/{local_size} device: {device}')

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setup stage_id based on rank
    assert world_size % args.num_stages == 0
    num_ranks_per_stage = int(world_size / args.num_stages)
    stage_id = rank // num_ranks_per_stage
    stage_to_ranks_map = {_stage_id: [] for _stage_id in range(args.num_stages)}
    for _rank in range(world_size):
        _stage_id = _rank // num_ranks_per_stage
        stage_to_ranks_map[_stage_id].append(_rank)
    is_stage_master = rank % num_ranks_per_stage == 0

    # Prepare BERT pipeline stage
    config = BertConfig.from_json_file(args.bert_config_path)
    stage_module = get_stage_bert_for_pretraining(stage_id, args.num_stages, config)
    stage_module.to(device)
    grad_sync_group = dist.new_group(stage_to_ranks_map[stage_id]) if num_ranks_per_stage > 1 else None
    stage = PipelineStage(stage_id=stage_id,
                          num_stages=args.num_stages,
                          stage_module=stage_module,
                          num_batch_dims=2,  # batch_size, max_seq_length
                          prev_rank=rank-num_ranks_per_stage if stage_id > 0 else None,
                          next_rank=rank+num_ranks_per_stage if stage_id < args.num_stages-1 else None,
                          grad_sync_group=grad_sync_group,
                          pipeline_method=args.pipeline_method)

    # Prepare BERT dataset
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=args.do_lower_case)
    train_dataset = BERTDataset(args.corpus_path,
                                tokenizer,
                                seq_len=args.max_seq_length,
                                corpus_lines=args.corpus_lines,
                                encoding='latin-1',
                                on_memory=args.on_memory)
    num_replicas = num_ranks_per_stage
    if num_replicas > 1:
        rank_in_stage = rank % num_ranks_per_stage
        train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank_in_stage)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=args.micro_batch_size,
                              drop_last=True,
                              num_workers=args.num_workers)

    # Set the number of optimization steps and epochs
    num_micro_batches_per_step = args.num_stages * args.gradient_accumulation_steps
    num_samples_per_step = num_micro_batches_per_step * args.micro_batch_size * num_replicas
    max_steps_per_epoch = len(train_dataset) // num_samples_per_step
    num_steps = args.num_optimization_steps
    if num_steps is None:
        assert args.num_epochs, 'num_optimization_steps or num_epochs needs to be specified.'
        num_epochs = args.num_epochs
        num_steps = max_steps_per_epoch * args.num_epochs
    else:
        num_samples = num_steps * num_samples_per_step
        num_epochs = math.ceil(num_samples / len(train_dataset))

    # Prepare optimizer.
    decay_param_group = {'params': [], 'weight_decay': args.weight_decay}
    no_decay_param_group = {'params': [], 'weight_decay': 0.}
    for module in stage_module.modules():
        if isinstance(module, nn.LayerNorm):
            no_decay_param_group['params'].extend(list(module.parameters()))
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, 'bias') and module.bias is not None:
                no_decay_param_group['params'].append(module.bias)
            decay_param_group['params'].append(module.weight)
    optimizer = BertAdam([decay_param_group, no_decay_param_group],
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_steps)

    dist.barrier()
    if is_master:
        print('============================')
        print(f'world_size: {world_size}')
        print(f'num_replica: {num_replicas}')
        print(f'num_epochs: {num_epochs}')
        print(f'num_optimization_steps: {num_steps}')
        print(f'num_micro_batches_per_step: {num_micro_batches_per_step}')
        print(f'num_ranks_per_stage: {num_ranks_per_stage}')
        for _stage_id in range(args.num_stages):
            print(f'stage{_stage_id}: ranks {stage_to_ranks_map[_stage_id]}')
        print('----------------------------')
        for key, value in vars(args).items():
            print(f'{key}: {value}')
        print('============================')

    main()
