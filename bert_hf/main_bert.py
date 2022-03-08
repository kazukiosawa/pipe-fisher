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

from pipeline import PipelineStage, PIPELINE_1F1B, PIPELINE_GPIPE, PIPELINE_CHIMERA
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
parser.add_argument('--pipeline_method', choices=[PIPELINE_1F1B, PIPELINE_GPIPE, PIPELINE_CHIMERA], default=PIPELINE_1F1B)
parser.add_argument('--recompute', action='store_true',
                    help='Recompute activations in backward pass')
parser.add_argument('--num_stages', type=int, default=4,
                    help='number of stages in configurable BERT model')
# Others
parser.add_argument('--checkpoint_dir', default=None, type=str,
                    help='path to directory to save checkpoints')
parser.add_argument('--seed', type=int, default=1,
                    help="random seed for initialization")
parser.add_argument('--p2p_backend', default=dist.Backend.GLOO, type=str)
parser.add_argument('--collective_backend', default=dist.Backend.NCCL, type=str)
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
                'state_dict': stage.stage_module.state_dict(),
                'optimizer': optimizers[0].state_dict(),
            }
            assert os.path.isdir(args.checkpoint_dir)
            ckpt_file_path = os.path.join(args.checkpoint_dir, f'epoch{epoch+1}_stage{rank_to_stage(rank)}.pt')
            torch.save(state, ckpt_file_path)
            print(f'Saved checkpoint to {ckpt_file_path}')

    if is_master:
        print('Finished.')


def train_one_epoch(epoch, step, num_steps_for_this_epoch):
    batch_sizes = (train_loader.batch_size, args.max_seq_length)
    if dual_pipelines:
        num_iterations = num_steps_for_this_epoch*num_micro_batches_per_step//2
        stage.start_comm_threads(num_iterations, batch_sizes)
        stage.stage_module.train()
        stage.up_pipe_stage.start_comm_threads(num_iterations, batch_sizes)
        stage.up_pipe_stage.stage_module.train()
    else:
        stage.start_comm_threads(num_steps_for_this_epoch*num_micro_batches_per_step, batch_sizes)
        stage.stage_module.train()

    train_iterator = iter(train_loader)

    for i in range(num_steps_for_this_epoch):
        dist.barrier()
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = stage.call_pipeline(train_iterator, num_micro_batches_per_step)

        for optimizer in optimizers:
            optimizer.step()

        loss = torch.tensor(loss, device=stage.device)
        dist.reduce(loss, dst=0)
        loss /= (num_replicas * num_micro_batches_per_step)
        if dual_pipelines:
            loss *= 2  # each pipeline handles half micro_batches
        if is_master:
            print(f'epoch{epoch+1} step{step+i+1} loss = {float(loss)}')


if __name__ == "__main__":
    args = parser.parse_args()

    # Setup rank and device
    local_rank, local_size, rank, world_size = init_dist_process_group(backend=args.p2p_backend)
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

    num_stages = args.num_stages
    recompute = args.recompute
    assert world_size % num_stages == 0
    num_ranks_per_stage = int(world_size / num_stages)
    num_replicas = num_ranks_per_stage
    dual_pipelines = args.pipeline_method == PIPELINE_CHIMERA
#    if dual_pipelines:
#        num_replicas *= 2

    def rank_to_stage(_rank, down_pipe=True):
        if down_pipe:
            return _rank // num_ranks_per_stage
        else:
            return (world_size - 1 - _rank) // num_ranks_per_stage

    stage_to_ranks = {_stage_id: [] for _stage_id in range(num_stages)}
    for _rank in range(world_size):
        stage_to_ranks[rank_to_stage(_rank)].append(_rank)
        if dual_pipelines:
            stage_to_ranks[rank_to_stage(_rank, down_pipe=False)].append(_rank)

    grad_sync_groups = []
    for _stage_id in range(num_stages):
        grad_sync_groups.append(dist.new_group(ranks=stage_to_ranks[_stage_id],
                                               backend=args.collective_backend))

    # Prepare BERT pipeline stages
    config = BertConfig.from_json_file(args.bert_config_path)

    def get_pipeline_stage(down_pipe=True):
        stage_id = rank_to_stage(rank, down_pipe=down_pipe)
        stage_module = get_stage_bert_for_pretraining(stage_id, num_stages, config).to(device)
        rank_interval = num_ranks_per_stage if down_pipe else -num_ranks_per_stage
        return PipelineStage(stage_id=stage_id,
                             num_stages=num_stages,
                             stage_module=stage_module,
                             pipeline_method=args.pipeline_method,
                             recompute=recompute,
                             prev_rank=rank-rank_interval if stage_id > 0 else None,
                             next_rank=rank+rank_interval if stage_id < num_stages-1 else None,
                             grad_sync_group=grad_sync_groups[stage_id],
                             is_up_pipe=not down_pipe,
                             up_pipe_stage=get_pipeline_stage(
                                 down_pipe=False) if down_pipe and dual_pipelines else None)

    stage = get_pipeline_stage()
    is_stage_master = rank % num_ranks_per_stage == 0

    # Prepare BERT dataset
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=args.do_lower_case)
    train_dataset = BERTDataset(args.corpus_path,
                                tokenizer,
                                seq_len=args.max_seq_length,
                                corpus_lines=args.corpus_lines,
                                encoding='latin-1',
                                on_memory=args.on_memory)
    if num_replicas > 1:
        rank_in_stage = rank % num_ranks_per_stage
        train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank_in_stage) # DistributedSampler for Chimera?
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=args.micro_batch_size,
                              drop_last=True,
                              num_workers=args.num_workers)

    # Set the number of optimization steps and epochs
    num_micro_batches_per_step = num_stages * args.gradient_accumulation_steps
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

    # Prepare optimizers.
    def get_optimizer(module):
        decay_param_group = {'params': [], 'weight_decay': args.weight_decay}
        no_decay_param_group = {'params': [], 'weight_decay': 0.}
        for m in module.modules():
            if isinstance(m, nn.LayerNorm):
                no_decay_param_group['params'].extend(list(m.parameters()))
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                if hasattr(m, 'bias') and m.bias is not None:
                    no_decay_param_group['params'].append(m.bias)
                decay_param_group['params'].append(m.weight)
        return BertAdam([decay_param_group, no_decay_param_group],
                        lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                        t_total=num_steps)

    optimizers = [get_optimizer(stage.stage_module)]
    if dual_pipelines:
        optimizers.append(get_optimizer(stage.up_pipe_stage.stage_module))

    dist.barrier()
    if is_master:
        print('============================')
        print(f'pipeline_method: {args.pipeline_method}')
        print(f'world_size: {world_size}')
        print(f'num_replica: {num_replicas}')
        print(f'num_epochs: {num_epochs}')
        print(f'num_optimization_steps: {num_steps}')
        print(f'recompute: {recompute}')
        print(f'num_micro_batches_per_step: {num_micro_batches_per_step}')
        print(f'num_ranks_per_stage: {num_ranks_per_stage}')
        for _stage_id in range(num_stages):
            print(f'stage{_stage_id}: ranks {stage_to_ranks[_stage_id]}')
        print('----------------------------')
        for key, value in vars(args).items():
            print(f'{key}: {value}')
        print('============================')

    main()
