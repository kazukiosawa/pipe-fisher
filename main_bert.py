import argparse
import os
import random
import math
import pickle
from contextlib import nullcontext
import yaml

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda import nvtx

from transformers import BertTokenizer, BertConfig, BertLayer

from pipeline import PipelineStage, PIPELINE_1F1B, PIPELINE_GPIPE, PIPELINE_CHIMERA, PIPELINE_INTER, PIPELINE_GPIPE_NGD
from utils import init_dist_process_group
from bert_optim import BertAdam
from bert_dataset import BERTDataset
from bert_model import get_stage_bert_for_pretraining
import auto_schedule

import asdfghjkl as asdl
#import sys
#sys.stdout.flush()

try:
    import wandb
except ImportError:
    wandb = None


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
parser.add_argument("--adam_learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--ngd_learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for NGD.")
parser.add_argument("--adam_max_grad_norm", type=float, default=1.)
parser.add_argument("--ngd_max_grad_norm", type=float, default=100.)
parser.add_argument("--beta1", default=0.9, type=float,
                    help="beta1 for Adam.")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for.")
parser.add_argument("--damping", type=float, default=0.01)
# Pipeline
parser.add_argument('--pipeline_method', choices=[PIPELINE_1F1B, PIPELINE_GPIPE, PIPELINE_CHIMERA, PIPELINE_INTER, PIPELINE_GPIPE_NGD], default=PIPELINE_1F1B)
parser.add_argument("--chunks", default=2, type=int,
                    help="Number of chunks for interleaved 1f1b.")
parser.add_argument('--recompute', action='store_true',
                    help='Recompute activations in backward pass')
parser.add_argument('--num_stages', type=int, default=4,
                    help='number of stages in configurable BERT model')
# Others
parser.add_argument('--checkpoint_dir', default=None, type=str,
                    help='path to directory to save checkpoints')
parser.add_argument('--save_checkpoint_steps', type=int, default=200)
parser.add_argument('--seed', type=int, default=1,
                    help="random seed for initialization")
parser.add_argument('--p2p_backend', default=dist.Backend.GLOO, type=str)
parser.add_argument('--collective_backend', default=dist.Backend.NCCL, type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--profile', action='store_true')
parser.add_argument('--record_ngd', action='store_true')
parser.add_argument('--ngd_schedule_path', type=str, default=None)
parser.add_argument('--observe_norm', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--wandb', action='store_true')


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

    if is_master:
        print('Finished.')


def train_one_epoch(epoch, step, num_steps_for_this_epoch):

    num_p2p_comm = num_steps_for_this_epoch * num_micro_batches_per_step
    if interleaved_pipelines:
        stage.start_interleaved_pipeline_comm_threads(num_p2p_comm)
    else:
        stage.start_comm_threads(num_p2p_comm)
    
    stage.stage_module.train()
    if dual_pipelines:
        stage.up_pipe_stage.start_comm_threads(num_p2p_comm)
        stage.up_pipe_stage.stage_module.train()

    if interleaved_pipelines:
        for inter_stage in stage.interleaved_stages:
            inter_stage.start_interleaved_pipeline_comm_threads(num_p2p_comm)
            inter_stage.stage_module.train()


    train_iterator = iter(train_loader)
    train_iterator_for_up_pipe = iter(train_loader_for_up_pipe) if dual_pipelines else None

    save_cxt = nullcontext()
    save_cxt_up_pipe = nullcontext()
    if is_ngd_training:
        save_cxt = asdl.save_inputs_outgrads(stage.stage_module,
                                             ignore_modules=ngd.ignore_modules)
        if dual_pipelines:
            save_cxt_up_pipe = asdl.save_inputs_outgrads(stage.up_pipe_stage.stage_module,
                                                         ignore_modules=ngd_up_pipe.ignore_modules)

    with save_cxt as cxt:
        with save_cxt_up_pipe as cxt_up_pipe:
            if is_ngd_training:
                cxt.set_input_scale(1/np.sqrt(total_num_samples_per_step))
                cxt.set_output_scale(micro_batch_size/np.sqrt(total_num_samples_per_step))
                if dual_pipelines:
                    cxt_up_pipe.set_input_scale(1/np.sqrt(total_num_samples_per_step))
                    cxt_up_pipe.set_output_scale(micro_batch_size/np.sqrt(total_num_samples_per_step))
            for i in range(num_steps_for_this_epoch):
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # 1 pipeline iteration
                if ngd_schedule is not None:
                    if step+i == 0:
                        loss = call_one_ngd_step(train_iterator, cxt, train_iterator_for_up_pipe, cxt_up_pipe)
                    else:
                        loss = stage.call_scheduled_pipeline(ngd_schedule[(step+i-1) % len(ngd_schedule)],
                                                             data_iterator=train_iterator,
                                                             ngd=ngd,
                                                             cxt=cxt,
                                                             data_iterator_up_pipe=train_iterator_for_up_pipe,
                                                             ngd_up_pipe=ngd_up_pipe,
                                                             cxt_up_pipe=cxt_up_pipe,
                                                             up_side_down=not first_half and dual_pipelines)
                else:
                    dist.barrier()
                    if args.record_ngd:
                        loss = call_one_ngd_step(train_iterator, cxt, train_iterator_for_up_pipe, cxt_up_pipe)
                    else:
                        loss = stage.call_pipeline(train_iterator,
                                                   num_micro_batches=num_micro_batches_per_step,
                                                   data_iterator_for_up_pipe=train_iterator_for_up_pipe,
                                                   ngd=ngd,
                                                   iteration=step+i)

                for optimizer in optimizers:
                    optimizer.step()

                if (step+i) % args.log_interval == 0:
                    loss = torch.tensor(loss, device=stage.device)
                    dist.reduce(loss, dst=0)
                    loss /= total_num_micro_batches_per_step
                    if dual_pipelines:
                        loss *= 2
                    if is_master:
                        print(f'epoch{epoch+1} step{step+i+1} loss = {float(loss)}')
                        if args.wandb:
                            log = {'epoch': epoch+1, 'step': step+i+1, 'loss': float(loss),
                                   'adam_learning_rate': optimizers[0].get_lr()[0]}
                            if args.observe_norm:
                                if is_ngd_training:
                                    log['p_norm '] = np.sqrt(sum([float(p.data.norm()) ** 2 for p in non_ngd_params]))
                                    log['g_norm'] = np.sqrt(sum([float(p.grad.norm()) ** 2 for p in non_ngd_params]))
                                    log['ngd_p_norm '] = np.sqrt(sum([float(p.data.norm()) ** 2 for p in kfac_params]))
                                    log['ngd_g_norm'] = np.sqrt(sum([float(p.grad.norm()) ** 2 for p in kfac_params]))
                                    log['ngd_learning_rate'] = optimizers[0].get_lr()[-1]
                                else:
                                    log['p_norm'] = np.sqrt(sum([float(p.data.norm()) ** 2 for p in stage.stage_module.parameters()]))
                                    log['g_norm'] = np.sqrt(sum([float(p.grad.norm()) ** 2 for p in stage.stage_module.parameters()]))
                            wandb.log(log)

                if args.checkpoint_dir is not None and (step+i+1) % args.save_checkpoint_steps == 0 and is_stage_master:
                    state = {
                        'epoch': epoch + 1,
                        'model': stage.stage_module.state_dict(),
                        'optimizer': optimizers[0].state_dict()
                    }
                    assert os.path.isdir(args.checkpoint_dir)
                    ckpt_file_path = os.path.join(args.checkpoint_dir, f'epoch{epoch+1}_step{step+i+1}_stage{rank_to_stage(rank)}.pt')
                    torch.save(state, ckpt_file_path)
                    print(f'Saved checkpoint to {ckpt_file_path}')


@nvtx.range('one_ngd_step')
def call_one_ngd_step(train_iterator, cxt, train_iterator_for_up_pipe=None, cxt_up_pipe=None):
    loss = stage.call_pipeline(train_iterator,
                               num_micro_batches=num_micro_batches_per_step,
                               data_iterator_for_up_pipe=train_iterator_for_up_pipe)
    if dual_pipelines:
        ngd.accumulate_curvature(cxt=cxt)
        ngd_up_pipe.accumulate_curvature(cxt=cxt_up_pipe)

        for module_name, module in ngd.named_modules_for('kron'):
            with nvtx.range(f'sync_kron_A:{module_name}'):
                ngd.sync_curvature(module_name=module_name, kron=['A'], async_op=True)
                ngd_up_pipe.sync_curvature(module_name=module_name, kron=['A'], async_op=True)
                ngd.wait_all_curvature_sync()
                ngd_up_pipe.wait_all_curvature_sync()
            with nvtx.range(f'sync_kron_B:{module_name}'):
                ngd.sync_curvature(module_name=module_name, kron=['B'], async_op=True)
                ngd_up_pipe.sync_curvature(module_name=module_name, kron=['B'], async_op=True)
                ngd.wait_all_curvature_sync()
                ngd_up_pipe.wait_all_curvature_sync()

        ngd.update_inv(kron=['A'], zero_curvature=True)
        ngd.update_inv(kron=['B'], zero_curvature=True)
        ngd_up_pipe.update_inv(kron=['A'], zero_curvature=True)
        ngd_up_pipe.update_inv(kron=['B'], zero_curvature=True)

        ngd.sync_grad_pre_precondition(enabled=is_distributed, async_op=True)
        ngd_up_pipe.sync_grad_pre_precondition(enabled=is_distributed, async_op=True)
        ngd.wait_all_grad_sync()
        ngd_up_pipe.wait_all_grad_sync()

        ngd.precondition()
        ngd_up_pipe.precondition()

        ngd.sync_grad_post_precondition(enabled=is_distributed, async_op=True)
        ngd_up_pipe.sync_grad_post_precondition(enabled=is_distributed, async_op=True)
        ngd.wait_all_grad_sync()
        ngd_up_pipe.wait_all_grad_sync()
    else:
        ngd.accumulate_curvature(cxt=cxt)

        for module_name, _ in ngd.named_modules_for('kron'):
            with nvtx.range(f'sync_kron_A:{module_name}'):
                ngd.sync_curvature(module_name=module_name, kron=['A'], enabled=is_distributed)
            with nvtx.range(f'sync_kron_B:{module_name}'):
                ngd.sync_curvature(module_name=module_name, kron=['B'], enabled=is_distributed)

        ngd.update_inv(kron=['A'], zero_curvature=True)
        ngd.update_inv(kron=['B'], zero_curvature=True)
        ngd.sync_grad_pre_precondition(enabled=is_distributed)
        ngd.precondition()
        ngd.sync_grad_post_precondition(enabled=is_distributed)

    return loss


if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)
    if args.config is not None:
        dict_args.update(yaml.safe_load(open(args.config, 'r')))

    # Setup rank and device
    local_rank, local_size, rank, world_size = init_dist_process_group(backend=args.p2p_backend)
    assert local_size <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    assert world_size > 1
    is_master = rank == 0

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_stages = args.num_stages
    recompute = args.recompute
    chunks = args.chunks
    
    dual_pipelines = args.pipeline_method == PIPELINE_CHIMERA
    interleaved_pipelines = args.pipeline_method == PIPELINE_INTER
    if interleaved_pipelines:
        assert chunks > 1
        assert num_stages % chunks == 0
        assert world_size % (num_stages // chunks) == 0
    else:
        assert world_size % num_stages == 0

    num_ranks_per_stage = int(world_size / num_stages)
    if interleaved_pipelines:
        num_ranks_per_stage = world_size // (num_stages // chunks)
    num_replicas = num_ranks_per_stage

    if dual_pipelines:
        num_replicas *= 2
    is_distributed = num_replicas > 1

    def rank_to_stage(_rank, down_pipe=True):
        if down_pipe:
            return _rank // num_ranks_per_stage
        else:
            return (world_size - 1 - _rank) // num_ranks_per_stage

    def rank_to_stages(_rank, down_pipe=True):
        stages_per_chunk = num_stages // chunks
        stages = []
        for _chunk in range(chunks):
            stages.append(_rank // num_ranks_per_stage + stages_per_chunk * _chunk)
        return stages

    stage_to_ranks = {_stage_id: [] for _stage_id in range(num_stages)}

    for _rank in range(world_size):
        if interleaved_pipelines:
            stages_per_chunk = num_stages // chunks
            for _chunk in range(chunks):
                stage_to_ranks[_rank // num_ranks_per_stage + _chunk * stages_per_chunk].append(_rank)
        else:
            stage_to_ranks[rank_to_stage(_rank)].append(_rank)
            if dual_pipelines:
                stage_to_ranks[rank_to_stage(_rank, down_pipe=False)].append(_rank)

    grad_sync_groups = []
    for _stage_id in range(num_stages):
        grad_sync_groups.append(dist.new_group(ranks=stage_to_ranks[_stage_id],
                                               backend=args.collective_backend))

    # Prepare BERT pipeline stages
    bert_config = BertConfig.from_json_file(args.bert_config_path)
    micro_batch_size = args.micro_batch_size
    max_seq_length = args.max_seq_length
    is_ngd_training = (args.record_ngd or args.ngd_schedule_path is not None or args.pipeline_method == PIPELINE_GPIPE_NGD) and not interleaved_pipelines

    def get_pipeline_stage(down_pipe=True):
        stage_id = rank_to_stage(rank, down_pipe=down_pipe)
        stage_module = get_stage_bert_for_pretraining(stage_id,
                                                      num_stages,
                                                      bert_config).to(device)
        rank_interval = num_ranks_per_stage if down_pipe else -num_ranks_per_stage
        return PipelineStage(stage_id=stage_id,
                             num_stages=num_stages,
                             stage_module=stage_module,
                             batch_sizes=(micro_batch_size, max_seq_length),
                             pipeline_method=args.pipeline_method,
                             recompute=recompute,
                             prev_rank=rank-rank_interval if stage_id > 0 else None,
                             next_rank=rank+rank_interval if stage_id < num_stages-1 else None,
                             rank=rank,
                             grad_sync_group=grad_sync_groups[stage_id],
                             is_up_pipe=not down_pipe,
                             up_pipe_stage=get_pipeline_stage(
                                 down_pipe=False) if down_pipe and dual_pipelines else None,
                             interleaved_stages=[],
                             chunks=chunks,
                             nvtx_tag='' if down_pipe else auto_schedule.TAG_UP_PIPE)

    def get_interleaved_pipeline_stages(down_pipe=True):
        stage_ids = rank_to_stages(rank, down_pipe=down_pipe)
        rank_interval = num_ranks_per_stage if down_pipe else -num_ranks_per_stage
        stages = []
        for i, stage_id in enumerate(stage_ids):
            if i>0:
                stage_module = get_stage_bert_for_pretraining(stage_id,
                                                              num_stages,
                                                              bert_config).to(device)
                inter_stage = PipelineStage(stage_id=stage_id,
                                            num_stages=num_stages,
                                            stage_module=stage_module,
                                            batch_sizes=(micro_batch_size, max_seq_length),
                                            pipeline_method=args.pipeline_method,
                                            recompute=recompute,
                                            prev_rank=(rank-rank_interval+world_size)%world_size if stage_id > 0 else None,
                                            next_rank=(rank+rank_interval)%world_size if stage_id < num_stages-1 else None,
                                            rank=rank,
                                            grad_sync_group=grad_sync_groups[stage_id],
                                            is_up_pipe=not down_pipe,
                                            up_pipe_stage=None,
                                            interleaved_stages=[],
                                            chunks=chunks,
                                            nvtx_tag='' if down_pipe else auto_schedule.TAG_UP_PIPE)
                stages.append(inter_stage)


        first_stage_id = stage_ids[0]
        stage_module = get_stage_bert_for_pretraining(first_stage_id,
                                                      num_stages,
                                                      bert_config).to(device)

        return PipelineStage(stage_id=first_stage_id,
                             num_stages=num_stages,
                             stage_module=stage_module,
                             batch_sizes=(micro_batch_size, max_seq_length),
                             pipeline_method=args.pipeline_method,
                             recompute=recompute,
                             prev_rank=(rank-rank_interval+world_size)%world_size if first_stage_id > 0 else None,
                             next_rank=(rank+rank_interval)%world_size if first_stage_id < num_stages-1 else None,
                             rank=rank,
                             grad_sync_group=grad_sync_groups[first_stage_id],
                             is_up_pipe=not down_pipe,
                             up_pipe_stage=None,
                             interleaved_stages=stages,
                             chunks=chunks,
                             nvtx_tag='' if down_pipe else auto_schedule.TAG_UP_PIPE)

    if interleaved_pipelines:
        stage = get_interleaved_pipeline_stages()
    else:
        stage = get_pipeline_stage()

    is_stage_master = rank % num_ranks_per_stage == 0

    # Prepare BERT dataset
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=args.do_lower_case)
    train_dataset = BERTDataset(args.corpus_path,
                                tokenizer,
                                seq_len=max_seq_length,
                                corpus_lines=args.corpus_lines,
                                encoding='latin-1',
                                on_memory=args.on_memory)

    def get_train_loader(down_pipe=True):
        sampler = None
        if num_replicas > 1:
            rank_in_replicas = rank_in_stage = rank % num_ranks_per_stage
            if dual_pipelines:
                rank_in_replicas = 2 * rank_in_stage + int(not down_pipe)
            sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank_in_replicas)
        return DataLoader(train_dataset,
                          sampler=sampler,
                          batch_size=micro_batch_size,
                          drop_last=True,
                          num_workers=args.num_workers)

    train_loader = get_train_loader()
    train_loader_for_up_pipe = get_train_loader(down_pipe=False) if dual_pipelines else None

    # Set the number of optimization steps and epochs
    num_micro_batches_per_step = num_stages * args.gradient_accumulation_steps
    if dual_pipelines:
        num_micro_batches_per_step //= 2  # each pipeline handles half micro_batches
    total_num_micro_batches_per_step = num_replicas * num_micro_batches_per_step
    total_num_samples_per_step = total_num_micro_batches_per_step * micro_batch_size
    max_steps_per_epoch = len(train_dataset) // total_num_samples_per_step
    num_steps = args.num_optimization_steps
    if num_steps is None:
        assert args.num_epochs, 'num_optimization_steps or num_epochs needs to be specified.'
        num_epochs = args.num_epochs
        num_steps = max_steps_per_epoch * args.num_epochs
    else:
        total_num_samples = num_steps * total_num_samples_per_step
        num_epochs = math.ceil(total_num_samples / len(train_dataset))

    ngd_schedule = None
    first_half = rank_to_stage(rank) // (num_stages // 2) == 0
    if args.ngd_schedule_path is not None:
        with open(args.ngd_schedule_path, 'rb') as f:
            ngd_schedules = pickle.load(f)
        assert len(ngd_schedules) == num_stages
        if dual_pipelines:
            if first_half:
                ngd_schedule = ngd_schedules[rank_to_stage(rank)]
            else:
                ngd_schedule = ngd_schedules[rank_to_stage(rank, down_pipe=False)]
        elif interleaved_pipelines:
            print("ngd not supported for interleaved pipelines")
        else:
            ngd_schedule = ngd_schedules[rank_to_stage(rank)]

    # Prepare natural gradient preconditioners
    ngd = ngd_up_pipe = None
    kfac_params = []
    non_ngd_params = []
    ngd_modules = []
    if is_ngd_training:
        def get_ngd(module, nvtx_tag='', down_pipe=True):
            module_partitions = None
            stage_id = rank_to_stage(rank, down_pipe=down_pipe)
            if num_replicas > 1:
                bert_layers = [m for m in module.modules() if isinstance(m, BertLayer)]
                partition_size = int(len(bert_layers) / num_replicas)  # floor
                if partition_size > 0:
                    module_partitions = []
                    for i in range(num_replicas):
                        module_list = nn.ModuleList(bert_layers[partition_size * i: partition_size * i + partition_size])
                        module_partitions.append([m for m in module_list.modules() if isinstance(m, nn.Linear)])
            return asdl.EmpiricalNaturalGradient(module,
                                                 fisher_shape=[(nn.Linear, asdl.SHAPE_KRON),
                                                               (nn.LayerNorm, asdl.SHAPE_UNIT_WISE),
                                                               (nn.Embedding, asdl.SHAPE_KRON)],
                                                 damping=args.damping,
                                                 grad_scale=1/total_num_micro_batches_per_step,
                                                 ignore_modules=['cls', nn.LayerNorm, nn.Embedding, 'pooler'],
                                                 sync_group=grad_sync_groups[stage_id],
                                                 sync_group_ranks=stage_to_ranks[stage_id],
                                                 module_partitions=module_partitions,
                                                 record_mode=args.record_ngd,
                                                 nvtx_tag=nvtx_tag)

        def register_params(_ngd: asdl.NaturalGradient):
            global kfac_params, non_ngd_params, ngd_modules
            for module in _ngd.model.modules():
                if module in _ngd.modules_for(asdl.SHAPE_KRON):
                    kfac_params += [p for p in module.parameters() if p.requires_grad]
                if module in _ngd.modules_for_curvature:
                    ngd_modules.append(module)
                else:
                    non_ngd_params += list(module.parameters())

        ngd = get_ngd(stage.stage_module)
        register_params(ngd)
        if dual_pipelines:
            ngd_up_pipe = get_ngd(stage.up_pipe_stage.stage_module, nvtx_tag=auto_schedule.TAG_UP_PIPE, down_pipe=False)
            register_params(ngd_up_pipe)

    # Prepare optimizers
    def get_optimizer(module):
        ngd_decay_param_group = {'params': [], 'weight_decay': args.weight_decay, 'b2': -1,
                                 'lr': args.ngd_learning_rate, 'max_grad_norm': args.ngd_max_grad_norm}
        ngd_no_decay_param_group = {'params': [], 'weight_decay': 0., 'b2': -1,
                                    'lr': args.ngd_learning_rate, 'max_grad_norm': args.ngd_max_grad_norm}
        decay_param_group = {'params': [], 'weight_decay': args.weight_decay}
        no_decay_param_group = {'params': [], 'weight_decay': 0.}
        for m in module.modules():
            if isinstance(m, nn.LayerNorm):
                if m in ngd_modules:
                    ngd_no_decay_param_group['params'] += list(m.parameters())
                else:
                    no_decay_param_group['params'] += list(m.parameters())
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                if hasattr(m, 'bias') and m.bias is not None:
                    if m in ngd_modules:
                        ngd_no_decay_param_group['params'].append(m.bias)
                    else:
                        no_decay_param_group['params'].append(m.bias)
                if m in ngd_modules:
                    ngd_decay_param_group['params'].append(m.weight)
                else:
                    decay_param_group['params'].append(m.weight)
        params = [decay_param_group, no_decay_param_group]
        if is_ngd_training:
            params += [ngd_decay_param_group, ngd_no_decay_param_group]
        return BertAdam(params,
                        lr=args.adam_learning_rate,
                        b1=args.beta1,
                        warmup=args.warmup_proportion,
                        t_total=num_steps,
                        max_grad_norm=args.adam_max_grad_norm)


    optimizers = [get_optimizer(stage.stage_module)]
    if dual_pipelines:
        optimizers.append(get_optimizer(stage.up_pipe_stage.stage_module))

    if interleaved_pipelines:
        for inter_stage in stage.interleaved_stages:
            optimizers.append(get_optimizer(inter_stage.stage_module))


    if not is_ngd_training:
        unused_keys = ['ngd_learning_rate', 'ngd_max_grad_norm', 'damping']
        for key in unused_keys:
            dict_args.pop(key)

    dist.barrier()
    if is_master:
        if args.wandb:
            wandb.init(entity=os.getenv('WANDB_ENTITY'),
                       project=os.getenv('WANDB_PROJECT'))
            wandb.config.update(dict_args)
        print('============================')
        print(f'pipeline_method: {args.pipeline_method}')
        print(f'num_epochs: {num_epochs}')
        print(f'num_optimization_steps: {num_steps}')
        print(f'world_size: {world_size}')
        print(f'num_replica: {num_replicas}')
        print(f'num_micro_batches_per_step: {num_micro_batches_per_step}')
        print(f'recompute: {recompute}')
        for _stage_id in range(num_stages):
            print(f'stage{_stage_id}: ranks {stage_to_ranks[_stage_id]}')
        print('----------------------------')
        for key, value in dict_args.items():
            print(f'{key}: {value}')
        print('============================')

    if args.profile:
        with torch.cuda.profiler.profile():
            main()
    else:
        main()
