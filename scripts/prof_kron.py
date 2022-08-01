import argparse
import time
from collections import deque
import csv

import numpy as np

import torch
from torch import nn
import asdfghjkl as asdl
from transformers.models.bert import BertLayer, BertConfig
from transformer_layers import OPTDecoderLayer, OPTConfig
from transformer_layers import T5Block, T5Config


parser = argparse.ArgumentParser()
parser.add_argument('--arch', choices=['bert', 'opt', 't5'], default='bert')
parser.add_argument('--config', type=str)
parser.add_argument('--batch_sizes', type=str, default='1,2,4,8,16')
parser.add_argument('--n_batches', type=int, default=4)
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--num_iters', type=int, default=10)
parser.add_argument('--num_warmups', type=int, default=5)
parser.add_argument('--csv_name', type=str, default='time_memory.csv')


def generate_batch(batch_size, seq_len, embed_size):
    """Generate batch data in (batch, sequence, embedding) order."""
    return torch.rand((batch_size, seq_len, embed_size))


def gen_attention_mask(q_seq_len, k_seq_len):
    """Generate a causal attention mask."""
    return torch.triu(
        torch.full((q_seq_len, k_seq_len), float('-inf')),
        diagonal=1)


def time_f(f):
    n_iters = args.num_iters
    n_warmups = args.num_warmups
    for _ in range(n_warmups):
        f()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.time()
        f()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    max_memory = torch.cuda.max_memory_allocated()
    times = np.array(times)
    results[f.__name__] = {'time': times.mean() * 1000, 'memory': max_memory/float(1<<20)}


def main(batch_size):

    if args.arch == 'bert':
        config_cls, block_cls = BertConfig, BertLayer
    elif args.arch == 'opt':
        config_cls, block_cls = OPTConfig, OPTDecoderLayer
    else:
        config_cls, block_cls = T5Config, T5Block
    config = config_cls.from_json_file(args.config)
    model = block_cls(config).to(device)

    x = {
        'hidden_states': generate_batch(batch_size, args.max_seq_len, config.hidden_size).to(device),
        'attention_mask': gen_attention_mask(args.max_seq_len, args.max_seq_len).to(device)
    }
    if args.arch == 'opt':
        x['attention_mask'] = x['attention_mask'].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def nothing():
        pass

    time_f(nothing)

    n_batches = args.n_batches

    def fwd_pipeline():
        queue = deque()
        for _ in range(n_batches):
            y = model(**x)[0]
            queue.append(y)

    time_f(fwd_pipeline)

    dy = generate_batch(batch_size, args.max_seq_len, config.hidden_size).to(device)

    def pipeline():
        queue = deque()
        for _ in range(n_batches):
            y = model(**x)[0]
            queue.append(y)
        for _ in range(n_batches):
            y = queue.popleft()
            y.backward(dy, retain_graph=True)

    time_f(pipeline)

    ignore_modules = [nn.LayerNorm]
    ngd = asdl.EmpiricalNaturalGradient(model,
                                        fisher_shape=[(nn.Linear, asdl.SHAPE_KRON)],
                                        ignore_modules=ignore_modules,
                                        damping=1.)


    with asdl.save_inputs_outgrads(model, ignore_modules=ignore_modules) as cxt:
        pipeline()
        numel_err = 0
        numel_kron = 0
        for module in model.modules():
            if cxt.is_operation_registered(module) and cxt.out_grads(module) is not None:
                numel_err += sum([g.numel() for g in cxt.out_grads(module)])
                cxt.calc_cov_kron(module)
                numel_kron += cxt.cov_kron(module)['A'].numel() + cxt.cov_kron(module)['B'].numel()

    def pipeline_and_curvature():
        with asdl.save_inputs_outgrads(model, ignore_modules=ignore_modules) as cxt:
            pipeline()
            ngd.update_curvature(cxt=cxt)

    pipeline_and_curvature()

    time_f(pipeline_and_curvature)

    def inv():
        ngd.update_inv()

    def precondition():
        ngd.precondition()

    time_f(inv)
    time_f(precondition)

    writer.writerow([
        batch_size,  # bs
        (results["fwd_pipeline"]["time"] - results["nothing"]["time"]) / n_batches,  # time_f
        (results["pipeline"]["time"] - results["fwd_pipeline"]["time"]) / n_batches,  # time_b
        (results["pipeline_and_curvature"]["time"] - results["pipeline"]["time"]) / n_batches,  # time_kron
        results["inv"]["time"],  # time_inv
        results["precondition"]["time"],  # time_prec
        results["nothing"]["memory"],  # mem_param
        (results["fwd_pipeline"]["memory"] - results["nothing"]["memory"]) / n_batches,  # mem_act
        results["pipeline"]["memory"] - results["fwd_pipeline"]["memory"],  # mem_peak_err
        (numel_err * 4 / float(1 << 20)) / n_batches,  # mem_save_err
        numel_kron * 4 / float(1 << 20)  # mem_kron
    ])
#    print(f'{batch_size},'
#          f'{(results["fwd_pipeline"]["time"] - results["nothing"]["time"]) / n_batches},'  # time_f
#          f'{(results["pipeline"]["time"] - results["fwd_pipeline"]["time"]) / n_batches},'  # time_b
#          f'{(results["pipeline_and_curvature"]["time"] - results["pipeline"]["time"]) / n_batches},'  # time_kron
#          f'{results["inv"]["time"]},'  # time_inv
#          f'{results["precondition"]["time"]},'  # time_prec
#          f'{results["nothing"]["memory"]},'  # mem_param
#          f'{(results["fwd_pipeline"]["memory"] - results["nothing"]["memory"]) / n_batches},'  # mem_act
#          f'{results["pipeline"]["memory"] - results["fwd_pipeline"]["memory"]},'  # mem_peak_err
#          f'{(numel_err * 4 /float(1<<20)) / n_batches},'  # mem_save_err
#          f'{numel_kron * 4 /float(1<<20)}'  # mem_kron
#    )


if __name__ == '__main__':
    args = parser.parse_args()
    assert torch.cuda.is_available()
    device = 'cuda:0'
    results = {}
    batch_sizes = [int(s) for s in args.batch_sizes.split(',')]
    header = ['bs', 'time_f', 'time_b', 'time_kron', 'time_inv', 'time_prec',
              'mem_param', 'mem_act', 'mem_peak_err', 'mem_save_err', 'mem_kron']
    with open(args.csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for bs in batch_sizes:
            print('bs', bs)
            main(bs)
