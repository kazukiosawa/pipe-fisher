#!/bin/bash

arch=bert
model=bert_base
#model=bert_large
config=configs/bert_config_${model}-uncased.json

python scripts/prof_kron.py \
    --arch $arch \
    --config $config \
    --n_batches 1 \
    --num_iters 10 \
    --num_warmups 5 \
    --max_seq_len 128 \
    --batch_sizes '1,2,4,8,16,32' \
