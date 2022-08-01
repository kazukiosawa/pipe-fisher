#!/bin/bash

arch=bert
model=bert-base
#model=bert-large
config=configs/bert_config_${model}-uncased.json
max_seq_len=128
batch_sizes='1,2,4,8,16,32,64'

#arch=t5
##model=t5-base
#model=t5-large
#max_seq_len=512
#config=configs/${model}.json
#batch_sizes='1,2,4,8,16,32,64'

#arch=opt
#model=opt-125m
##model=opt-350m
#max_seq_len=2048
#config=configs/${model}.json
#batch_sizes='1,2,4,8'

csv_name=prof/${model}_time_memory.csv

python scripts/prof_kron.py \
    --arch $arch \
    --config $config \
    --n_batches 1 \
    --num_iters 10 \
    --num_warmups 5 \
    --max_seq_len $max_seq_len \
    --batch_sizes $batch_sizes \
    --csv_name $csv_name \
