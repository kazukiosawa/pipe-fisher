#!/bin/bash

model=bert-base
#model=bert-large
batch_sizes='1,2,4,8,16,32,64'

#model=t5-base
#model=t5-large
#batch_sizes='1,2,4,8,16,32,64'

#model=opt-125m
#model=opt-350m
#batch_sizes='1,2,4,8'

data_path=prof/${model}_time_memory.csv

fig_path=figs/${model}_throughput2_gpipe_1f1b.pdf
python scripts/plot_throughput2.py \
    --data_path $data_path \
    --fig_path $fig_path \
    --batch_sizes $batch_sizes \

imgcat $fig_path

fig_path=figs/${model}_throughput2_chimera.pdf
python scripts/plot_throughput2.py \
    --data_path $data_path \
    --fig_path $fig_path \
    --chimera \
    --batch_sizes $batch_sizes \

imgcat $fig_path
