#!/bin/bash

#model=bert_base
#model=bert_large
model=t5-base
data_path=prof/${model}_time_memory.csv

fig_path=figs/${model}_throughput_gpipe_1f1b.pdf
python scripts/plot_throughput.py \
    --data_path $data_path \
    --fig_path $fig_path \

imgcat $fig_path

fig_path=figs/${model}_throughput_chimera.pdf
python scripts/plot_throughput.py \
    --data_path $data_path \
    --fig_path $fig_path \
    --chimera \

imgcat $fig_path
