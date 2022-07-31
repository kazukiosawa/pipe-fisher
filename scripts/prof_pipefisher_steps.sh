#!/bin/bash -l
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:02:00

export MASTER_ADDR=$(hostname)

#model=bert-base
model=bert-large
#pipeline='gpipe'
#pipeline='1f1b'
pipeline='chimera'
stages=8
ngpus=8
microbs=32
acc=1
export NSYS_NODE_INTERVAL=$((ngpus/stages))
export NSYS_OUTPUT=bert_prof/${model}_${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}_pipefisher
schedule_path=bert_prof/${model}_${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}_kfac_schedule.pickle

srun --wait=0 scripts/nsys_wrap.sh \
    python main_bert.py \
            --num_stages $stages \
            --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
            --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
            --corpus_lines 10000 \
            --do_lower_case \
            --bert_config_path ./configs/bert_config_${model}-uncased.json \
            --max_seq_length 128 \
            --micro_batch_size $microbs \
            --num_optimization_steps 8 \
            --gradient_accumulation_steps $acc \
            --pipeline_method $pipeline \
            --p2p_backend 'gloo' \
            --collective_backend 'nccl' \
            --profile \
            --ngd_schedule_path $schedule_path \

