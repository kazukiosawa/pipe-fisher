#!/bin/bash -l
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:05:00

export MASTER_ADDR=$(hostname)
export MASTER_PORT=1234

model=bert-base

# phase1
bs=8192
acc=64
seq_len=128
lr=6.e-3
nsteps=7038
ckpt_steps=1750
warmup=0.08525

srun python main_bert_simple.py \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --do_lower_case \
        --bert_config_path ./bert_configs/bert_config_${model}-uncased.json \
        --num_workers 8 \
        --batch_size $bs \
        --gradient_accumulation_steps $acc \
        --weight_decay 0.01 \
        --collective_backend 'nccl' \
        --log_interval 1 \
        --seed 0 \
        --learning_rate $lr \
        --max_seq_length $seq_len \
        --num_optimization_steps $nsteps \
        --warmup_proportion $warmup \
        --ngd_training \
        --ngd_with_lamb \
        --damping 1e-3 \
        --inv_interval 10 \

