#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --account=g34
#SBATCH --output=results/%j.o


module load daint-gpu
conda activate py38_kfac
#source ${HOME}/myvenv/bin/activate

link_name=latest.o
if [ -L $link_name ]; then
    unlink $link_name
fi
ln -s results/${SLURM_JOB_ID}.o $link_name

export MASTER_ADDR=$(hostname)

srun python main_bert.py \
        --num_stages 2 \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --corpus_lines 4000 \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --do_lower_case \
        --bert_config_path configs/bert_config_bert-base-uncased.json \
        --max_seq_length 128 \
        --micro_batch_size 32 \
        --num_optimization_steps 8 \
        --pipeline_method '1f1b' \
        --seed 2 \
	--dist_backend 'gloo' \

