#!/bin/bash -l
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --account=g34
#SBATCH --output=bert_large_1f1b_16b_4wx4d.txt


module load daint-gpu
#module load PyTorch

__conda_setup="$('/project/g34/shigang/anaconda38/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/project/g34/shigang/anaconda38/etc/profile.d/conda.sh" ]; then
        . "/project/g34/shigang/anaconda38/etc/profile.d/conda.sh"
    else
        export PATH="/project/g34/shigang/anaconda38/bin:$PATH"
    fi
fi
unset __conda_setup

which nvcc
nvidia-smi

which python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export NCCL_DEBUG=INFO
#export NCCL_IB_HCA=ipogif0
#export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_SOCKET_IFNAME=ipogif0

#export CUDA_LAUNCH_BLOCKING=1

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
echo $MASTER_ADDR

srun python main_bert.py \
        --module models.bert24.depth=4 \
        --max_seq_length 128 \
        --train_batch_size 16 \
        --train_path ./bert_dataset/wikipedia.segmented.nltk.txt \
        --bert_config_path configs/bert_config_bert-large-uncased.json \
        --vocab_path ./bert_dataset/bert-large-uncased-vocab.txt \
        --do_train \
        --do_lower_case \
        --num_minibatches 128 \
        --gradient_accumulation_steps 1 --oneFoneB --config_path tests/depth=4/conf_16nodes.json
