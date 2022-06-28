# PipeFisher

## Setup

### Data preparation
https://github.com/microsoft/AzureML-BERT/blob/master/docs/dataprep.md

Please store `wikipedia.segmented.nltk.txt` file under the `bert_data/` directory.

### Installation
```
pip install -r requirements.txt
pip install asdfghjkl/
```
For training, we use `apex.optimizers.FusedLAMB` of [NVIDIA's Apex library](https://github.com/NVIDIA/apex). Please follow the [instruction](https://github.com/NVIDIA/apex#installation) for installing `apex`. 

For profiling, we use [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems). Please make sure you can execute `nsys` command.

Our scripts are intended to run through the SLURM workload manager on a GPU cluster with 1 GPU per node.

## Training

Phase 1 pretraining of BERT-Base on the English Wikipedia by NVLAMB on 32 GPUs
```
sbatch scripts/train.sh
```

Phase 1 pretraining of BERT-Base on the English Wikipedia by K-FAC on 32 GPUs
```
sbatch scripts/train_kfac.sh
```

<img width="677" alt="image" src="https://user-images.githubusercontent.com/7961228/170479576-ce39e926-7bf3-4f11-8918-145278596917.png">


## Profiling

### Step 0. Profiling **Chimera** with 8 stages for BERT-Large on 8 GPUs 
```
sbatch scripts/prof_steps.sh
```
```
sh scripts/plot_cuda_timeline.sh
```
output: `bert_prof/bert-large_chimera_8stages_8gpus_microbs32_acc1.pdf`

### Step 1. Profiling **Chimera with K-FAC** with 8 stages for BERT-Large on 8 GPUs 
```
sbatch scripts/prof_kfac_steps.sh
```
```
sh scripts/plot_cuda_timeline_kfac.sh
```
output: `bert_prof/bert-large_chimera_8stages_8gpus_microbs32_acc1_kfac.pdf`

### Step 2. Automatic work assignments
```
sh scripts/auto_schedule.sh
```
output: `bert-large_chimera_8stages_8gpus_microbs32_acc1_kfac_schedule.pickle`

### Step 3. Profiling **Chimera with PipeFisher** with 8 stages for BERT-Large on 8 GPUs 
```
sbatch scripts/prof_pipefisher_steps.sh
```
```
sh scripts/plot_cuda_timeline_pipefisher.sh
```
output: `bert_prof/bert-large_chimera_8stages_8gpus_microbs32_acc1_pipefisher.pdf`

<img width="849" alt="image" src="https://user-images.githubusercontent.com/7961228/170479526-c13310e3-c644-4725-bb2a-5ef0829a1395.png">


By changing the settings of each script, you can run training/profiling on other BERT models, pipeline methods, number of pipeline stages, number of GPUs, etc.
