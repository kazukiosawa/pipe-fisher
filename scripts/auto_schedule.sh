#!/bin/bash

#model=bert-base
model=bert-large
#pipeline='gpipe'
#pipeline='1f1b'
pipeline='chimera'
stages=8
ngpus=8
microbs=32
acc=1

base_dir=bert_prof
name=${model}_${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}_kfac
main_event_text=one_ngd_step

sqlite_paths=$(find ${base_dir} -type f -name "${name}_node*.sqlite" | sort )
for sqlite_path in $sqlite_paths
do
    pickle_path_timeline=${base_dir}/$(basename ${sqlite_path} | cut -f 1 -d '.' )_timeline.pickle
    echo parse $sqlite_path
    python parse_nvtx_events.py \
        $sqlite_path \
        --pickle_path_timeline $pickle_path_timeline \
        --ignore_first_event \
        --main_event_indices '7' \
        --event_keywords call_forward,call_backward,cov_kron_A,cov_kron_B,inv_kron_A,inv_kron_B,sync_kron_A,sync_kron_B \
        --main_event_text $main_event_text
    rm -f $sqlite_path
    nsys_path=${base_dir}/$(basename ${sqlite_path} | cut -f 1 -d '.' ).nsys-rep
    rm -f $nsys_path
done
pickle_paths=""
for pickle_path in $(find ${base_dir} -type f -name "${name}_node*_timeline.pickle" | sort )
do
    pickle_paths+="${pickle_path},"
done
python auto_schedule.py $pickle_paths \
    --save_path=${base_dir}/${name}_schedule.pickle \
    --print_workloads \
