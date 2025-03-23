#!/bin/bash

# docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/tianchi/DLRM/run_batch.sh"

record_one_train() {
    rm -rf ./result/*
    path_cache_cap="${cache_cap// /_}"
    python3 train.py --data_location=/home/code/elem --smartstaged=True --storage_type=$storage_type --cache_sizes $cache_cap --no_eval 1> $log_dir/train_dlrm_log/dlrm_log_$path_cache_cap.txt 2>&1
    echo "Train with $storage_type and [$cache_cap] x100MB Cache done."
    rm -rf temp_emb/*
}


cd /home/code/aoxy/DeepRec/tianchi/DLRM/


log_dir=metrics/DRAM_SSDHASH/metrics_0
mkdir -p $log_dir/train_dlrm_log

storage_type="DRAM"
cache_cap="1 1 1 1"
record_one_train

storage_type="DRAM_SSDHASH"
cache_cap="10240 10240 10240 10240"
record_one_train

# EMBEDDING_COLS = ['user_id', 'district_id', 'times', 'timediff_list']
# SIZE = [1343323, 2557, 82589, 4987656]
# SIZE * 1552 / 1024 / 1024 = [1989, 4, 123, 7383]

for user_id_size in 398 597 796 994 1193 1392 1591 1790 1989
do
    cache_cap="$user_id_size 0 0 0"
    record_one_train
done

for district_id_size in 1 2 3 4
do
    cache_cap="0 $district_id_size 0 0"
    record_one_train
done

for times_size in 25 37 49 62 74 86 98 111 123
do
    cache_cap="0 0 $times_size 0"
    record_one_train
done

for timediff_list_size in 1477 2215 2953 3692 4430 5168 5906 6645 7383
do
    cache_cap="0 0 0 $timediff_list_size"
    record_one_train
done