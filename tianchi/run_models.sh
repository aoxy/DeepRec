#!/bin/bash

# docker stop axynetp
# docker start axynetp
# docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/reinstall_deeprc.sh" > script_output_hitrate.log 2>&1 &
# docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/tianchi/run_models.sh" > script_output_4models.log 2>&1 &


get_disk_read_written_sectors() {
    diskstats=$(cat /proc/diskstats | grep "nvme0n1p3"  | awk '{print $6, $10}')
    echo $diskstats
}

record_one_train() {
    rm -rf ./result/*
    path_cache_cap="${cache_sizes// /_}"
    echo "Initial disk read, written sectors(512 bytes): $(get_disk_read_written_sectors)" > $log_dir/train_disk_usage/disk_usage_$path_cache_cap.txt
    python3 train.py --no_eval --cache_sizes $cache_sizes --storage_type=$storage_type 1> $log_dir/train_dlrm_log/dlrm_log_$path_cache_cap.txt 2>&1 &
    cpp_pid=$!
    top -b -d 1 -p $cpp_pid > $log_dir/train_memory_usage/memory_usage_$path_cache_cap.txt &
    top_pid=$!
    wait $cpp_pid
    kill $top_pid
    echo "Final disk read, written sectors(512 bytes): $(get_disk_read_written_sectors)" >> $log_dir/train_disk_usage/disk_usage_$path_cache_cap.txt
    echo "Train with $storage_type and [$cache_sizes] x100MB Cache done. $(pwd)"
    rm -rf temp_emb/*
}

###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/DLRM/
log_dir=metrics4m/DRAM_SSDHASH/metrics_0
mkdir -p $log_dir/train_disk_usage
mkdir -p $log_dir/train_dlrm_log
mkdir -p $log_dir/train_memory_usage

storage_type="DRAM"
cache_sizes="1"
record_one_train

storage_type="DRAM_SSDHASH"
cache_sizes="10240"
record_one_train

# EMBEDDING_COLS = ['user_id', 'district_id', 'times', 'timediff_list']
# SIZE = [1343323, 2557, 82589, 4987656]
# SIZE * 1552 / 1024 / 1024 = [1989, 4, 123, 7383]
mkdir -p /home/code/aoxy/DeepRec/models_metrics/DLRM
mv $log_dir /home/code/aoxy/DeepRec/models_metrics/DLRM

###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/MMoE/
log_dir=metrics4m/DRAM_SSDHASH/metrics_0
mkdir -p $log_dir/train_disk_usage
mkdir -p $log_dir/train_dlrm_log
mkdir -p $log_dir/train_memory_usage

storage_type="DRAM"
cache_sizes="1"
record_one_train

storage_type="DRAM_SSDHASH"
cache_sizes="10240"
record_one_train

# EMBEDDING_COLS = ['user_id', 'district_id', 'times', 'timediff_list']
# SIZE = [1343323, 2557, 82589, 4987656]
# SIZE * 1552 / 1024 / 1024 = [1989, 4, 123, 7383]
mkdir -p /home/code/aoxy/DeepRec/models_metrics/MMoE
mv $log_dir /home/code/aoxy/DeepRec/models_metrics/MMoE

###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/WDL/
log_dir=metrics4m/DRAM_SSDHASH/metrics_0
mkdir -p $log_dir/train_disk_usage
mkdir -p $log_dir/train_dlrm_log
mkdir -p $log_dir/train_memory_usage

storage_type="DRAM"
cache_sizes="1"
record_one_train

storage_type="DRAM_SSDHASH"
cache_sizes="10240"
record_one_train

# EMBEDDING_COLS = ['user_id', 'district_id', 'times', 'timediff_list']
# SIZE = [1343323, 2557, 82589, 4987656]
# SIZE * 1552 / 1024 / 1024 = [1989, 4, 123, 7383]
mkdir -p /home/code/aoxy/DeepRec/models_metrics/WDL
mv $log_dir /home/code/aoxy/DeepRec/models_metrics/WDL

###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/DIEN/
log_dir=metrics4m/DRAM_SSDHASH/metrics_0
mkdir -p $log_dir/train_disk_usage
mkdir -p $log_dir/train_dlrm_log
mkdir -p $log_dir/train_memory_usage

storage_type="DRAM"
cache_sizes="1"
record_one_train

storage_type="DRAM_SSDHASH"
cache_sizes="10240"
record_one_train

# EMBEDDING_COLS = ['user_id', 'district_id', 'times', 'timediff_list']
# SIZE = [1343323, 2557, 82589, 4987656]
# SIZE * 1552 / 1024 / 1024 = [1989, 4, 123, 7383]
mkdir -p /home/code/aoxy/DeepRec/models_metrics/DIEN
mv $log_dir /home/code/aoxy/DeepRec/models_metrics/DIEN

###################################################################################################