#!/bin/bash

# docker stop axynetp
# docker start axynetp
# docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/reinstall_deeprc.sh" > script_output_hitrate.log 2>&1 &
# docker exec axynetp /bin/bash -c "/home/code/aoxy/DeepRec/tianchi/run_models.sh" > script_output_4models.log 2>&1 &


get_disk_read_written_sectors() {
    diskstats=$(cat /proc/diskstats | grep "sda"  | awk '{print $6, $10}')
    echo $diskstats
}

function log_message() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
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
    log_message "Train with $storage_type and [$cache_sizes] x100MB Cache done. $(pwd)"
    rm -rf temp_emb/*
}

record_one_eval() {
    path_cache_cap="${cache_sizes// /_}"
    echo "Initial disk read, written sectors(512 bytes): $(get_disk_read_written_sectors)" > $log_dir/eval_disk_usage/disk_usage_$path_cache_cap.txt
    python3 train.py --no_eval --eval_only=True --cache_sizes=$cache_sizes --storage_type=$storage_type 1> $log_dir/eval_dlrm_log/dlrm_log_$path_cache_cap.txt 2>&1 &
    cpp_pid=$!
    top -b -d 1 -p $cpp_pid > $log_dir/eval_memory_usage/memory_usage_$path_cache_cap.txt &
    top_pid=$!
    wait $cpp_pid
    kill $top_pid
    echo "Final disk read, written sectors(512 bytes): $(get_disk_read_written_sectors)" >> $log_dir/eval_disk_usage/disk_usage_$path_cache_cap.txt
    log_message "Evaluate with $storage_type and [$cache_sizes] x100MB Cache done."
    rm -rf temp_emb/*
}

prepare_dir() {
    mkdir -p $log_dir/train_disk_usage
    mkdir -p $log_dir/train_dlrm_log
    mkdir -p $log_dir/train_memory_usage
    mkdir -p $log_dir/eval_disk_usage
    mkdir -p $log_dir/eval_dlrm_log
    mkdir -p $log_dir/eval_memory_usage
}

log_dir=metrics4m/DRAM_SSDHASH/metrics_0
###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/DLRM/
prepare_dir

storage_type="DRAM"
cache_sizes="1"
# record_one_train
record_one_eval

# storage_type="DRAM_SSDHASH"
# cache_sizes="15240"
# record_one_train


###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/MMoE/
prepare_dir

storage_type="DRAM"
cache_sizes="1"
# record_one_train
record_one_eval

# storage_type="DRAM_SSDHASH"
# cache_sizes="5240"
# record_one_train


###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/WDL/
prepare_dir

storage_type="DRAM"
cache_sizes="1"
# record_one_train
record_one_eval

# storage_type="DRAM_SSDHASH"
# cache_sizes="15240"
# record_one_train


###################################################################################################

cd /home/code/aoxy/DeepRec/tianchi/DIEN/
prepare_dir

storage_type="DRAM"
cache_sizes="1"
# record_one_train
record_one_eval

# storage_type="DRAM_SSDHASH"
# cache_sizes="5240"
# record_one_train


###################################################################################################