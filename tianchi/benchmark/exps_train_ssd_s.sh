
model_name=$1
echo "Model Name: $model_name"

get_disk_read_written_sectors() {
    diskstats=$(cat /proc/diskstats | grep "nvme0n1p3"  | awk '{print $6, $10}')
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
    python3 train.py --cache_sizes $cache_sizes --storage_type=$storage_type --no_eval 1> $log_dir/train_dlrm_log/dlrm_log_$path_cache_cap.txt 2>&1 &
    cpp_pid=$!
    top -b -d 1 -p $cpp_pid > $log_dir/train_memory_usage/memory_usage_$path_cache_cap.txt &
    top_pid=$!
    wait $cpp_pid
    kill $top_pid
    echo "Final disk read, written sectors(512 bytes): $(get_disk_read_written_sectors)" >> $log_dir/train_disk_usage/disk_usage_$path_cache_cap.txt
    log_message "$model_name Train with $storage_type and [$cache_sizes] MB Cache done."
    rm -rf temp_emb/*
}

warm_up() {
    log_dir=temp_metrics
    mkdir -p $log_dir/train_disk_usage
    mkdir -p $log_dir/train_dlrm_log
    mkdir -p $log_dir/train_memory_usage
    storage_type="DRAM"
    cache_sizes=00
    record_one_train
    rm -rf $log_dir
}

cd /home/code/aoxy/DeepRec/tianchi/$model_name/

# warm_up
# echo "Warm Up done."
# [DLRM]="524 2 33 1941|625 625 625 625|625 4 120 1751|825 4 120 1551|425 4 120 1951|1125 4 120 1251|325 4 120 2051"
declare -A MODEL_CONFIG=(
    [DLRM]="2400"
    [MMoE]="510"
    [WDL]="2100"
    [DIEN]="650"
)


for rep in {0..0}
do
    log_dir=metrics/DRAM_SSDHASH/metrics_$rep
    mkdir -p $log_dir/train_disk_usage
    mkdir -p $log_dir/train_dlrm_log
    mkdir -p $log_dir/train_memory_usage

    storage_type="DRAM"
    cache_sizes=00
    record_one_train

    storage_type="DRAM_SSDHASH"
    IFS='|' read -ra config_groups <<< "${MODEL_CONFIG[$model_name]}"
    for group in "${config_groups[@]}"; do
        IFS=' ' read -ra cache_sizes_ls <<< "$group"
        cache_sizes="${cache_sizes_ls[*]}"
        record_one_train
    done
done
