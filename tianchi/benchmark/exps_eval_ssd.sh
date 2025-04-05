
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

record_one_eval() {
    path_cache_cap="${cache_sizes// /_}"
    rm -rf /home/code/aoxy/DeepRec/tianchi/checkpoints/$model_name/eval/
    echo "Initial disk read, written sectors(512 bytes): $(get_disk_read_written_sectors)" > $log_dir/eval_disk_usage/disk_usage_$path_cache_cap.txt
    python3 train.py --no_eval --eval_only=True --cache_sizes=$cache_sizes --storage_type=$storage_type 1> $log_dir/eval_dlrm_log/dlrm_log_$path_cache_cap.txt 2>&1 &
    cpp_pid=$!
    top -b -d 1 -p $cpp_pid > $log_dir/eval_memory_usage/memory_usage_$path_cache_cap.txt &
    top_pid=$!
    wait $cpp_pid
    kill $top_pid
    echo "Final disk read, written sectors(512 bytes): $(get_disk_read_written_sectors)" >> $log_dir/eval_disk_usage/disk_usage_$path_cache_cap.txt
    log_message "Evaluate with $storage_type and [$cache_sizes] MB Cache done. $(pwd)"
    rm -rf temp_emb/*
}

warm_up() {
    log_dir=temp_metrics
    mkdir -p $log_dir/eval_disk_usage
    mkdir -p $log_dir/eval_dlrm_log
    mkdir -p $log_dir/eval_memory_usage
    storage_type="DRAM"
    cache_sizes=00
    record_one_eval
    rm -rf $log_dir
}

cd /home/code/aoxy/DeepRec/tianchi/$model_name/

# warm_up
# echo "Warm Up done."

declare -A MODEL_CONFIG=(
    [DLRM]="3000 4000 5000 6000"
    [MMoE]="510"
    [WDL]="4000 6000"
    [DIEN]="650"
)

cache_sizes_ls=(${MODEL_CONFIG["${model_name:-DLRM}"]})


for rep in {0..0}
do
    log_dir=metrics/DRAM_SSDHASH/metrics_$rep
    mkdir -p $log_dir/eval_disk_usage
    mkdir -p $log_dir/eval_dlrm_log
    mkdir -p $log_dir/eval_memory_usage

    storage_type="DRAM"
    cache_sizes=00
    record_one_eval

    storage_type="DRAM_SSDHASH"
    for cache_sizes in "${cache_sizes_ls[@]}"
    do
        record_one_eval
    done
done
