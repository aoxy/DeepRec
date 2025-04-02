#!/bin/bash

model_name=$1
echo "Model Name: $model_name"
git_head=$2
gitlab_dir=/home/axy/code/aoxy/air_data3
deeprec_dir=/home/axy/code/aoxy/DeepRec
metrics_dir=tianchi/$model_name/metrics
tar_dir=$deeprec_dir/tianchi/benchmark/archives
cd $deeprec_dir
mkdir -p $tar_dir

docker stop axynetp
docker update --memory "160g" --memory-swap "170g" axynetp
docker start axynetp

function log_message() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
}

function check_tar_exists() {
    local tar_file="$1"
    local task_name="$2"
    if [ -f "$tar_file" ]; then
        log_message "Tar file $tar_file already exists, skipping task: $task_name."
        return 0
    else
        return 1
    fi
}

function run_task() {
    local memory="$1"
    local script="$2"
    local tar_name="$3"
    local task_name="$4"

    # 计算 swap 值
    local swap=$(echo "$memory + 12" | bc)

    # 构建完整的tar文件路径
    local tar_file="$tar_dir/$tar_name"

    # 检查tar文件是否存在
    if check_tar_exists "$tar_file" "$task_name"; then
        return 0
    fi

    log_message "=================================== Starting task: $task_name ==================================="
    log_message "Memory: ${memory}g, MemorySwap: ${swap}g, Script: $script $model_name, Output Tar: $tar_name"

    log_message "Stopping and updating container axynetp..."
    docker stop axynetp
    docker update --memory "${memory}g" --memory-swap "${swap}g" axynetp
    docker start axynetp

    log_message "Executing script: $script $model_name"
    docker exec axynetp /bin/bash -c "$script $model_name"

    log_message "Creating tar archive: $tar_name"
    tar -czf "$tar_file" "$metrics_dir"

    cp "$tar_file" "$gitlab_dir"
    cd "$gitlab_dir"
    git add "$tar_name"
    git commit -m "[$git_head] Add $tar_name"
    sudo -u axy git push
    cd $deeprec_dir
    
    log_message "Removing metrics directory: $metrics_dir"
    rm -rf "$metrics_dir"

    log_message "Task completed: $task_name"
}

Train_LMem=8
Train_MMem=7
Train_SMem=6
Eval_LMem=6
Eval_MMem=5
Eval_SMem=4

case "$model_name" in
    DLRM)
        Train_LMem=8
        Train_MMem=7
        Train_SMem=6
        Eval_LMem=6
        Eval_MMem=5
        Eval_SMem=4
        ;;
    MMoE)
        Train_LMem=1.2
        Train_MMem=0.9
        Train_SMem=0.6
        Eval_LMem=6
        Eval_MMem=5
        Eval_SMem=4
        ;;
    WDL)
        Train_LMem=6.4
        Train_MMem=5.3
        Train_SMem=4.2
        Eval_LMem=6
        Eval_MMem=5
        Eval_SMem=4
        ;;
    DIEN)
        Train_LMem=1.6
        Train_MMem=1.2
        Train_SMem=0.8
        Eval_LMem=6
        Eval_MMem=5
        Eval_SMem=4
        ;;
esac

echo "Train    L: $Train_LMem, M: $Train_MMem, S: $Train_SMem" > $tar_dir/config.txt
echo "Eval     L: $Eval_LMem, M: $Eval_MMem, S: $Eval_SMem" >> $tar_dir/config.txt


# DRAM train + eval + timeline
run_task "160" "/home/code/aoxy/DeepRec/tianchi/benchmark/exps_dram.sh" "metrics_dram.tar.gz" "DRAM Train + Eval + Timeline ($Train_LMem)"

run_task "$Train_LMem" "/home/code/aoxy/DeepRec/tianchi/benchmark/exps_train_ssd_l.sh" "metrics_l_ssd_train.tar.gz" "SSD Train ($Train_LMem)"
run_task "$Eval_LMem" "/home/code/aoxy/DeepRec/tianchi/benchmark/exps_eval_ssd.sh" "metrics_l_ssd_eval.tar.gz" "SSD Eval ($Eval_LMem)"

run_task "$Train_MMem" "/home/code/aoxy/DeepRec/tianchi/benchmark/exps_train_ssd_m.sh" "metrics_m_ssd_train.tar.gz" "SSD Train ($Train_MMem)"
run_task "$Eval_MMem" "/home/code/aoxy/DeepRec/tianchi/benchmark/exps_eval_ssd.sh" "metrics_m_ssd_eval.tar.gz" "SSD Eval ($Eval_MMem)"

run_task "$Train_SMem" "/home/code/aoxy/DeepRec/tianchi/benchmark/exps_train_ssd_s.sh" "metrics_s_ssd_train.tar.gz" "SSD Train ($Train_SMem)"
run_task "$Eval_SMem" "/home/code/aoxy/DeepRec/tianchi/benchmark/exps_eval_ssd.sh" "metrics_s_ssd_eval.tar.gz" "SSD Eval ($Eval_SMem)"

