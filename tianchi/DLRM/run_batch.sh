#!/bin/bash

# 循环执行命令 30 次
for i in {1..50}
do
    # 执行命令并将输出重定向到对应的文件中
    python3 train.py --data_location=/home/code/elem --smartstaged=True 1> dlrm_log_$i.txt 2>&1
done
