import matplotlib.pyplot as plt
import numpy as np
import re

def readMemory(mem_path):
    memory_data = []
    swap_data = []
    swap_used_extractor = re.compile(r"MiB Swap:\s+[0-9.]+ total,\s+[0-9.]+ free,\s+([0-9.]+) used")
    with open(mem_path) as f:
        lines = f.readlines()
        for log in lines:
            log = log.strip()
            swap_used_res = swap_used_extractor.findall(log)
            if swap_used_res:
                swap_data.append(float(swap_used_res[0]))
            if log.endswith("python3"):
                metrics = log.split(" ")
                metrics = [m.strip() for m in metrics]
                metrics = [m for m in metrics if len(m) > 0]
                res = metrics[5]
                if res.endswith("g"):
                    res = float(res[:-1]) * 1024
                elif res.endswith("m"):
                    res = float(res[:-1])
                else:
                    res = float(res) / 1024
                memory_data.append(res)
    return np.array(memory_data), np.array(swap_data)

# mem, swap = readMemory("models_metrics_counts/DIEN/metrics_0/train_memory_usage/memory_usage_1_1_1_1.txt") # 3174.4 110.8
# mem, swap = readMemory("models_metrics_counts/DLRM/metrics_0/train_memory_usage/memory_usage_1_1_1_1.txt") # 14233.6 1273.0
# mem, swap = readMemory("models_metrics_counts/WDL/metrics_0/train_memory_usage/memory_usage_1_1_1_1.txt") # 13721.6 111.8
mem, swap = readMemory("/mnt/data/aoxy/code/aoxy/DeepRec/tianchi/WDL/metrics4m/DRAM_SSDHASH/metrics_0/eval_memory_usage/memory_usage_1.txt") # 2252.8 0.0

# 表大小
# DLRM: 9398.620
# WDL:  11080.967
# DIEN: 613.956
# MMoE: 503.179

# 剩余
# DLRM: 14233.6 - 9398.620 = 4834.98
# WDL:  13721.6 - 11080.967 = 2640.633
# DIEN: 3174.4 - 613.956 = 2560.0
# MMoE  2252.8 - 503.179 = 1749.621

# 三个等级
# 剩余 + 表大小* 【0.35 0.25 0.15】
# DLRM: 8g 7g 6g
# WDL:  6.4g 5.3g 4.2g
# DIEN: 2.7g 2.3g 1.9g
# MMoE: 1.8g 1.5g 1.2g

# plot men and swap line
# plt.plot(mem, label='mem')
# plt.plot(swap, label='swap')
# plt.legend()
# plt.show()
print(mem.max(), swap.max())
# 12288.0, 33.8

# Eval
# 10342.4 867.5

# EMBEDDING_DIMENSIONS = 128
# LocklessHashMap Size = 4987656
# val_len_ = 1552
# 表大小 = 1552 * 4987656 = 7382.2 MiB
# Res = 12288.0 - 7382.2 = 4905.8 MiB
# Level1 = 9 GiB
# Level2 = 8 GiB
# Level3 = 7 GiB