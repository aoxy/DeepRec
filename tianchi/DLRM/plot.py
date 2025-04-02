import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_distribution(name, distances):
    # 过滤首次出现的情况
    
    plt.figure(figsize=(10,6))
    plt.hist(distances, bins=50, log=True)
    plt.xlabel('Reuse Distance')
    plt.ylabel('Frequency')
    # plt.title('Reuse Distance Distribution')
    plt.savefig(f"figures/{name}_distance.png",  bbox_inches="tight", dpi=600)
    plt.savefig(f"figures/{name}_distance.pdf",  bbox_inches="tight")

def plot_visit(name, visit):
    # 过滤首次出现的情况
    v2id = dict()
    for v, step in visit:
        if v not in v2id:
            v2id[v] = len(v2id) + 1
    x = [v2id[v] for v, _ in  visit]
    y = [step for _,step in  visit]
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, s=1, alpha=0.05)
    plt.xlabel('Embedding')
    plt.ylabel('Step')
    # plt.title('Reuse Distance Distribution')
    plt.savefig(f"figures/sample_{name}_visit.png",  bbox_inches="tight", dpi=600)
    plt.savefig(f"figures/sample_{name}_visit.pdf",  bbox_inches="tight")

reuse_distances = pickle.load(open("reuse_distances.pk", "rb"))
EMBEDDING_COLS = pickle.load(open("EMBEDDING_COLS.pk", "rb"))
visit_list = pickle.load(open("visit_list.pk", "rb"))

for i, reuse_distance in enumerate(reuse_distances):
    plot_distribution(EMBEDDING_COLS[i], reuse_distance)

MAX_LENGTH = 1_000_000

for i, visit in enumerate(visit_list):
    if len(visit) > MAX_LENGTH:
        # 生成等间隔采样索引（对齐采样）
        indices = np.linspace(0, len(visit)-1, num=MAX_LENGTH, dtype=int)
        # 执行高效向量化采样
        visit = np.array(visit)[indices]  # 假设visit是numpy数组
    print(EMBEDDING_COLS[i], len(visit))
    plot_visit(EMBEDDING_COLS[i], visit)


