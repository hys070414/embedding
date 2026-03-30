"""
全量实验运行脚本（非交互式）
- 删除旧缓存，用全量1787道题重新生成token embeddings和results_data.pkl
- 不需要训练注意力层
"""
import sys, io
# Windows GBK终端兼容
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, numpy as np

# 删除旧缓存，强制重新生成
CACHE_FILE = "token_embeddings_cache.npy"
if os.path.exists(CACHE_FILE):
    os.remove(CACHE_FILE)
    print(f"已删除旧缓存: {CACHE_FILE}")

from config import *
from experiment import run_experiment
from visualization import plot_heatmaps

print("=" * 60)
print("全量实验 (Token × Chunk) - 1787道化学题目")
print("指标: EOSk, Recall@10, Separation, Silhouette")
print("=" * 60)

# 直接运行，不训练注意力
results, t_names, c_names = run_experiment(None, None)

print("\n实验结果矩阵形状:", results.shape)
print("Token方法:", t_names)
print("Chunk方法:", c_names)

# 可视化
plot_heatmaps(results, t_names, c_names)

# 保存结果矩阵
np.save("results_matrix.npy", results)
print("结果矩阵已保存为 results_matrix.npy")
print("\n全量实验完成！")
