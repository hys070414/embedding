import os
import sys
from config import *
from train_attention import train_attention_model
from experiment import run_experiment
from visualization import plot_heatmaps

def main():
    print("="*50)
    print("双层拼接组合评估实验 (Token × Chunk)")
    print("指标: EOSk, Recall@5, Separation, Silhouette")
    print("="*50)

    # 可选：训练可学习注意力层
    token_attn, chunk_attn = None, None
    train_attn = input("是否训练可学习注意力层？(y/n): ").strip().lower() == 'y'
    if train_attn:
        print("\n开始训练注意力层...")
        token_attn, chunk_attn = train_attention_model()
    else:
        print("\n跳过训练，使用无参数拼接方法。")

    # 运行实验
    results, t_names, c_names = run_experiment(token_attn, chunk_attn)

    print("\n实验结果矩阵形状:", results.shape)
    print("Token方法:", t_names)
    print("Chunk方法:", c_names)

    # 可视化
    plot_heatmaps(results, t_names, c_names)

    # 保存结果矩阵
    np.save("results_matrix.npy", results)
    print("结果矩阵已保存为 results_matrix.npy")

if __name__ == "__main__":
    main()