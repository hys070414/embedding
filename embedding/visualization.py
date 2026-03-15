import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import RETRIEVAL_K

def plot_heatmaps(results, t_names, c_names, save_prefix="composition"):
    metrics = ['EOSk', f'Recall@{RETRIEVAL_K}', 'Separation', 'Silhouette']
    for m_idx, metric in enumerate(metrics):
        plt.figure(figsize=(12, 10))
        data = results[:, :, m_idx]
        data_plot = np.nan_to_num(data, nan=0.0)
        sns.heatmap(data_plot, annot=True, fmt='.3f',
                    xticklabels=c_names, yticklabels=t_names,
                    cmap='viridis', cbar_kws={'label': metric})
        plt.xlabel('Chunk-level Methods')
        plt.ylabel('Token-level Methods')
        plt.title(f'{metric} for Token×Chunk Combinations')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{metric}.png', dpi=150)
        plt.show()
        print(f"热力图已保存为 {save_prefix}_{metric}.png")