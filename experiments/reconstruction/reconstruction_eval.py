"""
向量→文本重建评估实验
====================
将各拼接组合得到的知识点向量"重建"为文本，评估重建质量作为第5个指标。

方法1: KNN检索 (从语料库中找最相近文本)
方法2: Qwen API向量引导生成 (利用大模型生成与向量最相似的文本)

评估指标:
  - 语义余弦相似度 (Cosine Sim，主要指标)
  - ROUGE-L (词汇重叠)
  - BERTScore代理 (用SentenceTransformer的cosine similarity近似)

输出:
  - reconstruction_results.npy  (t_names × c_names × 3 个子指标)
  - reconstruction_combined.npy (综合分数矩阵)
  - reconstruction_heatmap_full.png (热力图)
  - reconstruction_report.html   (详细报告)
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pickle
import numpy as np
import random
import json
import re
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer as rouge_scorer_lib
from sentence_transformers import SentenceTransformer
import dashscope
from dashscope import Generation

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
EMBED_DIM      = 384
SAMPLE_SIZE    = 10        # 每个组合随机抽取的样本数
ENCODER_NAME   = "all-MiniLM-L6-v2"
QWEN_API_KEY   = "sk-83da80fde3fd412a97a5668b8e1f8104"
USE_QWEN_GEN   = True      # 是否使用Qwen生成方案（可关闭只用KNN）
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

dashscope.api_key = QWEN_API_KEY

# ==================== 工具函数 ====================

def project_to_d(vec, d):
    """将向量投影到 d 维（分段平均）"""
    vec = np.array(vec, dtype=np.float32).flatten()
    total_len = vec.shape[0]
    if total_len == d:
        return vec
    if total_len % d == 0:
        return vec.reshape(-1, d).mean(axis=0)
    # 如果不整除，用插值截取或 pad
    if total_len > d:
        # 尽量均匀采样
        indices = np.linspace(0, total_len - 1, d).astype(int)
        return vec[indices]
    else:
        # 用零填充
        out = np.zeros(d, dtype=np.float32)
        out[:total_len] = vec
        return out


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


# ==================== 加载资源 ====================
print("=" * 60)
print("向量→文本重建评估实验")
print("=" * 60)

print(f"\n[1] 加载编码器 {ENCODER_NAME} ...")
encoder = SentenceTransformer(ENCODER_NAME)

print("[2] 加载 results_data.pkl ...")
with open('results_data.pkl', 'rb') as f:
    results_data = pickle.load(f)
print(f"    共 {len(results_data)} 个拼接组合")

# 构建语料库（所有原始题目文本，去重）
all_texts_set = set()
for data in results_data.values():
    for t in data['texts']:
        all_texts_set.add(t)
corpus = list(all_texts_set)
print(f"[3] 语料库大小: {len(corpus)} 道题目")

print("[4] 计算语料库嵌入 (KNN索引)...")
corpus_embeddings = encoder.encode(corpus, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=True)

knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
knn.fit(corpus_embeddings)


# ==================== KNN 重建 ====================
def knn_reconstruct(vec_384, top_k=1):
    """给定384维向量，返回句子库中最相似的文本"""
    vec_norm = normalize(vec_384).reshape(1, -1)
    distances, indices = knn.kneighbors(vec_norm, n_neighbors=top_k)
    return [corpus[idx] for idx in indices[0]]


# ==================== Qwen向量引导生成 ====================
def qwen_vector_guided_generate(vec_384, orig_text, max_retries=2):
    """
    利用Qwen API，通过以下策略将向量"还原"为文本：
    1. 先用KNN找到最近邻文本作为参考上下文
    2. 结合KNN结果和向量的统计特征，让模型重写
    """
    # 先获取KNN候选
    knn_candidates = knn_reconstruct(vec_384, top_k=3)
    candidates_str = "\n".join([f"- {c}" for c in knn_candidates])
    
    # 提取向量统计特征（辅助信息）
    vec_stats = {
        "mean": float(np.mean(vec_384)),
        "std":  float(np.std(vec_384)),
        "max":  float(np.max(vec_384)),
        "min":  float(np.min(vec_384)),
    }
    
    prompt = f"""你是一个文本嵌入向量解码器。给定一个384维文本嵌入向量，你需要推断它对应的原始化学题目文本。

向量统计特征（用于辅助判断）：
- 均值: {vec_stats['mean']:.4f}, 标准差: {vec_stats['std']:.4f}
- 最大值: {vec_stats['max']:.4f}, 最小值: {vec_stats['min']:.4f}

从语义相似度最高的候选文本中，选择最可能对应该向量的一个，并进行适当改写使其更准确：
{candidates_str}

请直接输出一个化学题目，不要任何额外解释，不要任何前缀（例如不要输出"重建文本："等）。
只输出一个化学题目文本。"""
    
    for attempt in range(max_retries):
        try:
            response = Generation.call(
                model="qwen-plus",
                prompt=prompt,
                temperature=0.2,
                max_tokens=200,
                result_format='message'
            )
            if response.status_code == 200:
                content = response.output.choices[0].message.content.strip()
                # 简单清理
                content = re.sub(r'^(重建文本[:：]|输出[:：]|答[:：])\s*', '', content)
                return content
            time.sleep(1)
        except Exception as e:
            print(f"  Qwen生成失败 (尝试 {attempt+1}): {e}")
            time.sleep(2)
    
    # 失败时退化到KNN
    return knn_candidates[0]


# ==================== 评估指标计算 ====================
rouge = rouge_scorer_lib.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)


def compute_reconstruction_metrics(orig_text, recon_text):
    """
    计算重建质量的多维指标：
    - cos_sim: 语义余弦相似度（主要指标）
    - rouge_l:  ROUGE-L F1
    - rouge1:   ROUGE-1 F1
    """
    # 语义相似度
    emb_orig  = encoder.encode([orig_text],  convert_to_numpy=True, normalize_embeddings=True)[0]
    emb_recon = encoder.encode([recon_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    cos_sim = float(np.dot(emb_orig, emb_recon))
    
    # ROUGE（对英文题目有效）
    r_scores = rouge.score(orig_text, recon_text)
    rouge_l = r_scores['rougeL'].fmeasure
    rouge1  = r_scores['rouge1'].fmeasure
    
    return cos_sim, rouge_l, rouge1


# ==================== 主实验循环 ====================
t_names = ['T_mean', 'T_max', 'T_min', 'T_concat', 'T_weighted',
           'T_product', 'T_first', 'T_last']
c_names = ['C_mean', 'C_max', 'C_min', 'C_concat', 'C_weighted',
           'C_product', 'C_diff', 'C_combined']

# 过滤掉 results_data 中没有的方法名
available_keys = set(results_data.keys())
t_names = [t for t in t_names if any(k[0] == t for k in available_keys)]
c_names = [c for c in c_names if any(k[1] == c for k in available_keys)]
print(f"\n有效Token方法: {t_names}")
print(f"有效Chunk方法: {c_names}")

# 结果矩阵: (t, c, 3) -> [cos_sim_knn, rouge_l_knn, cos_sim_qwen]
n_metrics = 4  # knn_cos, knn_rouge_l, knn_rouge1, qwen_cos
results_mat = np.full((len(t_names), len(c_names), n_metrics), np.nan)

# 详细数据，用于报告
detail_records = {}

print(f"\n[5] 开始重建评估实验 (SAMPLE_SIZE={SAMPLE_SIZE} per combination)...\n")
total_combos = sum(1 for t in t_names for c in c_names if (t, c) in results_data)
pbar = tqdm(total=total_combos, desc="组合进度")

for ti, t_name in enumerate(t_names):
    for ci, c_name in enumerate(c_names):
        key = (t_name, c_name)
        if key not in results_data:
            pbar.update(1)
            continue
        
        data     = results_data[key]
        vectors  = data['vectors']
        texts    = data['texts']
        
        if len(vectors) == 0:
            pbar.update(1)
            continue
        
        n_samp = min(SAMPLE_SIZE, len(vectors))
        indices = random.sample(range(len(vectors)), n_samp)
        
        knn_cos_scores  = []
        knn_rougeL_scores = []
        knn_rouge1_scores = []
        qwen_cos_scores = []
        
        samples_detail = []
        
        for idx in indices:
            vec  = vectors[idx]
            orig = texts[idx]
            
            # 投影到 384 维
            try:
                vec_384 = project_to_d(np.array(vec).flatten(), EMBED_DIM)
                vec_384 = normalize(vec_384)
            except Exception as e:
                print(f"  投影失败 {key} idx={idx}: {e}")
                continue
            
            # --- KNN重建 ---
            try:
                knn_recon = knn_reconstruct(vec_384, top_k=1)[0]
                knn_cos, knn_rl, knn_r1 = compute_reconstruction_metrics(orig, knn_recon)
                knn_cos_scores.append(knn_cos)
                knn_rougeL_scores.append(knn_rl)
                knn_rouge1_scores.append(knn_r1)
            except Exception as e:
                print(f"  KNN重建失败 {key} idx={idx}: {e}")
                knn_recon = ""
                knn_cos = knn_rl = knn_r1 = np.nan
            
            # --- Qwen生成重建 ---
            qwen_cos = np.nan
            qwen_recon = ""
            if USE_QWEN_GEN:
                try:
                    qwen_recon = qwen_vector_guided_generate(vec_384, orig)
                    qwen_cos, _, _ = compute_reconstruction_metrics(orig, qwen_recon)
                    qwen_cos_scores.append(qwen_cos)
                    time.sleep(0.3)  # 避免限流
                except Exception as e:
                    print(f"  Qwen生成失败 {key} idx={idx}: {e}")
            
            samples_detail.append({
                'idx': idx,
                'orig': orig[:120],
                'knn_recon': knn_recon[:120] if knn_recon else "",
                'qwen_recon': qwen_recon[:120] if qwen_recon else "",
                'knn_cos': float(knn_cos) if not np.isnan(knn_cos) else None,
                'knn_rougeL': float(knn_rl) if not np.isnan(knn_rl) else None,
                'qwen_cos': float(qwen_cos) if not np.isnan(qwen_cos) else None,
            })
        
        # 填写结果矩阵
        if knn_cos_scores:
            results_mat[ti, ci, 0] = np.mean(knn_cos_scores)
        if knn_rougeL_scores:
            results_mat[ti, ci, 1] = np.mean(knn_rougeL_scores)
        if knn_rouge1_scores:
            results_mat[ti, ci, 2] = np.mean(knn_rouge1_scores)
        if qwen_cos_scores:
            results_mat[ti, ci, 3] = np.mean(qwen_cos_scores)
        
        detail_records[key] = samples_detail
        pbar.update(1)
        pbar.set_postfix({
            'KNN_cos': f"{results_mat[ti,ci,0]:.3f}" if not np.isnan(results_mat[ti,ci,0]) else "nan",
            'Qwen_cos': f"{results_mat[ti,ci,3]:.3f}" if not np.isnan(results_mat[ti,ci,3]) else "nan"
        })

pbar.close()

# 综合重建分数（加权平均，以KNN_cos为主要指标）
w_knn_cos   = 0.40
w_knn_rougeL = 0.20
w_knn_rouge1 = 0.10
w_qwen_cos  = 0.30 if USE_QWEN_GEN else 0.0
if not USE_QWEN_GEN:
    w_knn_cos, w_knn_rougeL, w_knn_rouge1 = 0.50, 0.30, 0.20

combined_scores = (
    w_knn_cos   * np.nan_to_num(results_mat[:, :, 0]) +
    w_knn_rougeL * np.nan_to_num(results_mat[:, :, 1]) +
    w_knn_rouge1 * np.nan_to_num(results_mat[:, :, 2]) +
    w_qwen_cos  * np.nan_to_num(results_mat[:, :, 3])
)
# 恢复 nan（若该组合完全没有数据）
all_nan_mask = np.all(np.isnan(results_mat), axis=2)
combined_scores[all_nan_mask] = np.nan

# ==================== 保存结果 ====================
np.save("reconstruction_results.npy", results_mat)
np.save("reconstruction_combined.npy", combined_scores)
with open("reconstruction_detail.pkl", "wb") as f:
    pickle.dump({'detail': detail_records, 't_names': t_names, 'c_names': c_names}, f)
print("\n[6] 结果已保存")

# ==================== 绘制热力图 ====================
metric_labels = ['KNN余弦相似度', 'KNN ROUGE-L', 'KNN ROUGE-1', 'Qwen余弦相似度']
if not USE_QWEN_GEN:
    metric_labels = metric_labels[:3]
    results_mat_plot = results_mat[:, :, :3]
else:
    results_mat_plot = results_mat

fig, axes = plt.subplots(1, len(metric_labels) + 1,
                         figsize=(5 * (len(metric_labels) + 1), 6))

for m_idx, label in enumerate(metric_labels):
    ax = axes[m_idx]
    data = results_mat_plot[:, :, m_idx]
    data_plot = np.nan_to_num(data, nan=0.0)
    sns.heatmap(data_plot, annot=True, fmt='.3f',
                xticklabels=c_names, yticklabels=t_names,
                cmap='YlOrRd', ax=ax,
                cbar_kws={'label': label},
                vmin=0, vmax=1)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel('Chunk-level', fontsize=9)
    ax.set_ylabel('Token-level', fontsize=9)

# 综合分数
ax = axes[-1]
combined_plot = np.nan_to_num(combined_scores, nan=0.0)
sns.heatmap(combined_plot, annot=True, fmt='.3f',
            xticklabels=c_names, yticklabels=t_names,
            cmap='RdYlGn', ax=ax,
            cbar_kws={'label': 'Combined Score'},
            vmin=0, vmax=1)
ax.set_title('综合重建分数', fontsize=11)
ax.set_xlabel('Chunk-level', fontsize=9)
ax.set_ylabel('Token-level', fontsize=9)

plt.suptitle('向量→文本重建质量评估 (Token × Chunk Combinations)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('reconstruction_heatmap_full.png', dpi=150, bbox_inches='tight')
plt.close()
print("[7] 热力图已保存: reconstruction_heatmap_full.png")

# ==================== 打印结果摘要 ====================
print("\n" + "=" * 60)
print("实验结果摘要")
print("=" * 60)
print(f"\nKNN余弦相似度:")
print(f"  均值: {np.nanmean(results_mat[:,:,0]):.4f}")
print(f"  最大: {np.nanmax(results_mat[:,:,0]):.4f}")
print(f"  最小: {np.nanmin(results_mat[:,:,0]):.4f}")
best_ti, best_ci = np.unravel_index(np.nanargmax(results_mat[:,:,0]), results_mat[:,:,0].shape)
print(f"  最佳组合: {t_names[best_ti]} × {c_names[best_ci]} = {results_mat[best_ti,best_ci,0]:.4f}")

print(f"\n综合重建分数:")
print(f"  均值: {np.nanmean(combined_scores):.4f}")
print(f"  最大: {np.nanmax(combined_scores):.4f}")
best_ti, best_ci = np.unravel_index(np.nanargmax(combined_scores), combined_scores.shape)
print(f"  最佳组合: {t_names[best_ti]} × {c_names[best_ci]} = {combined_scores[best_ti,best_ci]:.4f}")

# ==================== 生成详细报告数据 ====================
report_data = {
    't_names': t_names,
    'c_names': c_names,
    'results_mat': results_mat.tolist(),
    'combined_scores': combined_scores.tolist(),
    'metric_labels': metric_labels,
    'detail_records': {str(k): v for k, v in detail_records.items()}
}
with open('reconstruction_report_data.json', 'w', encoding='utf-8') as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)
print("[8] 报告数据已保存: reconstruction_report_data.json")
print("\n实验完成！")
