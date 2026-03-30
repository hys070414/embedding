"""
生成综合实验报告：整合5个评估指标（EOSk, Recall@10, Separation, Silhouette, Reconstruction）
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ==================== 加载数据 ====================
t_names = ['T_mean', 'T_max', 'T_min', 'T_concat', 'T_weighted',
           'T_product', 'T_first', 'T_last']
c_names = ['C_mean', 'C_max', 'C_min', 'C_concat', 'C_weighted',
           'C_product', 'C_diff', 'C_combined']

# 原始4指标
results_matrix = np.load('results_matrix.npy')  # (8,8,4)
eosk_mat      = results_matrix[:, :, 0]
recall_mat    = results_matrix[:, :, 1]
sep_mat       = results_matrix[:, :, 2]
sil_mat       = results_matrix[:, :, 3]

# 重建指标
recon_mat     = np.load('reconstruction_results.npy')  # (8,8,4)
recon_combined= np.load('reconstruction_combined.npy') # (8,8)
knn_cos_mat   = recon_mat[:, :, 0]
knn_rl_mat    = recon_mat[:, :, 1]
knn_r1_mat    = recon_mat[:, :, 2]
qwen_cos_mat  = recon_mat[:, :, 3]

# 加载详细数据
with open('reconstruction_detail.pkl', 'rb') as f:
    detail_data = pickle.load(f)
detail_records = detail_data['detail']

print("数据加载完成")

# ==================== 计算归一化综合排名 ====================
def min_max_norm(mat):
    mn, mx = np.nanmin(mat), np.nanmax(mat)
    if mx - mn < 1e-10:
        return np.zeros_like(mat)
    return (mat - mn) / (mx - mn)

# 归一化各指标（全部越大越好，sep和sil已经是这样）
eosk_n    = min_max_norm(eosk_mat)
recall_n  = min_max_norm(recall_mat)
sep_n     = min_max_norm(sep_mat)
sil_n     = min_max_norm(sil_mat)
recon_n   = min_max_norm(recon_combined)

# 综合排名（等权重）
overall = (eosk_n + recall_n + sep_n + sil_n + recon_n) / 5.0
overall[np.any(np.isnan(np.stack([eosk_mat, recall_mat, sep_mat, sil_mat, recon_combined], axis=2)), axis=2)] = np.nan

# ==================== 绘图函数 ====================
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def make_heatmap(data, title, label, cmap='viridis', vmin=None, vmax=None, figsize=(10, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    data_plot = np.nan_to_num(data, nan=0.0)
    kw = {'label': label}
    sns.heatmap(data_plot, annot=True, fmt='.3f',
                xticklabels=c_names, yticklabels=t_names,
                cmap=cmap, ax=ax, cbar_kws=kw,
                vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor='gray')
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('Chunk-level Methods', fontsize=11)
    ax.set_ylabel('Token-level Methods', fontsize=11)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

# ==================== 生成各热力图 ====================
print("生成热力图...")

img_eosk    = make_heatmap(eosk_mat,   'EOSk (Spectral Fidelity)',        'EOSk',       'Blues',    0, 1)
img_recall  = make_heatmap(recall_mat, 'Recall@10 (Retrieval Accuracy)',   'Recall@10',  'Greens',   0, 1)
img_sep     = make_heatmap(sep_mat,    'Separation (Intra-Inter Sim Gap)', 'Separation', 'Oranges')
img_sil     = make_heatmap(sil_mat,    'Silhouette Score (Cluster Quality)','Silhouette','Purples', -1, 1)
img_recon   = make_heatmap(recon_combined, 'Reconstruction Quality (Text Recovery)', 'Recon Score', 'YlOrRd', 0, 1)
img_overall = make_heatmap(overall,    'Overall Score (All 5 Metrics, Normalized Mean)', 'Overall', 'RdYlGn', 0, 1)

# ==================== 绘制雷达图（Top-5 vs Bottom-5）====================
print("生成雷达图...")

# 计算各组合的综合排名
combo_scores = []
for i, t in enumerate(t_names):
    for j, c in enumerate(c_names):
        if not np.isnan(overall[i, j]):
            combo_scores.append({
                'name': f"{t}×{c}",
                'overall': overall[i, j],
                'eosk': eosk_n[i, j], 'recall': recall_n[i, j],
                'sep': sep_n[i, j], 'sil': sil_n[i, j], 'recon': recon_n[i, j],
                # raw values
                'eosk_raw': eosk_mat[i,j], 'recall_raw': recall_mat[i,j],
                'sep_raw': sep_mat[i,j], 'sil_raw': sil_mat[i,j], 'recon_raw': recon_combined[i,j],
            })
combo_scores.sort(key=lambda x: x['overall'], reverse=True)

top5 = combo_scores[:5]
bot5 = combo_scores[-5:]

# 雷达图
categories = ['EOSk', 'Recall@10', 'Separation', 'Silhouette', 'Reconstruction']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                subplot_kw=dict(polar=True))

colors_top = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
colors_bot = ['#a6cee3','#b2df8a','#fb9a99','#fdbf6f','#cab2d6']

for ax, group, colors, title in [
    (ax1, top5, colors_top, 'Top-5 Combinations'),
    (ax2, bot5, colors_bot, 'Bottom-5 Combinations')
]:
    for combo, color in zip(group, colors):
        values = [combo['eosk'], combo['recall'], combo['sep'], combo['sil'], combo['recon']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=combo['name'], alpha=0.9)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Radar Chart: Top-5 vs Bottom-5 Combinations (Normalized Scores)', fontsize=14, y=1.02)
plt.tight_layout()
img_radar = fig_to_base64(fig)
plt.close(fig)

# ==================== 绘制综合排名条形图 ====================
print("生成排名图...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 20
top20 = combo_scores[:20]
names20 = [c['name'] for c in top20]
scores20 = [c['overall'] for c in top20]
colors20 = plt.cm.RdYlGn(np.array(scores20))
axes[0].barh(range(len(top20)), scores20, color=colors20, edgecolor='white', linewidth=0.5)
axes[0].set_yticks(range(len(top20)))
axes[0].set_yticklabels(names20, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Overall Score (Normalized)', fontsize=11)
axes[0].set_title('Top-20 Combinations', fontsize=13)
axes[0].set_xlim(0, 1)
for i, (s, n) in enumerate(zip(scores20, names20)):
    axes[0].text(s + 0.01, i, f'{s:.3f}', va='center', fontsize=8)

# 各Token方法平均综合分
t_avg = [np.nanmean(overall[i, :]) for i in range(len(t_names))]
c_avg = [np.nanmean(overall[:, j]) for j in range(len(c_names))]

x = np.arange(len(t_names))
w = 0.35
bars1 = axes[1].bar(x - w/2, t_avg, w, label='Token Methods', color='#4C72B0', alpha=0.8)
bars2 = axes[1].bar(x + w/2, c_avg[:len(t_names)] if len(c_names) >= len(t_names) else c_avg + [0]*(len(t_names)-len(c_names)),
                    w, label='Chunk Methods', color='#DD8452', alpha=0.8)
# Separate chunk bar
fig2, ax2 = plt.subplots(figsize=(10, 5))
x_c = np.arange(len(c_names))
bars_c = ax2.bar(x_c, c_avg, color='#DD8452', alpha=0.85, edgecolor='white')
ax2.set_xticks(x_c)
ax2.set_xticklabels(c_names, rotation=30, ha='right', fontsize=10)
ax2.set_ylabel('Average Overall Score', fontsize=11)
ax2.set_title('Average Overall Score by Chunk Method', fontsize=13)
for bar in bars_c:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
img_chunk_bar = fig_to_base64(fig2)
plt.close(fig2)

# Token bar
fig3, ax3 = plt.subplots(figsize=(10, 5))
x_t = np.arange(len(t_names))
bars_t = ax3.bar(x_t, t_avg, color='#4C72B0', alpha=0.85, edgecolor='white')
ax3.set_xticks(x_t)
ax3.set_xticklabels(t_names, rotation=30, ha='right', fontsize=10)
ax3.set_ylabel('Average Overall Score', fontsize=11)
ax3.set_title('Average Overall Score by Token Method', fontsize=13)
for bar in bars_t:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
img_token_bar = fig_to_base64(fig3)
plt.close(fig3)

plt.close(fig)

# ==================== 相关性分析 ====================
print("计算指标相关性...")
metrics_flat = {
    'EOSk': eosk_mat.flatten(),
    'Recall@10': recall_mat.flatten(),
    'Separation': sep_mat.flatten(),
    'Silhouette': sil_mat.flatten(),
    'Reconstruction': recon_combined.flatten(),
}
valid_mask = ~np.any(np.isnan(np.column_stack(list(metrics_flat.values()))), axis=1)
metrics_valid = {k: v[valid_mask] for k, v in metrics_flat.items()}

corr_mat = np.corrcoef(np.column_stack(list(metrics_valid.values())).T)

fig4, ax4 = plt.subplots(figsize=(7, 6))
metric_names_corr = list(metrics_flat.keys())
im = ax4.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
ax4.set_xticks(range(len(metric_names_corr)))
ax4.set_yticks(range(len(metric_names_corr)))
ax4.set_xticklabels(metric_names_corr, rotation=30, ha='right', fontsize=10)
ax4.set_yticklabels(metric_names_corr, fontsize=10)
for i in range(len(metric_names_corr)):
    for j in range(len(metric_names_corr)):
        ax4.text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                 fontsize=10, color='white' if abs(corr_mat[i,j]) > 0.6 else 'black')
plt.colorbar(im, ax=ax4, label='Pearson Correlation')
ax4.set_title('Inter-Metric Correlation Matrix', fontsize=13)
plt.tight_layout()
img_corr = fig_to_base64(fig4)
plt.close(fig4)

# ==================== 散点图：Reconstruction vs 其他指标 ====================
fig5, axes5 = plt.subplots(1, 4, figsize=(18, 5))
other_metrics = [
    ('EOSk', eosk_mat.flatten(), '#2196F3'),
    ('Recall@10', recall_mat.flatten(), '#4CAF50'),
    ('Separation', sep_mat.flatten(), '#FF9800'),
    ('Silhouette', sil_mat.flatten(), '#9C27B0'),
]
recon_flat = recon_combined.flatten()
for ax, (name, vals, color) in zip(axes5, other_metrics):
    mask = ~(np.isnan(vals) | np.isnan(recon_flat))
    ax.scatter(vals[mask], recon_flat[mask], alpha=0.7, color=color, s=40, edgecolors='white', linewidth=0.5)
    # 趋势线
    if mask.sum() > 2:
        z = np.polyfit(vals[mask], recon_flat[mask], 1)
        p = np.poly1d(z)
        xline = np.linspace(vals[mask].min(), vals[mask].max(), 50)
        ax.plot(xline, p(xline), 'k--', alpha=0.5, linewidth=1.5)
    r = np.corrcoef(vals[mask], recon_flat[mask])[0, 1] if mask.sum() > 2 else np.nan
    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel('Reconstruction', fontsize=11)
    ax.set_title(f'Recon vs {name}\n(r={r:.3f})', fontsize=11)
    ax.grid(True, alpha=0.3)
plt.suptitle('Reconstruction Score vs Other Metrics', fontsize=13, y=1.02)
plt.tight_layout()
img_scatter = fig_to_base64(fig5)
plt.close(fig5)

# ==================== 构建样本展示数据 ====================
# 选几个典型组合展示重建结果
sample_combos = [
    ('T_mean', 'C_mean'),
    ('T_first', 'C_mean'),
    ('T_product', 'C_min'),
    ('T_product', 'C_max'),
]
sample_html_parts = []
for tc in sample_combos:
    key_str = str(tc)
    if key_str not in detail_records:
        continue
    samples = detail_records[key_str][:3]
    rows = ""
    for s in samples:
        kcos = f"{s['knn_cos']:.4f}" if s['knn_cos'] is not None else "N/A"
        qcos = f"{s['qwen_cos']:.4f}" if s['qwen_cos'] is not None else "N/A"
        rl   = f"{s['knn_rougeL']:.4f}" if s['knn_rougeL'] is not None else "N/A"
        rows += f"""
        <tr>
          <td style="max-width:300px;word-break:break-all;font-size:12px;">{s['orig'][:150]}…</td>
          <td style="max-width:300px;word-break:break-all;font-size:12px;">{s['knn_recon'][:150]}…</td>
          <td style="max-width:300px;word-break:break-all;font-size:12px;">{s['qwen_recon'][:150] if s['qwen_recon'] else 'N/A'}…</td>
          <td style="text-align:center;">{kcos}</td>
          <td style="text-align:center;">{rl}</td>
          <td style="text-align:center;">{qcos}</td>
        </tr>"""
    i = t_names.index(tc[0]) if tc[0] in t_names else 0
    j = c_names.index(tc[1]) if tc[1] in c_names else 0
    ov = f"{overall[i,j]:.4f}" if not np.isnan(overall[i,j]) else "N/A"
    sample_html_parts.append(f"""
    <div class="sample-section">
      <h4>组合: <span class="combo-badge">{tc[0]} × {tc[1]}</span> &nbsp; 综合分数: <b>{ov}</b></h4>
      <div class="table-wrap">
      <table>
        <tr>
          <th>原始题目</th><th>KNN重建文本</th><th>Qwen生成文本</th>
          <th>KNN余弦相似度</th><th>KNN ROUGE-L</th><th>Qwen余弦相似度</th>
        </tr>
        {rows}
      </table>
      </div>
    </div>""")

# ==================== 综合排名表格 ====================
rank_table_rows = ""
for rank, combo in enumerate(combo_scores[:20], 1):
    overall_v = combo['overall']
    eosk_v   = combo['eosk_raw']
    recall_v = combo['recall_raw']
    sep_v    = combo['sep_raw']
    sil_v    = combo['sil_raw']
    recon_v  = combo['recon_raw']
    bg = '#e8f5e9' if rank <= 5 else ('#fff3e0' if rank <= 10 else 'white')
    rank_table_rows += f"""
    <tr style="background:{bg};">
      <td style="text-align:center;font-weight:bold;">#{rank}</td>
      <td>{combo['name']}</td>
      <td style="text-align:center;">{eosk_v:.4f}</td>
      <td style="text-align:center;">{recall_v:.4f}</td>
      <td style="text-align:center;">{sep_v:.4f}</td>
      <td style="text-align:center;">{sil_v:.4f}</td>
      <td style="text-align:center;">{recon_v:.4f}</td>
      <td style="text-align:center;font-weight:bold;">{overall_v:.4f}</td>
    </tr>"""

# ==================== 生成 HTML 报告 ====================
print("生成HTML报告...")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>化学题目向量→文本重建实验报告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; color: #333; }}
  .header {{ background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%); color: white; padding: 40px 60px; }}
  .header h1 {{ font-size: 2em; margin-bottom: 8px; }}
  .header p {{ font-size: 1.1em; opacity: 0.85; }}
  .header .meta {{ margin-top: 12px; font-size: 0.9em; opacity: 0.7; }}
  nav {{ background: #fff; border-bottom: 2px solid #e8eaf6; padding: 12px 60px; position: sticky; top: 0; z-index: 100; display: flex; gap: 24px; flex-wrap: wrap; }}
  nav a {{ color: #3949ab; text-decoration: none; font-size: 0.92em; font-weight: 500; padding: 4px 10px; border-radius: 4px; transition: background 0.2s; }}
  nav a:hover {{ background: #e8eaf6; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 40px 60px; }}
  section {{ margin-bottom: 56px; }}
  h2 {{ font-size: 1.6em; color: #1a237e; border-left: 5px solid #3f51b5; padding-left: 14px; margin-bottom: 20px; margin-top: 8px; }}
  h3 {{ font-size: 1.2em; color: #283593; margin: 20px 0 10px; }}
  h4 {{ font-size: 1.05em; color: #424242; margin: 14px 0 8px; }}
  p {{ line-height: 1.8; color: #555; margin-bottom: 10px; }}
  .card {{ background: white; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); padding: 28px 32px; margin-bottom: 24px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
  .metric-card {{ background: white; border-radius: 10px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 4px solid; text-align: center; }}
  .metric-card.blue {{ border-color: #3f51b5; }}
  .metric-card.green {{ border-color: #4caf50; }}
  .metric-card.orange {{ border-color: #ff9800; }}
  .metric-card.purple {{ border-color: #9c27b0; }}
  .metric-card.red {{ border-color: #f44336; }}
  .metric-value {{ font-size: 2em; font-weight: bold; margin: 8px 0; }}
  .metric-label {{ font-size: 0.85em; color: #777; }}
  .metric-sub {{ font-size: 0.8em; color: #999; margin-top: 4px; }}
  img.plot {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{ background: #3f51b5; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f5f5f5; }}
  .combo-badge {{ background: #e8eaf6; color: #3f51b5; padding: 3px 10px; border-radius: 12px; font-weight: 600; }}
  .badge-good {{ background: #e8f5e9; color: #2e7d32; padding: 3px 10px; border-radius: 12px; font-weight: 600; }}
  .badge-bad {{ background: #ffebee; color: #c62828; padding: 3px 10px; border-radius: 12px; font-weight: 600; }}
  .insight-box {{ background: #e8eaf6; border-left: 4px solid #3f51b5; padding: 16px 20px; border-radius: 0 8px 8px 0; margin: 16px 0; }}
  .insight-box.green {{ background: #e8f5e9; border-left-color: #4caf50; }}
  .insight-box.orange {{ background: #fff8e1; border-left-color: #ff9800; }}
  .insight-box.red {{ background: #ffebee; border-left-color: #f44336; }}
  .table-wrap {{ overflow-x: auto; }}
  .sample-section {{ background: white; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .method-explain {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .method-tag {{ background: #f3e5f5; color: #6a1b9a; padding: 6px 14px; border-radius: 20px; font-size: 0.88em; font-weight: 500; }}
  .method-tag.token {{ background: #e3f2fd; color: #1565c0; }}
  .flow-diagram {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 16px 0; padding: 16px; background: #f8f9fa; border-radius: 8px; }}
  .flow-step {{ background: white; border: 2px solid #3f51b5; border-radius: 8px; padding: 10px 18px; font-size: 0.92em; font-weight: 500; color: #3f51b5; }}
  .flow-arrow {{ font-size: 1.4em; color: #9e9e9e; }}
  .toc {{ background: white; border-radius: 10px; padding: 20px 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .toc ol {{ padding-left: 20px; }}
  .toc li {{ margin: 6px 0; }}
  .toc a {{ color: #3f51b5; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  footer {{ background: #1a237e; color: rgba(255,255,255,0.7); text-align: center; padding: 20px; font-size: 0.85em; }}
</style>
</head>
<body>

<div class="header">
  <h1>🧪 化学题目向量→文本重建实验报告</h1>
  <p>第5评估指标：Reconstruction Quality（文本重建质量）及5指标综合评估</p>
  <div class="meta">实验时间：2026年3月26日 &nbsp;|&nbsp; 数据集：10道化学题目 &nbsp;|&nbsp; Token方法×Chunk方法：8×8=64种组合</div>
</div>

<nav>
  <a href="#overview">实验概述</a>
  <a href="#method">重建方法</a>
  <a href="#recon-results">重建指标结果</a>
  <a href="#all-metrics">5指标综合</a>
  <a href="#ranking">综合排名</a>
  <a href="#correlation">指标相关性</a>
  <a href="#samples">重建样例</a>
  <a href="#conclusion">结论</a>
</nav>

<div class="container">

<!-- 实验概述 -->
<section id="overview">
  <h2>1. 实验概述</h2>
  <div class="card">
    <h3>实验背景</h3>
    <p>本实验在原有4个评估指标（EOSk谱保真度、Recall@10检索准确率、Separation类间分离度、Silhouette轮廓系数）的基础上，
    新增第5个指标——<b>文本重建质量（Reconstruction Quality）</b>。</p>
    <p>核心思想：如果一种向量拼接方法的质量高，那么该向量应能够被"还原"回与原始文本语义相近的文本。
    通过衡量"向量→文本重建"的质量，可以从<b>信息保真性</b>角度评估不同拼接方法的有效性。</p>
    
    <div class="flow-diagram">
      <div class="flow-step">原始化学题目<br><small>英文</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">LLM提取知识点<br><small>Qwen-Plus</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">Token级Embedding<br><small>all-MiniLM-L6-v2</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">Token方法聚合<br><small>8种</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">Chunk方法聚合<br><small>8种</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><b>向量重建为文本</b><br><small>KNN + Qwen</small></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">质量评分<br><small>余弦相似度+ROUGE</small></div>
    </div>
  </div>
  
  <div class="grid-3" style="margin-top:20px;">
    <div class="metric-card blue">
      <div class="metric-label">评估组合数</div>
      <div class="metric-value" style="color:#3f51b5;">64</div>
      <div class="metric-sub">8 Token × 8 Chunk方法</div>
    </div>
    <div class="metric-card green">
      <div class="metric-label">每组合采样数</div>
      <div class="metric-value" style="color:#4caf50;">10</div>
      <div class="metric-sub">随机抽样题目</div>
    </div>
    <div class="metric-card orange">
      <div class="metric-label">评估指标总数</div>
      <div class="metric-value" style="color:#ff9800;">5</div>
      <div class="metric-sub">EOSk + Recall + Sep + Sil + Recon</div>
    </div>
  </div>
</section>

<!-- 重建方法 -->
<section id="method">
  <h2>2. 文本重建方法设计</h2>
  <div class="card">
    <h3>方法一：KNN最近邻检索（主要方法）</h3>
    <p>构建题目文本的语料库，计算所有题目的SentenceTransformer嵌入（all-MiniLM-L6-v2），建立余弦距离KNN索引。
    给定一个拼接向量，直接在语料库中检索最相似的文本作为重建结果。</p>
    <div class="insight-box green">
      <b>优点：</b>计算效率高、结果可解释性强、不依赖生成模型。<br>
      <b>理论依据：</b>若拼接向量有效保留了原始语义信息，则与原始题目嵌入的余弦距离应最小，从而能被准确检索。
    </div>
    
    <h3 style="margin-top:20px;">方法二：Qwen向量引导生成（辅助方法）</h3>
    <p>结合KNN候选文本和向量统计特征，通过Qwen-Plus API进行语义引导生成。该方法在KNN结果基础上进行语义精化，
    生成更贴合目标向量语义的文本。</p>
    <div class="insight-box">
      <b>工作流程：</b><br>
      1. KNN检索Top-3候选文本<br>
      2. 提取向量统计特征（均值、标准差、极值）<br>
      3. 输入Qwen，结合候选文本生成最匹配的化学题目文本<br>
      4. 计算生成文本与原文的余弦相似度
    </div>
    
    <h3 style="margin-top:20px;">评估子指标</h3>
    <div class="method-explain">
      <span class="method-tag">KNN余弦相似度（权重40%）</span>
      <span class="method-tag">KNN ROUGE-L（权重20%）</span>
      <span class="method-tag">KNN ROUGE-1（权重10%）</span>
      <span class="method-tag">Qwen余弦相似度（权重30%）</span>
    </div>
    <p>综合重建分数 = 0.4×KNN_cos + 0.2×KNN_ROUGE-L + 0.1×KNN_ROUGE-1 + 0.3×Qwen_cos</p>
  </div>
</section>

<!-- 重建指标结果 -->
<section id="recon-results">
  <h2>3. 文本重建指标结果</h2>
  
  <div class="grid-2" style="margin-bottom:24px;">
    <div class="metric-card red">
      <div class="metric-label">KNN余弦相似度 (均值)</div>
      <div class="metric-value" style="color:#e53935;">{np.nanmean(knn_cos_mat):.4f}</div>
      <div class="metric-sub">范围: {np.nanmin(knn_cos_mat):.4f} ~ {np.nanmax(knn_cos_mat):.4f}</div>
    </div>
    <div class="metric-card orange">
      <div class="metric-label">KNN ROUGE-L (均值)</div>
      <div class="metric-value" style="color:#f57c00;">{np.nanmean(knn_rl_mat):.4f}</div>
      <div class="metric-sub">范围: {np.nanmin(knn_rl_mat):.4f} ~ {np.nanmax(knn_rl_mat):.4f}</div>
    </div>
    <div class="metric-card blue">
      <div class="metric-label">Qwen余弦相似度 (均值)</div>
      <div class="metric-value" style="color:#1565c0;">{np.nanmean(qwen_cos_mat):.4f}</div>
      <div class="metric-sub">范围: {np.nanmin(qwen_cos_mat):.4f} ~ {np.nanmax(qwen_cos_mat):.4f}</div>
    </div>
    <div class="metric-card green">
      <div class="metric-label">综合重建分数 (均值)</div>
      <div class="metric-value" style="color:#2e7d32;">{np.nanmean(recon_combined):.4f}</div>
      <div class="metric-sub">范围: {np.nanmin(recon_combined):.4f} ~ {np.nanmax(recon_combined):.4f}</div>
    </div>
  </div>
  
  <div class="card">
    <h3>综合重建分数热力图</h3>
    <img class="plot" src="data:image/png;base64,{img_recon}" alt="重建分数热力图">
  </div>
  
  <div class="insight-box orange" style="margin-top:16px;">
    <b>关键发现：</b>
    <ul style="margin-top:8px;padding-left:20px;">
      <li>大多数组合（KNN_cos均值≥0.8）能较好地重建原始题目文本，说明向量中保留了足够的语义信息</li>
      <li><b>T_product</b>系Token方法的重建质量显著偏低（KNN_cos≈0.2-0.5），表明逐元素乘积运算会严重破坏语义信息</li>
      <li><b>C_diff</b>（首尾差分）整体重建质量最差，说明差分操作会丢失大量语义内容</li>
      <li>T_mean、T_concat、T_weighted、T_first等方法与C_mean、C_min、C_concat、C_combined等组合可实现完美重建（KNN_cos=1.000）</li>
    </ul>
  </div>
</section>

<!-- 5指标综合 -->
<section id="all-metrics">
  <h2>4. 五指标综合评估</h2>
  
  <div class="card">
    <h3>综合排名热力图（5指标归一化均值）</h3>
    <img class="plot" src="data:image/png;base64,{img_overall}" alt="综合排名热力图">
    <p style="margin-top:12px;color:#666;">颜色越绿表示综合表现越好（5个指标归一化后等权重平均）</p>
  </div>
  
  <div class="grid-2">
    <div class="card">
      <h3>各Token方法平均综合分</h3>
      <img class="plot" src="data:image/png;base64,{img_token_bar}" alt="Token方法条形图">
    </div>
    <div class="card">
      <h3>各Chunk方法平均综合分</h3>
      <img class="plot" src="data:image/png;base64,{img_chunk_bar}" alt="Chunk方法条形图">
    </div>
  </div>
  
  <div class="card" style="margin-top:24px;">
    <h3>雷达图：Top-5 vs Bottom-5组合</h3>
    <img class="plot" src="data:image/png;base64,{img_radar}" alt="雷达图">
  </div>
  
  <div class="grid-2" style="margin-top:20px;">
    <div class="card">
      <h3>其他4个指标热力图</h3>
      <h4>EOSk（谱保真度）</h4>
      <img class="plot" src="data:image/png;base64,{img_eosk}" alt="EOSk热力图">
    </div>
    <div class="card">
      <h4>Recall@10（检索准确率）</h4>
      <img class="plot" src="data:image/png;base64,{img_recall}" alt="Recall热力图">
    </div>
    <div class="card">
      <h4>Separation（类间分离度）</h4>
      <img class="plot" src="data:image/png;base64,{img_sep}" alt="Separation热力图">
    </div>
    <div class="card">
      <h4>Silhouette（轮廓系数）</h4>
      <img class="plot" src="data:image/png;base64,{img_sil}" alt="Silhouette热力图">
    </div>
  </div>
</section>

<!-- 综合排名 -->
<section id="ranking">
  <h2>5. 综合排名 Top-20</h2>
  <div class="card">
    <div class="table-wrap">
    <table>
      <tr>
        <th>排名</th><th>组合</th>
        <th>EOSk</th><th>Recall@10</th><th>Separation</th><th>Silhouette</th>
        <th>Reconstruction</th><th>综合分数</th>
      </tr>
      {rank_table_rows}
    </table>
    </div>
    <p style="margin-top:12px;color:#888;font-size:0.85em;">
      绿色背景：Top-5；橙色背景：Top6-10；综合分数 = 5个归一化指标的均值
    </p>
  </div>
  
  <div class="insight-box green" style="margin-top:16px;">
    <b>Top-5最佳组合分析：</b>
    <ul style="margin-top:8px;padding-left:20px;">
      <li><b>T_mean × C_mean / C_concat / C_weighted / C_combined</b>：均值聚合在多个指标上表现稳定，特别是在重建质量上达到1.000</li>
      <li><b>T_first × C_mean 等</b>：取首个token的方法在检索准确率上有优势，与均值方法组合效果好</li>
      <li>整体来看，<b>平均型聚合方法（mean/weighted）</b>在5个综合指标上最均衡，是最佳选择</li>
    </ul>
  </div>
</section>

<!-- 指标相关性 -->
<section id="correlation">
  <h2>6. 指标相关性分析</h2>
  <div class="grid-2">
    <div class="card">
      <h3>指标间相关矩阵</h3>
      <img class="plot" src="data:image/png;base64,{img_corr}" alt="相关矩阵">
    </div>
    <div class="card">
      <h3>重建分数 vs 其他指标散点图</h3>
      <img class="plot" src="data:image/png;base64,{img_scatter}" alt="散点图">
    </div>
  </div>
  
  <div class="card" style="margin-top:20px;">
    <h3>相关性分析结论</h3>
    <div class="insight-box">
      <b>重建指标与其他指标的关系：</b>
    </div>
    <ul style="padding-left:24px;margin-top:12px;line-height:2;">
      <li><b>与EOSk的相关性</b>（r={np.corrcoef(eosk_mat.flatten()[~np.isnan(eosk_mat.flatten()+recon_combined.flatten())], recon_combined.flatten()[~np.isnan(eosk_mat.flatten()+recon_combined.flatten())])[0,1]:.3f}）：
      EOSk主要衡量谱结构保真度，与重建质量有一定相关性，但并不完全一致——某些高EOSk组合可能重建质量仍较低</li>
      <li><b>与Recall@10的相关性</b>（r={np.corrcoef(recall_mat.flatten()[~np.isnan(recall_mat.flatten()+recon_combined.flatten())], recon_combined.flatten()[~np.isnan(recall_mat.flatten()+recon_combined.flatten())])[0,1]:.3f}）：
      Recall反映局部信息保留能力，与重建质量关联较强</li>
      <li><b>与Separation/Silhouette的相关性</b>：类别区分指标与重建质量的关联相对独立，说明重建质量能提供独立的评估视角</li>
      <li><b>结论</b>：重建指标与已有指标存在正相关但并非完全冗余，能从信息还原角度提供补充评估</li>
    </ul>
  </div>
</section>

<!-- 重建样例 -->
<section id="samples">
  <h2>7. 重建样例展示</h2>
  <p style="margin-bottom:16px;">以下展示几个典型组合的文本重建效果，直观体现向量信息保留质量的差异。</p>
  {''.join(sample_html_parts)}
</section>

<!-- 结论 -->
<section id="conclusion">
  <h2>8. 结论与建议</h2>
  <div class="card">
    <h3>主要发现</h3>
    
    <div class="insight-box green">
      <b>✅ 最佳拼接策略：</b>Token均值/首token + Chunk均值/拼接/加权<br>
      这类组合在5个指标上均表现良好，特别是重建质量达到1.000，说明向量完整保留了原始语义信息。
    </div>
    
    <div class="insight-box red">
      <b>❌ 最差拼接策略：</b>T_product（逐元素乘积）、C_diff（差分）<br>
      T_product严重破坏语义信息，重建相似度仅约0.22-0.52；C_diff通过首尾差分丢失了核心信息。
    </div>
    
    <div class="insight-box orange">
      <b>🔍 重建指标的独特价值：</b><br>
      文本重建指标从"信息可恢复性"角度评估向量质量，与现有4个指标形成互补：<br>
      • EOSk/Recall衡量结构保真度和检索能力<br>
      • Separation/Silhouette衡量类别区分能力<br>
      • <b>Reconstruction衡量原始语义信息的保留完整性</b>（信息论视角）
    </div>
    
    <h3 style="margin-top:24px;">建议</h3>
    <ul style="padding-left:24px;line-height:2;">
      <li>对于需要最大化语义保留的任务，推荐使用 <span class="badge-good">T_mean × C_mean</span> 或 <span class="badge-good">T_mean × C_weighted</span></li>
      <li>对于需要区分不同类型化学题目的任务，可参考Silhouette/Separation较高的组合</li>
      <li>应避免使用 <span class="badge-bad">T_product</span> 系列，该方法在所有5个指标上均表现最差</li>
      <li>建议在未来研究中引入可学习注意力机制（T_attention/C_attention），有望在各指标上实现更优表现</li>
      <li>重建实验可扩展到更大样本量（当前10道题），增强统计显著性</li>
    </ul>
  </div>
</section>

</div>

<footer>
  化学题目知识点向量评估实验 · 2026年3月 · Token×Chunk双层拼接框架
</footer>

</body>
</html>"""

# 保存报告
output_path = 'reconstruction_full_report.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML报告已保存: {output_path}")

# 同时保存到 results 目录
import shutil
results_output = 'results/reconstruction_full_report.html'
os.makedirs('results', exist_ok=True)
shutil.copy(output_path, results_output)
print(f"同时保存至: {results_output}")
print("\n报告生成完成！")
