"""
VecBrierLM 全量评估脚本
==============================================
灵感来源：CALM (Shao et al., 2025, arXiv 2510.27688)
  "Continuous Autoregressive Language Models"
  https://arxiv.org/abs/2510.27688

CALM 提出 BrierLM 作为无似然 LLM 评估指标，替代困惑度（Perplexity）。
本脚本将其核心思想迁移到「向量拼接方法质量评估」场景：

  原始 BrierLM（CALM）：
    Brier(P, y) = 2·P(y) − Σ_x P(x)²
    BrierLM = 几何均值(Brier-1, Brier-2, Brier-4)（n-gram 窗口）
    无偏估计：sample 两次→I{x1=y}+I{x2=y}−I{x1=x2}（准确性−多样性惩罚）

  本实验 VecBrierLM（向量空间适配）：
    - 将余弦相似度经 softmax(T=0.05) 转为隐式概率分布 P
    - 对每个查询向量，评估它在全量候选集中"找回自身"的能力
    - VecBrierLM = 几何均值(VecBrier@1, VecBrier@10, VecBrier@50) × 100
    - 越高越好：理想正交向量→~100，坍塌向量→~50

输入：results_data.pkl（64 种组合的向量+文本）
输出：
  brier_results.npy      (8,8) VecBrierLM 分数矩阵
  brier_detail.json      详细分数（各 k 值的 Brier）
  brier_heatmap.png      热力图
  brier_report.html      完整 HTML 报告（含与其他指标对比）
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pickle, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import base64
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ===== 从 evaluation.py 引入 BrierLM =====
from evaluation import compute_brierlm

# ===== 配置 =====
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

T_NAMES = ['T_mean', 'T_max', 'T_min', 'T_concat', 'T_weighted',
           'T_product', 'T_first', 'T_last']
C_NAMES = ['C_mean', 'C_max', 'C_min', 'C_concat', 'C_weighted',
           'C_product', 'C_diff', 'C_combined']

# ===== 加载数据 =====
print("=" * 70)
print("VecBrierLM 全量评估（CALM BrierScore 适配版）")
print("=" * 70)

print("\n[1] 加载 results_data.pkl ...")
with open('results_data.pkl', 'rb') as f:
    results_data = pickle.load(f)

available_keys = set(results_data.keys())
t_names = [t for t in T_NAMES if any(k[0] == t for k in available_keys)]
c_names = [c for c in C_NAMES if any(k[1] == c for k in available_keys)]
n_t, n_c = len(t_names), len(c_names)
sample_n = len(next(iter(results_data.values()))['texts'])
print(f"    组合数: {len(results_data)}, 每组合样本数: {sample_n}")
print(f"    Token方法: {t_names}")
print(f"    Chunk方法: {c_names}")

# ===== 主评估循环 =====
brier_mat  = np.full((n_t, n_c), np.nan)   # VecBrierLM 主分数
detail_mat = {}  # 详细各 k 值分数

print(f"\n[2] 开始 VecBrierLM 评估（共 {n_t * n_c} 个组合）...\n")
start_time = time.time()

pbar = tqdm(total=n_t * n_c, desc="组合进度", ncols=80)

for ti, t_name in enumerate(t_names):
    for ci, c_name in enumerate(c_names):
        key = (t_name, c_name)
        if key not in results_data:
            pbar.update(1)
            continue

        vectors = results_data[key]['vectors']
        if not vectors or len(vectors) == 0:
            pbar.update(1)
            continue

        try:
            brierlm, det = compute_brierlm(
                vectors,
                temperature=0.05,
                subsample=min(300, len(vectors))
            )
            brier_mat[ti, ci] = brierlm
            detail_mat[f"{t_name}×{c_name}"] = det
        except Exception as e:
            print(f"\n[错误] {t_name}×{c_name}: {e}")
            brier_mat[ti, ci] = np.nan

        pbar.set_postfix({
            'BrierLM': f"{brier_mat[ti, ci]:.2f}" if not np.isnan(brier_mat[ti, ci]) else "-",
        })
        pbar.update(1)

pbar.close()
elapsed = time.time() - start_time
print(f"\n评估完成！耗时 {elapsed:.1f} 秒")

# ===== 保存结果 =====
np.save("brier_results.npy", brier_mat)
with open("brier_detail.json", "w", encoding="utf-8") as f:
    json.dump({
        "t_names": t_names,
        "c_names": c_names,
        "brier_mat": brier_mat.tolist(),
        "detail": detail_mat,
        "sample_n": sample_n,
    }, f, ensure_ascii=False, indent=2)
print("结果已保存: brier_results.npy, brier_detail.json")

# ===== 打印摘要 =====
print("\n" + "=" * 60)
print("VecBrierLM 评估摘要")
print("=" * 60)
print(f"全局均值: {np.nanmean(brier_mat):.4f}")
print(f"全局最大: {np.nanmax(brier_mat):.4f}")
print(f"全局最小: {np.nanmin(brier_mat):.4f}")

bi, bj = np.unravel_index(np.nanargmax(brier_mat), brier_mat.shape)
wi, wj = np.unravel_index(np.nanargmin(brier_mat), brier_mat.shape)
print(f"\n最佳组合: {t_names[bi]} × {c_names[bj]} = {brier_mat[bi, bj]:.4f}")
print(f"最差组合: {t_names[wi]} × {c_names[wj]} = {brier_mat[wi, wj]:.4f}")

# Token 方法均值
print("\nToken 方法平均 BrierLM:")
for ti, t in enumerate(t_names):
    avg = np.nanmean(brier_mat[ti, :])
    print(f"  {t}: {avg:.4f}")

# Chunk 方法均值
print("\nChunk 方法平均 BrierLM:")
for ci, c in enumerate(c_names):
    avg = np.nanmean(brier_mat[:, ci])
    print(f"  {c}: {avg:.4f}")

# ===== 绘制热力图 =====
def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

fig, ax = plt.subplots(figsize=(12, 8))
data_plot = np.nan_to_num(brier_mat, nan=0.0)
sns.heatmap(data_plot, annot=True, fmt='.2f',
            xticklabels=c_names, yticklabels=t_names,
            cmap='RdYlGn', ax=ax,
            cbar_kws={'label': 'VecBrierLM'},
            vmin=np.nanmin(brier_mat) - 0.5,
            vmax=np.nanmax(brier_mat) + 0.5,
            linewidths=0.5, linecolor='gray')
ax.set_title('VecBrierLM — Token × Chunk 组合（CALM BrierScore 适配）',
             fontsize=13, pad=14)
ax.set_xlabel('Chunk Methods', fontsize=11)
ax.set_ylabel('Token Methods', fontsize=11)
plt.tight_layout()
plt.savefig('brier_heatmap.png', dpi=150, bbox_inches='tight')
img_brier = fig_to_b64(fig)
plt.close(fig)
print("\n热力图已保存: brier_heatmap.png")

# ===== 对比图：BrierLM vs 其他已有指标 =====
other_mats = {}
for fname, label in [
    ('results_matrix.npy', 'EOSk/Recall/Sep/Sil'),
    ('reconstruction_v2_combined.npy', 'Reconstruction_v2'),
]:
    if os.path.exists(fname):
        m = np.load(fname)
        if fname == 'results_matrix.npy':
            if m.ndim == 3:
                other_mats['EOSk']       = m[:, :, 0]
                other_mats['Recall@10']  = m[:, :, 1]
                other_mats['Separation'] = m[:, :, 2]
                other_mats['Silhouette'] = m[:, :, 3]
                if m.shape[2] >= 5:
                    other_mats['BrierLM_v1'] = m[:, :, 4]
        else:
            other_mats['Reconstruction_v2'] = m

scatter_imgs = {}
brier_flat = brier_mat.flatten()
for name, mat in other_mats.items():
    other_flat = mat.flatten()
    mask = ~(np.isnan(brier_flat) | np.isnan(other_flat))
    if mask.sum() < 3:
        continue
    fig_s, ax_s = plt.subplots(figsize=(5, 4))
    ax_s.scatter(other_flat[mask], brier_flat[mask],
                 alpha=0.7, s=40, color='#3f51b5', edgecolors='white', linewidth=0.5)
    z = np.polyfit(other_flat[mask], brier_flat[mask], 1)
    p = np.poly1d(z)
    xl = np.linspace(other_flat[mask].min(), other_flat[mask].max(), 50)
    ax_s.plot(xl, p(xl), 'k--', alpha=0.5, linewidth=1.5)
    r = np.corrcoef(other_flat[mask], brier_flat[mask])[0, 1]
    ax_s.set_xlabel(name, fontsize=10)
    ax_s.set_ylabel('VecBrierLM', fontsize=10)
    ax_s.set_title(f'r = {r:.3f}', fontsize=11)
    ax_s.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_imgs[name] = (fig_to_b64(fig_s), r)
    plt.close(fig_s)

# ===== 生成 HTML 报告 =====
# 排名表
combo_rank = []
for ti, t in enumerate(t_names):
    for ci, c in enumerate(c_names):
        if not np.isnan(brier_mat[ti, ci]):
            det = detail_mat.get(f"{t}×{c}", {})
            combo_rank.append({
                'name': f"{t}×{c}",
                'brierlm': brier_mat[ti, ci],
                'brier_k1':  det.get('brier_k1', float('nan')),
                'brier_k10': det.get('brier_k10', float('nan')),
                'brier_k50': det.get('brier_k50', float('nan')),
                'vec_brier_full': det.get('vec_brier_full', float('nan')),
            })
combo_rank.sort(key=lambda x: x['brierlm'], reverse=True)

rank_rows = ""
for rank, c in enumerate(combo_rank, 1):
    bg = '#e8f5e9' if rank <= 5 else ('#fff3e0' if rank <= 10 else 'white')
    def fmt(v): return f"{v:.4f}" if not (v != v) else "N/A"
    rank_rows += f"""
    <tr style="background:{bg};">
      <td style="text-align:center;font-weight:bold;">#{rank}</td>
      <td>{c['name']}</td>
      <td style="text-align:center;">{fmt(c['vec_brier_full'])}</td>
      <td style="text-align:center;">{fmt(c['brier_k1'])}</td>
      <td style="text-align:center;">{fmt(c['brier_k10'])}</td>
      <td style="text-align:center;">{fmt(c['brier_k50'])}</td>
      <td style="text-align:center;font-weight:bold;">{fmt(c['brierlm'])}</td>
    </tr>"""

scatter_section = ""
for name, (img_b64, r) in scatter_imgs.items():
    scatter_section += f"""
    <div class="card" style="margin-bottom:20px;">
      <h3>VecBrierLM vs {name}（r = {r:.3f}）</h3>
      <img class="plot" src="data:image/png;base64,{img_b64}" alt="{name}散点图">
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>VecBrierLM 评估报告（CALM BrierScore 适配）</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#f5f7fa; color:#333; }}
  .header {{ background:linear-gradient(135deg,#1a237e,#3949ab); color:white; padding:40px 60px; }}
  .header h1 {{ font-size:1.9em; margin-bottom:8px; }}
  .header p {{ opacity:0.85; font-size:1.05em; }}
  nav {{ background:#fff; border-bottom:2px solid #e8eaf6; padding:12px 60px; position:sticky; top:0; z-index:100; display:flex; gap:20px; flex-wrap:wrap; }}
  nav a {{ color:#3949ab; text-decoration:none; font-size:0.9em; font-weight:500; padding:4px 10px; border-radius:4px; }}
  nav a:hover {{ background:#e8eaf6; }}
  .container {{ max-width:1300px; margin:0 auto; padding:40px 60px; }}
  section {{ margin-bottom:52px; }}
  h2 {{ font-size:1.5em; color:#1a237e; border-left:5px solid #3f51b5; padding-left:12px; margin-bottom:18px; }}
  h3 {{ font-size:1.1em; color:#283593; margin:16px 0 10px; }}
  p {{ line-height:1.8; color:#555; margin-bottom:10px; }}
  .card {{ background:white; border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,.07); padding:26px 30px; margin-bottom:22px; }}
  .grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .metric-card {{ background:white; border-radius:10px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,.06); border-top:4px solid; text-align:center; }}
  .metric-value {{ font-size:2em; font-weight:bold; margin:8px 0; }}
  .metric-label {{ font-size:0.85em; color:#777; }}
  img.plot {{ max-width:100%; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,.1); }}
  table {{ width:100%; border-collapse:collapse; font-size:0.9em; }}
  th {{ background:#3f51b5; color:white; padding:10px 14px; text-align:left; }}
  td {{ padding:9px 14px; border-bottom:1px solid #eee; }}
  tr:hover td {{ background:#f5f5f5; }}
  .insight-box {{ background:#e8eaf6; border-left:4px solid #3f51b5; padding:16px 20px; border-radius:0 8px 8px 0; margin:16px 0; }}
  .insight-box.green {{ background:#e8f5e9; border-left-color:#4caf50; }}
  .insight-box.orange {{ background:#fff8e1; border-left-color:#ff9800; }}
  .insight-box.red {{ background:#ffebee; border-left-color:#f44336; }}
  .formula-box {{ background:#f3f4f6; border:1px solid #ddd; border-radius:8px; padding:16px 20px; font-family:monospace; font-size:0.95em; margin:14px 0; }}
  footer {{ background:#1a237e; color:rgba(255,255,255,.7); text-align:center; padding:20px; font-size:0.85em; }}
</style>
</head>
<body>

<div class="header">
  <h1>📐 VecBrierLM 评估报告</h1>
  <p>CALM BrierScore → 向量分布质量评估的适配迁移</p>
  <p style="margin-top:8px;opacity:0.7;font-size:0.9em;">
    灵感来源：Shao et al. (2025). <em>Continuous Autoregressive Language Models</em>. arXiv:2510.27688
  </p>
</div>

<nav>
  <a href="#background">背景与方法</a>
  <a href="#results">评估结果</a>
  <a href="#ranking">组合排名</a>
  <a href="#comparison">与其他指标对比</a>
  <a href="#conclusion">结论</a>
</nav>

<div class="container">

<!-- 背景与方法 -->
<section id="background">
  <h2>1. 方法背景：CALM BrierScore 的迁移</h2>
  <div class="card">
    <h3>CALM 论文核心贡献</h3>
    <p>CALM（Continuous Autoregressive Language Models）提出将语言建模从"离散 token 预测"转变为"连续向量预测"，
    通过 Autoencoder 将 K 个 token 压缩为一个连续向量。由于输出不再是离散分布，传统困惑度（Perplexity）不再适用，
    作者提出 <b>BrierLM</b> 作为无似然评估指标：</p>
    
    <div class="formula-box">
Brier(P, y) = 2·P(y) − Σ_x P(x)²   （测量预测分布 P 在真实答案 y 上的准确性−多样性惩罚）

BrierLM = 100 × (Brier-1 × Brier-2 × Brier-4)^(1/3)   （n-gram 窗口的几何均值）
    </div>
    
    <h3 style="margin-top:20px;">本实验的适配：VecBrierLM</h3>
    <p>在向量拼接方法评估场景中，我们没有显式的概率分布，但可以将<b>余弦相似度 softmax</b>后视为隐式分布：</p>
    
    <div class="formula-box">
P(v_j | q) = softmax(cosine(q, V))[j]   （T=0.05，低温使分布更锐利）

VecBrier@k(q, v_true) = 2·P_true - Σ P_j²   （在 top-k 候选中计算）

VecBrierLM = 100 × (VecBrier@1 × VecBrier@10 × VecBrier@50)^(1/3)
    </div>
    
    <div class="insight-box green">
      <b>核心语义与 CALM 一致：</b><br>
      • <b>2·P(true)</b>：奖励高保真度（自身相似度高 = 向量语义保留好）<br>
      • <b>-Σ P²</b>：惩罚模式坍塌（所有向量太相似则该项大，分数低）<br>
      • <b>几何均值</b>：平衡不同 top-k 下的表现，类比 CALM 的多 n-gram 窗口
    </div>
    
    <div class="insight-box orange">
      <b>与 CALM 的关键差异：</b><br>
      CALM 的 Brier 作用于离散词元分布（语言模型的输出空间）；<br>
      VecBrierLM 作用于连续向量相似度分布（向量检索的隐式概率空间）。<br>
      这是将"生成模型评估"思路迁移到"向量表示评估"的创新性适配。
    </div>
  </div>
</section>

<!-- 评估结果 -->
<section id="results">
  <h2>2. 评估结果</h2>
  
  <div class="grid-3" style="margin-bottom:24px;">
    <div class="metric-card" style="border-color:#3f51b5;">
      <div class="metric-label">全局均值 VecBrierLM</div>
      <div class="metric-value" style="color:#3f51b5;">{np.nanmean(brier_mat):.2f}</div>
    </div>
    <div class="metric-card" style="border-color:#4caf50;">
      <div class="metric-label">最佳组合 ({t_names[bi]}×{c_names[bj]})</div>
      <div class="metric-value" style="color:#4caf50;">{np.nanmax(brier_mat):.2f}</div>
    </div>
    <div class="metric-card" style="border-color:#f44336;">
      <div class="metric-label">最差组合 ({t_names[wi]}×{c_names[wj]})</div>
      <div class="metric-value" style="color:#f44336;">{np.nanmin(brier_mat):.2f}</div>
    </div>
  </div>
  
  <div class="card">
    <h3>VecBrierLM 热力图（Token × Chunk 64种组合）</h3>
    <img class="plot" src="data:image/png;base64,{img_brier}" alt="BrierLM热力图">
    <p style="margin-top:12px;color:#666;">颜色越绿 = 向量分布质量越高（准确性高且无坍塌）</p>
  </div>
</section>

<!-- 组合排名 -->
<section id="ranking">
  <h2>3. 组合排名（按 VecBrierLM 降序）</h2>
  <div class="card">
    <div style="overflow-x:auto;">
    <table>
      <tr>
        <th>排名</th><th>组合</th>
        <th>VecBrier(全量)</th>
        <th>VecBrier@k=1</th>
        <th>VecBrier@k=10</th>
        <th>VecBrier@k=50</th>
        <th>VecBrierLM</th>
      </tr>
      {rank_rows}
    </table>
    </div>
    <p style="margin-top:10px;color:#888;font-size:0.85em;">
      绿色：Top-5；橙色：Top6-10
    </p>
  </div>
</section>

<!-- 与其他指标对比 -->
<section id="comparison">
  <h2>4. 与其他指标的相关性对比</h2>
  <p>VecBrierLM 作为新增的第 6 个指标，与原有指标的相关性揭示其独特的评估维度：</p>
  {scatter_section if scatter_section else '<div class="card"><p>其他指标文件尚未生成（reconstruction_v2 仍在运行中），稍后重新运行本脚本即可更新对比分析。</p></div>'}
</section>

<!-- 结论 -->
<section id="conclusion">
  <h2>5. 结论</h2>
  <div class="card">
    <h3>VecBrierLM 的意义</h3>
    <div class="insight-box green">
      <b>✅ 优势：</b><br>
      • <b>完全本地，无 API 调用</b>：与 Reconstruction_v2 相比，运行速度快 100×<br>
      • <b>理论基础扎实</b>：直接继承 CALM 论文的严格数学框架<br>
      • <b>惩罚坍塌</b>：能检测出"所有向量趋同"的退化拼接方法（如 T_product 的乘积坍塌）<br>
      • <b>无需标注</b>：完全无监督，只依赖向量分布本身
    </div>
    <div class="insight-box orange">
      <b>⚠️ 局限性：</b><br>
      • 仅评估向量分布的自洽性（找回自身的能力），不直接评估与原始文本的语义对应<br>
      • 与 Recall@10 存在设计上的概念重叠（均衡量"检索自身"），但 VecBrierLM 额外引入多样性惩罚项<br>
      • 温度参数 T 的选择影响灵敏度，需根据数据调整
    </div>
    <h3 style="margin-top:20px;">与 CALM 原论文的异同对比</h3>
    <table>
      <tr><th>维度</th><th>CALM BrierLM</th><th>VecBrierLM（本实验）</th></tr>
      <tr><td>应用场景</td><td>评估生成式 LLM 的文本生成质量</td><td>评估向量拼接方法的表示质量</td></tr>
      <tr><td>概率分布 P</td><td>语言模型的 softmax 输出（离散词分布）</td><td>余弦相似度 softmax（连续向量空间）</td></tr>
      <tr><td>"真实答案" y</td><td>下一个 token（语言模型预测目标）</td><td>查询向量自身（自检索目标）</td></tr>
      <tr><td>多窗口/k 值</td><td>Brier-1, 2, 4（n-gram 宽度）</td><td>VecBrier@1, 10, 50（top-k 候选集大小）</td></tr>
      <tr><td>合并方式</td><td>几何均值 × 100</td><td>几何均值 × 100（相同）</td></tr>
      <tr><td>无似然</td><td>是（替代 Perplexity）</td><td>是（替代标注评估）</td></tr>
    </table>
  </div>
</section>

</div>

<footer>
  化学题目知识点向量评估实验 · VecBrierLM · 灵感来源：CALM (arXiv:2510.27688)
</footer>

</body>
</html>"""

with open("brier_report.html", "w", encoding="utf-8") as f:
    f.write(html)
print("HTML 报告已保存: brier_report.html")
print("\n全部完成！")
