"""
生成 Word 文档所需的各图片，保存到 docx_figs/ 目录
"""
import json, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = r"C:\Users\hwm20\OneDrive\桌面\Research\embedding"
OUT_DIR = os.path.join(BASE, "docx_figs")
os.makedirs(OUT_DIR, exist_ok=True)

brier_mat = np.load(os.path.join(BASE, "brier_results.npy"))
with open(os.path.join(BASE, "brier_detail.json"), "r", encoding="utf-8") as f:
    det = json.load(f)
t_names = det["t_names"]
c_names = det["c_names"]
detail = det["detail"]
N = det["sample_n"]

full_metrics_path = os.path.join(BASE, "weighted_full_metrics.json")
with open(full_metrics_path, "r", encoding="utf-8") as f:
    full_metrics = json.load(f)

weighted_results_path = os.path.join(BASE, "weighted_results.json")
with open(weighted_results_path, "r", encoding="utf-8") as f:
    weighted_results = json.load(f)

def get_avg_metrics(method_name, is_new=True):
    if is_new:
        return {mn: full_metrics.get(f"AVG_{method_name}_{mn}", float("nan"))
                for mn in ["eosk", "recall_10", "separation", "silhouette", "brierlm"]}
    else:
        return {mn: full_metrics.get(f"BASELINE_{method_name}_{mn}", float("nan"))
                for mn in ["eosk", "recall_10", "separation", "silhouette", "brierlm"]}

# ── 图1：热力图 ──
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(brier_mat, annot=True, fmt=".1f", cmap="RdYlGn",
            xticklabels=c_names, yticklabels=t_names,
            vmin=0, vmax=100, ax=ax, linewidths=0.5,
            annot_kws={"size": 8})
ax.set_xlabel("Chunk Method", fontsize=10)
ax.set_ylabel("Token Method", fontsize=10)
ax.set_title(f"VecBrierLM Heatmap (N={N})", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig1_heatmap.png done")

# ── 图2：柱状图（Token均值 + Chunk均值）──
t_avgs = brier_mat.mean(axis=1)
c_avgs = brier_mat.mean(axis=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

def bar_color(v):
    return "#e74c3c" if v < 70 else ("#f39c12" if v < 88 else "#27ae60")

for ax, vals, labels, title in [
    (axes[0], t_avgs, t_names, "Token Method Average"),
    (axes[1], c_avgs, c_names, "Chunk Method Average")
]:
    cols = [bar_color(v) for v in vals]
    bars = ax.bar(labels, vals, color=cols, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v+0.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=25, labelsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_bar.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig2_bar.png done")

# ── 图3：散点图（Brier@k=1 vs @k=50）──
SEP = "\u00d7"
try:
    with open(os.path.join(BASE, "brier_detail.json"), "r", encoding="utf-8") as f:
        bdet = json.load(f)
    det_d = bdet["detail"]
    xs, ys, scores, labels_sc = [], [], [], []
    for key, v in det_d.items():
        if isinstance(v, dict) and "brier_k1" in v and "brier_k50" in v:
            xs.append(v["brier_k1"])
            ys.append(v["brier_k50"])
            scores.append(v.get("brierlm", 0))
            labels_sc.append(key)

    if xs:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(xs, ys, c=scores, cmap="RdYlGn", vmin=0, vmax=100,
                        s=40, alpha=0.8, edgecolors="white", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="VecBrierLM")
        ax.set_xlabel("Brier@k=1 (Short-range)", fontsize=10)
        ax.set_ylabel("Brier@k=50 (Long-range)", fontsize=10)
        ax.set_title("Short-range vs Long-range Discriminability", fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig3_scatter.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("fig3_scatter.png done")
    else:
        # 用现有热力图代替
        import shutil
        shutil.copy(os.path.join(BASE, "brier_heatmap.png"),
                    os.path.join(OUT_DIR, "fig3_scatter.png"))
        print("fig3_scatter.png (using heatmap fallback)")
except Exception as e:
    print(f"scatter fallback: {e}")
    import shutil
    shutil.copy(os.path.join(BASE, "brier_heatmap.png"),
                os.path.join(OUT_DIR, "fig3_scatter.png"))

# ── 图4：5指标归一化多方法对比 ──
methods = ["C_mean", "C_sim_weighted", "C_idf_weighted", "C_combined"]
palettes = {"C_mean": "#95a5a6", "C_sim_weighted": "#e67e22",
            "C_idf_weighted": "#3498db", "C_combined": "#27ae60"}
metric_labels = ["EOSk", "Recall@10", "Separation", "Silhouette\n(norm)", "BrierLM\n(/100)"]

data_all = {}
for m in methods:
    is_new = m in ["C_sim_weighted", "C_idf_weighted"]
    avgs = get_avg_metrics(m, is_new=is_new)
    data_all[m] = [
        avgs.get("eosk", float("nan")),
        avgs.get("recall_10", float("nan")),
        avgs.get("separation", float("nan")),
        (avgs.get("silhouette", float("nan")) + 1) / 2,
        avgs.get("brierlm", float("nan")) / 100.0,
    ]

x = np.arange(len(metric_labels))
w = 0.18
fig, ax = plt.subplots(figsize=(12, 5))
for i, m in enumerate(methods):
    vals = data_all[m]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m,
                  color=palettes[m], alpha=0.85, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylim(0, 1.35)
ax.set_ylabel("Normalized Score", fontsize=10)
ax.set_title("5-Metric Comparison (8-Token Average, Normalized)", fontsize=10)
ax.legend(fontsize=9, loc="upper right")
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_5metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig4_5metrics.png done")

# ── 图5：per-token BrierLM 对比 ──
compared = {
    "C_mean":         [detail.get(f"{t}{SEP}C_mean",  {}).get("brierlm", 0) for t in t_names],
    "C_sim_weighted": [weighted_results.get(f"{t}{SEP}C_sim_weighted", {}).get("brierlm", 0) for t in t_names],
    "C_idf_weighted": [weighted_results.get(f"{t}{SEP}C_idf_weighted", {}).get("brierlm", 0) for t in t_names],
    "C_combined":     [detail.get(f"{t}{SEP}C_combined", {}).get("brierlm", 0) for t in t_names],
}
xp = np.arange(len(t_names))
wp = 0.2
fig, ax = plt.subplots(figsize=(11, 5))
pals = {"C_mean": "#95a5a6", "C_sim_weighted": "#e67e22",
        "C_idf_weighted": "#3498db", "C_combined": "#27ae60"}
for i, (label, vals) in enumerate(compared.items()):
    ax.bar(xp + (i - 1.5) * wp, vals, wp, label=label,
           color=pals[label], alpha=0.85, edgecolor="white", lw=0.5)
ax.set_xticks(xp)
ax.set_xticklabels(t_names, rotation=30, fontsize=9)
ax.set_ylim(0, 108)
ax.set_ylabel("VecBrierLM", fontsize=10)
ax.set_title("Per-Token BrierLM: C_mean vs Weighted vs C_combined", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_per_token.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig5_per_token.png done")

print("All figures saved to:", OUT_DIR)
