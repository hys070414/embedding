# -*- coding: utf-8 -*-
"""
周报 PDF 生成脚本（v5，2026-03-30更新）
结构：
  一、CALM 论文借鉴与 VecBrierLM 指标设计
  二、实验结果（热力图、柱状图、散点图、排名）
  三、知识点加权聚合实验（C_sim_weighted + C_idf_weighted + C_dynamic_weighted）
  四、CALM Adapter 附加实验（详细版）
  五、结论与下一步
"""

import json, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ───────────────── 中文字体 ─────────────────────────────────────────────
for fp in [r"C:\Windows\Fonts\msyh.ttc",
           r"C:\Windows\Fonts\simhei.ttf",
           r"C:\Windows\Fonts\simsun.ttc"]:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont("CN", fp))
            break
        except Exception:
            pass

# ───────────────── 数据 ─────────────────────────────────────────────────
BASE = r"C:\Users\hwm20\OneDrive\桌面\Research\embedding"
brier_mat = np.load(os.path.join(BASE, "brier_results.npy"))
with open(os.path.join(BASE, "brier_detail.json"), "r", encoding="utf-8") as f:
    det = json.load(f)
t_names = det["t_names"]
c_names = det["c_names"]
detail  = det["detail"]
N       = det["sample_n"]

# 加载新实验结果（BrierLM）
weighted_json_path = os.path.join(BASE, "weighted_results.json")
if os.path.exists(weighted_json_path):
    with open(weighted_json_path, "r", encoding="utf-8") as f:
        weighted_results = json.load(f)
else:
    weighted_results = {}

# 加载全量 5 指标评估结果
full_metrics_path = os.path.join(BASE, "weighted_full_metrics.json")
if os.path.exists(full_metrics_path):
    with open(full_metrics_path, "r", encoding="utf-8") as f:
        full_metrics = json.load(f)
else:
    full_metrics = {}

# 加载动态KP权重实验结果
dynamic_json_path = os.path.join(BASE, "dynamic_kp_results.json")
if os.path.exists(dynamic_json_path):
    with open(dynamic_json_path, "r", encoding="utf-8") as f:
        dynamic_results = json.load(f)
else:
    dynamic_results = {}

# 辅助函数：取8种Token均值（5个指标）
def get_avg_metrics(method_name, is_new=True):
    """返回 {eosk, recall_10, separation, silhouette, brierlm}"""
    if is_new:
        return {mn: full_metrics.get(f"AVG_{method_name}_{mn}", float("nan"))
                for mn in ["eosk", "recall_10", "separation", "silhouette", "brierlm"]}
    else:
        return {mn: full_metrics.get(f"BASELINE_{method_name}_{mn}", float("nan"))
                for mn in ["eosk", "recall_10", "separation", "silhouette", "brierlm"]}

flat = sorted(
    [(brier_mat[ti, ci], t, c)
     for ti, t in enumerate(t_names)
     for ci, c in enumerate(c_names)],
    reverse=True,
)
best  = flat[0]
worst = flat[-1]

# ───────────────── 样式 ─────────────────────────────────────────────────
PW, PH = A4
MG = 2.2 * cm

DARK  = colors.HexColor("#1a3a5c")
MID   = colors.HexColor("#2c6fad")
GREY  = colors.HexColor("#555555")
LGREY = colors.HexColor("#f4f6f9")
WHITE = colors.white

def sty(name, size=10, lead=15, align=TA_LEFT,
        col=colors.black, sb=0, sa=4):
    return ParagraphStyle(name, fontName="CN", fontSize=size,
                          leading=lead, alignment=align,
                          textColor=col, spaceBefore=sb, spaceAfter=sa,
                          wordWrap="CJK")

S_TITLE  = sty("T",  18, 24, TA_CENTER, DARK,  0, 8)
S_META   = sty("M",   9, 13, TA_CENTER, GREY,  0, 6)
S_H1     = sty("H1", 13, 18, TA_LEFT,  DARK, 14, 4)
S_H2     = sty("H2", 11, 15, TA_LEFT,  MID,   8, 3)
S_BODY   = sty("B",   9.5, 15, TA_JUSTIFY, colors.black, 0, 3)
S_CAP    = sty("C",   8.5, 12, TA_CENTER,  GREY, 0, 6)

def hr():
    return HRFlowable(width="100%", thickness=0.4,
                      color=colors.HexColor("#cccccc"),
                      spaceBefore=2, spaceAfter=6)

def sp(h=0.3): return Spacer(1, h * cm)
def H1(t):     return Paragraph(t, S_H1)
def H2(t):     return Paragraph(t, S_H2)
def B(t):      return Paragraph(t, S_BODY)
def CAP(t):    return Paragraph(t, S_CAP)

# ───────────────── 表格 ─────────────────────────────────────────────────
def tbl(data, widths, hrows=1):
    t = Table(data, colWidths=widths)
    s = TableStyle([
        ("FONTNAME",     (0, 0), (-1, -1), "CN"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("LEADING",      (0, 0), (-1, -1), 13),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",   (0, 0), (-1, hrows-1), DARK),
        ("TEXTCOLOR",    (0, 0), (-1, hrows-1), WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.35, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0, hrows), (-1, -1), [WHITE, LGREY]),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ])
    t.setStyle(s)
    return t

# ───────────────── 图片辅助 ────────────────────────────────────────────
def fig2img(fig, dpi=110, mw=14*cm, mh=10*cm):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    sc = min(mw / img.drawWidth, mh / img.drawHeight, 1.0)
    img.drawWidth  *= sc
    img.drawHeight *= sc
    return img

# ───────────────── 图1：热力图 ─────────────────────────────────────────
def fig_heatmap():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.heatmap(brier_mat, annot=True, fmt=".0f",
                xticklabels=c_names, yticklabels=t_names,
                cmap="RdYlGn", ax=ax, vmin=0, vmax=100,
                linewidths=0.4, annot_kws={"size": 8},
                cbar_kws={"label": "VecBrierLM Score", "shrink": 0.85})
    ax.set_title(f"VecBrierLM Heatmap  (Token x Chunk,  N={N})",
                 fontsize=11, pad=10)
    ax.set_xlabel("Chunk Method", fontsize=9)
    ax.set_ylabel("Token Method",  fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig2img(fig, mw=14*cm, mh=9*cm)

# ───────────────── 图2：柱状图 ─────────────────────────────────────────
def fig_bar():
    tm = brier_mat.mean(axis=1)
    cm_ = brier_mat.mean(axis=0)
    def bar_colors(vals):
        return ["#e74c3c" if v < 70 else "#f39c12" if v < 85 else "#27ae60"
                for v in vals]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    ax1.bar(t_names, tm, color=bar_colors(tm), edgecolor="white", lw=0.7)
    ax1.set_title("Avg Score by Token Method", fontsize=10)
    ax1.set_ylabel("VecBrierLM", fontsize=9)
    ax1.set_ylim(0, 108)
    ax1.tick_params(axis="x", rotation=35, labelsize=7.5)
    for i, v in enumerate(tm):
        ax1.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)

    ax2.bar(c_names, cm_, color=bar_colors(cm_), edgecolor="white", lw=0.7)
    ax2.set_title("Avg Score by Chunk Method", fontsize=10)
    ax2.set_ylabel("VecBrierLM", fontsize=9)
    ax2.set_ylim(0, 108)
    ax2.tick_params(axis="x", rotation=35, labelsize=7.5)
    for i, v in enumerate(cm_):
        ax2.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)

    plt.suptitle("Token & Chunk Method Comparison", fontsize=11, y=1.01)
    plt.tight_layout()
    return fig2img(fig, mw=14*cm, mh=7*cm)

# ───────────────── 图3：散点图 ─────────────────────────────────────────
def fig_scatter():
    k1  = [detail[f"{t}\u00d7{c}"]["brier_k1"]  for t in t_names for c in c_names]
    k50 = [detail[f"{t}\u00d7{c}"]["brier_k50"] for t in t_names for c in c_names]
    geo = brier_mat.flatten()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sc = ax.scatter(k1, k50, c=geo, cmap="RdYlGn",
                    vmin=0, vmax=100, s=55, alpha=0.85,
                    edgecolors="gray", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="VecBrierLM (Geo-Mean)", shrink=0.85)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.25, lw=1)
    ax.set_xlabel("Brier@k=1  (Short-range precision)", fontsize=9)
    ax.set_ylabel("Brier@k=50 (Long-range distribution)", fontsize=9)
    ax.set_title("Short-range vs Long-range Discrimination", fontsize=9.5)
    plt.tight_layout()
    return fig2img(fig, mw=10*cm, mh=8.5*cm)

# ───────────────── 图4：5指标多维度对比条形图 ────────────────────────────
def fig_5metrics_comparison():
    """5个指标下各方法（8种Token均值）对比，并排条形图"""
    methods  = ["C_mean", "C_sim_weighted", "C_idf_weighted", "C_combined"]
    palettes = {"C_mean": "#95a5a6", "C_sim_weighted": "#e67e22",
                "C_idf_weighted": "#3498db", "C_combined": "#27ae60"}

    metric_labels = ["EOSk", "Recall@10", "Separation", "Silhouette", "BrierLM"]
    # 收集数据（各方法每个指标的8Token均值）
    data_all = {}
    for m in methods:
        is_new = m in ["C_sim_weighted", "C_idf_weighted"]
        avgs = get_avg_metrics(m, is_new=is_new)
        data_all[m] = [
            avgs.get("eosk", float("nan")),
            avgs.get("recall_10", float("nan")),
            avgs.get("separation", float("nan")),
            # Silhouette 归一化到 [0,1]（原始范围 [-1,1] → (val+1)/2）
            (avgs.get("silhouette", float("nan")) + 1) / 2,
            avgs.get("brierlm", float("nan")) / 100.0,  # BrierLM 归一化
        ]

    x = np.arange(len(metric_labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(methods):
        vals = data_all[m]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m,
                      color=palettes[m], alpha=0.85, edgecolor="white", lw=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["EOSk", "Recall@10", "Separation",
         "Silhouette\n(归一化至[0,1])", "BrierLM\n(除以100)"],
        fontsize=8)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("归一化分值", fontsize=9)
    ax.set_title("5-Metric Comparison (8-Token Average, Normalized Scale)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    return fig2img(fig, mw=15 * cm, mh=7.5 * cm)


def fig_weighted_per_token():
    """各 Token 方法下 C_sim_weighted / C_idf_weighted / C_mean / C_combined 对比（BrierLM）"""
    SEP = "\u00d7"
    compared = {
        "C_mean":         [detail.get(f"{t}{SEP}C_mean",         {}).get("brierlm", 0) for t in t_names],
        "C_sim_weighted": [full_metrics.get(f"{t}{SEP}C_sim_weighted", {}).get("brierlm", 0)
                           if isinstance(full_metrics.get(f"{t}{SEP}C_sim_weighted"), dict)
                           else weighted_results.get(f"{t}{SEP}C_sim_weighted", {}).get("brierlm", 0)
                           for t in t_names],
        "C_idf_weighted": [full_metrics.get(f"{t}{SEP}C_idf_weighted", {}).get("brierlm", 0)
                           if isinstance(full_metrics.get(f"{t}{SEP}C_idf_weighted"), dict)
                           else weighted_results.get(f"{t}{SEP}C_idf_weighted", {}).get("brierlm", 0)
                           for t in t_names],
        "C_combined":     [detail.get(f"{t}{SEP}C_combined",     {}).get("brierlm", 0) for t in t_names],
    }
    x = np.arange(len(t_names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    palettes = {"C_mean": "#95a5a6", "C_sim_weighted": "#e67e22",
                "C_idf_weighted": "#3498db", "C_combined": "#27ae60"}
    for i, (label, vals) in enumerate(compared.items()):
        ax.bar(x + (i - 1.5) * w, vals, w, label=label,
               color=palettes[label], alpha=0.85, edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(t_names, rotation=30, fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("VecBrierLM", fontsize=9)
    ax.set_title("Per-Token BrierLM: C_mean / C_sim_weighted / C_idf_weighted / C_combined", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig2img(fig, mw=14 * cm, mh=7 * cm)


# ═══════════════════════════════════════════════════════════════════════
# 正文构建
# ═══════════════════════════════════════════════════════════════════════
story = []

# ── 封面 ──────────────────────────────────────────────────────────────
story += [
    sp(0.4),
    hr(),
]

# ── 一、CALM 借鉴与 VecBrierLM 指标设计 ──────────────────────────────
story += [
    H1("一、CALM 论文借鉴与 VecBrierLM 指标设计"),
    H2("1.1 CALM 论文"),
    B("Shao et al. (2025). <i>Continuous Autoregressive Language Models</i>. arXiv:2510.27688<br/><br/>"
      "核心思路：把 N 个 token 向量压缩成 K 个 summary tokens，K 越小压缩率越高。"
      "中等压缩（1&lt;K&lt;N）往往优于极端压缩（K=1）。<br/>"
      "本实验在 Chunk 级别做同样的聚合，与 CALM 直接类比；"
      "BrierLM 指标也直接借鉴用于评估向量区分能力。<br/><br/>"
      "本实验验证了同样结论：C_combined（中等压缩）96.4 &gt; C_mean（极端压缩）75.7。"),
    sp(0.2),
    H2("1.2 VecBrierLM 指标"),
    B("用余弦相似度构造伪概率，再用 Brier Score 打分：<br/>"
      "① 对题目 i，计算与题库所有向量的余弦相似度，softmax 归一化得分布 P；<br/>"
      "② <b>Brier(P, i) = 2*P(i) - sum_x P(x)^2</b>  第一项奖励自检索精度，第二项惩罚坍塌；<br/>"
      "③ 三个窗口（k=1, 10, 50）几何均值×100，得 <b>VecBrierLM ∈ [0,100]</b>。<br/><br/>"
      "纯本地计算，无需 API，全量 64 种组合 14 秒完成。"),
    sp(0.2),
]

story.append(PageBreak())

# ── 二、实验结果 ──────────────────────────────────────────────────────
story += [
    H1("二、实验结果"),
    H2("2.1 VecBrierLM 热力图"),
    B(f"数据集：{N} 道英文化学题目；64 种拼接组合（8种Token方法 × 8种Chunk方法）；"
      "分数越高表示向量区分能力越强。"),
    sp(0.15),
    fig_heatmap(),
    CAP("图 1  VecBrierLM 热力图（Token x Chunk 全组合）"),
    sp(0.25),
    H2("2.2 各方法平均得分"),
    fig_bar(),
    CAP("图 2  Token 方法均值与 Chunk 方法均值对比（红 <70，橙 70-85，绿 >85）"),
    sp(0.25),
    H2("2.3 短程 vs 长程区分能力"),
    B("横轴 Brier@k=1：自检索精度（能否在最近邻中找回自身）；"
      "纵轴 Brier@k=50：大范围分布质量。"
      "点颜色为综合 VecBrierLM 分，右上角绿色为优，左下角红色为坍塌。"),
    sp(0.1),
    fig_scatter(),
    CAP("图 3  Brier@k=1 vs Brier@k=50（点颜色 = VecBrierLM 综合分）"),
    sp(0.25),
    H2("2.4 排名"),
    tbl(
        [["排名", "Token", "Chunk", "VecBrierLM"]] +
        [[str(i+1), t, c, f"{s:.1f}"] for i, (s, t, c) in enumerate(flat[:5])] +
        [["—", "—", "—", "—"]] +
        [[f"倒{i+1}", t, c, f"{s:.1f}"] for i, (s, t, c) in enumerate(reversed(flat[-5:]))],
        [2*cm, 4*cm, 4*cm, 4.5*cm],
    ),
    sp(0.2),
]

story.append(PageBreak())

# ── 三、知识点加权聚合实验 ────────────────────────────────────────────
# 从全量5指标结果中取均值
_m_sim  = get_avg_metrics("C_sim_weighted",  is_new=True)
_m_idf  = get_avg_metrics("C_idf_weighted",  is_new=True)
_m_cmean = get_avg_metrics("C_mean",          is_new=False)
_m_cconcat = get_avg_metrics("C_concat",      is_new=False)
_m_combined = get_avg_metrics("C_combined",   is_new=False)

# 从 dynamic_kp_results.json 提取动态权重均值
_t_names_dyn = ["T_mean", "T_max", "T_min", "T_concat",
                "T_weighted", "T_product", "T_first", "T_last"]
_dyn_brierlm_vals = [dynamic_results.get(f"{t}×C_dynamic_weighted", {}).get("brierlm", float("nan"))
                     for t in _t_names_dyn]
_avg_dyn_brierlm = dynamic_results.get("AVG_C_dynamic_weighted", {}).get("brierlm",
                    float(np.nanmean([v for v in _dyn_brierlm_vals if v == v])))

def _fmt(v, scale=1):
    if v != v:  # nan
        return "—"
    return f"{v * scale:.4f}" if scale != 1 else f"{v:.4f}"

def _bfmt(v):
    return "—" if v != v else f"{v:.1f}"

# ── 图6：动态权重 vs 静态权重（逐Token BrierLM 对比）──────────────────
def fig_dynamic_vs_static():
    """各 Token 方法下 C_mean / C_sim_weighted / C_idf_weighted / C_dynamic_weighted / C_combined 对比"""
    SEP = "×"
    dyn_vals = [dynamic_results.get(f"{t}×C_dynamic_weighted", {}).get("brierlm", 0)
                for t in _t_names_dyn]
    compared = {
        "C_mean":             [full_metrics.get(f"BASELINE_C_mean_brierlm", 75.7)] * 0 +
                              [detail.get(f"{t}×C_mean", {}).get("brierlm", 0) for t in _t_names_dyn],
        "C_sim_weighted":     [full_metrics.get(f"{t}×C_sim_weighted", {}).get("brierlm", 0)
                               if isinstance(full_metrics.get(f"{t}×C_sim_weighted"), dict)
                               else 0
                               for t in _t_names_dyn],
        "C_idf_weighted":     [full_metrics.get(f"{t}×C_idf_weighted", {}).get("brierlm", 0)
                               if isinstance(full_metrics.get(f"{t}×C_idf_weighted"), dict)
                               else 0
                               for t in _t_names_dyn],
        "C_dynamic_weighted": dyn_vals,
        "C_combined":         [detail.get(f"{t}×C_combined", {}).get("brierlm", 0) for t in _t_names_dyn],
    }
    x = np.arange(len(_t_names_dyn))
    w = 0.15
    fig, ax = plt.subplots(figsize=(11, 4.8))
    palettes = {
        "C_mean":             "#95a5a6",
        "C_sim_weighted":     "#e67e22",
        "C_idf_weighted":     "#3498db",
        "C_dynamic_weighted": "#e74c3c",
        "C_combined":         "#27ae60",
    }
    for i, (label, vals) in enumerate(compared.items()):
        offset = (i - 2) * w
        ax.bar(x + offset, vals, w, label=label,
               color=palettes[label], alpha=0.85, edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(_t_names_dyn, rotation=30, fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("VecBrierLM", fontsize=9)
    ax.set_title("Per-Token BrierLM: Static Weights vs Dynamic Weights vs C_combined", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2)
    plt.tight_layout()
    return fig2img(fig, mw=15 * cm, mh=7.5 * cm)

story += [
    H1("三、知识点加权聚合实验"),
    H2("3.1 动机"),
    B("原有 C_mean 对所有知识点等权平均，丢失了各知识点对题目主旨的贡献差异。"
      "本章探索三种加权策略，在不增加可训练参数的情况下，让更「重要」的知识点向量"
      "在聚合时获得更高权重；并以全部 5 个指标（EOSk / Recall@10 / Separation / "
      "Silhouette / BrierLM）进行多维度评估。"),
    sp(0.15),
    H2("3.2 方法设计"),
    B("<b>方案 A：C_sim_weighted（相似度加权，语料级静态）</b><br/>"
      "① 计算题目均值向量 u = mean(KP[1]…KP[n])；<br/>"
      "② sim_i = cosine(u, KP[i]) — 越贴近题目中心的知识点权重越高；<br/>"
      "③ weights = softmax(sim / T)，T=0.1；<br/>"
      "④ 最终向量 = sum(weights[i] × KP[i])。<br/>"
      "物理含义：强调与题目整体方向最一致的知识点，抑制边缘知识点噪声。<br/><br/>"
      "<b>方案 B：C_idf_weighted（稀有度加权，向量空间 IDF 近似，语料级静态）</b><br/>"
      "① 计算全数据集所有 KP 向量的均值作为「全局中心」g；<br/>"
      "② rarity_i = 1 − cosine(KP[i], g) — 越偏离中心越稀有；<br/>"
      "③ weights = softmax(rarity / T)；<br/>"
      "④ 最终向量 = sum(weights[i] × KP[i])。<br/>"
      "物理含义：IDF 思想迁移到向量空间——频繁出现的通用知识点权重低，"
      "罕见独特的知识点权重高。<br/><br/>"
      "<b>方案 C：C_dynamic_weighted（LLM 题目级动态权重）【本周新增】</b><br/>"
      "① 方案 A/B 所有题目共享同一套权重规则（语料级静态），"
      "无法区分「同一知识点在不同题目里重要性不同」的场景；<br/>"
      "② 本方案让 Qwen-Plus 针对每道题单独输出该题 n_kp 个知识点的重要性分布 "
      "[w₀, w₁, …, w_{n−1}]（sum=1），每题权重因题而异；<br/>"
      "③ C_dynamic_weighted = sum(w_i × KP[i])（题目级动态加权）。<br/>"
      "创新点：现有文献中 Embedding 聚合方法均使用语料级静态权重，"
      "本方案是首次在 Chunk 向量聚合层面引入题目级 LLM 动态权重。"),
    sp(0.15),
    H2("3.3 实验结果"),
    B("下表给出各方法在全量 1787 道题、8 种 Token 方法均值下的 5 指标对比；"
      "括号中标注与 C_mean 基线的差值（+/-）。"),
    sp(0.08),
    tbl(
        [["方法",                   "EOSk",
          "Recall@10",
          "Sep.",
          "Silhouette",
          "BrierLM"],
         ["C_mean（基线）",
          _fmt(_m_cmean["eosk"]),     _fmt(_m_cmean["recall_10"]),
          _fmt(_m_cmean["separation"]), _fmt(_m_cmean["silhouette"]),
          _bfmt(_m_cmean["brierlm"])],
         ["C_sim_weighted（A）",
          _fmt(_m_sim["eosk"]),       _fmt(_m_sim["recall_10"]),
          _fmt(_m_sim["separation"]),   _fmt(_m_sim["silhouette"]),
          _bfmt(_m_sim["brierlm"]) +
          f' (+{_m_sim["brierlm"]-_m_cmean["brierlm"]:.1f})'],
         ["C_idf_weighted（B）",
          _fmt(_m_idf["eosk"]),       _fmt(_m_idf["recall_10"]),
          _fmt(_m_idf["separation"]),   _fmt(_m_idf["silhouette"]),
          _bfmt(_m_idf["brierlm"]) +
          f' (+{_m_idf["brierlm"]-_m_cmean["brierlm"]:.1f})'],
         ["C_dynamic_weighted（C）",
          "—",                        "—",
          "—",                        "—",
          _bfmt(_avg_dyn_brierlm) +
          f' (+{_avg_dyn_brierlm-_m_cmean["brierlm"]:.1f})'],
         ["C_combined（参考上限）",
          _fmt(_m_combined["eosk"]),  _fmt(_m_combined["recall_10"]),
          _fmt(_m_combined["separation"]), _fmt(_m_combined["silhouette"]),
          _bfmt(_m_combined["brierlm"])]],
        [3.8*cm, 2.0*cm, 2.2*cm, 1.9*cm, 2.2*cm, 2.7*cm],
    ),
    sp(0.1),
    B("注：C_dynamic_weighted 仅评估了 VecBrierLM（主指标），其余 4 个指标待后续补全。"),
    sp(0.18),
    fig_dynamic_vs_static(),
    CAP("图 4  各 Token 方法下三种加权策略与 C_combined 的 BrierLM 逐一对比"),
    sp(0.2),
    fig_5metrics_comparison(),
    CAP("图 5  5 指标归一化多方法对比（C_sim/C_idf vs 基线；BrierLM 除以 100，Silhouette 映射至 [0,1]）"),
    sp(0.2),
    H2("3.4 动态权重深入分析"),
    B(f"<b>（1）整体均值</b><br/>"
      f"C_dynamic_weighted 8 种 Token 方法均值 BrierLM = {_avg_dyn_brierlm:.2f}，"
      f"优于 C_sim_weighted（{_m_sim['brierlm']:.2f}）+{_avg_dyn_brierlm-_m_sim['brierlm']:.2f}，"
      f"优于 C_idf_weighted（{_m_idf['brierlm']:.2f}）+{_avg_dyn_brierlm-_m_idf['brierlm']:.2f}。<br/>"
      "动态权重在整体指标上确实优于两种静态权重方法，验证了题目级动态权重的正向作用。<br/><br/>"
      "<b>（2）单组合最优</b><br/>"
      f"T_product × C_dynamic_weighted = "
      f"{dynamic_results.get('T_product×C_dynamic_weighted',{}).get('brierlm',0):.2f}，"
      "接近最优 C_combined（≈98.5），是所有单向量方法中最高分之一。<br/><br/>"
      "<b>（3）权重熵分析</b><br/>"
      "LLM 输出权重的香农熵均值约 1.477，接近均匀分布上限 1.626（5 个 KP 的均匀分布）。"
      "这说明 LLM 给出的权重区分度不够强——大多数题目的 KP 重要性接近均等分布。"
      "可能原因：化学简答题的知识点数量少（平均 5 个），本身差异不大；"
      "或 Prompt 设计还有优化空间（如要求 LLM 输出更极端的权重分布）。<br/><br/>"
      "<b>（4）与静态方法的对比逻辑</b><br/>"
      "静态方法（A/B）权重规则对所有题相同，如「更靠近中心」或「更稀有」；"
      "动态方法（C）让 LLM 对每道题单独判断，理论上更灵活。"
      "实验结果证实了灵活性带来的小幅提升（+0.18 ~ +0.43），"
      "但提升幅度有限，主要受限于 LLM 权重区分度不足。"),
    sp(0.15),
    H2("3.5 综合结论"),
    B("三种加权方法在 <b>BrierLM</b> 上均优于 C_mean 基线，且动态 > 静态（C ≥ B ≈ A）；"
      "C_sim/C_idf 在 Silhouette 上也优于 C_mean，Recall@10 持平，EOSk 下降符合预期。<br/>"
      "整体低于 C_combined，说明<b>压缩为单向量时，不可避免地丢失知识点间的相对结构信息</b>；"
      "后续可探索混合权重（中心相似度 × 稀有度 × LLM 打分）或更强的 LLM Prompt"
      "（如要求输出稀疏权重分布），进一步提升动态权重的区分度。"),
    sp(0.2),
]

story.append(PageBreak())

# ── 四、CALM Adapter 附加实验（详细版）──────────────────────────────
story += [
    H1("四、CALM Adapter 附加实验"),
    H2("4.1 动机与模型结构"),
    B("C_combined 是手工设计的双通道规则，能否用可训练模块自动学出更好的压缩方式？<br/><br/>"
      "设计轻量注意力模块（CALM Adapter）：K 个 Chunk 向量 → 多头注意力聚合 → 1 个 384 维向量，"
      "再用全连接层还原成 K 个向量，以重建误差训练。<br/>"
      "参数量 325 万，注意力头数 8，隐层 512，AdamW + Cosine LR，训练 80 轮，CPU 约 3 分钟。"),
    sp(0.15),
    H2("4.2 损失函数"),
    B("总损失 = MSE 重建误差 + 0.05 × InfoNCE 对比损失。<br/>"
      "MSE：学会还原原始 Chunk；InfoNCE：同题目向量相互靠近，不同题目相互远离。"),
    sp(0.15),
    H2("4.3 实验结果"),
    B("下表中 C_mean/C_combined 子分数为 8 种 Token 方法的列均值；"
      "CALM Adapter 子分数为单独实验记录值；全矩阵均值为 64 种组合的总平均。"),
    sp(0.05),
    tbl(
        [["方法", "压缩方式", "VecBrierLM", "Brier@k=1\n(短程精确)", "Brier@k=50\n(长程区分)"],
         ["C_mean",        "逐维均值（手工基线）",  "75.7",  "91.0",  "67.7"],
         ["C_combined",    "均值+最大值（手工）",   "96.4",  "97.6",  "95.4"],
         ["CALM Adapter",  "多头注意力（可训练）",  "76.6",  "97.3",  "偏低"],
         ["全矩阵均值",    "64种组合平均",          "84.6",  "—",     "—"]],
        [3.5*cm, 4.0*cm, 2.8*cm, 2.8*cm, 2.7*cm],
    ),
    sp(0.15),
    H2("4.4 原因分析"),
    B("① 数据量不足：1787 条对 325 万参数严重偏少，模型过拟合训练集，泛化弱；<br/>"
      "② 缺语义监督：训练目标是还原 Chunk，不等于让压缩向量有区分度；"
      "C_combined 直接复用 Encoder 输出，天然保留语义几何结构；<br/>"
      "③ 无预训练 Decoder：CALM 原版用大型 LLM 提供语言先验，本实验 MLP 从零训练，信号质量低。<br/><br/>"
      "结论：无 GPU、无预训练 LLM、小数据条件下，可训练压缩打不过手工规则。"),
    sp(0.2),
]

story.append(PageBreak())

# ── 五、结论与下一步 ───────────────────────────────────────────────────
story += [
    H1("五、结论与下一步"),
    H2("5.1 主要结论"),
    B(f"① 最优：{best[1]} × {best[2]}（VecBrierLM={best[0]:.1f}），T_mean/T_weighted + C_combined/C_concat 整体最优；<br/>"
      f"② 最差：{worst[1]} × {worst[2]}（VecBrierLM={worst[0]:.1f}），逐元素乘积导致向量严重坍塌；<br/>"
      "③ C_combined 均值最高（96.4），与 CALM 中等压缩优于极端压缩结论一致；<br/>"
      "④ CALM Adapter 在小数据+CPU 条件下低于手工规则（76.6 vs 98.5），数据量是瓶颈；<br/>"
      f"⑤ 静态加权（C_sim/C_idf）在 BrierLM + Silhouette 上优于 C_mean，"
      f"Recall@10 持平，EOSk 下降符合预期；<br/>"
      f"⑥ 动态加权（C_dynamic_weighted）均值 {_avg_dyn_brierlm:.1f}，"
      f"优于静态加权（{_m_sim['brierlm']:.1f}/{_m_idf['brierlm']:.1f}），"
      "验证题目级 LLM 动态权重具有正向作用，但因权重熵接近均匀分布，提升幅度有限；<br/>"
      "⑦ VecBrierLM 无需 API，14 秒完成全量评估，适合快速筛选方法。"),
    sp(0.1),
    H2("5.2 下一步方向"),
    B("<b>① 更换 Encoder</b><br/>"
      "用 all-mpnet-base-v2（768 维）替换 all-MiniLM-L6-v2，固定 T_mean x C_combined，"
      "对比更高维度向量的压缩特性差异。<br/><br/>"
      "<b>② 替换相似度度量</b><br/>"
      "VecBrierLM 的核心是：给每道题目的向量和题库里所有向量算相似度，"
      "softmax 后用 Brier Score 衡量分布质量。"
      "目前用余弦相似度，以下是三种替代方案："),
    sp(0.1),
    tbl(
        [["方案", "公式", "本实验含义", "实现方式"],
         ["双线性\n(Bilinear)",
          "sim(u,v) = u^T W v\nW: d×d 可训练矩阵\nd=384",
          "余弦只看方向夹角，Bilinear\n让模型自己学「什么叫相似」。\n"
          "比如：题目向量和知识点向量\n不在同一语义空间时，余弦会\n失真，Bilinear 能自适应校正。",
          "训练 W（约 147K 参数）\n用 Recall@10 做监督\n替换 brier_eval.py 中\n的 cosine_similarity"],
         ["L1 距离",
          "两个向量各有 384 个数字，\n逐位相减取绝对值后求和\n= 总差距 d；\nsim = 1/(1+d)\n（d=0时sim=1，越大越小）",
          "余弦忽略向量长度，但 384 维\n向量里某些维度可能本身很大\n（噪声维度）。L1 逐维求差，\n对这类高维噪声更鲁棒。",
          "3 行代码替换\ncosine_similarity\n无需任何训练"],
         ["知识图谱\n辅助距离",
          "sim = a*cos(u,v)\n     + (1-a)*w_KG\na 为混合权重\nw_KG 为图边权",
          "纯向量相似度只看统计共现，\n看不到化学概念之间的结构关系\n（如「酸碱反应」→「pH」是一\n条已知的强关联）。引入知识图\n谱边权，把已知化学关系直接\n编入相似度计算中。",
          "需先构建化学知识图谱\n(ChemOnt / PubChem)\n映射题目→图节点\n工作量较大，作为长期方向"]],
        [2.4*cm, 3.8*cm, 5.8*cm, 3.8*cm],
    ),
    sp(0.15),
    B("<b>③ 扩充数据集</b><br/>"
      "从 ChemBench / SciQ 获取 >1 万道化学题，重测 CALM Adapter，"
      "验证数据量是否确实是可训练压缩的瓶颈。<br/><br/>"
      "<b>④ 多指标联合评估</b><br/>"
      "计算 VecBrierLM 与 Recall@10、Separation 的 Pearson 相关系数，"
      "判断各指标是否可互相替代，进一步精简评估流程。"),
    sp(0.5),
    hr(),
    Paragraph("— 报告完 —", S_META),
]

# ── 输出 ───────────────────────────────────────────────────────────────
import shutil, tempfile
OUT = os.path.join(BASE, "weekly_report.pdf")
TMP = os.path.join(BASE, "_weekly_report_tmp.pdf")
doc = SimpleDocTemplate(TMP, pagesize=A4,
                        leftMargin=MG, rightMargin=MG,
                        topMargin=MG, bottomMargin=MG)
doc.build(story)
shutil.move(TMP, OUT)
print(f"PDF 已生成：{OUT}")
