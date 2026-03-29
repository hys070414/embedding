"""
生成 VecBrierLM 实验分析 PDF 报告
包含：实验图表 + CALM方法借鉴分析 + 压缩方法讨论
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 字体注册（中文支持）──────────────────────────────────────────
def register_fonts():
    """尝试注册中文字体，失败则用英文"""
    font_paths = [
        (r"C:\Windows\Fonts\simhei.ttf", "SimHei"),
        (r"C:\Windows\Fonts\msyh.ttc", "MsYaHei"),
        (r"C:\Windows\Fonts\simsun.ttc", "SimSun"),
    ]
    for path, name in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return None

CJK_FONT = register_fonts()
BASE_FONT  = CJK_FONT if CJK_FONT else "Helvetica"
BOLD_FONT  = CJK_FONT if CJK_FONT else "Helvetica-Bold"

# ── 加载数据 ──────────────────────────────────────────────────────
with open("brier_detail.json", encoding="utf-8") as f:
    data = json.load(f)

t_names = data["t_names"]
c_names = data["c_names"]
brier_mat = np.array(data["brier_mat"])
detail    = data["detail"]
sample_n  = data["sample_n"]

# 各k值矩阵
bk1_mat = np.zeros((8,8)); bk10_mat = np.zeros((8,8))
bk50_mat = np.zeros((8,8)); full_mat = np.zeros((8,8))
for ti, t in enumerate(t_names):
    for ci, c in enumerate(c_names):
        key = f"{t}\u00d7{c}"
        d = detail.get(key, {})
        bk1_mat[ti,ci]  = d.get("brier_k1",  0)
        bk10_mat[ti,ci] = d.get("brier_k10", 0)
        bk50_mat[ti,ci] = d.get("brier_k50", 0)
        full_mat[ti,ci] = d.get("vec_brier_full", 0)

# ── 生成图表 ──────────────────────────────────────────────────────
def fig_to_image(fig, dpi=130, max_width=16*cm, max_height=19*cm):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    # 按比例缩放以适应页面
    w, h = img.drawWidth, img.drawHeight
    scale = min(max_width / w, max_height / h, 1.0)
    img.drawWidth  = w * scale
    img.drawHeight = h * scale
    return img

def make_main_heatmap():
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.zeros_like(brier_mat, dtype=bool)
    im = sns.heatmap(
        brier_mat,
        annot=True, fmt='.1f',
        xticklabels=c_names, yticklabels=t_names,
        cmap='RdYlGn',
        ax=ax,
        vmin=0, vmax=100,
        linewidths=0.4, linecolor='gray',
        cbar_kws={'label': 'VecBrierLM Score', 'shrink': 0.85},
        annot_kws={'size': 8}
    )
    ax.set_title('VecBrierLM Score Heatmap (8 Token Methods x 8 Chunk Methods, N=1787)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Chunk Method', fontsize=10)
    ax.set_ylabel('Token Method', fontsize=10)
    plt.tight_layout()
    return fig_to_image(fig)

def make_k_comparison():
    """2x2子图：BrierLM, Brier@k=1,10,50"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    mats   = [brier_mat, bk1_mat, bk10_mat, bk50_mat]
    titles = ['VecBrierLM (Geo-Mean)', 'Brier@k=1', 'Brier@k=10', 'Brier@k=50']
    for ax, mat, title in zip(axes.flatten(), mats, titles):
        sns.heatmap(mat, annot=True, fmt='.0f',
                    xticklabels=c_names, yticklabels=t_names,
                    cmap='RdYlGn', ax=ax, vmin=0, vmax=100,
                    cbar=True, linewidths=0.3,
                    annot_kws={'size': 7},
                    cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Chunk', fontsize=8)
        ax.set_ylabel('Token', fontsize=8)
        ax.tick_params(labelsize=7.5)
    plt.suptitle('VecBrierLM Decomposition: Brier@k=1/10/50 + Overall', fontsize=12, y=1.01)
    plt.tight_layout()
    return fig_to_image(fig, dpi=100)

def make_row_col_bar():
    """按Token方法和Chunk方法分组的均值柱状图"""
    t_means = brier_mat.mean(axis=1)
    c_means = brier_mat.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors_t = ['#d73027' if v < 50 else '#fee08b' if v < 80 else '#1a9850' for v in t_means]
    colors_c = ['#d73027' if v < 50 else '#fee08b' if v < 80 else '#1a9850' for v in c_means]

    bars1 = ax1.barh(t_names, t_means, color=colors_t, edgecolor='gray', linewidth=0.5)
    ax1.set_title('Average VecBrierLM by Token Method', fontweight='bold')
    ax1.set_xlabel('VecBrierLM Score')
    ax1.set_xlim(0, 105)
    for bar, val in zip(bars1, t_means):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                 va='center', fontsize=8.5)
    ax1.axvline(x=80, color='orange', linestyle='--', alpha=0.6, label='Threshold=80')
    ax1.legend(fontsize=8)

    bars2 = ax2.barh(c_names, c_means, color=colors_c, edgecolor='gray', linewidth=0.5)
    ax2.set_title('Average VecBrierLM by Chunk Method', fontweight='bold')
    ax2.set_xlabel('VecBrierLM Score')
    ax2.set_xlim(0, 105)
    for bar, val in zip(bars2, c_means):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                 va='center', fontsize=8.5)
    ax2.axvline(x=80, color='orange', linestyle='--', alpha=0.6, label='Threshold=80')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig_to_image(fig)

def make_scatter_k1_k50():
    """散点图：Brier@k=1 vs Brier@k=50，体现向量质量的两个维度"""
    k1_flat  = bk1_mat.flatten()
    k50_flat = bk50_mat.flatten()
    labels   = [f"{t}x{c}" for t in t_names for c in c_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(k1_flat, k50_flat, c=brier_mat.flatten(),
                         cmap='RdYlGn', vmin=0, vmax=100,
                         s=60, alpha=0.8, edgecolors='gray', linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label='VecBrierLM Score')

    # 标注极端点
    for i, (x, y, lbl) in enumerate(zip(k1_flat, k50_flat, labels)):
        bm = brier_mat.flatten()[i]
        if bm < 35 or bm > 98.4:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, alpha=0.85)

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='k1=k50 (no decay)')
    ax.set_xlabel('Brier@k=1 (immediate accuracy)', fontsize=10)
    ax.set_ylabel('Brier@k=50 (broad coverage)', fontsize=10)
    ax.set_title('Brier@k=1 vs Brier@k=50:\nAccuracy vs Coverage Trade-off', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 102); ax.set_ylim(0, 102)
    plt.tight_layout()
    return fig_to_image(fig)

# ── 构建 PDF ──────────────────────────────────────────────────────
OUTPUT = "VecBrierLM_Analysis_Report.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2.2*cm, bottomMargin=2.2*cm
)

styles = getSampleStyleSheet()

# 自定义样式
title_style = ParagraphStyle(
    'CustomTitle', parent=styles['Title'],
    fontName=BOLD_FONT, fontSize=18, spaceAfter=6,
    textColor=colors.HexColor('#1a3a5c'), alignment=TA_CENTER
)
h1_style = ParagraphStyle(
    'H1', parent=styles['Heading1'],
    fontName=BOLD_FONT, fontSize=14, spaceAfter=4, spaceBefore=14,
    textColor=colors.HexColor('#1a3a5c'),
    borderPad=4,
)
h2_style = ParagraphStyle(
    'H2', parent=styles['Heading2'],
    fontName=BOLD_FONT, fontSize=11, spaceAfter=3, spaceBefore=8,
    textColor=colors.HexColor('#2c5f8a'),
)
body_style = ParagraphStyle(
    'Body', parent=styles['Normal'],
    fontName=BASE_FONT, fontSize=9.5, leading=15,
    spaceAfter=5, alignment=TA_JUSTIFY
)
caption_style = ParagraphStyle(
    'Caption', parent=styles['Normal'],
    fontName=BASE_FONT, fontSize=8.5, leading=12,
    textColor=colors.HexColor('#555555'), alignment=TA_CENTER,
    spaceBefore=2, spaceAfter=8
)
bullet_style = ParagraphStyle(
    'Bullet', parent=body_style,
    leftIndent=12, spaceBefore=2
)
formula_style = ParagraphStyle(
    'Formula', parent=styles['Normal'],
    fontName='Courier', fontSize=9.5, leading=14,
    leftIndent=20, spaceAfter=6, spaceBefore=4,
    backColor=colors.HexColor('#f5f5f5'),
    borderPad=6
)
highlight_style = ParagraphStyle(
    'Highlight', parent=body_style,
    backColor=colors.HexColor('#fffbe6'),
    borderPad=5, leftIndent=8, rightIndent=8,
    borderColor=colors.HexColor('#f0a500'),
    spaceBefore=4, spaceAfter=6
)

story = []

# ════════════════════════════════════════════════════════
# 封面
# ════════════════════════════════════════════════════════
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph("VecBrierLM Experiment Report", title_style))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Evaluating Chemical Question Embedding Quality via CALM-Inspired Brier Score",
    ParagraphStyle('subtitle', parent=body_style, fontSize=11,
                   alignment=TA_CENTER, textColor=colors.HexColor('#444444'))
))
story.append(Spacer(1, 0.4*cm))
story.append(HRFlowable(width="90%", thickness=1.5, color=colors.HexColor('#1a3a5c')))
story.append(Spacer(1, 0.3*cm))

meta_data = [
    ["Dataset", f"dataset_without_preference.jsonl  |  N = {sample_n} questions"],
    ["Embedding Model", "all-MiniLM-L6-v2  (384 dim)"],
    ["Experiment Scale", "8 Token Methods x 8 Chunk Methods = 64 combinations"],
    ["Evaluation", "VecBrierLM (Brier@k=1, 10, 50, full)  —  no API, local only"],
    ["Reference", "CALM: Continuous Autoregressive LM (Shao et al., arXiv:2510.27688)"],
    ["Date", "2026-03-28"],
]
meta_table = Table(meta_data, colWidths=[4.2*cm, 12.5*cm])
meta_table.setStyle(TableStyle([
    ('FONTNAME',    (0,0), (-1,-1), BASE_FONT),
    ('FONTSIZE',    (0,0), (-1,-1), 9),
    ('FONTNAME',    (0,0), (0,-1), BOLD_FONT),
    ('BACKGROUND',  (0,0), (0,-1), colors.HexColor('#e8f0fb')),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')]),
    ('TOPPADDING',  (0,0), (-1,-1), 5),
    ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
]))
story.append(meta_table)
story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 第1节：方法来源——CALM论文
# ════════════════════════════════════════════════════════
story.append(Paragraph("1. Method Background: CALM Paper", h1_style))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("1.1  What is CALM?", h2_style))
story.append(Paragraph(
    "<b>CALM (Continuous Autoregressive Language Models)</b> is proposed by Shao et al. (2025, arXiv:2510.27688). "
    "It transforms a standard LLM from predicting the next <i>discrete token</i> into predicting the next "
    "<i>continuous vector</i>. The core architecture compresses K tokens into a single dense vector using a "
    "high-fidelity autoencoder (reconstruction accuracy >99.9%), and the language model learns to predict "
    "the sequence of such vectors autoregressively.",
    body_style
))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("1.2  BrierLM: The Key Evaluation Metric", h2_style))
story.append(Paragraph(
    "Because CALM operates in continuous vector space, standard perplexity (which requires discrete token probabilities) "
    "is no longer applicable. CALM introduces <b>BrierLM</b> as a likelihood-free evaluation metric:",
    body_style
))
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph(
    "Brier(P, y) = 2 * P(y) - SUM_x P(x)^2",
    formula_style
))
story.append(Paragraph(
    "BrierLM = GeoMean( Brier-1, Brier-2, Brier-4 ) * 100",
    formula_style
))
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph(
    "The formula has two elegant components: <b>+2·P(y)</b> rewards high probability assigned to the correct target, "
    "while <b>-SUM P(x)²</b> penalizes distribution collapse (when all predictions converge to a single point). "
    "The geometric mean over multiple granularities (K=1,2,4) measures multi-scale quality.",
    body_style
))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("1.3  Compression Method: Can It Be Borrowed?", h2_style))
story.append(Paragraph(
    "CALM's autoencoder compresses K consecutive tokens into one vector. The compression ratio K is a key hyperparameter "
    "— it defines the <i>semantic bandwidth</i> of a single vector. This is directly analogous to a question in "
    "our experiment:",
    body_style
))

borrow_data = [
    ["CALM Concept", "Our Experiment Analog", "Can Borrow?"],
    ["K tokens → 1 vector", "Chunk methods: chunk size K", "YES (design)"],
    ["Autoencoder bottleneck dim l", "Final embedding dim (384)", "PARTIAL"],
    ["Autoencoder reconstruction loss", "EOSk metric", "YES (concept)"],
    ["BrierLM (likelihood-free eval)", "VecBrierLM (this work)", "YES (implemented)"],
    ["Semantic bandwidth scaling", "C_concat expands dim", "YES (future work)"],
]
borrow_table = Table(borrow_data, colWidths=[5*cm, 5.5*cm, 3.5*cm])
borrow_table.setStyle(TableStyle([
    ('FONTNAME',    (0,0), (-1,-1), BASE_FONT),
    ('FONTNAME',    (0,0), (-1,0), BOLD_FONT),
    ('FONTSIZE',    (0,0), (-1,-1), 9),
    ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#1a3a5c')),
    ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f5ff')]),
    ('TOPPADDING',  (0,0), (-1,-1), 5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ('LEFTPADDING', (0,0), (-1,-1), 7),
    # 绿色标注 YES 行
    ('BACKGROUND', (2,1), (2,1), colors.HexColor('#c8e6c9')),
    ('BACKGROUND', (2,3), (2,3), colors.HexColor('#c8e6c9')),
    ('BACKGROUND', (2,4), (2,4), colors.HexColor('#c8e6c9')),
    ('BACKGROUND', (2,2), (2,2), colors.HexColor('#fff9c4')),
    ('BACKGROUND', (2,5), (2,5), colors.HexColor('#fff9c4')),
]))
story.append(borrow_table)
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph(
    "<b>On the compression method specifically:</b> CALM's autoencoder (K tokens → 1 vector) is most analogous "
    "to our Chunk methods, especially <i>C_mean</i> (average K token embeddings into 1) and "
    "<i>C_concat</i> (preserve all K embeddings by concatenation). "
    "The key insight from CALM is that the compression ratio K is itself a <i>scaling axis</i> — "
    "larger K encodes more context but demands higher fidelity from the downstream model. "
    "In our setting, <b>C_concat is the loss-free \"lossless compression\"</b> (it keeps all information), "
    "while <b>C_mean is \"lossy compression\"</b> (it averages, inevitably losing some diversity). "
    "VecBrierLM confirms this: C_concat scores ~98.4 while C_mean scores ~75.7 on average.",
    body_style
))
story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 第2节：VecBrierLM 指标定义
# ════════════════════════════════════════════════════════
story.append(Paragraph("2. VecBrierLM: Adapting CALM to Vector Evaluation", h1_style))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("2.1  Adaptation Design", h2_style))
story.append(Paragraph(
    "The original BrierLM uses the LLM's softmax output as the probability distribution P. "
    "In our setting, there is no generative model — we have only a set of dense vectors. "
    "We adapt the idea by using <b>cosine similarity softmax</b> as the implicit distribution:",
    body_style
))
story.append(Paragraph(
    "sim_i(j)  = cosine( v_i, v_j )",
    formula_style
))
story.append(Paragraph(
    "P_i(j)  = softmax( sim_i / temperature )[j]     (temperature = 0.05)",
    formula_style
))
story.append(Paragraph(
    "VecBrier@k(i) = 2 * P_i(i_true) - SUM_j P_i(j)^2    (top-k candidates)",
    formula_style
))
story.append(Paragraph(
    "VecBrierLM = GeoMean( VecBrier@1, VecBrier@10, VecBrier@50 ) * 100",
    formula_style
))
story.append(Spacer(1, 0.1*cm))

story.append(Paragraph(
    "<b>Interpretation:</b> For each question vector v_i, we throw it into the pool of all 1,787 "
    "question vectors and ask: <i>\"can it find itself?\"</i>  "
    "A high score means the vector is both accurate (finds itself at top) "
    "and diverse (not confused with other vectors). A low score indicates "
    "either poor retrieval accuracy or distribution collapse.",
    highlight_style
))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("2.2  Differences from Existing Evaluation Methods", h2_style))
compare_data = [
    ["Metric", "Requires API", "Speed (64 combos)", "Detects Collapse", "Measures"],
    ["Reconstruction_v2\n(KP-F1+LLM-Judge)", "YES (Qwen)", "~3 hours", "Indirect", "Text regeneration quality"],
    ["Recall@10", "No", "Fast", "No", "Retrieval hit rate (KP)"],
    ["EOSk", "No", "Fast", "No", "Subspace alignment"],
    ["Silhouette", "No", "Fast", "No", "Cluster quality"],
    ["VecBrierLM\n(this work)", "No", "14 seconds", "DIRECT", "Vector ID-ability + diversity"],
]
compare_table = Table(compare_data, colWidths=[3.8*cm, 2.5*cm, 3*cm, 2.5*cm, 4.8*cm])
compare_table.setStyle(TableStyle([
    ('FONTNAME',    (0,0), (-1,-1), BASE_FONT),
    ('FONTNAME',    (0,0), (-1,0), BOLD_FONT),
    ('FONTSIZE',    (0,0), (-1,-1), 8.5),
    ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#1a3a5c')),
    ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ('TOPPADDING',  (0,0), (-1,-1), 5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    # VecBrierLM行高亮
    ('BACKGROUND',  (0,5), (-1,5), colors.HexColor('#e8f5e9')),
    ('FONTNAME',    (0,5), (-1,5), BOLD_FONT),
]))
story.append(compare_table)
story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 第3节：实验结果图表
# ════════════════════════════════════════════════════════
story.append(Paragraph("3. Experimental Results", h1_style))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("3.1  Overall VecBrierLM Heatmap", h2_style))
story.append(Paragraph(
    "The heatmap below shows VecBrierLM scores for all 64 Token×Chunk combinations "
    "(N=1,787 questions, color range 0-100). Green = high quality, Red = collapse/low quality.",
    body_style
))
story.append(Spacer(1, 0.2*cm))
story.append(make_main_heatmap())
story.append(Paragraph(
    "Figure 1: VecBrierLM heatmap (8x8 combinations). "
    "T_max/T_min x C_mean/C_weighted (top-left region) show severe collapse (~33); "
    "T_product x C_product scores 17.9 (worst); "
    "nearly all C_concat/C_diff/C_combined combinations score >97.",
    caption_style
))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("3.2  Grouped Average Performance", h2_style))
story.append(make_row_col_bar())
story.append(Paragraph(
    "Figure 2: Average VecBrierLM score by Token method (left) and Chunk method (right). "
    "Orange dashed line marks threshold=80.",
    caption_style
))

story.append(PageBreak())

story.append(Paragraph("3.3  Multi-Scale Decomposition (Brier@k=1/10/50)", h2_style))
story.append(Paragraph(
    "VecBrierLM is the geometric mean of three scale-specific Brier scores. "
    "Examining each scale separately reveals how quickly vector quality degrades as the candidate pool grows:",
    body_style
))
story.append(Spacer(1, 0.1*cm))
story.append(make_k_comparison())
story.append(Paragraph(
    "Figure 3: Multi-scale Brier scores. Note that T_max/T_min x C_mean combinations score well at k=1 "
    "(~80) but collapse drastically at k=10 (~35) and k=50 (~13), indicating that these vectors can "
    "identify the 'obvious' nearest neighbor but fail to maintain discriminability in the broader pool.",
    caption_style
))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("3.4  Accuracy vs Coverage Trade-off", h2_style))
story.append(Paragraph(
    "The scatter plot below contrasts immediate accuracy (Brier@k=1) against broad coverage (Brier@k=50). "
    "Points above the diagonal indicate vectors that maintain high quality even in large candidate pools.",
    body_style
))
story.append(make_scatter_k1_k50())
story.append(Paragraph(
    "Figure 4: Brier@k=1 vs Brier@k=50 scatter. Points near the diagonal maintain consistent quality "
    "across scales. The catastrophically collapsed combinations (T_max/T_min x C_mean/C_weighted) "
    "appear at the lower-right, showing high k=1 accuracy but near-zero k=50 coverage.",
    caption_style
))

story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 第4节：数值结果汇总表
# ════════════════════════════════════════════════════════
story.append(Paragraph("4. Numerical Summary", h1_style))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("4.1  Top-10 and Bottom-10 Combinations", h2_style))

# 排序
flat_scores = [(brier_mat[ti,ci], t_names[ti], c_names[ci]) for ti in range(8) for ci in range(8)]
flat_scores.sort(key=lambda x: x[0], reverse=True)

top_data = [["Rank", "Token Method", "Chunk Method", "VecBrierLM", "Brier@k=1", "Brier@k=50"]]
for rank, (score, t, c) in enumerate(flat_scores[:10], 1):
    key = f"{t}\u00d7{c}"
    d = detail[key]
    top_data.append([str(rank), t, c, f"{score:.2f}", f"{d['brier_k1']:.2f}", f"{d['brier_k50']:.2f}"])

top_table = Table(top_data, colWidths=[1.2*cm, 3.5*cm, 3.5*cm, 3*cm, 3*cm, 3*cm])
top_table.setStyle(TableStyle([
    ('FONTNAME',    (0,0), (-1,-1), BASE_FONT),
    ('FONTNAME',    (0,0), (-1,0), BOLD_FONT),
    ('FONTSIZE',    (0,0), (-1,-1), 8.5),
    ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#2c6e2e')),
    ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f1faf1'), colors.white]),
    ('TOPPADDING',  (0,0), (-1,-1), 4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('ALIGN', (3,1), (-1,-1), 'CENTER'),
]))
story.append(top_table)
story.append(Paragraph("Table 1: Top-10 combinations by VecBrierLM score.", caption_style))

story.append(Spacer(1, 0.3*cm))

bottom_data = [["Rank", "Token Method", "Chunk Method", "VecBrierLM", "Brier@k=1", "Brier@k=50"]]
for rank, (score, t, c) in enumerate(reversed(flat_scores[-10:]), 1):
    key = f"{t}\u00d7{c}"
    d = detail[key]
    bottom_data.append([str(rank), t, c, f"{score:.2f}", f"{d['brier_k1']:.2f}", f"{d['brier_k50']:.2f}"])

bot_table = Table(bottom_data, colWidths=[1.2*cm, 3.5*cm, 3.5*cm, 3*cm, 3*cm, 3*cm])
bot_table.setStyle(TableStyle([
    ('FONTNAME',    (0,0), (-1,-1), BASE_FONT),
    ('FONTNAME',    (0,0), (-1,0), BOLD_FONT),
    ('FONTSIZE',    (0,0), (-1,-1), 8.5),
    ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#8b2020')),
    ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fff5f5'), colors.white]),
    ('TOPPADDING',  (0,0), (-1,-1), 4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('ALIGN', (3,1), (-1,-1), 'CENTER'),
]))
story.append(bot_table)
story.append(Paragraph("Table 2: Bottom-10 combinations (worst performing).", caption_style))

story.append(PageBreak())

# ════════════════════════════════════════════════════════
# 第5节：分析与结论
# ════════════════════════════════════════════════════════
story.append(Paragraph("5. Analysis and Conclusions", h1_style))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.2*cm))

story.append(Paragraph("5.1  Key Findings", h2_style))

findings = [
    ("<b>Finding 1: Chunk method dominates, Token method plays a secondary role.</b>  "
     "C_concat / C_diff / C_combined consistently achieve VecBrierLM > 97 regardless of Token method. "
     "These \"lossless\" chunk methods preserve the full information content of K token embeddings. "
     "This mirrors CALM's insight: concatenation (like CALM's K=1, no compression) maximally preserves "
     "the semantic bandwidth of each token."),
    ("<b>Finding 2: T_max / T_min combined with C_mean / C_weighted cause catastrophic collapse.</b>  "
     "These combinations score ~33-34 on VecBrierLM. The pattern is revealing: Brier@k=1 remains ~80 "
     "(short-range accuracy survives), but Brier@k=50 drops to ~13 (long-range discriminability collapses). "
     "This is precisely the collapse behavior that CALM's BrierLM was designed to detect."),
    ("<b>Finding 3: T_product x C_product achieves the absolute minimum (17.9).</b>  "
     "Element-wise multiplication of embeddings drives many dimensions toward zero, creating "
     "near-degenerate vectors. CALM's -SUM P(x)^2 term directly penalizes this: "
     "when all vectors look the same, the softmax distributes probability uniformly, "
     "and the Brier score approaches 0."),
    ("<b>Finding 4: T_mean / T_weighted / T_first / T_last are consistently safe Token methods.</b>  "
     "These methods average or select single token embeddings that preserve the typical embedding "
     "distribution. Combined with any non-degenerate Chunk method, they achieve VecBrierLM > 85."),
    ("<b>Finding 5: Multi-scale evaluation is essential.</b>  "
     "Using only Brier@k=1 would mask the collapse of T_max/T_min combinations "
     "(they score ~80 at k=1 but ~13 at k=50). The geometric mean penalizes "
     "poor performance at any scale — a key advantage over single-scale metrics like Recall@10."),
]
for f in findings:
    story.append(Paragraph(f"• {f}", bullet_style))
    story.append(Spacer(1, 0.15*cm))

story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("5.2  On Borrowing CALM's Compression Method", h2_style))
story.append(Paragraph(
    "CALM uses a high-fidelity autoencoder to compress K tokens into a single l-dimensional vector, "
    "with reconstruction accuracy >99.9%. The key question for our experiment is: "
    "<b>can we train a similar autoencoder for chemistry question embeddings?</b>",
    body_style
))
story.append(Spacer(1, 0.1*cm))

story.append(Paragraph("<b>What can be directly borrowed:</b>", body_style))
borrow_items = [
    "The K-scaling idea: systematically vary K (number of token embeddings to compress) "
    "and measure how VecBrierLM changes with K. This would reveal the \"semantic bandwidth\" "
    "of chemistry question embeddings.",
    "The bottleneck dimension l: test whether projecting to smaller dimensions (e.g., 64, 128) "
    "before combining improves VecBrierLM by forcing more discriminative representations.",
    "The evaluation philosophy: BrierLM as a likelihood-free metric is directly applicable "
    "to any vector space evaluation problem where generating text is expensive or unavailable.",
]
for item in borrow_items:
    story.append(Paragraph(f"  ✓  {item}", bullet_style))

story.append(Spacer(1, 0.15*cm))
story.append(Paragraph("<b>What requires adaptation:</b>", body_style))
adapt_items = [
    "CALM's autoencoder is trained end-to-end with a generative LM. In our setting, "
    "the base encoder (all-MiniLM-L6-v2) is frozen, so we can only apply post-hoc "
    "aggregation (Token/Chunk methods) rather than learned compression.",
    "CALM's K tokens are consecutive in a natural text sequence. Our K \"tokens\" are "
    "actually semantically related knowledge point chunks — a different kind of sequence "
    "where position matters less than semantic content.",
    "Training an autoencoder from scratch for chemistry embeddings is feasible future work, "
    "but would require a large chemistry corpus aligned with our dataset format.",
]
for item in adapt_items:
    story.append(Paragraph(f"  ○  {item}", bullet_style))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("5.3  Recommended Combinations", h2_style))

rec_data = [
    ["Category", "Recommended Combinations", "VecBrierLM"],
    ["Best (lossless)", "Any T x C_concat / C_diff / C_combined", "97-99"],
    ["Good (balanced)", "T_mean/T_weighted/T_first/T_last x C_mean/C_weighted", "91-95"],
    ["Acceptable", "T_product x C_combined", "90.8"],
    ["Poor (avoid)", "T_max/T_min x C_mean/C_weighted", "33-34"],
    ["Worst (avoid)", "T_product x C_product", "17.9"],
]
rec_table = Table(rec_data, colWidths=[3.5*cm, 8.5*cm, 3*cm])
rec_table.setStyle(TableStyle([
    ('FONTNAME',    (0,0), (-1,-1), BASE_FONT),
    ('FONTNAME',    (0,0), (-1,0), BOLD_FONT),
    ('FONTSIZE',    (0,0), (-1,-1), 9),
    ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#1a3a5c')),
    ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
    ('BACKGROUND',  (0,1), (-1,1), colors.HexColor('#c8e6c9')),
    ('BACKGROUND',  (0,2), (-1,2), colors.HexColor('#dcedc8')),
    ('BACKGROUND',  (0,3), (-1,3), colors.HexColor('#fff9c4')),
    ('BACKGROUND',  (0,4), (-1,4), colors.HexColor('#ffccbc')),
    ('BACKGROUND',  (0,5), (-1,5), colors.HexColor('#ffcdd2')),
    ('TOPPADDING',  (0,0), (-1,-1), 5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ('LEFTPADDING', (0,0), (-1,-1), 7),
]))
story.append(rec_table)

story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph(
    "<b>Reference:</b> Shao, C., et al. (2025). Continuous Autoregressive Language Models. "
    "arXiv preprint arXiv:2510.27688.",
    ParagraphStyle('ref', parent=body_style, fontSize=8.5, textColor=colors.HexColor('#555555'))
))

# ── 生成 PDF ──────────────────────────────────────────────────────
doc.build(story)
print(f"PDF saved: {OUTPUT}")
