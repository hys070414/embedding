// generate_weekly_docx.js
// 生成周报 Word 文档

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, HorizontalPositionAlign,
  PageBreak, LevelFormat, TableOfContents,
} = require('docx');
const fs = require('fs');
const path = require('path');

const BASE = 'C:\\Users\\hwm20\\OneDrive\\桌面\\Research\\embedding';
const FIGS = path.join(BASE, 'docx_figs');
const OUT  = path.join(BASE, 'weekly_report.docx');

// ── 颜色/尺寸常量 ──────────────────────────────────────────────────────
const CLR_TITLE   = '1a3a5c';
const CLR_H1      = '2E75B6';
const CLR_H2      = '2E75B6';
const CLR_BODY    = '000000';
const CLR_CAPTION = '555555';
const CLR_BORDER  = 'BBBBBB';
const PAGE_W = 11906;  // A4 宽 DXA
const PAGE_H = 16838;  // A4 高 DXA
const MARGIN = 1134;   // ~2cm
const CONTENT_W = PAGE_W - MARGIN * 2;  // ~9638

// ── 辅助函数 ──────────────────────────────────────────────────────────
const sp = (pt) => new Paragraph({ spacing: { before: 0, after: pt * 20 } });

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text, bold: true, size: 28, color: CLR_H1, font: 'Arial' })],
  spacing: { before: 320, after: 120 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: CLR_H1, space: 4 } },
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text, bold: true, size: 24, color: CLR_H2, font: 'Arial' })],
  spacing: { before: 200, after: 80 },
});

const B = (text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text, size: 21, font: 'Arial', color: CLR_BODY })],
  spacing: { before: 0, after: 80 },
});

// 支持多段的 body（接收 TextRun 数组）
const BP = (runs) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  children: runs,
  spacing: { before: 0, after: 80 },
});

const TR = (text, opts={}) => new TextRun({ text, size: 21, font: 'Arial', color: CLR_BODY, ...opts });
const CAP = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text, size: 18, italics: true, color: CLR_CAPTION, font: 'Arial' })],
  spacing: { before: 60, after: 160 },
});

const HR = () => new Paragraph({
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: CLR_H1, space: 1 } },
  spacing: { before: 0, after: 160 },
  children: [],
});

const cellBorder = { style: BorderStyle.SINGLE, size: 1, color: CLR_BORDER };
const allBorders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

// 加载图片
function loadImg(name) {
  const fp = path.join(FIGS, name);
  return fs.readFileSync(fp);
}

function imgParagraph(name, wPx, hPx, caption) {
  const ratio = hPx / wPx;
  const wDxa = Math.min(CONTENT_W, Math.round(wPx * 9.525));  // 1px≈9.525 DXA @96dpi
  const hDxa = Math.round(wDxa * ratio);
  const wEmu = wDxa * 914400 / 1440;
  const hEmu = hDxa * 914400 / 1440;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({
        type: 'png',
        data: loadImg(name),
        transformation: { width: Math.round(wEmu/914400*96), height: Math.round(hEmu/914400*96) },
        altText: { title: caption, description: caption, name: caption },
      })],
      spacing: { before: 100, after: 60 },
    }),
    CAP(caption),
  ];
}

// 表格构建
function makeTable(rows, colWidths) {
  const tblRows = rows.map((row, ri) => {
    const cells = row.map((cell, ci) => {
      const isHeader = ri === 0;
      const paras = String(cell).split('\n').map(line =>
        new Paragraph({
          children: [new TextRun({
            text: line,
            size: 18,
            bold: isHeader,
            font: 'Arial',
            color: isHeader ? 'FFFFFF' : CLR_BODY,
          })],
          spacing: { before: 40, after: 40 },
        })
      );
      return new TableCell({
        borders: allBorders,
        width: { size: colWidths[ci], type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        shading: isHeader
          ? { fill: CLR_H1, type: ShadingType.CLEAR }
          : (ri % 2 === 0 ? { fill: 'F5F8FC', type: ShadingType.CLEAR } : {}),
        verticalAlign: VerticalAlign.CENTER,
        children: paras,
      });
    });
    return new TableRow({ children: cells });
  });

  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tblRows,
  });
}

// ── 正文构建 ─────────────────────────────────────────────────────────
const children = [];
const add = (...items) => items.forEach(i => {
  if (Array.isArray(i)) i.forEach(x => children.push(x));
  else children.push(i);
});

// ── 封面 ──
add(
  sp(240),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: '化学题目向量评估实验', bold: true, size: 48, color: CLR_TITLE, font: 'Arial' })],
    spacing: { before: 0, after: 120 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Embedding Strategy Research Report', size: 28, color: '666666', font: 'Arial' })],
    spacing: { before: 0, after: 240 },
  }),
  HR(),
  sp(160),
);

// ── 一、CALM 论文借鉴与 VecBrierLM ──
add(
  H1('一、CALM 论文借鉴与 VecBrierLM 指标设计'),
  H2('1.1 CALM 论文'),
  B('Shao et al. (2025). Continuous Autoregressive Language Models. arXiv:2510.27688'),
  sp(60),
  B('核心思路：把 N 个 token 向量压缩成 K 个 summary tokens，K 越小压缩率越高。中等压缩（1<K<N）往往优于极端压缩（K=1）。本实验在 Chunk 级别做同样的聚合，与 CALM 直接类比；BrierLM 指标也直接借鉴用于评估向量区分能力。'),
  sp(40),
  B('本实验验证了同样结论：C_combined（中等压缩）96.4 > C_mean（极端压缩）75.7。'),
  sp(120),
  H2('1.2 VecBrierLM 指标'),
  B('用余弦相似度构造伪概率，再用 Brier Score 打分：'),
  BP([TR('① 对题目 i，计算与题库所有向量的余弦相似度，softmax 归一化得分布 P；')]),
  BP([TR('② '), TR('Brier(P, i) = 2*P(i) - sum_x P(x)^2', { bold: true }), TR('  第一项奖励自检索精度，第二项惩罚坍塌；')]),
  BP([TR('③ 三个窗口（k=1, 10, 50）几何均值×100，得 '), TR('VecBrierLM ∈ [0,100]', { bold: true }), TR('。')]),
  sp(40),
  B('纯本地计算，无需 API，全量 64 种组合 14 秒完成。'),
  sp(120),
  new Paragraph({ children: [new PageBreak()] }),
);

// ── 二、实验结果 ──
add(
  H1('二、实验结果'),
  H2('2.1 VecBrierLM 热力图'),
  B('数据集：1787 道英文化学题目；64 种拼接组合（8种Token方法 × 8种Chunk方法）；分数越高表示向量区分能力越强。'),
  sp(80),
  ...imgParagraph('fig1_heatmap.png', 1400, 840, '图 1  VecBrierLM 热力图（Token × Chunk 全组合）'),
  sp(120),
  H2('2.2 各方法平均得分'),
  ...imgParagraph('fig2_bar.png', 1600, 530, '图 2  Token 方法均值与 Chunk 方法均值对比（红 <70，橙 70-85，绿 >85）'),
  sp(120),
  H2('2.3 短程 vs 长程区分能力'),
  B('横轴 Brier@k=1：自检索精度（能否在最近邻中找回自身）；纵轴 Brier@k=50：大范围分布质量。点颜色为综合 VecBrierLM 分，右上角绿色为优，左下角红色为坍塌。'),
  sp(80),
  ...imgParagraph('fig3_scatter.png', 1100, 830, '图 3  Brier@k=1 vs Brier@k=50（点颜色 = VecBrierLM 综合分）'),
  sp(120),
  H2('2.4 排名'),
  makeTable(
    [
      ['排名', 'Token 方法', 'Chunk 方法', 'VecBrierLM'],
      ['1', 'T_max', 'C_diff', '98.5'],
      ['2', 'T_min', 'C_diff', '98.5'],
      ['3', 'T_last', 'C_combined', '98.5'],
      ['4', 'T_mean', 'C_diff', '98.5'],
      ['5', 'T_concat', 'C_diff', '98.5'],
      ['—', '—', '—', '—'],
      ['倒1', 'T_product', 'C_product', '17.9'],
      ['倒2', 'T_max', 'C_max', '33.2'],
      ['倒3', 'T_min', 'C_min', '33.5'],
      ['倒4', 'T_max', 'C_weighted', '33.7'],
      ['倒5', 'T_max', 'C_mean', '33.7'],
    ],
    [1440, 2880, 2880, 3238]
  ),
  sp(200),
  new Paragraph({ children: [new PageBreak()] }),
);

// ── 三、CALM Adapter 附加实验 ──
add(
  H1('三、CALM Adapter 附加实验'),
  H2('3.1 动机与模型结构'),
  B('C_combined 是手工设计的双通道规则，能否用可训练模块自动学出更好的压缩方式？'),
  sp(60),
  B('设计轻量注意力模块（CALM Adapter）：K 个 Chunk 向量 -> 多头注意力聚合 -> 1 个 384 维向量，再用全连接层还原成 K 个向量，以重建误差训练。参数量 325 万，注意力头数 8，隐层 512，AdamW + Cosine LR，训练 80 轮，CPU 约 3 分钟。'),
  sp(120),
  H2('3.2 损失函数'),
  B('总损失 = MSE 重建误差 + 0.05 × InfoNCE 对比损失。'),
  B('MSE：学会还原原始 Chunk；InfoNCE：同题目向量相互靠近，不同题目相互远离。'),
  sp(120),
  H2('3.3 实验结果'),
  B('下表中 C_mean/C_combined 子分数为 8 种 Token 方法的列均值；CALM Adapter 子分数为单独实验记录值；全矩阵均值为 64 种组合的总平均。'),
  sp(80),
  makeTable(
    [
      ['方法', '压缩方式', 'VecBrierLM', 'Brier@k=1\n(短程精确)', 'Brier@k=50\n(长程区分)'],
      ['C_mean',        '逐维均值（手工基线）',  '75.7',  '91.0',  '67.7'],
      ['C_combined',    '均值+最大值（手工）',   '96.4',  '97.6',  '95.4'],
      ['CALM Adapter',  '多头注意力（可训练）',  '76.6',  '97.3',  '偏低'],
      ['全矩阵均值',    '64种组合平均',          '84.6',  '—',     '—'],
    ],
    [2520, 2880, 2016, 2016, 1944]
  ),
  sp(120),
  H2('3.4 原因分析'),
  B('① 数据量不足：1787 条对 325 万参数严重偏少，模型过拟合训练集，泛化弱；'),
  B('② 缺语义监督：训练目标是还原 Chunk，不等于让压缩向量有区分度；C_combined 直接复用 Encoder 输出，天然保留语义几何结构；'),
  B('③ 无预训练 Decoder：CALM 原版用大型 LLM 提供语言先验，本实验 MLP 从零训练，信号质量低。'),
  sp(60),
  B('结论：无 GPU、无预训练 LLM、小数据条件下，可训练压缩打不过手工规则。'),
  sp(120),
  new Paragraph({ children: [new PageBreak()] }),
);

// ── 四、知识点加权聚合实验 ──
add(
  H1('四、知识点加权聚合实验'),
  H2('4.1 动机'),
  B('原有 C_mean 对所有知识点等权平均，丢失了各知识点对题目主旨的贡献差异。本实验探索两种无参数加权策略，在不增加任何可训练参数的情况下，让更"重要"的知识点向量在聚合时获得更高权重；并用全部 5 个指标（EOSk / Recall@10 / Separation / Silhouette / BrierLM）进行多维度评估。'),
  sp(120),
  H2('4.2 方法设计'),
  BP([TR('方案 C：C_sim_weighted（相似度加权）', { bold: true })]),
  B('① 计算题目均值向量 u = mean(KP[1]...KP[n])；'),
  B('② sim_i = cosine(u, KP[i])  —  越贴近题目中心的知识点权重越高；'),
  B('③ weights = softmax(sim / T),  T=0.1；'),
  B('④ 最终向量 = sum(weights[i] * KP[i])。'),
  B('物理含义：强调与题目整体方向最一致的知识点，抑制边缘知识点噪声。'),
  sp(80),
  BP([TR('方案 D：C_idf_weighted（稀有度加权，向量空间 IDF 近似）', { bold: true })]),
  B('① 计算全数据集所有知识点向量的均值作为「全局中心」g；'),
  B('② rarity_i = 1 - cosine(KP[i], g)  —  越偏离中心越稀有；'),
  B('③ weights = softmax(rarity / T)；'),
  B('④ 最终向量 = sum(weights[i] * KP[i])。'),
  B('物理含义：IDF 思想迁移到向量空间——频繁出现、语义平庸的知识点（近全局中心）权重低，罕见独特的知识点（远离中心）权重高。'),
  sp(120),
  H2('4.3 实验结果（全量 5 指标，8 种 Token 方法均值）'),
  B('全量 1787 道题目，8 种 Token 方法均值如下。EOSk/Recall@10/Separation 越高越好；Silhouette 越接近 1 越好（负值表明类间/类内区分弱）；BrierLM 越高越好。'),
  sp(80),
  makeTable(
    [
      ['方法',           'EOSk',   'Recall@10', 'Sep.',   'Silhouette', 'BrierLM'],
      ['C_mean（基线）', '1.0000', '0.3033',    '0.0494', '-0.0371',    '75.7'],
      ['C_sim_weighted', '0.6572', '0.2973',    '0.0490', '-0.0110',    '78.7'],
      ['C_idf_weighted', '0.6507', '0.2949',    '0.0494', '-0.0127',    '79.0'],
      ['C_concat（参考）','1.0000','0.3033',    '0.0494', '-0.0371',    '94.0'],
      ['C_combined（最优）','0.7157','0.2977',  '0.0501', '-0.0620',    '96.4'],
    ],
    [2736, 1440, 1584, 1368, 1584, 1512]
  ),
  sp(120),
  ...imgParagraph('fig4_5metrics.png', 1600, 660,
    '图 4  5 个指标归一化后的多方法对比（BrierLM 除以 100，Silhouette 映射到 [0,1]）'),
  ...imgParagraph('fig5_per_token.png', 1500, 680,
    '图 5  8 种 Token 方法下 C_mean / C_sim_weighted / C_idf_weighted / C_combined 的 BrierLM 逐一对比'),
  sp(120),
  H2('4.4 指标维度分析'),
  BP([TR('① BrierLM：新方法对 C_mean 显著提升', { bold: true })]),
  B('C_sim_weighted = 78.7，C_idf_weighted = 79.0，均高于 C_mean (75.7)，提升约 +3.1 点。加权策略保留了更强的语义区分信息，验证了加权的正向作用。'),
  sp(60),
  BP([TR('② EOSk：新方法低于 C_mean（0.6572 vs 1.0000）', { bold: true })]),
  B('EOSk 衡量的是与「原始等权均值」的谱子空间对齐程度；加权改变了聚合方向，自然偏离原始基准，故 EOSk 下降属预期行为，并非质量变差，而是空间结构与 C_mean 不同。'),
  sp(60),
  BP([TR('③ Recall@10：新方法与 C_mean 相当（0.2973 vs 0.3033）', { bold: true })]),
  B('用题目向量检索 KP 块的准确率基本持平，说明加权未破坏题目对知识点的检索能力。'),
  sp(60),
  BP([TR('④ Silhouette：新方法显著优于 C_mean（-0.0110 vs -0.0371）', { bold: true })]),
  B('Silhouette 衡量类内紧密度与类间分离度；新方法轮廓系数更高（负得更少），表明加权聚合后同类题目的向量分布更聚集，类间分离更明显。'),
  sp(60),
  BP([TR('⑤ Separation：各方法相近（0.0490 / 0.0494 / C_mean=0.0494）', { bold: true })]),
  B('类内-类间余弦差在所有单向量方法（C_mean / sim / idf）之间差距极小，说明加权策略对题目类型区分度的影响有限；C_combined 略高（0.0501）得益于多分量设计。'),
  sp(120),
  H2('4.5 综合结论'),
  B('两种加权方法在 BrierLM + Silhouette 两个指标上均优于 C_mean，Recall@10 持平，EOSk 下降符合预期。整体优于朴素均值，但仍低于保留完整 KP 向量的 C_concat / C_combined，说明压缩为单向量时不可避免丢失知识点间的相对结构信息。后续可探索混合加权（中心相似度 × 稀有度）或引入 KP 拓扑结构进一步改进。'),
  sp(120),
  new Paragraph({ children: [new PageBreak()] }),
);

// ── 五、结论与下一步 ──
add(
  H1('五、结论与下一步'),
  H2('5.1 主要结论'),
  B('① 最优：T_max × C_diff（VecBrierLM=98.5），T_mean/T_weighted + C_combined/C_concat 整体最优；'),
  B('② 最差：T_product × C_product（VecBrierLM=17.9），逐元素乘积导致向量严重坍塌；'),
  B('③ C_combined 均值最高（96.4），与 CALM 中等压缩优于极端压缩结论一致；'),
  B('④ CALM Adapter 在小数据+CPU 条件下低于手工规则（76.6 vs 98.5），数据量是瓶颈；'),
  B('⑤ 新加权方法多指标表现：BrierLM 和 Silhouette 均高于 C_mean，Recall@10 持平，EOSk 下降符合预期（偏离等权基准的结构变化）；'),
  B('⑥ VecBrierLM 无需 API，14 秒完成全量评估，适合快速筛选方法。'),
  sp(120),
  H2('5.2 下一步方向'),
  BP([TR('① 更换 Encoder', { bold: true })]),
  B('用 all-mpnet-base-v2（768 维）替换 all-MiniLM-L6-v2，固定 T_mean x C_combined，对比更高维度向量的压缩特性差异。'),
  sp(80),
  BP([TR('② 替换相似度度量', { bold: true })]),
  B('VecBrierLM 的核心是：给每道题目的向量和题库里所有向量算相似度，softmax 后用 Brier Score 衡量分布质量。目前用余弦相似度，三种替代方案：'),
  sp(80),
  makeTable(
    [
      ['方案', '公式说明', '本实验含义', '实现方式'],
      ['双线性 Bilinear',
       'sim(u,v) = u^T W v\nW: d×d 可训练矩阵, d=384',
       '余弦只看方向夹角，Bilinear 让模型自己学「什么叫相似」。题目向量和知识点向量不在同一语义空间时，余弦会失真，Bilinear 能自适应校正。',
       '训练 W（约147K参数）\n用 Recall@10 做监督\n替换 brier_eval.py 中的 cosine_similarity'],
      ['L1 距离',
       '逐位相减取绝对值后求和=d；sim = 1/(1+d)\n（d=0时sim=1）',
       '余弦忽略向量长度，但384维向量里某些维度可能本身很大（噪声维度）。L1 逐维求差，对这类高维噪声更鲁棒。',
       '3 行代码替换 cosine_similarity，无需任何训练'],
      ['知识图谱辅助距离',
       'sim = a*cos(u,v) + (1-a)*w_KG\na 为混合权重\nw_KG 为图边权',
       '纯向量相似度只看统计共现，看不到化学概念之间的结构关系（如「酸碱反应->pH」）。引入知识图谱边权，把已知化学关系直接编入相似度计算中。',
       '需先构建化学知识图谱 (ChemOnt/PubChem)，映射题目->图节点，工作量较大，作为长期方向'],
    ],
    [1728, 2160, 3240, 2160]
  ),
  sp(120),
  BP([TR('③ 扩充数据集', { bold: true })]),
  B('从 ChemBench / SciQ 获取 >1 万道化学题，重测 CALM Adapter，验证数据量是否确实是可训练压缩的瓶颈。'),
  sp(80),
  BP([TR('④ 多指标联合评估', { bold: true })]),
  B('计算 VecBrierLM 与 Recall@10、Separation 的 Pearson 相关系数，判断各指标是否可互相替代，进一步精简评估流程。'),
  sp(240),
  HR(),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: '— 报告完 —', size: 18, color: '888888', font: 'Arial', italics: true })],
    spacing: { before: 60, after: 0 },
  }),
);

// ── 生成文档 ──────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: 'Arial', size: 21 } },
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Arial', color: CLR_H1 },
        paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0 },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Arial', color: CLR_H2 },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [new TextRun({ text: '化学题目向量评估实验  |  研究周报', size: 18, color: '888888', font: 'Arial' })],
          border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: CLR_BORDER, space: 1 } },
          alignment: AlignmentType.RIGHT,
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          children: [
            new TextRun({ text: 'Page ', size: 18, color: '888888', font: 'Arial' }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '888888', font: 'Arial' }),
            new TextRun({ text: ' / ', size: 18, color: '888888', font: 'Arial' }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: '888888', font: 'Arial' }),
          ],
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUT, buffer);
  console.log('Word 文档已生成:', OUT);
  console.log('文件大小:', (buffer.length / 1024).toFixed(1), 'KB');
}).catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
