# Embedding Evaluation Experiments

化学题目向量表示评估实验，探究不同 Token 聚合方式与 Chunk 聚合方式对向量质量的影响。

## 项目结构

```
embedding/
├── experiments/
│   ├── core/              # 核心实验框架
│   │   ├── experiment.py      - 主实验入口（64种组合）
│   │   ├── evaluation.py      - 6项评估指标
│   │   ├── token_methods.py   - 8种Token聚合方法
│   │   ├── chunk_methods.py   - 8种Chunk聚合方法
│   │   ├── data_loader.py     - 数据集加载
│   │   └── config.py          - 全局配置
│   │
│   ├── brier/             # VecBrierLM 评估
│   │   ├── brier_eval.py      - BrierLM 评估模块
│   │   └── run_brier_eval.py  - 全量评估入口
│   │
│   ├── weighted_chunk/    # 知识点加权聚合实验
│   │   ├── weighted_chunk_exp.py        - C_sim_weighted / C_idf_weighted
│   │   ├── eval_weighted_all_metrics.py - 加权方法全量5指标评估
│   │   ├── dynamic_kp_weight.py         - C_dynamic_weighted（LLM动态权重）
│   │   └── eval_dynamic_all_metrics.py  - 动态权重全量5指标评估
│   │
│   ├── calm_adapter/      # CALM Adapter 压缩实验
│   │   ├── calm_adapter.py     - 标准注意力自编码器
│   │   └── calm_adapter_llm.py - LLM-in-the-Loop 版本
│   │
│   └── reconstruction/    # 向量重建质量评估
│       ├── reconstruction_v2.py   - KNN + Qwen 重建评估
│       └── reconstruction_eval.py - 重建评估工具函数
│
├── utils/                 # 工具脚本
│   ├── debug_kp.py            - 知识点提取调试
│   ├── generate_brier_pdf.py  - BrierLM 报告生成
│   ├── export_figs_for_docx.py - 图表导出
│   ├── generate_weekly_docx.js - 周报 Word 生成
│   └── weekly_report_pdf.py   - 周报 PDF 生成
│
├── dataset_without_preference.jsonl  # 数据集（1787道化学题）
└── README.md
```

## 实验概览

### 实验框架
- **数据集**：1787道英文化学题目（含题目类型字段）
- **向量模型**：all-MiniLM-L6-v2（384维，固定不训练）
- **知识点提取**：Qwen-Plus API
- **实验规模**：8种Token方法 × 8种Chunk方法 = 64种组合

### 评估指标（6项）

| 指标 | 含义 |
|------|------|
| EOSk | 谱保真度（子空间结构对齐） |
| Recall@10 | 检索准确率（题目→知识点） |
| Separation | 类间-类内余弦相似度差 |
| Silhouette | 聚类轮廓系数 |
| Reconstruction_v2 | 向量→文本重建质量（KP-F1+LLM-Judge+ROUGE-L） |
| VecBrierLM | 向量分布质量（CALM论文启发，纯本地无API） |

### 主要结论
- **最优策略**：`T_mean`/`T_weighted` + `C_combined`（6指标均衡最优）
- **最差策略**：`T_product`（逐元素乘积），VecBrierLM 最低，向量坍塌严重
- **加权聚合**：`C_sim_weighted`/`C_idf_weighted`/`C_dynamic_weighted` 比 `C_mean` 提升约 +3点，但低于 `C_combined`
- **CALM Adapter**：数据量不足（1787条）导致效果低于基线

## 依赖

```bash
pip install sentence-transformers numpy scikit-learn torch openai
```

需要配置 Qwen API Key（`config.py` 或环境变量 `QWEN_API_KEY`）。
