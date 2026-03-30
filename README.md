# Embedding Evaluation Experiments

化学题目向量表示评估实验，探究不同 Token 聚合方式与 Chunk 聚合方式对向量质量的影响。

## 项目结构

```
embedding/
├── experiments/
│   ├── core/                  # 核心实验框架
│   │   ├── experiment.py          - 主实验入口（64种组合）
│   │   ├── evaluation.py          - 6项评估指标
│   │   ├── token_methods.py       - 8种Token聚合方法
│   │   ├── chunk_methods.py       - 8种Chunk聚合方法
│   │   ├── data_loader.py         - 数据集加载
│   │   ├── config.py              - 全局配置
│   │   ├── run.py                 - 快速运行入口
│   │   ├── run_full.py            - 全量运行入口
│   │   ├── generate_report.py     - 报告生成
│   │   ├── visualization.py       - 结果可视化
│   │   ├── validate_pipeline.py   - 管道验证
│   │   └── show_weighted_results.py - 加权结果展示
│   │
│   ├── brier/                 # VecBrierLM 评估
│   │   ├── brier_eval.py          - BrierLM 评估模块
│   │   └── run_brier_eval.py      - 全量评估入口
│   │
│   ├── weighted_chunk/        # 知识点加权聚合实验
│   │   ├── weighted_chunk_exp.py        - C_sim_weighted / C_idf_weighted
│   │   ├── eval_weighted_all_metrics.py - 加权方法全量5指标评估
│   │   ├── dynamic_kp_weight.py         - C_dynamic_weighted（LLM动态权重）
│   │   └── eval_dynamic_all_metrics.py  - 动态权重全量5指标评估
│   │
│   ├── calm_adapter/          # CALM Adapter 压缩实验
│   │   ├── calm_adapter.py        - 标准注意力自编码器
│   │   ├── calm_adapter_llm.py    - LLM-in-the-Loop 版本
│   │   ├── train_attention.py     - 注意力模块训练
│   │   └── train_inverter.py      - Inverter 训练
│   │
│   └── reconstruction/        # 向量重建质量评估
│       ├── reconstruction_v2.py       - KNN + Qwen 重建评估
│       ├── reconstruction_eval.py     - 重建评估工具函数
│       └── reconstruction_beam.py.py  - Beam Search 重建
│
├── results/                   # 实验结果数据
│   ├── results_matrix.npy         - (8,8,5) 5指标矩阵
│   ├── brier_results.npy          - (8,8) VecBrierLM 分数矩阵
│   ├── brier_detail.json          - BrierLM 各k值细节
│   ├── calm_adapter_results.json  - Adapter 实验结果
│   ├── calm_adapter_llm_results.json - LLM Adapter 结果
│   ├── weighted_full_metrics.json - 加权方法5指标结果
│   ├── weighted_results.json      - 加权方法对比数据
│   ├── dynamic_full_metrics.json  - 动态权重5指标结果
│   ├── dynamic_kp_results.json    - 动态权重实验结果
│   └── dynamic_kp_weights_cache.json - 权重缓存
│
├── utils/                     # 工具与调试脚本
│   ├── test_api_kp.py             - API 知识点提取测试
│   ├── test_brier.py              - BrierLM 测试
│   ├── test_kp_fixed.py           - 知识点修复测试
│   ├── test_zsinvert.py           - ZSInvert 测试
│   └── debug_kp.py                - 知识点提取调试
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
