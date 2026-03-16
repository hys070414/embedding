Text Representation for Chemical Problems: A Compositional Study

This repository contains code and experimental results for studying how to best aggregate text representations at different granularities (token-level and chunk-level) for chemical problems. The goal is to find the most effective way to combine embeddings to preserve semantic information, which can be used in downstream tasks such as knowledge-enhanced reinforcement learning.



 Overview
Chemical questions often contain multiple knowledge points. To obtain a fixed‑dimensional representation of the whole question, we need to combine embeddings of these points. This study investigates **how different aggregation methods affect the information retained** in the final vector. Two levels are considered:

Chunk‑level: each knowledge point is already a textual unit (extracted by a large language model). We directly combine the embeddings of these points.
Token‑level + Chunk‑level: we first combine the tokens inside each knowledge point to obtain its embedding, and then combine the resulting point embeddings into a question vector.

We evaluate each combination with unsupervised metrics that measure structural fidelity, retrieval accuracy, and class separability.

---

 Experiments

Experiment 1: Single‑level Chunk Aggregation
**Goal**: Compare 9 methods for directly combining knowledge point embeddings.

- Data: 1,787 chemical questions. Knowledge points are extracted using Qwen (`qwen-plus`) from each question’s `input` field. Each point is embedded with Qwen `text-embedding-v2` (dimension 1536).
- Aggregation methods (applied to the list of point embeddings):
  - `horizontal` / `vertical`: concatenation (same result after flattening)
  - `average`
  - `weighted_average` (uniform weights, same as average)
  - `elementwise_product`
  - `diff` (first – last)
  - `combined` (concat + average + product)
  - `max_pool`
  - `min_pool`
- Metrics:
  - EOSk: spectral fidelity – alignment of residual subspaces after removing dominant components (higher = better).
  - Robustness: cosine similarity between the original vector and the vector obtained after randomly shuffling the order of points (measures order sensitivity).
  - Recall@5: fraction of a question’s own points that appear among the top‑5 retrieved by the question vector.

Results:  
- `average` and `max_pool` performed best (Recall@5 > 0.63, EOSk ≈ 1, robustness ≈ 1).
- `elementwise_product` and `diff` were poor (Recall@5 < 0.21) and should be avoided.

See [experiment_1_report](link-to-pdf) for details.

---

Experiment 2: Two‑level Token‑Chunk Aggregation
Goal: Investigate the combined effect of token‑level and chunk‑level aggregation (9 × 9 = 81 combinations). This mimics a scenario where only raw text is available and we must build representations from tokens upwards.

- Data: Same 1,787 questions. Knowledge points are again extracted by Qwen. For each point, we obtain token embeddings using `sentence-transformers/all-MiniLM-L6-v2` (dimension 384). A maximum of 50 tokens per point is kept.
- Token‑level methods (combine tokens inside one point into a point vector):
  - `T_mean`, `T_max`, `T_min`, `T_concat`, `T_weighted` (uniform), `T_product`, `T_first`, `T_last`, `T_attention` (trained on a question‑type classification task).
- Chunk‑level methods** (combine point vectors into a question vector):
  - `C_mean`, `C_max`, `C_min`, `C_concat`, `C_weighted`, `C_product`, `C_diff`, `C_combined`, `C_attention` (trained similarly).
- Metrics:
  - EOSk (projected to 384‑D)
  - Recall@10
  - Separation = intra‑class cosine similarity − inter‑class cosine similarity (higher = better class discrimination)
  - Silhouette coefficient (higher = better clustering)

Results:  
- The best overall combination is `T_mean` + `C_mean` (or `C_concat`/`C_weighted`). It achieves the highest Recall@10 (~0.359) and Separation (~0.089) while keeping EOSk = 1.
- `T_first`, `T_last` and any combination involving `C_product` or `C_diff` perform very poorly (negative Silhouette, low recall).
- The trained attention layers (`T_attention`, `C_attention`) do not outperform simple averaging, suggesting that for this task static averaging is already optimal.



---

 Metrics
| Metric        | Formula / Description                                                                 | Range      | Interpretation                     |
|---------------|---------------------------------------------------------------------------------------|------------|-------------------------------------|
|EOSk     | Subspace overlap after removing top‑2 components                                     | [0, 1]     | Higher means more structural preservation |
| Recall@k  | Fraction of a question’s own points retrieved in top‑k                               | [0, 1]     | Higher means better local detail retention |
| Separation| Intra‑class similarity minus inter‑class similarity                                   | [-1, 1]    | Higher means better class separation |
| Silhouette| (b - a) / max(a,b) averaged over all samples                                          | [-1, 1]    | >0 indicates sensible clustering    |

---

 Results Summary
| Experiment                | Best Method(s)          | Key Observations                                                                 |
|---------------------------|-------------------------|----------------------------------------------------------------------------------|
|Single‑level (points)| `average`, `max_pool`   | Recall@5 ≈ 0.63, EOSk ≈ 1; `elementwise_product` and `diff` fail.              |
| Two‑level (token×chunk)| `T_mean` + `C_mean`     | Recall@10 ≈ 0.359, Separation ≈ 0.089; simple averaging outperforms all others. |
| Avoid                 | `T_first`, `T_last`, `C_product`, `C_diff` | These methods destroy semantic structure (negative silhouette, low recall).      |

---

