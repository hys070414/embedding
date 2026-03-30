"""
BrierScore 评估模块
==========================================
来源灵感：CALM (Continuous Autoregressive Language Models, Shao et al. 2025)
论文链接：https://arxiv.org/abs/2510.27688

CALM 的核心贡献之一是提出 BrierLM 作为"无似然"语言模型评估指标，
替代传统困惑度（Perplexity）。本模块将此思路迁移到"向量拼接方法质量评估"场景：

==========  CALM 的 Brier Score 定义  ==========
对于一个预测分布 P 和真实结果 y（离散词元），Brier 分数：
    Brier(P, y) = 2·P(y) − Σ_x P(x)²
蒙特卡洛无偏估计（从 P 采两个样本 x1, x2）：
    Brier(P, y) ≈ I{x1=y} + I{x2=y} − I{x1=x2}
    前两项衡量"准确性"，后一项惩罚"模式坍塌"（多样性）。

==========  本实验的适配  ==========
在向量检索场景中，我们没有显式的 P(y)，但可以将"检索相似度分布"视为隐式概率分布：
- 将向量间的余弦相似度经 softmax 转化为概率分布 P
- 对每个查询向量 q，构造候选集（KNN 返回的 top-K 文本对应的向量），
  计算 Brier Score：即当前向量在候选集中"找到正确答案"的准确性 + 多样性惩罚项

具体实现（VecBrier）：
- 设查询向量为 q，候选向量矩阵为 V（K×d），真实目标为 v_true
- 相似度向量 s = cosine(q, V)，softmax(s) → P
- Brier(P, v_true) ≈ P[true] + sample_hit − P[s1=s2]（用 top-1/top-2 近似）

更直观的简化版（本模块采用）：
- 对 N 个查询向量，将余弦相似度 softmax 后，计算：
    score_i = 2·P[i, i] − Σ_j P[i,j]²
  （P[i,j] = softmax(cosine(q_i, V))[j]，i 是自身的索引）
- VecBrierLM = 均值 × 100

这个指标：
① 不需要任何 API 调用（纯本地）
② 对语义保真度敏感：好的向量应让自身相似度远高于其他
③ 惩罚模式坍塌（所有向量都很相似则 Σ P²大，分数低）
④ 与 CALM 的 BrierLM 在设计原则上一致
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def softmax(x, temperature=1.0):
    """数值稳定的 softmax"""
    x = np.array(x, dtype=np.float64) / temperature
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def compute_vec_brier(query_vectors, corpus_vectors=None, temperature=0.1, subsample=500):
    """
    计算 VecBrier 分数（CALM BrierScore 的向量空间适配版本）。

    参数
    ----
    query_vectors : (N, d) array-like
        查询向量矩阵（每行一个题目向量）
    corpus_vectors : (M, d) array-like, optional
        候选语料库向量。若为 None，则用 query_vectors 本身（自检索模式）
    temperature : float
        softmax 温度，控制分布锐度（默认 0.1，较低温度使分布更锐利）
    subsample : int
        为加速，最多使用 subsample 条查询（随机抽样）

    返回
    ----
    vec_brier : float
        VecBrier 综合分数（越高越好，理论上限约为 100）
    brier_n : dict
        {"brier_1": ..., "brier_2": ..., "brier_4": ...}
        类比 CALM 的 Brier-1/2/4（n-gram 窗口对应，本模块中为不同 top-K 下的 Brier）
    """
    Q = np.array(query_vectors, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)

    N = Q.shape[0]

    # 归一化
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    Q = Q / norms

    if corpus_vectors is None:
        C = Q  # 自检索模式：语料库 = 查询集
    else:
        C = np.array(corpus_vectors, dtype=np.float32)
        norms_c = np.linalg.norm(C, axis=1, keepdims=True)
        norms_c[norms_c < 1e-10] = 1.0
        C = C / norms_c

    # 子采样（避免超大矩阵计算卡死）
    if N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        Q_sub = Q[idx]
        true_idx = idx  # 在 corpus 中的真实位置（自检索时 = Q 的行号）
    else:
        Q_sub = Q
        true_idx = np.arange(N)

    # 计算余弦相似度矩阵 (n_sub, M)
    sim_matrix = Q_sub @ C.T  # 已归一化，直接内积 = 余弦相似度

    # 对每个 query，计算 Brier Score
    brier_scores = []
    for i, (q_sims, true_i) in enumerate(zip(sim_matrix, true_idx)):
        P = softmax(q_sims, temperature=temperature)
        P_true = P[true_i]
        sum_P_sq = np.sum(P ** 2)
        # Brier(P, y) = 2·P(y) − Σ P(x)²
        bs = 2.0 * P_true - sum_P_sq
        brier_scores.append(bs)

    vec_brier = float(np.mean(brier_scores)) * 100.0

    # 分 top-K 的 Brier 变体（类比 CALM 的 Brier-n）
    brier_n = {}
    for k_val, name in [(1, "brier_1"), (2, "brier_2"), (4, "brier_4")]:
        # 只保留 top-K 候选
        k_scores = []
        for i, (q_sims, true_i) in enumerate(zip(sim_matrix, true_idx)):
            top_k_idx = np.argpartition(q_sims, -k_val)[-k_val:]
            # 检查真实索引是否在 top-K 中（精确命中率分量）
            hit = float(true_i in top_k_idx)
            # softmax 只在 top-K 候选上
            top_k_sims = q_sims[top_k_idx]
            P_k = softmax(top_k_sims, temperature=temperature)
            sum_P_sq_k = np.sum(P_k ** 2)
            # 若命中则 P_true = max(P_k)，否则 0
            P_true_k = float(np.max(P_k)) * hit
            bs_k = 2.0 * P_true_k - sum_P_sq_k
            k_scores.append(bs_k)
        brier_n[name] = float(np.mean(k_scores)) * 100.0

    return vec_brier, brier_n


def compute_brierlm(query_vectors, corpus_vectors=None, temperature=0.1, subsample=500):
    """
    VecBrierLM：类比 CALM 的 BrierLM = 几何均值(Brier-1, Brier-2, Brier-4) × 100
    
    这是最终的汇总指标，用于比较不同 Token×Chunk 组合的向量质量。

    返回
    ----
    brierlm : float
        几何均值形式的 VecBrierLM（越高越好）
    detail : dict
        包含 vec_brier（全量版）和 brier_n（分档版）的详细分数
    """
    vec_brier, brier_n = compute_vec_brier(
        query_vectors, corpus_vectors, temperature=temperature, subsample=subsample
    )
    vals = [max(brier_n[k], 1e-6) for k in ["brier_1", "brier_2", "brier_4"]]
    brierlm = float(np.prod(vals) ** (1.0 / len(vals)))

    detail = {"vec_brier": vec_brier, **brier_n, "brierlm": brierlm}
    return brierlm, detail


# ===== 快速自测 =====
if __name__ == "__main__":
    np.random.seed(42)
    print("=== VecBrierLM 自测 ===\n")

    # Case 1: 理想向量（正交性好）
    N, d = 200, 384
    ideal_vecs = np.eye(N, d)  # 前 N 个标准基（极度不同）
    bl, det = compute_brierlm(ideal_vecs)
    print(f"[理想向量（近乎正交）] BrierLM = {bl:.4f}")
    print(f"  细节: {det}\n")

    # Case 2: 随机向量（中等分离度）
    rand_vecs = np.random.randn(N, d)
    bl2, det2 = compute_brierlm(rand_vecs)
    print(f"[随机向量] BrierLM = {bl2:.4f}")
    print(f"  细节: {det2}\n")

    # Case 3: 坍塌向量（所有向量相同 → 应得最低分）
    collapse_vecs = np.ones((N, d)) + np.random.randn(N, d) * 0.001
    bl3, det3 = compute_brierlm(collapse_vecs)
    print(f"[坍塌向量（近乎相同）] BrierLM = {bl3:.4f}")
    print(f"  细节: {det3}\n")

    print("预期: 理想 >> 随机 > 坍塌（BrierLM 应从高到低）")
