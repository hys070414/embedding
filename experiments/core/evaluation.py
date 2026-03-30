import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.linalg import svd

# ==================== CALM-inspired BrierLM ====================
# 灵感来源：CALM (Shao et al., 2025, arxiv 2510.27688)
# 原始 BrierLM 用于评估"无似然"LLM；此处适配为"向量分布质量"评估：
# 将余弦相似度经 softmax 转化为隐式概率分布，计算 Brier Score。
# 衡量：向量能否以高概率"从候选池中找到自身"（准确性）同时保持分布多样性（避免坍塌）

def _softmax(x, temperature=0.05):
    """数值稳定的 softmax（低温度 → 更锐利的分布，对质量更敏感）"""
    x = np.array(x, dtype=np.float64) / temperature
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()

def compute_brierlm(query_vectors, temperature=0.05, subsample=300):
    """
    计算 VecBrierLM（CALM BrierScore 的向量空间适配版本）。

    VecBrierLM = 几何均值(VecBrier_k=1, VecBrier_k=10, VecBrier_k=50) × 100
    其中 VecBrier_k：在 top-k 候选中，向量"找到自身"的 Brier Score 均值。

    设计原则（对应 CALM 论文）：
      - 2·P(true) 项：奖励高保真度（自身相似度高）
      - −Σ P(x)² 项：惩罚分布坍塌（所有向量太相似则该项大，分数低）
    
    参数
    ----
    query_vectors : (N, d) array-like
    temperature   : softmax 温度（默认 0.05，较低以增强区分度）
    subsample     : 最多使用的样本数
    
    返回
    ----
    brierlm : float  （越高越好，0~100）
    detail  : dict   {'brier_k1', 'brier_k10', 'brier_k50', 'vec_brier_full', 'brierlm'}
    """
    Q_list = []
    for v in query_vectors:
        v = np.array(v, dtype=np.float32).flatten()
        Q_list.append(v)
    
    # 统一维度：若向量维度不一致，取最小公倍数或分段平均投影到最常见维度
    dims = [v.shape[0] for v in Q_list]
    target_d = int(np.median(dims))  # 取中位数维度作为目标
    Q_uniform = []
    for v in Q_list:
        d = v.shape[0]
        if d == target_d:
            Q_uniform.append(v)
        elif d > target_d:
            if d % target_d == 0:
                Q_uniform.append(v.reshape(-1, target_d).mean(axis=0))
            else:
                idx = np.linspace(0, d - 1, target_d).astype(int)
                Q_uniform.append(v[idx])
        else:
            out = np.zeros(target_d, dtype=np.float32)
            out[:d] = v
            Q_uniform.append(out)
    Q = np.array(Q_uniform, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    N = Q.shape[0]

    # L2 归一化
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    Q = Q / norms

    # 子采样
    if N > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, subsample, replace=False)
        Q_sub = Q[idx]
        true_indices = idx  # 在全量 Q 中的真实位置
    else:
        Q_sub = Q
        true_indices = np.arange(N)

    n_sub = Q_sub.shape[0]

    # 相似度矩阵 (n_sub, N)
    sim_matrix = Q_sub @ Q.T

    def brier_topk(k):
        scores = []
        for i in range(n_sub):
            sims = sim_matrix[i]
            true_i = true_indices[i]
            # top-k 候选（排除自身以外的样本，然后重新加入自身）
            if N > k:
                top_idx = np.argpartition(sims, -(k + 1))[-(k + 1):]
            else:
                top_idx = np.arange(N)
            # 确保 true_i 在候选中
            if true_i not in top_idx:
                top_idx = np.append(top_idx[:-1], true_i)
            cand_sims = sims[top_idx]
            P = _softmax(cand_sims, temperature=temperature)
            true_pos = np.where(top_idx == true_i)[0][0]
            P_true = P[true_pos]
            sum_P_sq = np.sum(P ** 2)
            bs = 2.0 * P_true - sum_P_sq
            scores.append(bs)
        return float(np.mean(scores)) * 100.0

    bk1  = brier_topk(1)
    bk10 = brier_topk(10)
    bk50 = brier_topk(50)

    # 全量 Brier（无 top-k 限制）
    full_scores = []
    for i in range(n_sub):
        sims = sim_matrix[i]
        true_i = true_indices[i]
        P = _softmax(sims, temperature=temperature)
        P_true = P[true_i]
        sum_P_sq = np.sum(P ** 2)
        full_scores.append(2.0 * P_true - sum_P_sq)
    vec_brier_full = float(np.mean(full_scores)) * 100.0

    # 几何均值（类比 CALM BrierLM）
    vals = [max(v, 1e-6) for v in [bk1, bk10, bk50]]
    brierlm = float(np.prod(vals) ** (1.0 / 3.0))

    detail = {
        "brier_k1": bk1,
        "brier_k10": bk10,
        "brier_k50": bk50,
        "vec_brier_full": vec_brier_full,
        "brierlm": brierlm,
    }
    return brierlm, detail

def project_to_d(vec, d):
    """将向量分段平均投影到 d 维（要求总长度是 d 的整数倍）"""
    total_len = vec.shape[0]
    if total_len % d != 0:
        raise ValueError(f"向量长度 {total_len} 不是 {d} 的整数倍")
    n_segments = total_len // d
    return vec.reshape(n_segments, d).mean(axis=0)

def compute_eosk(orig_matrix, comp_matrix, k=2, n_sub=10):
    """计算 EOSk 谱保真度"""
    if orig_matrix.ndim != 2 or comp_matrix.ndim != 2:
        print(f"警告: orig_matrix 维度 {orig_matrix.ndim}, comp_matrix 维度 {comp_matrix.ndim}")
        return np.nan
    n_samples = orig_matrix.shape[0]
    if n_samples == 0:
        return np.nan
    if n_samples < k + n_sub:
        print(f"警告: 样本数 {n_samples} 小于 k+sub ({k+n_sub})，无法计算 EOSk")
        return np.nan

    def remove_top_components(X, k_):
        U, s, Vt = svd(X, full_matrices=False)
        X_res = X - U[:, :k_] @ np.diag(s[:k_]) @ Vt[:k_, :]
        return X_res

    X_res = remove_top_components(orig_matrix, k)
    Z_res = remove_top_components(comp_matrix, k)
    U_X, _, _ = svd(X_res, full_matrices=False)
    U_Z, _, _ = svd(Z_res, full_matrices=False)
    U_X_sub = U_X[:, :n_sub]
    U_Z_sub = U_Z[:, :n_sub]
    P_X = U_X_sub @ U_X_sub.T
    P_Z = U_Z_sub @ U_Z_sub.T
    overlap = np.trace(P_X @ P_Z) / n_sub
    return overlap

def compute_retrieval_recall(question_vectors, block_vectors, block_qid_list, k=5):
    """用题目向量检索知识点块，计算 Recall@k"""
    recall_sum = 0
    total_questions = len(question_vectors)
    block_mat = np.array(block_vectors)
    for i, q_vec in enumerate(question_vectors):
        sims = cosine_similarity([q_vec], block_mat)[0]
        top_k_idx = np.argsort(sims)[-k:][::-1]
        qid = i
        hits = sum(1 for idx in top_k_idx if block_qid_list[idx] == qid)
        recall_sum += hits / min(k, len(block_vectors))
    return recall_sum / total_questions

def compute_intra_inter_separation(question_vectors, labels):
    """
    计算类内平均相似度、类间平均相似度、分离度（intra - inter）以及轮廓系数。
    question_vectors: (n_samples, d) numpy array
    labels: list of ints, 每个题目的类型编码
    返回: (separation_score, silhouette_score)
    """
    if len(question_vectors) < 2 or len(set(labels)) < 2:
        return np.nan, np.nan

    # 类内平均相似度
    unique_labels = np.unique(labels)
    intra_sims = []
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        vecs = question_vectors[idx]
        sim_matrix = cosine_similarity(vecs)
        # 去掉自身
        np.fill_diagonal(sim_matrix, 0)
        intra_sim = sim_matrix.sum() / (len(idx) * (len(idx) - 1))
        intra_sims.append(intra_sim)
    intra_avg = np.mean(intra_sims) if intra_sims else np.nan

    # 类间平均相似度（不同类之间）
    inter_sims = []
    for i, lbl_i in enumerate(unique_labels):
        for j, lbl_j in enumerate(unique_labels):
            if i >= j:
                continue
            idx_i = np.where(labels == lbl_i)[0]
            idx_j = np.where(labels == lbl_j)[0]
            vecs_i = question_vectors[idx_i]
            vecs_j = question_vectors[idx_j]
            sims = cosine_similarity(vecs_i, vecs_j).flatten()
            inter_sims.append(np.mean(sims))
    inter_avg = np.mean(inter_sims) if inter_sims else np.nan

    separation = intra_avg - inter_avg
    # 轮廓系数
    sil = silhouette_score(question_vectors, labels)
    return separation, sil