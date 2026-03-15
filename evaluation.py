import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.linalg import svd

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