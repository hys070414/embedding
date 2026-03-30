"""
对两种新加权方法 C_sim_weighted / C_idf_weighted 运行全部 5 个指标评估：
  0. EOSk          谱保真度
  1. Recall@10     检索准确率（KP 块检索）
  2. Separation    类内-类间余弦相似度差
  3. Silhouette    轮廓系数
  4. BrierLM       CALM-inspired 向量分布质量

数据来源：results_data.pkl（64 组合，含 C_concat 供拆 KP 用）
类型标签：从 dataset_without_preference.jsonl 读取 type 字段

输出：
  weighted_full_metrics.json   — 所有组合的 5 指标详情
  weighted_full_metrics.npy    — (2, 8, 5) 矩阵 [新chunk方法, token方法, 指标]
"""

import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ── 项目本地模块 ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluation import (
    compute_eosk, compute_retrieval_recall,
    compute_intra_inter_separation, compute_brierlm
)
from data_loader import load_questions_from_jsonl

BASE = Path(__file__).parent
DIM  = 384
JSONL_PATH = BASE / "dataset_without_preference.jsonl"
DATA_PATH  = BASE / "results_data.pkl"
OUTPUT_JSON = BASE / "weighted_full_metrics.json"
OUTPUT_NPY  = BASE / "weighted_full_metrics.npy"


def softmax(x, temperature=0.1):
    x = np.array(x, dtype=np.float64) / temperature
    x -= x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-15)


def cosine_sim_to_mat(a, B):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return B_n @ a_n


def aggregate_sim_weighted(kp_vecs_list):
    result = []
    for vecs in kp_vecs_list:
        vecs = np.array(vecs, dtype=np.float32)
        if len(vecs) <= 1:
            result.append(vecs[0] if len(vecs) == 1 else np.zeros(DIM, dtype=np.float32))
            continue
        mean_v = vecs.mean(axis=0)
        sims   = cosine_sim_to_mat(mean_v, vecs)
        w      = softmax(sims, 0.1)
        result.append((w[:, None] * vecs).sum(axis=0).astype(np.float32))
    return np.array(result, dtype=np.float32)


def aggregate_idf_weighted(kp_vecs_list, global_center):
    result = []
    gc_n   = global_center / (np.linalg.norm(global_center) + 1e-10)
    for vecs in kp_vecs_list:
        vecs = np.array(vecs, dtype=np.float32)
        if len(vecs) <= 1:
            result.append(vecs[0] if len(vecs) == 1 else np.zeros(DIM, dtype=np.float32))
            continue
        v_n   = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        rarity = 1.0 - (v_n @ gc_n.astype(np.float32))
        w      = softmax(rarity, 0.1)
        result.append((w[:, None] * vecs).sum(axis=0).astype(np.float32))
    return np.array(result, dtype=np.float32)


def main():
    print("=" * 65)
    print("新加权方法全量 5 指标评估")
    print("=" * 65)

    # ── 1. 加载题目（获取 type 标签）────────────────────────────
    print("\n[1/6] 加载题目数据 ...")
    questions = load_questions_from_jsonl(str(JSONL_PATH), None)
    N = len(questions)
    print(f"  题目总数: {N}")
    types = [q.get("type", "unknown") for q in questions]
    le    = LabelEncoder()
    type_labels = le.fit_transform(types)
    print(f"  类型数: {len(le.classes_)}")

    # ── 2. 加载向量数据 ─────────────────────────────────────────
    print("\n[2/6] 加载 results_data.pkl ...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    vectors_data = {k: v["vectors"] for k, v in data.items()}
    print(f"  已加载 {len(vectors_data)} 个组合")

    t_names = ["T_mean", "T_max", "T_min", "T_concat",
               "T_weighted", "T_product", "T_first", "T_last"]

    # ── 3. 拆解 KP 向量（用 T_mean × C_concat）────────────────
    print("\n[3/6] 从 C_concat 拆解知识点向量 ...")
    cc_key = ("T_mean", "C_concat")
    cc_raws = vectors_data[cc_key]
    kp_vecs_list = []
    for raw in cc_raws:
        row   = np.array(raw, dtype=np.float32)
        n_kp  = row.shape[0] // DIM
        kp_vecs_list.append(row.reshape(n_kp, DIM))
    print(f"  共 {len(kp_vecs_list)} 道题目，知识点数：{set(len(v) for v in kp_vecs_list)}")

    # ── 4. 计算全局中心 & 两种加权 Chunk 向量 ──────────────────
    print("\n[4/6] 计算加权 Chunk 向量 ...")
    global_center = np.vstack(kp_vecs_list).mean(axis=0)
    chunk_sim  = aggregate_sim_weighted(kp_vecs_list)        # (N, 384)
    chunk_idf  = aggregate_idf_weighted(kp_vecs_list, global_center)  # (N, 384)
    print(f"  C_sim_weighted shape: {chunk_sim.shape}")
    print(f"  C_idf_weighted shape: {chunk_idf.shape}")

    # ── 5. 构建 block_vectors（投影后的 KP 向量，用于 Recall@10）
    # 直接用 C_concat 里的 KP 向量，每道题贡献 n_kp 个 KP 块
    print("\n[5/6] 构建 KP 块向量索引（用于 Recall@10）...")
    block_vectors  = []
    block_qid_list = []
    for qid, kps in enumerate(kp_vecs_list):
        for kp in kps:
            block_vectors.append(kp)
            block_qid_list.append(qid)
    print(f"  KP 块总数: {len(block_vectors)}")

    # ── 6. 逐 Token 方法 × 2 新 Chunk 方法 评估全部指标 ────────
    print("\n[6/6] 计算 5 个指标 ...")

    new_chunk_names = ["C_sim_weighted", "C_idf_weighted"]
    new_chunk_vecs  = {"C_sim_weighted": chunk_sim,
                       "C_idf_weighted": chunk_idf}

    # 结果矩阵 [c_idx, t_idx, metric_idx]
    results_mat  = np.full((2, 8, 5), np.nan)
    results_dict = {}

    # 用于 EOSk：orig_avg（T_mean × C_mean 作为原始均值基准）
    orig_avg = np.array(vectors_data[("T_mean", "C_mean")], dtype=np.float32)

    for ci, c_name in enumerate(new_chunk_names):
        c_chunk_vecs = new_chunk_vecs[c_name]   # (N, 384)

        for ti, t_name in enumerate(tqdm(t_names, desc=c_name, leave=False)):
            key_label = f"{t_name}×{c_name}"

            # 获取对应 Token 方法的 C_mean 向量，反推 token 层向量
            cm_key = (t_name, "C_mean")
            if cm_key not in vectors_data:
                print(f"  ⚠ {cm_key} 不存在，跳过")
                continue
            t_cm_vecs  = np.array(vectors_data[cm_key], dtype=np.float32)  # (N,384)
            # C_mean 即 KP 均值，反推 token_vecs = 2*combined - c_mean
            c_mean_vecs = np.array([kps.mean(axis=0) for kps in kp_vecs_list], dtype=np.float32)
            t_vecs      = 2.0 * t_cm_vecs - c_mean_vecs   # (N, 384)

            # 最终向量：token + new_chunk 的平均（与 C_mean 构造逻辑一致）
            doc_vecs = (t_vecs + c_chunk_vecs) / 2.0      # (N, 384)
            doc_mat  = doc_vecs.astype(np.float32)

            metric_result = {}

            # ① EOSk
            try:
                eosk = compute_eosk(orig_avg, doc_mat, k=2, n_sub=5)
                results_mat[ci, ti, 0] = eosk
                metric_result["eosk"] = float(eosk)
            except Exception as e:
                print(f"  EOSk 失败 {key_label}: {e}")

            # ② Recall@10
            try:
                recall = compute_retrieval_recall(
                    doc_mat, block_vectors, block_qid_list, k=10)
                results_mat[ci, ti, 1] = recall
                metric_result["recall_10"] = float(recall)
            except Exception as e:
                print(f"  Recall@10 失败 {key_label}: {e}")

            # ③ Separation  ④ Silhouette
            try:
                sep, sil = compute_intra_inter_separation(doc_mat, type_labels)
                results_mat[ci, ti, 2] = sep
                results_mat[ci, ti, 3] = sil
                metric_result["separation"] = float(sep)
                metric_result["silhouette"] = float(sil)
            except Exception as e:
                print(f"  Sep/Sil 失败 {key_label}: {e}")

            # ⑤ BrierLM
            try:
                bl, bdetail = compute_brierlm(doc_mat, subsample=500)
                results_mat[ci, ti, 4] = bl
                metric_result["brierlm"] = float(bl)
                metric_result.update({f"brier_{k}": v for k, v in bdetail.items()
                                      if k != "brierlm"})
            except Exception as e:
                print(f"  BrierLM 失败 {key_label}: {e}")

            results_dict[key_label] = metric_result
            tqdm.write(f"  {key_label}: EOSk={metric_result.get('eosk', float('nan')):.4f}  "
                       f"Recall@10={metric_result.get('recall_10', float('nan')):.4f}  "
                       f"Sep={metric_result.get('separation', float('nan')):.4f}  "
                       f"Sil={metric_result.get('silhouette', float('nan')):.4f}  "
                       f"BrierLM={metric_result.get('brierlm', float('nan')):.2f}")

    # ── 7. 计算各指标均值（8 种 Token 方法平均）────────────────
    metric_names = ["eosk", "recall_10", "separation", "silhouette", "brierlm"]
    for ci, c_name in enumerate(new_chunk_names):
        for mi, mn in enumerate(metric_names):
            vals = [results_dict.get(f"{t}×{c_name}", {}).get(mn, np.nan)
                    for t in t_names]
            valid = [v for v in vals if not np.isnan(v)]
            avg_key = f"AVG_{c_name}_{mn}"
            results_dict[avg_key] = float(np.mean(valid)) if valid else float("nan")

    # ── 8. 对比：从原始结果矩阵读取基线 ──────────────────────────
    orig_mat = np.load(BASE / "results_matrix.npy")    # (8, 8, 4)
    with open(BASE / "brier_detail.json", "r", encoding="utf-8") as f:
        bd = json.load(f)
    base_t_names = bd["t_names"]
    base_c_names = bd["c_names"]
    SEP = chr(0x00D7)

    for ref_c, ci_r in [(cn, i) for i, cn in enumerate(base_c_names)]:
        for mi, mn in enumerate(metric_names):
            if mi < 4:  # EOSk/Recall/Sep/Sil 来自 results_matrix.npy
                vals = [orig_mat[ti, ci_r, mi] for ti in range(8)]
            else:        # BrierLM 来自 brier_detail.json
                vals = [bd["detail"].get(f"{t}{SEP}{ref_c}", {}).get("brierlm", np.nan)
                        for t in base_t_names]
            valid = [v for v in vals if not np.isnan(v)]
            results_dict[f"BASELINE_{ref_c}_{mn}"] = float(np.mean(valid)) if valid else float("nan")

    # ── 9. 保存 ──────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    np.save(OUTPUT_NPY, results_mat)
    print(f"\n已保存: {OUTPUT_JSON}")
    print(f"已保存: {OUTPUT_NPY}")

    # ── 10. 打印汇总表 ───────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"{'方法':38s}  {'EOSk':>6}  {'Rec@10':>7}  {'Sep':>7}  {'Sil':>7}  {'BrierLM':>8}")
    print("-" * 85)
    for t in t_names:
        for c in ["C_sim_weighted", "C_idf_weighted"]:
            k = f"{t}×{c}"
            r = results_dict.get(k, {})
            print(f"  {k:38s}  "
                  f"{r.get('eosk', float('nan')):6.4f}  "
                  f"{r.get('recall_10', float('nan')):7.4f}  "
                  f"{r.get('separation', float('nan')):7.4f}  "
                  f"{r.get('silhouette', float('nan')):7.4f}  "
                  f"{r.get('brierlm', float('nan')):8.2f}")

    print("\n  ── 8种Token方法均值 ──")
    for c in ["C_sim_weighted", "C_idf_weighted"]:
        print(f"\n  {c}:")
        for mn in metric_names:
            v = results_dict.get(f"AVG_{c}_{mn}", float("nan"))
            print(f"    {mn:15s} = {v:.4f}")

    print("\n  ── 原始方法基线（8种Token均值）──")
    for ref_c in ["C_mean", "C_weighted", "C_concat", "C_diff", "C_combined"]:
        print(f"\n  {ref_c}:")
        for mn in metric_names:
            v = results_dict.get(f"BASELINE_{ref_c}_{mn}", float("nan"))
            print(f"    {mn:15s} = {v:.4f}")

    print("\n完成！")
    return results_dict, results_mat


if __name__ == "__main__":
    main()
