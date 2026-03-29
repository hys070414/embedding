"""
对 C_dynamic_weighted 方法运行全部 5 个指标评估：
  0. EOSk          谱保真度
  1. Recall@10     检索准确率（KP 块检索）
  2. Separation    类内-类间余弦相似度差
  3. Silhouette    轮廓系数
  4. BrierLM       CALM-inspired 向量分布质量

权重来源：dynamic_kp_weights_cache.json（已有缓存，无需重新调 API）
输出：dynamic_full_metrics.json — 5 指标详情
"""

import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from evaluation import (
    compute_eosk, compute_retrieval_recall,
    compute_intra_inter_separation, compute_brierlm
)
from data_loader import load_questions_from_jsonl

BASE = Path(__file__).parent
DIM  = 384
JSONL_PATH    = BASE / "dataset_without_preference.jsonl"
DATA_PATH     = BASE / "results_data.pkl"
WEIGHTS_CACHE = BASE / "dynamic_kp_weights_cache.json"
OUTPUT_JSON   = BASE / "dynamic_full_metrics.json"

T_NAMES = ["T_mean", "T_max", "T_min", "T_concat",
           "T_weighted", "T_product", "T_first", "T_last"]


def main():
    print("=" * 65)
    print("C_dynamic_weighted 全量 5 指标补测")
    print("=" * 65)

    # ── 1. 加载题目类型标签 ──────────────────────────────────────
    print("\n[1/6] 加载题目数据 ...")
    questions = load_questions_from_jsonl(str(JSONL_PATH), None)
    N = len(questions)
    print(f"  题目总数: {N}")
    types = [q.get("type", "unknown") for q in questions]
    le = LabelEncoder()
    type_labels = le.fit_transform(types)
    print(f"  类型数: {len(le.classes_)}")

    # ── 2. 加载向量数据 ─────────────────────────────────────────
    print("\n[2/6] 加载 results_data.pkl ...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    vectors_data = {k: v["vectors"] for k, v in data.items()}
    print(f"  已加载 {len(vectors_data)} 个组合")

    # ── 3. 拆解 KP 向量 ─────────────────────────────────────────
    print("\n[3/6] 从 C_concat 拆解知识点向量 ...")
    cc_key = ("T_mean", "C_concat")
    cc_raws = vectors_data[cc_key]
    kp_vecs_list = []
    for raw in cc_raws:
        row  = np.array(raw, dtype=np.float32)
        n_kp = row.shape[0] // DIM
        kp_vecs_list.append(row.reshape(n_kp, DIM))
    print(f"  共 {len(kp_vecs_list)} 道题目，KP 数：{set(len(v) for v in kp_vecs_list)}")

    # ── 4. 加载 LLM 动态权重并聚合 ──────────────────────────────
    print("\n[4/6] 加载 LLM 动态权重并聚合 ...")
    with open(WEIGHTS_CACHE, "r", encoding="utf-8") as f:
        raw_cache = json.load(f)
    all_weights = [np.array(raw_cache[str(i)], dtype=np.float32) for i in range(N)]
    print(f"  已加载 {len(all_weights)} 道题的权重")

    # 动态加权聚合
    c_dynamic_vecs = []
    for vecs, w in zip(kp_vecs_list, all_weights):
        vecs = np.array(vecs, dtype=np.float32)
        if len(vecs) == 1:
            c_dynamic_vecs.append(vecs[0])
            continue
        if len(w) != len(vecs):
            w = np.ones(len(vecs), dtype=np.float32) / len(vecs)
        agg = (w[:, None] * vecs).sum(axis=0)
        c_dynamic_vecs.append(agg)
    c_dynamic_vecs = np.array(c_dynamic_vecs, dtype=np.float32)
    print(f"  C_dynamic_weighted shape: {c_dynamic_vecs.shape}")

    # ── 5. 构建 KP 块向量索引（Recall@10 用）────────────────────
    print("\n[5/6] 构建 KP 块向量索引 ...")
    block_vectors  = []
    block_qid_list = []
    for qid, kps in enumerate(kp_vecs_list):
        for kp in kps:
            block_vectors.append(kp)
            block_qid_list.append(qid)
    print(f"  KP 块总数: {len(block_vectors)}")

    # 用于 EOSk：T_mean × C_mean 作为基准
    orig_avg = np.array(vectors_data[("T_mean", "C_mean")], dtype=np.float32)
    c_mean_vecs = np.array([kps.mean(axis=0) for kps in kp_vecs_list], dtype=np.float32)

    # ── 6. 逐 Token 方法评估全部指标 ─────────────────────────────
    print("\n[6/6] 计算 5 个指标 ...")
    results_dict = {}
    results_mat  = np.full((8, 5), np.nan)  # (t_idx, metric_idx)
    metric_names = ["eosk", "recall_10", "separation", "silhouette", "brierlm"]

    for ti, t_name in enumerate(tqdm(T_NAMES, desc="C_dynamic_weighted")):
        key_label = f"{t_name}×C_dynamic_weighted"

        cm_key = (t_name, "C_mean")
        if cm_key not in vectors_data:
            print(f"  ⚠ {cm_key} 不存在，跳过")
            continue
        t_cm_vecs = np.array(vectors_data[cm_key], dtype=np.float32)
        # 反推 token 向量
        t_vecs   = 2.0 * t_cm_vecs - c_mean_vecs
        doc_vecs = (t_vecs + c_dynamic_vecs) / 2.0
        doc_mat  = doc_vecs.astype(np.float32)

        metric_result = {}

        # ① EOSk
        try:
            eosk = compute_eosk(orig_avg, doc_mat, k=2, n_sub=5)
            results_mat[ti, 0] = eosk
            metric_result["eosk"] = float(eosk)
        except Exception as e:
            print(f"  EOSk 失败 {key_label}: {e}")

        # ② Recall@10
        try:
            recall = compute_retrieval_recall(doc_mat, block_vectors, block_qid_list, k=10)
            results_mat[ti, 1] = recall
            metric_result["recall_10"] = float(recall)
        except Exception as e:
            print(f"  Recall@10 失败 {key_label}: {e}")

        # ③ Separation  ④ Silhouette
        try:
            sep, sil = compute_intra_inter_separation(doc_mat, type_labels)
            results_mat[ti, 2] = sep
            results_mat[ti, 3] = sil
            metric_result["separation"] = float(sep)
            metric_result["silhouette"] = float(sil)
        except Exception as e:
            print(f"  Sep/Sil 失败 {key_label}: {e}")

        # ⑤ BrierLM
        try:
            bl, bdetail = compute_brierlm(doc_mat, subsample=500)
            results_mat[ti, 4] = bl
            metric_result["brierlm"] = float(bl)
        except Exception as e:
            print(f"  BrierLM 失败 {key_label}: {e}")

        results_dict[key_label] = metric_result
        tqdm.write(f"  {key_label}: EOSk={metric_result.get('eosk', float('nan')):.4f}  "
                   f"Recall@10={metric_result.get('recall_10', float('nan')):.4f}  "
                   f"Sep={metric_result.get('separation', float('nan')):.4f}  "
                   f"Sil={metric_result.get('silhouette', float('nan')):.4f}  "
                   f"BrierLM={metric_result.get('brierlm', float('nan')):.2f}")

    # ── 7. 计算 8 种 Token 方法均值 ────────────────────────────
    print("\n  ── 8 种 Token 方法均值 ──")
    for mi, mn in enumerate(metric_names):
        vals = [results_dict.get(f"{t}×C_dynamic_weighted", {}).get(mn, np.nan)
                for t in T_NAMES]
        valid = [v for v in vals if not np.isnan(v)]
        avg = float(np.mean(valid)) if valid else float("nan")
        results_dict[f"AVG_C_dynamic_weighted_{mn}"] = avg
        print(f"    {mn:15s} = {avg:.4f}")

    # ── 8. 保存 ─────────────────────────────────────────────────
    np.save(BASE / "dynamic_full_metrics.npy", results_mat)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {OUTPUT_JSON}")
    print(f"已保存: dynamic_full_metrics.npy")
    print("\n完成！")


if __name__ == "__main__":
    main()
