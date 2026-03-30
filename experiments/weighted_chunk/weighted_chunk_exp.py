"""
知识点加权聚合实验
==========================================
方案C：C_sim_weighted  —— 余弦相似度加权（无参数 Cross-Attention 近似）
方案D：C_idf_weighted  —— 向量空间 IDF 近似（偏离全局中心距离）

思路：
- 从 results_data.pkl 中 C_concat 向量拆出知识点级向量（每 384 维一个知识点）
- C_sim_weighted：题目均值向量 与 各 KP 做 cosine → softmax(T=0.1) → 加权均值
- C_idf_weighted：计算全局 KP 中心 → 1 - cos(KP, 全局中心) → softmax(T=0.1) → 加权均值
- 评估用 VecBrierLM

数据结构说明（results_data.pkl）：
  key: (t_name, c_name) 元组，如 ('T_mean', 'C_concat')
  value: {'vectors': [list of ndarray], 'texts': [list of str]}
  其中 C_concat 的每个向量是多个知识点向量拼接，维度 = n_kp × 384
"""

import pickle
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from brier_eval import compute_brierlm

DATA_PATH = Path(__file__).parent / "results_data.pkl"
OUTPUT_JSON = Path(__file__).parent / "weighted_results.json"
DIM = 384  # all-MiniLM-L6-v2 向量维度


def softmax(x, temperature=0.1):
    x = np.array(x, dtype=np.float64) / temperature
    x -= x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-15)


def cosine_sim_vec_to_mat(a, B):
    """向量 a (d,) 和矩阵 B (n, d) 的余弦相似度，返回 (n,)"""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return B_norm @ a_norm


def aggregate_sim_weighted(kp_vecs_list):
    """
    C_sim_weighted：用题目均值向量对各 KP 做余弦相似度加权。
    kp_vecs_list: list of (n_kp, 384) arrays，每题的知识点向量列表
    返回: (N, 384) 加权均值向量
    """
    result = []
    for vecs in kp_vecs_list:
        vecs = np.array(vecs, dtype=np.float32)
        if len(vecs) <= 1:
            result.append(vecs[0] if len(vecs) == 1 else np.zeros(DIM, dtype=np.float32))
            continue
        mean_v = vecs.mean(axis=0)
        sims = cosine_sim_vec_to_mat(mean_v, vecs)  # (n_kp,)
        weights = softmax(sims, temperature=0.1)
        agg = (weights[:, None] * vecs).sum(axis=0)
        result.append(agg.astype(np.float32))
    return np.array(result, dtype=np.float32)


def aggregate_idf_weighted(kp_vecs_list, global_center):
    """
    C_idf_weighted：用 1 - cos(KP, 全局中心) 作为每个 KP 的"稀有度"权重。
    kp_vecs_list: list of (n_kp, 384) arrays
    global_center: (384,) 全局平均知识点向量
    返回: (N, 384) 加权均值向量
    """
    result = []
    gc_norm = global_center / (np.linalg.norm(global_center) + 1e-10)
    for vecs in kp_vecs_list:
        vecs = np.array(vecs, dtype=np.float32)
        if len(vecs) <= 1:
            result.append(vecs[0] if len(vecs) == 1 else np.zeros(DIM, dtype=np.float32))
            continue
        v_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        cos_to_center = v_norm @ gc_norm.astype(np.float32)  # (n_kp,)
        rarity = 1.0 - cos_to_center     # 越远离中心 → 越稀有
        weights = softmax(rarity, temperature=0.1)
        agg = (weights[:, None] * vecs).sum(axis=0)
        result.append(agg.astype(np.float32))
    return np.array(result, dtype=np.float32)


def main():
    print("=" * 60)
    print("知识点加权聚合实验")
    print("=" * 60)

    # ── 加载数据 ──────────────────────────────────────────────
    print(f"\n[1/5] 加载 {DATA_PATH.name} ...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    # 探测数据结构
    sample_key = next(iter(data))
    print(f"  数据类型: {type(data)}, 样本 key: {sample_key}")
    sample_val = data[sample_key]
    if isinstance(sample_val, dict):
        print(f"  value 结构: dict, keys={list(sample_val.keys())}")
        vectors_data = {k: v['vectors'] for k, v in data.items()}
    else:
        print(f"  value 结构: {type(sample_val)}")
        vectors_data = data

    t_names = ["T_mean", "T_max", "T_min", "T_concat", "T_weighted",
               "T_product", "T_first", "T_last"]

    # ── 提取 C_concat 知识点向量 ────────────────────────────────
    print("\n[2/5] 从 T_mean × C_concat 拆解知识点向量 ...")

    # 找到 C_concat 的 key（可能是元组或字符串）
    cc_key = None
    for k in vectors_data:
        if isinstance(k, tuple) and k == ('T_mean', 'C_concat'):
            cc_key = k
            break
        elif isinstance(k, str) and 'T_mean' in k and 'C_concat' in k:
            cc_key = k
            break

    if cc_key is None:
        # 找任意 C_concat
        for k in vectors_data:
            if isinstance(k, tuple) and k[1] == 'C_concat':
                cc_key = k
                break
            elif isinstance(k, str) and 'C_concat' in k:
                cc_key = k
                break
    
    print(f"  使用 key: {cc_key}")
    cc_vecs_raw = vectors_data[cc_key]  # list of (n_kp_i * 384,) arrays，长度不等
    N = len(cc_vecs_raw)

    # 拆解每题的知识点向量列表（每题 n_kp 可能不同）
    kp_vecs_list = []
    kp_counts = []
    for i in range(N):
        row = np.array(cc_vecs_raw[i], dtype=np.float32)
        total_dim = row.shape[0]
        n_kp_i = total_dim // DIM
        kps = row.reshape(n_kp_i, DIM)
        kp_vecs_list.append(kps)
        kp_counts.append(n_kp_i)

    from collections import Counter
    cnt = Counter(kp_counts)
    print(f"  N={N}, 知识点数分布: {dict(cnt)}")

    # ── 计算全局知识点中心 ──────────────────────────────────────
    print("\n[3/5] 计算全局知识点中心 ...")
    all_kps = np.vstack(kp_vecs_list)  # (N*n_kp, 384)
    global_center = all_kps.mean(axis=0)
    print(f"  全局中心 norm={np.linalg.norm(global_center):.4f}")

    # ── 计算两种加权向量 ────────────────────────────────────────
    print("\n[4/5] 计算 C_sim_weighted 和 C_idf_weighted ...")
    c_sim_weighted = aggregate_sim_weighted(kp_vecs_list)
    print(f"  C_sim_weighted shape: {c_sim_weighted.shape}")
    c_idf_weighted = aggregate_idf_weighted(kp_vecs_list, global_center)
    print(f"  C_idf_weighted shape: {c_idf_weighted.shape}")

    # ── 对 8 种 Token 方法 × 2 新 Chunk 方法运行 VecBrierLM ───────
    print("\n[5/5] 评估 VecBrierLM ...")
    results = {}

    # 确定 C_mean key 格式
    def find_key(t_name, c_name):
        """查找 vectors_data 中对应的 key"""
        # 尝试元组格式
        if (t_name, c_name) in vectors_data:
            return (t_name, c_name)
        # 尝试字符串格式（含 × 或 _）
        for sep in ["×", "_x_", "×", "__"]:
            k = f"{t_name}{sep}{c_name}"
            if k in vectors_data:
                return k
        return None

    for t_name in t_names:
        # 获取对应 Token 方法的 C_mean 向量（用于反推 token 向量）
        cm_key = find_key(t_name, 'C_mean')
        if cm_key is None:
            print(f"  ⚠ 找不到 ({t_name}, C_mean)，跳过")
            continue

        t_cm_vecs = np.array(vectors_data[cm_key], dtype=np.float32)

        # 反推 token 向量：combined = (token_vec + chunk_vec) / 2
        # → token_vec = 2 * combined - chunk_vec
        c_mean_vecs = np.array([kps.mean(axis=0) for kps in kp_vecs_list], dtype=np.float32)
        t_vecs = 2.0 * t_cm_vecs - c_mean_vecs  # (N, 384)

        # 用新 Chunk 向量合成最终向量
        sim_combined = (t_vecs + c_sim_weighted) / 2.0
        idf_combined = (t_vecs + c_idf_weighted) / 2.0

        for label, vecs in [(f"{t_name}×C_sim_weighted", sim_combined),
                            (f"{t_name}×C_idf_weighted", idf_combined)]:
            bl, detail = compute_brierlm(vecs, subsample=500)
            results[label] = {"brierlm": bl, **detail}
            print(f"  {label}: BrierLM={bl:.2f}")

    # ── 计算各方法 8 种 Token 平均，并对比原始基线 ──────────────
    brier_json_path = Path(__file__).parent / "brier_detail.json"
    if brier_json_path.exists():
        with open(brier_json_path, "r") as f:
            brier_data = json.load(f)
        baseline_detail = brier_data.get("detail", {})

        for new_method in ["C_sim_weighted", "C_idf_weighted"]:
            scores = [v["brierlm"] for k, v in results.items() if new_method in k]
            if scores:
                results[f"AVG_{new_method}"] = {
                    "brierlm": float(np.mean(scores)),
                    "note": f"8种Token方法平均 (N={len(scores)})"
                }

        for ref_chunk in ["C_mean", "C_concat", "C_combined", "C_weighted", "C_diff"]:
            ref_scores = []
            for t in t_names:
                k = f"{t}×{ref_chunk}"
                if k in baseline_detail:
                    ref_scores.append(baseline_detail[k]["brierlm"])
            if ref_scores:
                results[f"AVG_{ref_chunk}"] = {
                    "brierlm": float(np.mean(ref_scores)),
                    "note": f"原始 {ref_chunk} 8种Token均值 (N={len(ref_scores)})"
                }

    # ── 保存结果 ────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {OUTPUT_JSON}")

    # ── 打印汇总表 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("实验汇总：新加权方法 vs 原始方法（VecBrierLM）")
    print("=" * 60)
    print(f"\n{'方法':42s}  {'BrierLM':>8}")
    print("-" * 54)
    for t in t_names:
        for suffix in ["C_sim_weighted", "C_idf_weighted"]:
            k = f"{t}×{suffix}"
            if k in results:
                print(f"  {k:42s}  {results[k]['brierlm']:>8.2f}")

    print("\n  ── 各方法8种Token平均 ──")
    for key in ["AVG_C_sim_weighted", "AVG_C_idf_weighted",
                "AVG_C_mean", "AVG_C_weighted", "AVG_C_concat",
                "AVG_C_combined", "AVG_C_diff"]:
        if key in results:
            note = results[key].get("note", "")
            print(f"  {key:35s}  {results[key]['brierlm']:>8.2f}  ({note})")

    print("\n实验完成！")
    return results


if __name__ == "__main__":
    main()
