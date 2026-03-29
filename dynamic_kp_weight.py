"""
动态知识点权重聚合实验（方向A）
=====================================================
创新点：现有方法（C_idf_weighted / C_sim_weighted）使用语料级静态权重，
       本方法让 LLM 针对每道题输出"题目级动态知识点重要性分布"，
       每道题的 KP 权重因题而异，而非所有题共享同一套权重规则。

流程：
  1. 从 results_data.pkl 提取每题的知识点向量（来自 C_concat）
  2. 用 Qwen-Plus 为每道题生成 KP 重要性分数 weights[i] ∈ [0,1]，sum=1
  3. 加权聚合 KP 向量 → C_dynamic_weighted
  4. 与 Token 向量拼接 → 最终题目向量
  5. 评估 VecBrierLM 等指标，与 C_idf_weighted / C_combined 对比

注意：KP 文本从 cache_data.npy 中提取（data_loader.py 生成的缓存）
     若缓存不存在则需从 dataset_without_preference.jsonl 重新提取
"""

import os, sys, io, json, re, time, pickle, warnings
import numpy as np
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import dashscope
from dashscope import Generation

sys.path.insert(0, str(Path(__file__).parent))
from brier_eval import compute_brierlm

# ── 配置 ─────────────────────────────────────────────────────────────────
QWEN_API_KEY = "sk-83da80fde3fd412a97a5668b8e1f8104"
dashscope.api_key = QWEN_API_KEY

DATA_PATH    = Path(__file__).parent / "results_data.pkl"
DATASET_PATH = Path(__file__).parent / "dataset_without_preference.jsonl"
CACHE_PATH   = Path(__file__).parent / "cache_data.npy"
WEIGHTS_CACHE = Path(__file__).parent / "dynamic_kp_weights_cache.json"  # LLM权重缓存，避免重复调用
OUTPUT_JSON  = Path(__file__).parent / "dynamic_kp_results.json"

DIM = 384
BATCH_SIZE = 8       # 每次并发几道题（实际串行，但一次prompt打包多道题）
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.3  # API调用间隔(s)

T_NAMES = ["T_mean", "T_max", "T_min", "T_concat", "T_weighted",
           "T_product", "T_first", "T_last"]


# ── 工具函数 ──────────────────────────────────────────────────────────────
def softmax(x, temperature=1.0):
    x = np.array(x, dtype=np.float64) / temperature
    x -= x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-15)


def cosine_sim_vec_to_mat(a, B):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return B_norm @ a_norm


def find_key(vectors_data, t_name, c_name):
    if (t_name, c_name) in vectors_data:
        return (t_name, c_name)
    for sep in ["×", "__", "_x_"]:
        k = f"{t_name}{sep}{c_name}"
        if k in vectors_data:
            return k
    return None


# ── 步骤1：从 pkl 提取知识点向量和题目文本 ────────────────────────────────
def load_kp_data(data):
    """从 results_data.pkl 提取每题的 KP 向量列表和题目文本"""
    vectors_data = {k: v['vectors'] for k, v in data.items()}
    texts_data   = {k: v['texts']   for k, v in data.items()}

    cc_key = find_key(vectors_data, 'T_mean', 'C_concat')
    assert cc_key is not None, "找不到 (T_mean, C_concat) key"
    print(f"  使用 key: {cc_key}")

    cc_vecs_raw = vectors_data[cc_key]
    texts       = texts_data[cc_key]
    N = len(cc_vecs_raw)

    kp_vecs_list = []
    kp_counts    = []
    for i in range(N):
        row   = np.array(cc_vecs_raw[i], dtype=np.float32)
        n_kp  = row.shape[0] // DIM
        kps   = row.reshape(n_kp, DIM)
        kp_vecs_list.append(kps)
        kp_counts.append(n_kp)

    from collections import Counter
    cnt = Counter(kp_counts)
    print(f"  N={N}, KP数分布: {dict(sorted(cnt.items()))}")
    return kp_vecs_list, texts, vectors_data, texts_data


# ── 步骤2：从缓存或 dataset 加载 KP 文本 ──────────────────────────────────
def load_kp_texts_from_cache():
    """尝试从 cache_data.npy 提取 KP 文本"""
    if not CACHE_PATH.exists():
        return None
    try:
        cache = np.load(CACHE_PATH, allow_pickle=True).item()
        # cache 结构: {'token_embs': {(qid,kid,tid):vec}, 'kp_token_counts': {(qid,kid):n}, ...}
        # 重建 kp_texts: {qid: [kp0_text, kp1_text, ...]}
        # cache 里没有直接存文本，只有向量，需要从 dataset 加载
        return None
    except Exception:
        return None


def load_kp_texts_from_dataset(N):
    """从 dataset_without_preference.jsonl 重新提取 KP 文本（若有缓存则跳过 LLM 调用）"""
    # 先尝试读取现有的 test_kp_fixed_result.json 之类的文件
    kp_cache_files = list(Path(__file__).parent.glob("*kp*result*.json")) + \
                     list(Path(__file__).parent.glob("kp_texts*.json"))
    
    # 加载 dataset
    questions = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    questions = questions[:N]
    return [q['input'] for q in questions]


# ── 步骤3：LLM 为每道题生成动态 KP 权重 ──────────────────────────────────
def build_kp_prompt_batch(questions_and_kp_counts):
    """
    构造批量 prompt：一次发送多道题（每道题含题目文本 + KP数量），
    LLM 返回每道题的权重列表。
    
    为简化LLM输出解析，每道题单独调用。
    """
    pass  # 见下方 get_llm_weights_single


def get_llm_weights_single(question_text, n_kp, qid, retries=MAX_RETRIES):
    """
    让 LLM 对单道题的 n_kp 个知识点分配重要性权重（归一化到 sum=1）。
    
    设计思路：
    - 不需要 LLM 知道具体知识点内容（因为我们只有向量没有文本标签）
    - 而是让 LLM 根据题目文本，输出"该题涉及的 n_kp 个概念的重要性排序"
    - 权重反映：哪些知识点对理解/检索该题更关键
    
    返回: numpy array shape (n_kp,), sum=1
    """
    prompt = f"""Given a chemistry question, assign importance weights to its {n_kp} knowledge points.
The knowledge points are indexed 0 to {n_kp-1} in order of extraction.
Return ONLY a JSON array of {n_kp} non-negative numbers that sum to 1.0.
Higher weight = more central to solving the question.

Question: {question_text[:300]}

Output format: [w0, w1, w2, ...]
Return only the JSON array, no explanation."""

    for attempt in range(retries):
        try:
            response = Generation.call(
                model="qwen-plus",
                prompt=prompt,
                temperature=0.1,
                max_tokens=128,
                result_format='message'
            )
            if response.status_code == 200:
                content = response.output.choices[0].message.content.strip()
                # 提取 JSON 数组
                match = re.search(r'\[[\d\s.,eE+\-]+\]', content)
                if match:
                    weights = json.loads(match.group())
                    if isinstance(weights, list) and len(weights) == n_kp:
                        w = np.array(weights, dtype=np.float64)
                        w = np.clip(w, 0, None)
                        s = w.sum()
                        if s > 1e-8:
                            return (w / s).astype(np.float32)
            time.sleep(0.5)
        except Exception as e:
            time.sleep(1)

    # 失败：退化为均匀权重
    return np.ones(n_kp, dtype=np.float32) / n_kp


def get_all_llm_weights(question_texts, kp_counts, weights_cache_path):
    """
    获取所有题目的 LLM 动态权重，支持断点续跑（已计算的从缓存加载）。
    
    返回: list of np.array, 每个 shape (n_kp_i,)
    """
    N = len(question_texts)
    
    # 加载已有缓存
    cache = {}
    if weights_cache_path.exists():
        with open(weights_cache_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        cache = {int(k): np.array(v, dtype=np.float32) for k, v in raw.items()}
        print(f"  从缓存加载 {len(cache)}/{N} 条权重")

    all_weights = [None] * N
    todo_ids = []
    for i in range(N):
        if i in cache:
            all_weights[i] = cache[i]
        else:
            todo_ids.append(i)

    print(f"  需要 LLM 计算: {len(todo_ids)} 道题")

    save_interval = 50
    for idx, i in enumerate(todo_ids):
        w = get_llm_weights_single(question_texts[i], kp_counts[i], i)
        all_weights[i] = w
        cache[i] = w.tolist()

        if (idx + 1) % save_interval == 0 or (idx + 1) == len(todo_ids):
            with open(weights_cache_path, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in cache.items()}, f)
            print(f"  进度: {idx+1}/{len(todo_ids)} (总{i+1}/{N}), 已保存缓存")

        time.sleep(SLEEP_BETWEEN_CALLS)

    return all_weights


# ── 步骤4：动态加权聚合 ────────────────────────────────────────────────────
def aggregate_dynamic_weighted(kp_vecs_list, all_weights):
    """
    C_dynamic_weighted：使用 LLM 输出的题目级动态权重聚合 KP 向量。
    kp_vecs_list: list of (n_kp_i, 384) arrays
    all_weights:  list of (n_kp_i,) weight arrays
    返回: (N, 384)
    """
    result = []
    for vecs, w in zip(kp_vecs_list, all_weights):
        vecs = np.array(vecs, dtype=np.float32)
        if len(vecs) == 1:
            result.append(vecs[0])
            continue
        w = np.array(w, dtype=np.float32)
        # 确保维度一致
        if len(w) != len(vecs):
            w = np.ones(len(vecs), dtype=np.float32) / len(vecs)
        agg = (w[:, None] * vecs).sum(axis=0)
        result.append(agg)
    return np.array(result, dtype=np.float32)


# ── 步骤5：评估 ───────────────────────────────────────────────────────────
def evaluate_all_tokens(c_dynamic_vecs, kp_vecs_list, vectors_data):
    """
    对 8 种 Token 方法 × C_dynamic_weighted 计算 VecBrierLM。
    使用与 weighted_chunk_exp.py 相同的"反推 Token 向量"策略。
    """
    results = {}
    c_mean_vecs_base = np.array(
        [kps.mean(axis=0) for kps in kp_vecs_list], dtype=np.float32
    )  # (N, 384) — 不含 Token

    for t_name in T_NAMES:
        cm_key = find_key(vectors_data, t_name, 'C_mean')
        if cm_key is None:
            print(f"  ⚠ 找不到 ({t_name}, C_mean)，跳过")
            continue
        t_cm_vecs  = np.array(vectors_data[cm_key], dtype=np.float32)
        # 反推 token 向量：t_vec = 2 * (T×C_mean) - C_mean_pure
        t_vecs     = 2.0 * t_cm_vecs - c_mean_vecs_base  # (N, 384)
        final_vecs = (t_vecs + c_dynamic_vecs) / 2.0     # (N, 384)

        bl, detail = compute_brierlm(final_vecs, subsample=500)
        results[f"{t_name}×C_dynamic_weighted"] = {"brierlm": bl, **detail}
        print(f"  {t_name}×C_dynamic_weighted: BrierLM={bl:.2f}")

    return results


# ── 主函数 ────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("方向A：题目级动态知识点权重聚合实验")
    print("=" * 65)

    # 1. 加载 pkl
    print(f"\n[1/5] 加载 {DATA_PATH.name} ...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    kp_vecs_list, texts, vectors_data, texts_data = load_kp_data(data)
    N = len(kp_vecs_list)
    kp_counts = [len(kps) for kps in kp_vecs_list]

    # 2. 加载题目文本（用于 LLM prompt）
    print(f"\n[2/5] 加载题目文本 ...")
    question_texts = load_kp_texts_from_dataset(N)
    assert len(question_texts) == N, f"文本数 {len(question_texts)} ≠ 向量数 {N}"
    print(f"  题目数: {N}")

    # 3. LLM 生成动态权重（支持断点续跑）
    print(f"\n[3/5] LLM 生成题目级动态 KP 权重 ...")
    print(f"  预计 API 调用: {N} 次，约 {N * SLEEP_BETWEEN_CALLS / 60:.1f} 分钟")
    all_weights = get_all_llm_weights(question_texts, kp_counts, WEIGHTS_CACHE)

    # 统计权重分布
    entropy_list = []
    for w in all_weights:
        w = np.array(w)
        e = -np.sum(w * np.log(w + 1e-15))
        entropy_list.append(e)
    print(f"  权重熵均值: {np.mean(entropy_list):.3f} (均匀分布={np.log(np.mean(kp_counts)):.3f})")
    print(f"  权重熵越低 → 越集中（LLM 认为某几个 KP 更重要）")

    # 4. 动态加权聚合
    print(f"\n[4/5] 动态加权聚合 ...")
    c_dynamic_vecs = aggregate_dynamic_weighted(kp_vecs_list, all_weights)
    print(f"  C_dynamic_weighted shape: {c_dynamic_vecs.shape}")

    # 5. 评估
    print(f"\n[5/5] 评估 VecBrierLM ...")
    results = evaluate_all_tokens(c_dynamic_vecs, kp_vecs_list, vectors_data)

    # 汇总均值
    dyn_scores = [v["brierlm"] for k, v in results.items() if "C_dynamic" in k]
    results["AVG_C_dynamic_weighted"] = {
        "brierlm": float(np.mean(dyn_scores)),
        "note": f"8种Token方法平均 (N={len(dyn_scores)})"
    }

    # 加载原始基线进行对比
    brier_json = Path(__file__).parent / "brier_detail.json"
    weighted_json = Path(__file__).parent / "weighted_results.json"
    baselines = {}
    if brier_json.exists():
        with open(brier_json, 'r') as f:
            bd = json.load(f)
        detail = bd.get("detail", {})
        for ref in ["C_mean", "C_concat", "C_combined", "C_weighted"]:
            scores = [detail[f"{t}×{ref}"]["brierlm"]
                      for t in T_NAMES if f"{t}×{ref}" in detail]
            if scores:
                baselines[f"AVG_{ref}"] = float(np.mean(scores))
    if weighted_json.exists():
        with open(weighted_json, 'r', encoding='utf-8') as f:
            wd = json.load(f)
        for ref in ["C_sim_weighted", "C_idf_weighted"]:
            k = f"AVG_{ref}"
            if k in wd:
                baselines[k] = wd[k]["brierlm"]

    # 保存结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUTPUT_JSON}")

    # 打印汇总
    print("\n" + "=" * 65)
    print("实验汇总：各方法 8种Token均值 VecBrierLM 对比")
    print("=" * 65)
    print(f"  {'方法':<30} {'BrierLM':>8}")
    print(f"  {'-'*40}")

    comparison = {**baselines,
                  "AVG_C_dynamic_weighted": results["AVG_C_dynamic_weighted"]["brierlm"]}
    # 按分数排序
    for k, v in sorted(comparison.items(), key=lambda x: -x[1]):
        marker = " ◀ 本方法" if "dynamic" in k else ""
        print(f"  {k:<30} {v:>8.2f}{marker}")

    delta = results["AVG_C_dynamic_weighted"]["brierlm"] - baselines.get("AVG_C_mean", 0)
    print(f"\n  vs C_mean 提升: {delta:+.2f}")
    delta2 = results["AVG_C_dynamic_weighted"]["brierlm"] - baselines.get("AVG_C_idf_weighted", 0)
    print(f"  vs C_idf_weighted 提升: {delta2:+.2f}")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
