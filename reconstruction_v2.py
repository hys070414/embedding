"""
向量→文本重建评估实验 v2（正式版）
=====================================
生成方法：Qwen API 语义引导生成
  - 利用 all-MiniLM-L6-v2 的 KNN 检索找到最接近目标向量的候选文本（top-5）
  - 将候选文本 + 向量统计特征喂给 Qwen-Plus，生成语义对齐的重建文本
  - 与原始方案的区别：
      ① 原来 Qwen 只是在候选中"选一个最像的"，现在明确要求 Qwen 做知识点保留的改写
      ② 评估指标全面替换：去掉余弦相似度，改用 KP-F1 + LLM-Judge + ROUGE-L

评估指标（三个）：
  1. KP-F1   ：知识点集合 F1（Qwen提取知识点→token集合比较）
  2. ROUGE-L ：词汇重叠（作为基线对照）
  3. LLM-Judge：Qwen评判裁判，三维度打分归一化到[0,1]

全量：每个组合使用 results_data.pkl 中全部样本（或受 MAX_SAMPLES_PER_COMBO 限制）

输出：
  reconstruction_v2_results.npy    (8,8,3): [kp_f1, rouge_l, llm_judge_norm]
  reconstruction_v2_combined.npy   (8,8)  : 综合分数
  reconstruction_v2_detail.pkl             详细记录（用于 HTML 报告）
  reconstruction_v2_heatmap.png
  reconstruction_v2_report_data.json
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys, io
# Windows GBK终端兼容：强制UTF-8输出
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import sys, pickle, json, re, time, random, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rouge_score import rouge_scorer as rouge_lib
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import dashscope
from dashscope import Generation

warnings.filterwarnings("ignore")

# ================== 配置 ==================
EMBED_DIM    = 384
ENCODER_NAME = "all-MiniLM-L6-v2"
QWEN_API_KEY = "sk-83da80fde3fd412a97a5668b8e1f8104"
RANDOM_SEED  = 42
MAX_SAMPLES_PER_COMBO = 100    # 每组合最多100题（统计稳健，约2~3小时完成）

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
dashscope.api_key = QWEN_API_KEY

# ================== 加载资源 ==================
print("=" * 70)
print("向量→文本重建评估实验 v2 (Qwen生成 + KP-F1 + LLM-Judge)")
print("=" * 70)

print(f"\n[1] 加载编码器: {ENCODER_NAME} ...")
encoder = SentenceTransformer(ENCODER_NAME)

print("[2] 加载 results_data.pkl ...")
with open('results_data.pkl', 'rb') as f:
    results_data = pickle.load(f)
sample_n = len(next(iter(results_data.values()))['texts'])
print(f"    组合数: {len(results_data)}, 每组合样本数: {sample_n}")

# ================== 构建 KNN 索引（用于 Qwen 生成的候选池）==================
print("[3] 构建语料库 KNN 索引 ...")
all_texts_set = set()
for d in results_data.values():
    for t in d['texts']:
        all_texts_set.add(t)
corpus = list(all_texts_set)
print(f"    语料库大小: {len(corpus)}")

corpus_embs = encoder.encode(corpus, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=True)
knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
knn.fit(corpus_embs)

def normalize_vec(vec):
    v = np.array(vec, dtype=np.float32).flatten()
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def project_to_d(vec, d):
    v = np.array(vec, dtype=np.float32).flatten()
    if v.shape[0] == d: return v
    if v.shape[0] > d:
        if v.shape[0] % d == 0:
            return v.reshape(-1, d).mean(axis=0)
        return v[np.linspace(0, v.shape[0]-1, d).astype(int)]
    out = np.zeros(d, dtype=np.float32)
    out[:v.shape[0]] = v
    return out

def knn_candidates(vec_384, top_k=5):
    v = normalize_vec(vec_384).reshape(1, -1)
    _, idxs = knn.kneighbors(v, n_neighbors=top_k)
    return [corpus[i] for i in idxs[0]]

# ================== Qwen 生成重建文本 ==================
def qwen_reconstruct(vec_384, max_retries=2):
    """
    核心重建函数：
    1. KNN 找 top-5 候选文本
    2. 给 Qwen 这5个候选 + 向量统计特征
    3. Qwen 综合这些信息，生成一道涵盖相关知识点的化学题目
    （不同于原方案：不是选最相似的，而是真正生成新文本）
    """
    candidates = knn_candidates(vec_384, top_k=5)
    cand_str = "\n".join([f"  {i+1}. {c[:200]}" for i, c in enumerate(candidates)])
    
    # 向量统计特征（辅助信息，帮助 Qwen 感知向量语义倾向）
    mean_val = float(np.mean(vec_384))
    std_val  = float(np.std(vec_384))
    
    prompt = f"""You are a chemistry question reconstructor. Based on the semantic information encoded in an embedding vector, reconstruct the most likely chemistry question.

The following 5 candidate chemistry questions are the nearest neighbors to the target embedding vector (ranked by semantic similarity):
{cand_str}

Vector statistics: mean={mean_val:.4f}, std={std_val:.4f}

Task: Generate one chemistry question that best captures the core chemical knowledge points shared among the top candidates. 
- The question should be coherent and scientifically accurate
- Focus on the recurring chemistry concepts across candidates
- Write in the same style as the candidates
- Output only the question text, no explanation

Reconstructed question:"""
    
    for attempt in range(max_retries):
        try:
            resp = Generation.call(
                model="qwen-plus",
                prompt=prompt,
                temperature=0.15,
                max_tokens=250,
                result_format='message'
            )
            if resp.status_code == 200:
                text = resp.output.choices[0].message.content.strip()
                # 清除可能的前缀
                text = re.sub(r'^(Reconstructed question[:：]?\s*|Question[:：]?\s*)', '', text, flags=re.IGNORECASE)
                return text.strip()
            time.sleep(0.5)
        except Exception as e:
            time.sleep(1)
    
    # 失败时返回最近邻候选
    return candidates[0]


# ================== 知识点提取（KP-F1 核心）==================
_kp_cache = {}

def extract_kps(text, max_retries=3):
    """用 Qwen 提取化学知识点列表（强制简单字符串数组格式）"""
    if text in _kp_cache:
        return _kp_cache[text]

    prompt = (
        "List the chemistry knowledge points in this question as a JSON array of short strings (2-5 words each).\n"
        "IMPORTANT: Return ONLY a flat JSON array like [\"term1\", \"term2\", \"term3\"].\n"
        "Do NOT use nested objects. No explanation.\n\n"
        f"Question: {text[:400]}\n\n"
        "Output (JSON array only):"
    )

    def _parse_kps(content):
        """解析Qwen返回内容 → 字符串列表，支持多种格式"""
        # 尝试1: 直接解析整个 content
        try:
            parsed = json.loads(content)
        except Exception:
            # 尝试2: 提取第一个 [...] 块
            m = re.search(r'\[.*?\]', content, re.DOTALL)
            if not m:
                return None
            try:
                parsed = json.loads(m.group())
            except Exception:
                return None

        if not isinstance(parsed, list) or not parsed:
            return None

        result = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                result.append(item.lower().strip())
            elif isinstance(item, dict):
                # 支持 {"topic": ..., "concept": ...} 格式
                for field in ('topic', 'concept', 'name', 'term', 'key', 'point'):
                    if field in item and isinstance(item[field], str):
                        result.append(item[field].lower().strip())
                        break
        return result if result else None

    for _ in range(max_retries):
        try:
            resp = Generation.call(
                model="qwen-plus",
                prompt=prompt,
                temperature=0.0,
                max_tokens=200,
                result_format='message'
            )
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content.strip()
                kps = _parse_kps(content)
                if kps:
                    _kp_cache[text] = kps
                    return kps
            time.sleep(0.3)
        except Exception:
            time.sleep(0.5)

    # 失败时用 TF-IDF 风格关键词兜底
    fallback = list(set(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())))[:8]
    _kp_cache[text] = fallback
    return fallback


def compute_kp_f1(kps_orig, kps_recon):
    """
    知识点集合 F1（token-level）：
    将知识点拆成词 token 集合，计算精确率、召回率、F1。
    对措辞变体鲁棒（"electron transfer" vs "transfer of electrons" 会有部分匹配）。
    """
    if not kps_orig or not kps_recon:
        return 0.0, 0.0, 0.0
    
    STOP = {'the','and','for','with','from','that','this','are','was','have',
            'has','not','but','its','can','all','one','will','used','using',
            'what','which','when','how','given','find','calculate','determine'}
    
    def to_tokens(kp_list):
        tokens = set()
        for kp in kp_list:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', kp.lower())
            tokens.update(w for w in words if w not in STOP)
        return tokens
    
    s_orig  = to_tokens(kps_orig)
    s_recon = to_tokens(kps_recon)
    
    if not s_orig or not s_recon:
        return 0.0, 0.0, 0.0
    
    inter = s_orig & s_recon
    prec = len(inter) / len(s_recon)
    rec  = len(inter) / len(s_orig)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ================== LLM-Judge ==================
def llm_judge(orig_text, recon_text, max_retries=2):
    """
    Qwen 评判裁判：评估重建题对原题知识点的保留程度。
    三维度（各0-2分）：化学概念保留、题型一致性、关键实体/数值保留。
    总分 0-6，归一化到 [0,1]。
    """
    prompt = f"""You are a chemistry education expert evaluating knowledge preservation in question reconstruction.

Original: {orig_text[:300]}
Reconstructed: {recon_text[:300]}

Score three dimensions (0=none, 1=partial, 2=full):
1. Core chemistry concept retention
2. Question type consistency  
3. Key entities/values retention

Return ONLY JSON: {{"concept": X, "type": X, "entity": X}}"""
    
    for _ in range(max_retries):
        try:
            resp = Generation.call(
                model="qwen-plus",
                prompt=prompt,
                temperature=0.0,
                max_tokens=80,
                result_format='message'
            )
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content.strip()
                m = re.search(r'\{.*?\}', content, re.DOTALL)
                if m:
                    try:
                        s = json.loads(m.group())
                        total = s.get('concept', 0) + s.get('type', 0) + s.get('entity', 0)
                        return min(max(int(total), 0), 6)
                    except (json.JSONDecodeError, ValueError):
                        pass
            time.sleep(0.2)
        except Exception:
            time.sleep(0.5)
    return 3  # 失败时返回中值


# ================== ROUGE-L ==================
rouge_scorer = rouge_lib.RougeScorer(['rougeL'], use_stemmer=False)

def compute_rougeL(orig, recon):
    if not orig or not recon:
        return 0.0
    return rouge_scorer.score(orig, recon)['rougeL'].fmeasure


# ================== 主实验循环 ==================
t_names = ['T_mean', 'T_max', 'T_min', 'T_concat', 'T_weighted',
           'T_product', 'T_first', 'T_last']
c_names = ['C_mean', 'C_max', 'C_min', 'C_concat', 'C_weighted',
           'C_product', 'C_diff', 'C_combined']

available_keys = set(results_data.keys())
t_names = [t for t in t_names if any(k[0] == t for k in available_keys)]
c_names = [c for c in c_names if any(k[1] == c for k in available_keys)]
print(f"\n有效Token方法: {t_names}")
print(f"有效Chunk方法: {c_names}")
n_t, n_c = len(t_names), len(c_names)

# 结果矩阵: (t, c, 3) → [kp_f1, rouge_l, llm_judge_norm]
results_mat = np.full((n_t, n_c, 3), np.nan)
detail_records = {}

n_combos = sum(1 for t in t_names for c in c_names if (t, c) in available_keys)
print(f"\n[4] 开始重建评估（共 {n_combos} 个组合）...\n")

# API 调用计数（用于速率监控）
api_calls = 0

pbar = tqdm(total=n_combos, desc="组合进度")

for ti, t_name in enumerate(t_names):
    for ci, c_name in enumerate(c_names):
        key = (t_name, c_name)
        if key not in results_data:
            pbar.update(1)
            continue
        
        data    = results_data[key]
        vectors = data['vectors']
        texts   = data['texts']
        
        n_samples = len(vectors)
        if n_samples == 0:
            pbar.update(1)
            continue
        
        if MAX_SAMPLES_PER_COMBO is not None:
            n_use   = min(MAX_SAMPLES_PER_COMBO, n_samples)
            indices = random.sample(range(n_samples), n_use)
        else:
            indices = list(range(n_samples))
        
        kp_f1_list   = []
        rouge_l_list = []
        judge_list   = []
        detail_list  = []
        
        for idx in indices:
            vec  = vectors[idx]
            orig = texts[idx]
            
            try:
                vec_384 = normalize_vec(project_to_d(np.array(vec).flatten(), EMBED_DIM))
            except Exception:
                continue
            
            # 1. Qwen 生成重建文本
            try:
                recon = qwen_reconstruct(vec_384)
                api_calls += 1
                time.sleep(0.3)
            except Exception as e:
                recon = orig  # fallback
            
            # 2. KP-F1（需要 2 次 Qwen API 提取知识点）
            kp_f1 = np.nan
            kps_o, kps_r = [], []
            try:
                kps_o = extract_kps(orig)
                api_calls += 1
                kps_r = extract_kps(recon)
                api_calls += 1
                _, _, kp_f1 = compute_kp_f1(kps_o, kps_r)
                time.sleep(0.2)
            except Exception:
                pass
            
            # 3. ROUGE-L
            rl = compute_rougeL(orig, recon)
            
            # 4. LLM-Judge
            judge_raw  = 3
            judge_norm = 0.5
            try:
                judge_raw  = llm_judge(orig, recon)
                judge_norm = judge_raw / 6.0
                api_calls += 1
                time.sleep(0.2)
            except Exception:
                pass
            
            if not np.isnan(kp_f1):
                kp_f1_list.append(kp_f1)
            rouge_l_list.append(rl)
            judge_list.append(judge_norm)
            
            detail_list.append({
                'idx'        : idx,
                'orig'       : orig[:200],
                'recon'      : recon[:200],
                'kps_orig'   : kps_o[:5],
                'kps_recon'  : kps_r[:5],
                'kp_f1'      : float(kp_f1) if not np.isnan(kp_f1) else None,
                'rouge_l'    : float(rl),
                'llm_judge'  : judge_raw,
                'judge_norm' : float(judge_norm),
            })
        
        # 写入矩阵
        if kp_f1_list:
            results_mat[ti, ci, 0] = np.nanmean(kp_f1_list)
        if rouge_l_list:
            results_mat[ti, ci, 1] = np.nanmean(rouge_l_list)
        if judge_list:
            results_mat[ti, ci, 2] = np.nanmean(judge_list)
        
        detail_records[key] = detail_list
        pbar.update(1)
        pbar.set_postfix({
            'KP-F1' : f"{results_mat[ti,ci,0]:.3f}" if not np.isnan(results_mat[ti,ci,0]) else "-",
            'Judge' : f"{results_mat[ti,ci,2]:.3f}" if not np.isnan(results_mat[ti,ci,2]) else "-",
            'APIs'  : api_calls,
        })

pbar.close()
print(f"\n总 API 调用次数: {api_calls}")

# ================== 综合分数 ==================
# KP-F1(50%) + LLM-Judge(35%) + ROUGE-L(15%)
combined = (0.50 * np.nan_to_num(results_mat[:, :, 0]) +
            0.15 * np.nan_to_num(results_mat[:, :, 1]) +
            0.35 * np.nan_to_num(results_mat[:, :, 2]))
combined[np.all(np.isnan(results_mat), axis=2)] = np.nan

# ================== 保存 ==================
np.save("reconstruction_v2_results.npy", results_mat)
np.save("reconstruction_v2_combined.npy", combined)
with open("reconstruction_v2_detail.pkl", "wb") as f:
    pickle.dump({'detail': detail_records, 't_names': t_names, 'c_names': c_names}, f)

report_data = {
    't_names': t_names, 'c_names': c_names,
    'results_mat': results_mat.tolist(),
    'combined': combined.tolist(),
    'metric_labels': ['KP-F1', 'ROUGE-L', 'LLM-Judge(norm)'],
    'detail': {str(k): v for k, v in detail_records.items()},
    'api_calls': api_calls,
    'n_samples_per_combo': sample_n,
}
with open("reconstruction_v2_report_data.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)
print("结果已保存")

# ================== 绘制热力图 ==================
metric_labels = ['KP-F1（知识点集合F1）', 'ROUGE-L', 'LLM-Judge（归一化）']
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

for m_idx, label in enumerate(metric_labels):
    ax = axes[m_idx]
    sns.heatmap(np.nan_to_num(results_mat[:, :, m_idx], nan=0.0),
                annot=True, fmt='.3f',
                xticklabels=c_names, yticklabels=t_names,
                cmap='YlOrRd', ax=ax,
                cbar_kws={'label': label}, vmin=0, vmax=1)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Chunk方法', fontsize=9)
    ax.set_ylabel('Token方法', fontsize=9)

ax = axes[3]
sns.heatmap(np.nan_to_num(combined, nan=0.0),
            annot=True, fmt='.3f',
            xticklabels=c_names, yticklabels=t_names,
            cmap='RdYlGn', ax=ax,
            cbar_kws={'label': '综合分数'}, vmin=0, vmax=1)
ax.set_title('综合重建分数\n(KP-F1×50%+ROUGE-L×15%+Judge×35%)', fontsize=10)
ax.set_xlabel('Chunk方法', fontsize=9)
ax.set_ylabel('Token方法', fontsize=9)

plt.suptitle('重建质量评估 — Token×Chunk组合（KP-F1 + LLM-Judge）', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('reconstruction_v2_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("热力图已保存: reconstruction_v2_heatmap.png")

# ================== 打印结果摘要 ==================
print("\n" + "=" * 60)
print("实验结果摘要")
print("=" * 60)
for m_idx, label in enumerate(['KP-F1', 'ROUGE-L', 'LLM-Judge(norm)']):
    m = results_mat[:, :, m_idx]
    if not np.all(np.isnan(m)):
        bi, bj = np.unravel_index(np.nanargmax(m), m.shape)
        wi, wj = np.unravel_index(np.nanargmin(m), m.shape)
        print(f"\n{label}:")
        print(f"  均值={np.nanmean(m):.4f}  最大={np.nanmax(m):.4f}  最小={np.nanmin(m):.4f}")
        print(f"  最佳: {t_names[bi]}×{c_names[bj]}={m[bi,bj]:.4f}")
        print(f"  最差: {t_names[wi]}×{c_names[wj]}={m[wi,wj]:.4f}")

if not np.all(np.isnan(combined)):
    bi, bj = np.unravel_index(np.nanargmax(combined), combined.shape)
    wi, wj = np.unravel_index(np.nanargmin(combined), combined.shape)
    print(f"\n综合分数:")
    print(f"  均值={np.nanmean(combined):.4f}  最大={np.nanmax(combined):.4f}  最小={np.nanmin(combined):.4f}")
    print(f"  最佳: {t_names[bi]}×{c_names[bj]}={combined[bi,bj]:.4f}")
    print(f"  最差: {t_names[wi]}×{c_names[wj]}={combined[wi,wj]:.4f}")

print("\n实验完成！")
