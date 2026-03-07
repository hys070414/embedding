import numpy as np
import dashscope
from dashscope import TextEmbedding, Generation
from tqdm import tqdm
import time
import os
import json
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import svd
import re
import matplotlib.pyplot as plt

# ==================== 配置 ====================
dashscope.api_key = "sk-cb52e47466e54085964e8239e3588105"        
EMBEDDING_MODEL = "text-embedding-v2"            # 千问 embedding 模型
GENERATION_MODEL = "qwen-plus"                    # 用于知识点提取的生成模型
CACHE_FILE = "chem_embeddings_with_llm.npy"       # 新的缓存文件（与之前区分）
TOP_K_EOSK = 2
N_SUB_EOSK = 10
RETRIEVAL_K = 5
USE_LLM_EXTRACTION = True                          # True 表示使用大模型提取知识点，False 则按句子切分

# 清除代理环境变量
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# ==================== 1. 从 JSONL 文件加载题库 ====================
def load_questions_from_jsonl(file_path, sample_size=None):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
                if 'input' in q:
                    questions.append(q)
            except json.JSONDecodeError:
                print(f"警告: 跳过无法解析的行: {line[:50]}...")
    print(f"共加载 {len(questions)} 道题目")
    if sample_size is not None and sample_size < len(questions):
        questions = random.sample(questions, sample_size)
        print(f"随机抽取 {sample_size} 道题目进行实验")
    return questions

questions = load_questions_from_jsonl('dataset_without_preference.jsonl', sample_size=None)  # 使用全部题目
if not questions:
    raise ValueError("题库为空，请检查文件路径和内容")

# ==================== 2. 知识点提取函数（使用大模型）====================
def extract_knowledge_points(question_text, max_retries=3):
    """
    调用千问生成模型，从题目文本中提取出若干个独立的知识点。
    返回一个字符串列表，每个元素是一个知识点。
    """
    prompt = f"""
请从以下化学题目中提取出解决该问题所需的独立知识点。每个知识点应是一个简洁的陈述句或短语，覆盖解题的关键概念、原理或步骤。将知识点以 JSON 数组形式返回，数组中的每个元素是一个字符串。只返回 JSON 数组，不要任何额外解释。

题目：{question_text}

示例输出格式：["知识点1", "知识点2", ...]
"""
    for attempt in range(max_retries):
        try:
            response = Generation.call(
                model=GENERATION_MODEL,
                prompt=prompt,
                temperature=0.3,
                max_tokens=512,
                result_format='message'
            )
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                # 尝试解析 JSON（可能被 markdown 包围）
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = content
                try:
                    points = json.loads(json_str)
                    if isinstance(points, list) and all(isinstance(p, str) for p in points):
                        return points
                    else:
                        print(f"返回内容不是字符串列表: {points}")
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}\n内容: {content}")
            else:
                print(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {response.message}")
        except Exception as e:
            print(f"异常: {e}")
        time.sleep(1)
    # 如果多次失败，返回原始文本作为一个整体知识点（保底）
    return [question_text]

# ==================== 2b. 备选：句子切分（如果 USE_LLM_EXTRACTION=False）====================
def split_into_sentences(text):
    sentences = re.split(r'[。！？.;!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]
    return sentences

# ==================== 3. 生成知识点 embedding（带缓存）====================
def get_embedding(text):
    resp = TextEmbedding.call(
        model=EMBEDDING_MODEL,
        input=text,
        text_type="query"
    )
    if resp.status_code == 200:
        emb = resp.output['embeddings'][0]['embedding']
        return np.array(emb, dtype=np.float32)
    else:
        raise Exception(f"API 调用失败: {resp.message}")

if os.path.exists(CACHE_FILE):
    chunk_embeddings = np.load(CACHE_FILE, allow_pickle=True).item()
    print(f"从 {CACHE_FILE} 加载缓存，共 {len(chunk_embeddings)} 个块")
else:
    chunk_embeddings = {}  # 格式: { (qid, chunk_idx): vector }
    for qid, item in enumerate(tqdm(questions, desc="提取知识点并获取embedding")):
        text = item['input']
        # 根据配置选择知识点提取方式
        if USE_LLM_EXTRACTION:
            points = extract_knowledge_points(text)
        else:
            points = split_into_sentences(text)
        for idx, point in enumerate(points):
            key = (qid, idx)
            try:
                emb = get_embedding(point)
                chunk_embeddings[key] = emb
                time.sleep(0.1)  # 控制 embedding API 调用频率
            except Exception as e:
                print(f"  跳过知识点 '{point[:30]}...' 错误: {e}")
        # 控制生成模型调用频率（若使用大模型提取，可适当增加间隔）
        if USE_LLM_EXTRACTION:
            time.sleep(0.5)  # 避免生成 API 限流
    np.save(CACHE_FILE, chunk_embeddings)

# 构建题目到其所有知识点向量的映射
question_to_chunks = {}
for (qid, idx), emb in chunk_embeddings.items():
    if qid not in question_to_chunks:
        question_to_chunks[qid] = []
    question_to_chunks[qid].append(emb)

valid_qids = [qid for qid, embs in question_to_chunks.items() if len(embs) > 0]
print(f"有效题目数量: {len(valid_qids)}")

# 构建块索引到 qid 的列表（按存储顺序）
block_qid_list = [key[0] for key in chunk_embeddings.keys()]
block_vectors = list(chunk_embeddings.values())

# ==================== 4. 定义拼接方式（保持不变）====================
def concat_horizontal(emb_list): return np.concatenate(emb_list)
def vertical_concat(emb_list): return np.concatenate(emb_list)
def average(emb_list): return np.mean(emb_list, axis=0)
def weighted_avg(emb_list, weights=None):
    if weights is None:
        weights = np.ones(len(emb_list)) / len(emb_list)
    return np.average(emb_list, axis=0, weights=weights)
def elementwise_product(emb_list):
    result = np.ones_like(emb_list[0])
    for emb in emb_list: result *= emb
    return result
def diff(emb_list):
    if len(emb_list) > 1: return emb_list[0] - emb_list[-1]
    else: return emb_list[0]
def combined(emb_list):
    concat = np.concatenate(emb_list)
    avg = np.mean(emb_list, axis=0)
    prod = np.ones_like(emb_list[0])
    for emb in emb_list: prod *= emb
    return np.concatenate([concat, avg, prod])
def max_pool(emb_list): return np.max(emb_list, axis=0)
def min_pool(emb_list): return np.min(emb_list, axis=0)

METHODS = {
    "horizontal": concat_horizontal,
    "vertical": vertical_concat,
    "average": average,
    "weighted_avg": lambda x: weighted_avg(x, weights=None),
    "elementwise_product": elementwise_product,
    "diff": diff,
    "combined": combined,
    "max_pool": max_pool,
    "min_pool": min_pool,
}

# ==================== 5. 辅助函数：投影到d维 ====================
def project_to_d(vec, d):
    total_len = vec.shape[0]
    if total_len % d != 0:
        raise ValueError(f"向量长度 {total_len} 不是 {d} 的整数倍，无法投影")
    n_segments = total_len // d
    return vec.reshape(n_segments, d).mean(axis=0)

# ==================== 6. 评估指标实现（保持不变）====================
def compute_eosk(orig_matrix, comp_matrix, k=2, n_sub=10):
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

def compute_robustness(method_name, method_func, chunk_lists, original_vectors, num_shuffles=5):
    similarities = []
    for i, embs in enumerate(chunk_lists):
        orig_vec = original_vectors[i]
        for _ in range(num_shuffles):
            shuffled = embs.copy()
            random.shuffle(shuffled)
            shuffled_vec = method_func(shuffled)
            if shuffled_vec.shape != orig_vec.shape:
                print(f"警告: 方法 {method_name} 在题目 {i} 上维度不匹配 (打乱后 {shuffled_vec.shape} vs 原始 {orig_vec.shape})，跳过")
                continue
            sim = cosine_similarity([shuffled_vec], [orig_vec])[0][0]
            similarities.append(sim)
    return np.mean(similarities) if similarities else np.nan

# ==================== 7. 主流程 ====================
def main():
    chunk_lists = [question_to_chunks[qid] for qid in valid_qids]
    orig_avg_vectors = np.array([np.mean(embs, axis=0) for embs in chunk_lists])

    method_vectors = {}
    print("\n生成各拼接方式的题目向量...")
    for method_name, method_func in tqdm(METHODS.items(), desc="拼接方式"):
        vecs = [method_func(embs) for embs in chunk_lists]
        method_vectors[method_name] = vecs

    results = {}
    for method_name, vecs in method_vectors.items():
        print(f"\n评估方法: {method_name}")
        d = orig_avg_vectors.shape[1]
        proj_vecs = None
        try:
            proj_vecs = [project_to_d(v, d) for v in vecs]
        except ValueError:
            pass
        if proj_vecs is not None:
            proj_mat = np.array(proj_vecs)
            eosk = compute_eosk(orig_avg_vectors, proj_mat, k=TOP_K_EOSK, n_sub=N_SUB_EOSK)
            results[f"{method_name}_EOSk"] = eosk
            print(f"  EOSk: {eosk:.4f}")
        else:
            results[f"{method_name}_EOSk"] = np.nan
            print(f"  EOSk: 无法投影 (维度不匹配)")

        robustness = compute_robustness(method_name, METHODS[method_name], chunk_lists, vecs, num_shuffles=5)
        results[f"{method_name}_Robustness"] = robustness
        print(f"  变换鲁棒性 (打乱顺序): {robustness:.4f}")

        if proj_vecs is not None:
            recall = compute_retrieval_recall(proj_mat, block_vectors, block_qid_list, k=RETRIEVAL_K)
            results[f"{method_name}_Recall@{RETRIEVAL_K}"] = recall
            print(f"  检索 Recall@{RETRIEVAL_K}: {recall:.4f}")
        else:
            results[f"{method_name}_Recall@{RETRIEVAL_K}"] = np.nan
            print(f"  检索 Recall@{RETRIEVAL_K}: 无法投影")

    # 输出汇总表格
    print("\n" + "="*60)
    print("各拼接方式评估结果汇总")
    print("="*60)
    methods_list = list(METHODS.keys())
    eosk_vals, rob_vals, recall_vals = [], [], []
    for method_name in methods_list:
        eosk = results.get(f"{method_name}_EOSk", np.nan)
        rob = results.get(f"{method_name}_Robustness", np.nan)
        recall = results.get(f"{method_name}_Recall@{RETRIEVAL_K}", np.nan)
        eosk_vals.append(eosk); rob_vals.append(rob); recall_vals.append(recall)
        print(f"\n方法: {method_name}")
        print(f"  EOSk: {eosk:.4f}")
        print(f"  鲁棒性: {rob:.4f}")
        print(f"  Recall@{RETRIEVAL_K}: {recall:.4f}")

    # 绘制图表
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    x = np.arange(len(methods_list))
    width = 0.6
    ax1 = axes[0]
    ax1.bar(x, eosk_vals, width, color='steelblue', edgecolor='black')
    ax1.set_ylabel('EOSk'); ax1.set_title('各拼接方式的谱保真度 (EOSk)')
    ax1.set_xticks(x); ax1.set_xticklabels(methods_list, rotation=45, ha='right')
    ax1.set_ylim(0, 1.05)
    for i, v in enumerate(eosk_vals):
        if not np.isnan(v):
            ax1.text(i, v+0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax2 = axes[1]
    ax2.bar(x, rob_vals, width, color='coral', edgecolor='black')
    ax2.set_ylabel('Robustness'); ax2.set_title('各拼接方式的变换鲁棒性 (顺序打乱)')
    ax2.set_xticks(x); ax2.set_xticklabels(methods_list, rotation=45, ha='right')
    ax2.set_ylim(0, 1.05)
    for i, v in enumerate(rob_vals):
        if not np.isnan(v):
            ax2.text(i, v+0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax3 = axes[2]
    ax3.bar(x, recall_vals, width, color='forestgreen', edgecolor='black')
    ax3.set_ylabel(f'Recall@{RETRIEVAL_K}')
    ax3.set_title(f'各拼接方式的检索准确率 (Recall@{RETRIEVAL_K})')
    ax3.set_xticks(x); ax3.set_xticklabels(methods_list, rotation=45, ha='right')
    ax3.set_ylim(0, 1.05)
    for i, v in enumerate(recall_vals):
        if not np.isnan(v):
            ax3.text(i, v+0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('拼接方式评估结果_LLM.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n图表已保存为 '拼接方式评估结果_LLM.png'")

if __name__ == "__main__":
    main()