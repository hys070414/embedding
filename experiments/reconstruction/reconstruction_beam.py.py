"""
评估脚本：基于 LLM 引导式生成（Beam Search）将向量重建为文本，并计算重建质量。
需要安装：torch, transformers, sentence-transformers, nltk, rouge-score, matplotlib, seaborn, scikit-learn
"""
from evaluation import project_to_d
import random
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import os
import numpy as np
import torch
import random
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
ENCODER_NAME = "C:\\Users\\hwm20\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2\\snapshots\\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"  # 与实验一致的编码器
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"                 # 语言模型（可根据显存调整）
BEAM_WIDTH =3                                         # 束搜索宽度
MAX_STEPS =20  
EMBED_DIM = 384                                        # 最大生成步数
SAMPLE_SIZE =10                                      # 每种组合随机抽取的样本数（避免全量过慢）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载编码器
print(f"加载编码器 {ENCODER_NAME} ...")
encoder = SentenceTransformer(ENCODER_NAME)
encoder = encoder.to(DEVICE)

# 加载语言模型
print(f"加载语言模型 {LLM_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(
    LLM_NAME,
    trust_remote_code=True,
    clean_up_tokenization_spaces=True   # 显式设置，消除警告
)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
)
if DEVICE == "cpu":
    model = model.to(DEVICE)
model.eval()

def embed_text(text):
    """返回归一化的嵌入向量"""
    emb = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return torch.tensor(emb, dtype=torch.float32).to(DEVICE)
test_text = "test"
test_emb = embed_text(test_text)
print("embed_text 测试成功，向量维度:", test_emb.shape)
def beam_search_invert(target_emb, prefix="The sentence is: ", beam_width=BEAM_WIDTH, max_steps=MAX_STEPS):
    """
    从目标嵌入生成文本
    target_emb: 目标向量 (numpy array 或 torch tensor)
    prefix: 可选的初始提示
    """
    # 编码前缀
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    beams = [(prefix_ids, 0.0)]   # (token_ids, score)

    for step in range(max_steps):
        candidates = []
        for token_ids, score in beams:
            if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                candidates.append((token_ids, score))
                continue

            input_ids = torch.tensor([token_ids], device=DEVICE)
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]  # (vocab_size)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            top_k_indices = np.argsort(probs)[-beam_width:][::-1]
            for next_token in top_k_indices:
                new_ids = token_ids + [next_token]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                emb = embed_text(text)
                sim = torch.dot(emb, target_emb).item()  # 余弦相似度（已归一化）
                candidates.append((new_ids, sim))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        if all(ids and ids[-1] == tokenizer.eos_token_id for ids, _ in beams):
            break

    best_ids, best_score = beams[0]
    best_text = tokenizer.decode(best_ids, skip_special_tokens=True)
    return best_text, best_score

def compute_similarity(orig, recon):
    """计算原始文本和重建文本之间的相似度"""
    # 语义余弦相似度
    emb_orig = encoder.encode(orig, convert_to_numpy=True, normalize_embeddings=True)
    emb_recon = encoder.encode(recon, convert_to_numpy=True, normalize_embeddings=True)
    cos_sim = np.dot(emb_orig, emb_recon)

    # ROUGE-L
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(orig, recon)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([orig.split()], recon.split(), smoothing_function=smoothie)

    return cos_sim, rouge1, rougeL, bleu

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 加载实验结果
    print("加载 results_data.pkl ...")
    with open('results_data.pkl', 'rb') as f:
        results_data = pickle.load(f)

    # 定义所有拼接组合
    t_names = ['T_mean', 'T_max', 'T_min', 'T_concat', 'T_weighted', 'T_product',
               'T_first', 'T_last', 'T_attention']
    c_names = ['C_mean', 'C_max', 'C_min', 'C_concat', 'C_weighted', 'C_product',
               'C_diff', 'C_combined', 'C_attention']

    # 存储重建分数矩阵
    reconstruction_scores = np.full((len(t_names), len(c_names)), np.nan)

    # 2. 遍历每个组合，随机抽取样本进行重建
    for ti, t_name in enumerate(tqdm(t_names, desc="Token Methods")):
        for ci, c_name in enumerate(c_names):
            key = (t_name, c_name)
            if key not in results_data:
                continue
            data = results_data[key]
            vectors = data['vectors']   # 列表，每个元素为 numpy 数组
            orig_texts = data['texts']  # 原始文本列表

            if len(vectors) == 0:
                continue

            # 随机抽样（如果样本数少于 SAMPLE_SIZE，则全部使用）
            n_samples = min(len(vectors), SAMPLE_SIZE)
            indices = random.sample(range(len(vectors)), n_samples)

            scores = []
            for idx in indices:
                vec = vectors[idx]
                orig = orig_texts[idx]
                 # 投影到 EMBED_DIM
            try:
                vec_proj = project_to_d(vec, EMBED_DIM)
            except ValueError as e:
                print(f"投影失败: {key}, 样本 {idx}, 错误: {e}")
                continue

                # 确保向量是 numpy 数组并转为 tensor
                if isinstance(vec, np.ndarray):
                    vec_tensor = torch.tensor(vec, dtype=torch.float32).to(DEVICE)
                else:
                    vec_tensor = vec.to(DEVICE) if torch.is_tensor(vec) else torch.tensor(vec, dtype=torch.float32).to(DEVICE)
                # 归一化（如果向量未归一化）
                vec_tensor = vec_tensor / torch.norm(vec_tensor)

                try:
                    recon, _ = beam_search_invert(vec_tensor, prefix="The sentence is: ")
                    cos_sim, rouge1, rougeL, bleu = compute_similarity(orig, recon)
                    combined = 0.25 * cos_sim + 0.25 * rouge1 + 0.25 * rougeL + 0.25 * bleu
                    scores.append(combined)
                except Exception as e:
                    print(f"Error processing {key}: {e}")
                    continue

            if scores:
                reconstruction_scores[ti, ci] = np.mean(scores)
            else:
                reconstruction_scores[ti, ci] = np.nan

    # 3. 保存结果矩阵并绘制热力图
    np.save("reconstruction_scores.npy", reconstruction_scores)

    plt.figure(figsize=(12, 10))
    data_plot = np.nan_to_num(reconstruction_scores, nan=0.0)
    sns.heatmap(data_plot, annot=True, fmt='.3f',
                xticklabels=c_names, yticklabels=t_names,
                cmap='viridis', cbar_kws={'label': 'Reconstruction Score'})
    plt.xlabel('Chunk-level Methods')
    plt.ylabel('Token-level Methods')
    plt.title('Reconstruction Score (Higher = Better)')
    plt.tight_layout()
    plt.savefig('reconstruction_heatmap.png', dpi=150)
    plt.show()
    print("热力图已保存为 reconstruction_heatmap.png")
    print("重建分数矩阵统计:")
print("最小值:", np.nanmin(reconstruction_scores))
print("最大值:", np.nanmax(reconstruction_scores))
print("非 NaN 数量:", np.count_nonzero(~np.isnan(reconstruction_scores)))