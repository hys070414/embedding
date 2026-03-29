import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import os
import json
import re
import time
import numpy as np
from tqdm import tqdm
import dashscope
from dashscope import Generation
from sentence_transformers import SentenceTransformer
import torch
from config import *

dashscope.api_key = QWEN_API_KEY

# ==================== 1. 从 jsonl 加载题目 ====================
def load_questions_from_jsonl(file_path, sample_size=None):
    questions = []
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        q = json.loads(line)
                        if 'input' in q:
                            questions.append(q)
                    except json.JSONDecodeError:
                        continue
            print(f"成功使用 {enc} 编码加载文件")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法使用常见编码读取文件 {file_path}")
    print(f"共加载 {len(questions)} 道题目")
    if sample_size is not None and sample_size < len(questions):
        questions = random.sample(questions, sample_size)
        print(f"随机抽取 {sample_size} 道题目进行实验")
    return questions
# ==================== 2. 知识点提取函数（使用大模型）====================
def extract_knowledge_points(question_text, max_retries=3):
    """调用千问生成模型，从题目中提取独立知识点，返回字符串列表"""
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
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = content
                try:
                    points = json.loads(json_str)
                    if isinstance(points, list) and all(isinstance(p, str) for p in points):
                        return points
                except json.JSONDecodeError:
                    pass
            time.sleep(1)
        except Exception:
            time.sleep(2)
    return [question_text]  # 保底返回原文本

# ==================== 2b. 备选：句子切分（如果 USE_LLM_EXTRACTION=False）====================
def split_into_sentences(text):
    sentences = re.split(r'[。！？.;!?]', text)
    return [s.strip() for s in sentences if s.strip()] or [text]

# ==================== 3. 加载本地 token embedding 模型 ====================
print("加载本地 token embedding 模型...")
token_model = SentenceTransformer(LOCAL_EMBED_MODEL)
# 确保模型在 CPU 上（可根据需要调整）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
token_model = token_model.to(device)

def get_token_embeddings(text):
    """
    返回每个 token 的 embedding，形状 (num_tokens, embed_dim)
    使用 sentence-transformers 内部 tokenization 和模型前向获取 token 级向量。
    """
    # tokenize
    tokens = token_model.tokenize([text])
    # 模型前向
    with torch.no_grad():
        # 返回的 output 包含 token_embeddings (batch, seq_len, hidden)
        output = token_model.forward(tokens)
    token_embs = output['token_embeddings'][0].cpu().numpy()  # (seq_len, embed_dim)
    # 去除特殊 token（如 [CLS] 和 [SEP]），通常前两个 token 是特殊标记，这里简单去掉第一个和最后一个
    # 注意：不同模型特殊 token 位置不同，all-MiniLM-L6-v2 第一个是 [CLS]，最后一个是 [SEP]
    if token_embs.shape[0] > 2:
        token_embs = token_embs[1:-1]  # 去掉 [CLS] 和 [SEP]
    else:
        token_embs = token_embs  # 可能只有一个 token？
    return token_embs.astype(np.float32)

# ==================== 4. 生成所有 token 级 embedding 并缓存 ====================
def prepare_token_embeddings(questions):
    """生成 token 级 embedding，返回字典和辅助信息"""
    if os.path.exists(CACHE_FILE):
        data = np.load(CACHE_FILE, allow_pickle=True).item()
        print(f"从 {CACHE_FILE} 加载缓存")
        return data

    data = {
        'token_embs': {},           # key: (qid, kp_idx, token_idx) -> vector
        'kp_token_counts': {},       # key: (qid, kp_idx) -> token_count
        'questions': questions,      # 保存原始题目（可选）
    }

    for qid, item in enumerate(tqdm(questions, desc="提取知识点并生成token embedding")):
        text = item['input']
        # 提取知识点
        if USE_LLM_EXTRACTION:
            kp_list = extract_knowledge_points(text)[:MAX_KNOWLEDGE_POINTS]
        else:
            kp_list = split_into_sentences(text)[:MAX_KNOWLEDGE_POINTS]
            print(f"题目 {qid} 提取到 {len(kp_list)} 个知识点") 

        for kid, kp in enumerate(kp_list):
            try:
                token_embs = get_token_embeddings(kp)
                # 限制 token 数
                if token_embs.shape[0] > MAX_TOKENS_PER_POINT:
                    token_embs = token_embs[:MAX_TOKENS_PER_POINT]
                token_count = token_embs.shape[0]
                data['kp_token_counts'][(qid, kid)] = token_count
                for tid in range(token_count):
                    data['token_embs'][(qid, kid, tid)] = token_embs[tid]
            except Exception as e:
                print(f"  跳过知识点 '{kp[:30]}...' 错误: {e}")

        if USE_LLM_EXTRACTION:
            time.sleep(0.5)  # 避免限流

    np.save(CACHE_FILE, data)
    print(f"缓存已保存至 {CACHE_FILE}")
    return data