import os, sys, json, re, time, random, numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pickle, dashscope
from dashscope import Generation
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from rouge_score import rouge_scorer as rouge_lib

QWEN_API_KEY = 'sk-83da80fde3fd412a97a5668b8e1f8104'
dashscope.api_key = QWEN_API_KEY

print('加载编码器...')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

data = pickle.load(open('results_data.pkl','rb'))
key = ('T_mean', 'C_mean')
vecs = data[key]['vectors']
texts = data[key]['texts']

# 构建 KNN 索引
corpus = list(set(t for d in data.values() for t in d['texts']))
corpus_embs = encoder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
knn.fit(corpus_embs)

def normalize_vec(vec):
    v = np.array(vec, dtype=np.float32).flatten()
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

# 测一道题
vec_384 = normalize_vec(np.array(vecs[0]).flatten())
orig = texts[0]
print('原题:', orig[:100])

# KNN candidates
dists, idxs = knn.kneighbors(vec_384.reshape(1,-1), n_neighbors=5)
candidates = [corpus[i] for i in idxs[0]]
cand_str = '\n'.join([f'{i+1}. {c[:150]}' for i,c in enumerate(candidates)])

prompt = f'You are a chemistry question reconstructor. Based on semantic candidates, generate the most likely chemistry question.\n\nTop 5 candidates:\n{cand_str}\n\nGenerate one chemistry question capturing the core knowledge. Output only the question:'
resp = Generation.call(model='qwen-plus', prompt=prompt, temperature=0.15, max_tokens=200, result_format='message')
recon = resp.output.choices[0].message.content.strip() if resp.status_code == 200 else candidates[0]
print('重建:', recon[:100])

# KP提取
def extract_kps(text):
    p = f'Extract chemistry knowledge points as JSON array. Return only JSON. Question: {text[:300]}'
    r = Generation.call(model='qwen-plus', prompt=p, temperature=0, max_tokens=150, result_format='message')
    if r.status_code == 200:
        m = re.search(r'\[.*?\]', r.output.choices[0].message.content, re.DOTALL)
        if m:
            try: return [k.lower() for k in json.loads(m.group()) if isinstance(k,str)]
            except: pass
    return []

kps_o = extract_kps(orig)
kps_r = extract_kps(recon)
print('原题知识点:', kps_o)
print('重建知识点:', kps_r)

# KP-F1
STOP = {'the','and','for','with','from','that','this','are','was','have','has','not','but','its','can','all','one','will'}
def to_tokens(kps):
    tokens = set()
    for kp in kps:
        tokens.update(w for w in re.findall(r'\b[a-zA-Z]{3,}\b', kp) if w not in STOP)
    return tokens

s_o = to_tokens(kps_o)
s_r = to_tokens(kps_r)
inter = s_o & s_r
prec = len(inter)/len(s_r) if s_r else 0
rec  = len(inter)/len(s_o) if s_o else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f'KP-F1: P={prec:.3f} R={rec:.3f} F1={f1:.3f}')
print(f'交集词汇: {inter}')

# ROUGE-L
rs = rouge_lib.RougeScorer(['rougeL'],use_stemmer=False)
rl = rs.score(orig, recon)['rougeL'].fmeasure
print(f'ROUGE-L: {rl:.3f}')
print('验证成功!')
