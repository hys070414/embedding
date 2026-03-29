import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from config import *
from data_loader import load_questions_from_jsonl, prepare_token_embeddings
from token_methods import get_token_methods_with_attention
from chunk_methods import get_chunk_methods_with_attention
from evaluation import compute_eosk, compute_retrieval_recall, compute_intra_inter_separation, project_to_d, compute_brierlm

def run_experiment(token_attn_layer=None, chunk_attn_layer=None):
    # 1. 加载数据
    questions = load_questions_from_jsonl(JSONL_PATH, SAMPLE_SIZE)
    print(f"加载题目数: {len(questions)}") 
    token_data = prepare_token_embeddings(questions)
    print("token_data 键:", token_data.keys()) 
    token_embs = token_data['token_embs']
    kp_token_counts = token_data.get('kp_token_counts', {})

    # 提取类型标签并编码
    types = [q.get('type', 'unknown') for q in questions]
    le = LabelEncoder()
    type_labels = le.fit_transform(types)   # 每个题目对应的整数标签

    all_qids = sorted(set([key[0] for key in kp_token_counts.keys()]))
    print(f"有效题目数量: {len(all_qids)}")
    if len(all_qids) == 0:
        print("没有有效题目，实验终止")
        t_names = list(get_token_methods_with_attention(token_attn_layer).keys())
        c_names = list(get_chunk_methods_with_attention(chunk_attn_layer).keys())
        return np.full((len(t_names), len(c_names), 4), np.nan), t_names, c_names

   
    # 2. 获取拼接方法
    token_methods = get_token_methods_with_attention(token_attn_layer)
    chunk_methods = get_chunk_methods_with_attention(chunk_attn_layer)

    t_names = list(token_methods.keys())
    c_names = list(chunk_methods.keys())

    # 存储结果矩阵： (t_idx, c_idx, metric)
    # metric: 0=EOSk, 1=Recall@k, 2=Separation, 3=Silhouette, 4=BrierLM (CALM-inspired)
    results = np.full((len(t_names), len(c_names), 5), np.nan)
  # ========== 新增：初始化用于保存数据的字典 ==========
    results_data = {}
    # 3. 主循环
    for ti, t_name in enumerate(tqdm(t_names, desc="Token方法")):
        t_func = token_methods[t_name]

        # 对每个题目，生成投影后的知识点向量列表（维度均为 EMBED_DIM）
        chunk_vectors_proj = {}  # qid -> list of projected knowledge point vectors
        for qid in all_qids:
            raw_vecs = []
            kid = 0
            while True:
                token_count = kp_token_counts.get((qid, kid), 0)
                if token_count == 0:
                    break
                token_vecs = []
                for tid in range(token_count):
                    key = (qid, kid, tid)
                    if key in token_embs:
                        token_vecs.append(token_embs[key])
                if token_vecs:
                    try:
                        vec = t_func(token_vecs)          # 原始维度可能不同
                        raw_vecs.append(vec)
                    except Exception as e:
                        print(f"token 方法 {t_name} 失败 (qid={qid},kid={kid}): {e}")
                        # 失败时添加零向量（维度为 EMBED_DIM 的投影版本）
                        raw_vecs.append(np.zeros(EMBED_DIM))
                kid += 1
            # 将每个知识点向量投影到 EMBED_DIM
            proj_vecs = []
            for v in raw_vecs:
                try:
                    pv = project_to_d(v, EMBED_DIM)
                    proj_vecs.append(pv)
                except ValueError:
                    # 如果投影失败（极少见），用零向量代替
                    proj_vecs.append(np.zeros(EMBED_DIM))
            chunk_vectors_proj[qid] = proj_vecs

        # 构建用于检索的块向量列表（投影后的向量）
        block_vectors = []
        block_qid_list = []
        for qid in all_qids:
            for vec in chunk_vectors_proj[qid]:
                block_vectors.append(vec)
                block_qid_list.append(qid)

        # 原始平均向量（基于投影后的向量）
        orig_avg = np.array([np.mean(chunk_vectors_proj[qid], axis=0) for qid in all_qids])

        # 对每种 chunk 方法
        for ci, c_name in enumerate(c_names):
            c_func = chunk_methods[c_name]

            # 生成题目整体向量（输入为投影后的知识点向量）
            doc_vectors = []
            for qid in all_qids:
                proj_vecs = chunk_vectors_proj[qid]
                try:
                    doc_vec = c_func(proj_vecs)               # 输入已统一维度
                    doc_vectors.append(doc_vec)
                except Exception as e:
                    print(f"chunk 方法 {c_name} 失败 (qid={qid}): {e}")
                    doc_vectors.append(np.zeros(EMBED_DIM))   # 保底

            # 投影题目向量（若某些 chunk 方法输出维度可能不是 EMBED_DIM，但这里我们强制投影）
            proj_doc_vecs = []
            for v in doc_vectors:
                try:
                    pv = project_to_d(v, EMBED_DIM)
                    proj_doc_vecs.append(pv)
                except ValueError:
                    proj_doc_vecs = None
                    break
            if proj_doc_vecs is None:
                # 如果投影失败，该组合所有指标设为 nan
                results[ti, ci, :] = np.nan
                continue

            proj_mat = np.array(proj_doc_vecs)

            # 计算 EOSk
            eosk = compute_eosk(orig_avg, proj_mat, k=TOP_K_EOSK, n_sub=N_SUB_EOSK)
            results[ti, ci, 0] = eosk

            # 计算 Recall@k
            recall = compute_retrieval_recall(proj_mat, block_vectors, block_qid_list, k=RETRIEVAL_K)
            results[ti, ci, 1] = recall

            # 分离度和轮廓系数
            labels_for_qids = np.array([type_labels[qid] for qid in all_qids])
            separation, sil = compute_intra_inter_separation(proj_mat, labels_for_qids)
            results[ti, ci, 2] = separation
            results[ti, ci, 3] = sil

            # ========== 新增：BrierLM（CALM-inspired，无API，纯本地） ==========
            # 衡量：拼接后的向量能否从语料库中"找回自身"（准确性+多样性惩罚）
            try:
                brierlm_score, brier_detail = compute_brierlm(proj_mat)
                results[ti, ci, 4] = brierlm_score
            except Exception as e:
                print(f"BrierLM 计算失败 ({t_name}×{c_name}): {e}")
                results[ti, ci, 4] = np.nan


            # ========== 新增：保存当前组合的向量和原始文本 ==========
            key = (t_name, c_name)
            # 原始文本按 all_qids 顺序提取
            orig_texts = [questions[qid]['input'] for qid in all_qids]
            # 注意：doc_vectors 是列表，每个元素是 numpy 数组（可能维度不同），直接保存列表
            results_data[key] = {
                'vectors': doc_vectors,   # 列表形式，保留原始维度
                'texts': orig_texts
            }
    
    # ========== 新增：所有组合处理完后保存 ==========
    import pickle
    with open('results_data.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    print(f"已保存 {len(results_data)} 个组合的结果到 results_data.pkl")
    
    return results, t_names, c_names
