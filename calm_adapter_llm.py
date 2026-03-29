# -*- coding: utf-8 -*-
"""
CALM Adapter with LLM-in-the-Loop Training (方向一)
====================================================
改进点（相对于 calm_adapter.py）：
  1. LLM-Judge 作为训练信号（每 N 步调用 Qwen-Plus 给压缩向量打语义保留分）
  2. KNN 对齐损失：压缩向量检索 top-K 文本，要求与原始题目高度重叠
  3. 对比学习（InfoNCE）：保留，增强向量区分度

训练损失 = L_recon + λ1·L_contra + λ2·L_llm_align
           重建损失   对比损失         LLM 对齐损失（稀疏，每 align_interval 步更新一次）

LLM 对齐损失（L_llm_align）设计：
  - 用当前 Adapter 压缩向量检索 top-3 候选文本
  - 将候选文本和原始题目文本同时送给 Qwen，询问语义保留程度（0~1）
  - 分数高 → 梯度鼓励当前向量；分数低 → 梯度推动向量向 C_mean 方向靠拢
  - 实现：LLM 给出 score，通过 (1 - score) * ||z - z_mean||² 反向传播

注：LLM 调用是"延迟监督"（每 align_interval=5 个 batch 调一次，batch_size_llm=8 条），
    整体 API 消耗可控（约 200~400 次调用 / epoch）
"""

import os, sys, io, pickle, json, re, time, random, warnings
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTHONIOENCODING'] = 'utf-8'
# 强制无缓冲输出（日志实时写入）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

warnings.filterwarnings("ignore")

BASE   = r"C:\Users\hwm20\OneDrive\桌面\Research\embedding"
DIM    = 384
KMAX   = 7
DEVICE = "cpu"
QWEN_API_KEY = "sk-83da80fde3fd412a97a5668b8e1f8104"

sys.path.insert(0, BASE)
from brier_eval import compute_brierlm as _compute_brierlm

# ─────────────────────────────────────────────────────────────────
# 1. Qwen API（复用 reconstruction_v2.py 风格）
# ─────────────────────────────────────────────────────────────────
import dashscope
from dashscope import Generation

dashscope.api_key = QWEN_API_KEY


def llm_judge_batch(orig_texts, cand_texts_list, max_retries=3):
    """
    批量 LLM-Judge：对每道题，判断候选文本相比原文保留了多少关键语义。
    
    orig_texts      : List[str], 原始题目文本
    cand_texts_list : List[List[str]], 每题的 top-K 候选文本列表
    返回: List[float], 每题的语义保留分数 [0, 1]
    """
    scores = []
    for orig, cands in zip(orig_texts, cand_texts_list):
        cand_str = "\n".join([f"  候选{i+1}: {c[:200]}" for i, c in enumerate(cands)])
        prompt = f"""你是一个化学教育专家。请判断以下候选题目与原题目的语义相似程度。

原题目：{orig[:300]}

检索到的候选题目：
{cand_str}

请从以下角度综合打分（0到1之间的小数）：
1. 知识点覆盖：候选题目是否包含原题目的核心化学概念？
2. 题型一致：候选题目的解题类型与原题相似？
3. 难度相当：解题难度相近？

只返回一个0到1之间的小数，不要其他内容。例如：0.75"""
        
        for attempt in range(max_retries):
            try:
                resp = Generation.call(
                    model="qwen-plus",
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=10,
                )
                if resp.status_code == 200:
                    text = resp.output.text.strip()
                    match = re.search(r'\d+\.?\d*', text)
                    if match:
                        val = float(match.group())
                        scores.append(min(max(val, 0.0), 1.0))
                    else:
                        scores.append(0.5)
                    break
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(2)
        else:
            scores.append(0.5)  # 失败时默认中等分数
    
    return scores


# ─────────────────────────────────────────────────────────────────
# 2. 数据加载（复用 calm_adapter.py 风格，另外保存文本）
# ─────────────────────────────────────────────────────────────────
def load_data(pkl_path, token_method="T_mean"):
    """
    返回:
        chunk_seqs  : List[np.ndarray (K, 384)]
        cmean_vecs  : np.ndarray (N, 384)
        texts       : List[str]   原始题目文本
        all_vecs    : np.ndarray (N, 384)  用于 KNN 构建语料库（= C_mean）
    """
    print(f"加载 results_data.pkl ...")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    concat_key = (token_method, "C_concat")
    mean_key   = (token_method, "C_mean")

    concat_data = d[concat_key]["vectors"]
    mean_data   = np.array(d[mean_key]["vectors"], dtype=np.float32)
    texts       = d[mean_key]["texts"]  # 原始题目文本

    chunk_seqs = []
    for raw in concat_data:
        v = np.array(raw, dtype=np.float32)
        K = len(v) // DIM
        if K == 0:
            chunk_seqs.append(np.zeros((1, DIM), dtype=np.float32))
        else:
            chunk_seqs.append(v.reshape(K, DIM))

    print(f"  题目数: {len(chunk_seqs)}, 平均K={np.mean([c.shape[0] for c in chunk_seqs]):.2f}")
    return chunk_seqs, mean_data, texts, mean_data.copy()


# ─────────────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────────────
class ChunkDataset(Dataset):
    def __init__(self, chunk_seqs, indices):
        self.data    = chunk_seqs
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.tensor(self.data[idx], dtype=torch.float32), idx


def collate_pad(batch):
    """pad到KMAX，返回 (B, KMAX, 384), Ks, indices"""
    seqs, idxs = zip(*batch)
    Ks = [x.shape[0] for x in seqs]
    padded = torch.zeros(len(seqs), KMAX, DIM)
    for i, (x, k) in enumerate(zip(seqs, Ks)):
        padded[i, :k] = x
    return padded, torch.tensor(Ks, dtype=torch.long), list(idxs)


# ─────────────────────────────────────────────────────────────────
# 4. 模型（与 calm_adapter.py 完全一致）
# ─────────────────────────────────────────────────────────────────
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, dim=DIM, heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm  = nn.LayerNorm(dim)
        self.proj  = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, key_padding_mask=None):
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        out = self.norm(out.squeeze(1))
        return self.proj(out)


class ChunkReconstructor(nn.Module):
    def __init__(self, dim=DIM, kmax=KMAX):
        super().__init__()
        self.kmax = kmax
        self.expand = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * kmax),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, z):
        B = z.shape[0]
        out = self.expand(z).reshape(B, self.kmax, -1)
        return self.norm(out)


class CALMAdapterLLM(nn.Module):
    def __init__(self, dim=DIM, heads=4, kmax=KMAX):
        super().__init__()
        self.compressor    = MultiHeadAttentionPooling(dim, heads)
        self.reconstructor = ChunkReconstructor(dim, kmax)

    def forward(self, x, Ks):
        B, K, D = x.shape
        mask = torch.arange(K, device=x.device).unsqueeze(0) >= Ks.unsqueeze(1)
        z    = self.compressor(x, key_padding_mask=mask)
        recon = self.reconstructor(z)
        return z, recon


# ─────────────────────────────────────────────────────────────────
# 5. 损失函数
# ─────────────────────────────────────────────────────────────────
def recon_loss(recon, x, Ks):
    B, K, D = x.shape
    loss = 0.0; cnt = 0
    for i in range(B):
        k = Ks[i].item()
        loss += ((recon[i, :k] - x[i, :k]) ** 2).mean()
        cnt += 1
    return loss / max(cnt, 1)


def info_nce_loss(z, temperature=0.07):
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0)
    z_norm = nn.functional.normalize(z, dim=-1)
    sim = z_norm @ z_norm.T / temperature
    labels = torch.arange(B, device=z.device)
    sim_copy = sim.clone()
    sim_copy.fill_diagonal_(-1e9)
    loss = -torch.log(
        torch.exp(torch.diagonal((z_norm @ z_norm.T) / temperature)) /
        torch.exp(sim_copy).sum(dim=-1).clamp(min=1e-8)
    ).mean()
    return loss.clamp(min=0.0)


def llm_align_loss(z, z_mean_tensor, llm_scores):
    """
    LLM 对齐损失：
      score 高 → 当前向量好，loss = (1-score) * 0（不惩罚）
      score 低 → 当前向量差，推向 C_mean 方向
    
    L_align = mean( (1 - score_i) * ||z_i - z_mean_i||² )
    
    z             : (B, 384) 当前压缩向量（requires_grad=True）
    z_mean_tensor : (B, 384) 对应的 C_mean 向量（无梯度，作为目标锚点）
    llm_scores    : List[float] 每条的 LLM-Judge 分数 [0, 1]
    """
    scores = torch.tensor(llm_scores, dtype=torch.float32, device=z.device)  # (B,)
    weights = (1.0 - scores)  # 分数低 → 权重大 → 惩罚大
    diff_sq = ((z - z_mean_tensor) ** 2).sum(dim=-1)  # (B,)
    loss = (weights * diff_sq).mean()
    return loss


# ─────────────────────────────────────────────────────────────────
# 6. KNN 检索辅助（构建语料库用于 LLM-Judge）
# ─────────────────────────────────────────────────────────────────
class KNNRetriever:
    def __init__(self, corpus_vecs, corpus_texts, k=3):
        self.texts = corpus_texts
        norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self.vecs = corpus_vecs / norms
        self.knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        self.knn.fit(self.vecs)
        self.k = k

    def retrieve(self, query_vecs):
        """返回每条的 top-K 文本列表 List[List[str]]"""
        norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        q = query_vecs / norms
        _, idxs = self.knn.kneighbors(q)
        return [[self.texts[i] for i in row] for row in idxs]


# ─────────────────────────────────────────────────────────────────
# 7. 主训练流程
# ─────────────────────────────────────────────────────────────────
def train(
    epochs=60,
    batch_size=64,
    lr=3e-4,
    lam1=0.05,          # InfoNCE 权重
    lam2=0.3,           # LLM 对齐损失权重
    patience=12,
    align_interval=5,   # 每 N 个 batch 调用一次 LLM-Judge
    llm_batch=8,        # 每次 LLM 调用处理的样本数
    token_method="T_mean",
):
    print("=" * 60)
    print("CALM Adapter with LLM-in-the-Loop Training")
    print("=" * 60)
    print(f"配置: lam1(InfoNCE)={lam1}, lam2(LLM-align)={lam2}, "
          f"align_interval={align_interval}, llm_batch={llm_batch}")

    # ── 数据 ──────────────────────────────────────────────────────
    pkl_path = os.path.join(BASE, "results_data.pkl")
    chunk_seqs, cmean_vecs, texts, corpus_vecs = load_data(pkl_path, token_method)
    N = len(chunk_seqs)

    # KNN 检索器（用 C_mean 向量作为语料库，文本即为原始题目文本）
    print("构建 KNN 检索器...")
    retriever = KNNRetriever(corpus_vecs, texts, k=3)

    # 划分训练/验证集
    all_idx = list(range(N))
    tr_idx, va_idx = train_test_split(all_idx, test_size=0.2, random_state=42)

    tr_ds = ChunkDataset(chunk_seqs, tr_idx)
    va_ds = ChunkDataset(chunk_seqs, va_idx)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       collate_fn=collate_pad, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       collate_fn=collate_pad)

    # ── 模型 ──────────────────────────────────────────────────────
    model = CALMAdapterLLM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # C_mean 向量（作为 LLM 对齐损失的锚点）
    cmean_tensor = torch.tensor(cmean_vecs, dtype=torch.float32)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    # 统计 LLM 调用次数
    llm_call_count = 0
    llm_score_log  = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_losses, tr_recon, tr_nce, tr_align = [], [], [], []
        batch_step = 0

        for xb, Ks, idxs in tr_dl:
            xb = xb.to(DEVICE)
            Ks = Ks.to(DEVICE)
            z, recon = model(xb, Ks)

            l_r = recon_loss(recon, xb, Ks)
            l_c = info_nce_loss(z)
            l_a = torch.tensor(0.0)

            # ── LLM-in-the-loop（每 align_interval 步） ──────────
            do_llm = (batch_step % align_interval == 0)
            if do_llm and lam2 > 0:
                # 选 llm_batch 个样本做 LLM-Judge
                b_size = min(llm_batch, len(idxs))
                sel_local = random.sample(range(len(idxs)), b_size)
                sel_global = [idxs[i] for i in sel_local]

                # 用当前 z 检索候选文本
                z_np = z[sel_local].detach().cpu().numpy()
                cand_texts_list = retriever.retrieve(z_np)
                orig_texts_sel  = [texts[i] for i in sel_global]

                # 调用 Qwen-Plus 打分
                scores = llm_judge_batch(orig_texts_sel, cand_texts_list)
                llm_call_count += b_size
                llm_score_log.append(np.mean(scores))

                # 计算 LLM 对齐损失（只对选中的样本）
                z_sel        = z[sel_local]                         # (b_size, 384)
                zmean_sel    = cmean_tensor[sel_global].to(DEVICE)  # (b_size, 384)
                l_a = llm_align_loss(z_sel, zmean_sel, scores)

            loss = l_r + lam1 * l_c + lam2 * l_a
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tr_losses.append(loss.item())
            tr_recon.append(l_r.item())
            tr_nce.append(l_c.item())
            tr_align.append(l_a.item())
            batch_step += 1

        sched.step()

        # ── 验证 ──────────────────────────────────────────────────
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, Ks, _ in va_dl:
                xb, Ks = xb.to(DEVICE), Ks.to(DEVICE)
                z, recon = model(xb, Ks)
                l_r = recon_loss(recon, xb, Ks)
                l_c = info_nce_loss(z)
                va_losses.append((l_r + lam1 * l_c).item())

        tr_l = np.mean(tr_losses)
        va_l = np.mean(va_losses)

        if ep % 5 == 0 or ep == 1:
            avg_llm = np.mean(llm_score_log[-20:]) if llm_score_log else 0.0
            elapsed = time.time() - t0
            print(
                f"Ep {ep:3d}/{epochs}  train={tr_l:.4f}  val={va_l:.4f}  "
                f"recon={np.mean(tr_recon):.4f}  nce={np.mean(tr_nce):.4f}  "
                f"align={np.mean(tr_align):.4f}  llm_score={avg_llm:.3f}  "
                f"llm_calls={llm_call_count}  [{elapsed:.0f}s]"
            )

        if va_l < best_val - 1e-5:
            best_val   = va_l
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {ep}  (patience={patience})")
                break

    model.load_state_dict(best_state)
    print(f"\n训练完成。最佳 val_loss = {best_val:.4f}，LLM 调用总次数: {llm_call_count}")

    # ── 全量推理 ──────────────────────────────────────────────────
    print("\n全量推理，提取压缩向量...")
    full_ds = ChunkDataset(chunk_seqs, list(range(N)))
    full_dl = DataLoader(full_ds, batch_size=128, shuffle=False, collate_fn=collate_pad)
    model.eval()
    all_z = []
    with torch.no_grad():
        for xb, Ks, _ in full_dl:
            z, _ = model(xb.to(DEVICE), Ks.to(DEVICE))
            all_z.append(z.cpu().numpy())
    compressed_vecs = np.concatenate(all_z, axis=0)
    print(f"压缩向量 shape: {compressed_vecs.shape}")

    # ── 评估 ──────────────────────────────────────────────────────
    print("\n计算 VecBrierLM...")
    score_llm,   detail_llm   = _compute_brierlm(compressed_vecs, temperature=0.1, subsample=N)
    score_cmean, detail_cmean = _compute_brierlm(cmean_vecs,      temperature=0.1, subsample=N)

    brier_mat    = np.load(os.path.join(BASE, "brier_results.npy"))
    best_existing = float(brier_mat.max())

    # 加载原始 CALM Adapter（无LLM）结果作为对比
    orig_adapter_score = None
    orig_adapter_path = os.path.join(BASE, "calm_adapter_results.json")
    if os.path.exists(orig_adapter_path):
        with open(orig_adapter_path) as f:
            orig_data = json.load(f)
        orig_adapter_score = orig_data.get("score_adapter")

    print(f"\n{'='*55}")
    print(f"  VecBrierLM 对比结果")
    print(f"  C_mean 基线                : {score_cmean:.2f}")
    if orig_adapter_score:
        print(f"  CALM Adapter (无LLM)       : {orig_adapter_score:.2f}")
    print(f"  CALM Adapter + LLM-in-loop : {score_llm:.2f}")
    print(f"  现有最优 (T_mean×C_combined): {best_existing:.2f}")
    print(f"  LLM-Adapter 细节: {detail_llm}")
    print(f"{'='*55}")

    # ── 保存结果 ──────────────────────────────────────────────────
    result = {
        "score_llm_adapter":   float(score_llm),
        "score_cmean":         float(score_cmean),
        "score_orig_adapter":  orig_adapter_score,
        "best_existing":       best_existing,
        "detail_llm":          detail_llm,
        "detail_cmean":        detail_cmean,
        "n_params":            n_params,
        "epochs_trained":      ep,
        "best_val_loss":       float(best_val),
        "llm_call_count":      llm_call_count,
        "avg_llm_score":       float(np.mean(llm_score_log)) if llm_score_log else None,
        "config": {
            "lam1": lam1, "lam2": lam2,
            "align_interval": align_interval,
            "llm_batch": llm_batch,
            "token_method": token_method,
            "batch_size": batch_size,
            "lr": lr,
        }
    }

    out_vecs  = os.path.join(BASE, "calm_adapter_llm_vecs.npy")
    out_json  = os.path.join(BASE, "calm_adapter_llm_results.json")
    out_model = os.path.join(BASE, "calm_adapter_llm_weights.pt")

    np.save(out_vecs, compressed_vecs)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    torch.save(best_state, out_model)

    print(f"\n结果已保存:")
    print(f"  {out_vecs}")
    print(f"  {out_json}")
    print(f"  {out_model}")

    return result


# ─────────────────────────────────────────────────────────────────
# 8. 入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = train(
        epochs=60,
        batch_size=64,
        lr=3e-4,
        lam1=0.05,           # InfoNCE 权重（轻）
        lam2=0.3,            # LLM 对齐损失权重
        patience=12,
        align_interval=5,    # 每 5 个 batch 调用一次 LLM
        llm_batch=8,         # 每次 LLM 处理 8 条样本
        token_method="T_mean",
    )
