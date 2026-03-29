# -*- coding: utf-8 -*-
"""
CALM-inspired Lightweight Adapter
===================================
灵感来源：CALM (Shao et al., 2025) 的向量压缩思路
本实现：纯本地 CPU，无需 LLM

架构：
  Compressor : [K, 384] -> [1, 384]   (多chunk → 单向量，类比 CALM K→1)
  Reconstructor: [1, 384] -> [K, 384]  (单向量 → 还原各chunk，类比 CALM Decoder)

训练目标：
  L_recon  = MSE(reconstructed_chunks, original_chunks)   重建损失
  L_contra = InfoNCE(compressed, anchor)                   对比损失（区分度）
  L_total  = L_recon + λ * L_contra

评估：
  - VecBrierLM：压缩后向量在语料库中的区分能力（与C_mean对比）
  - Reconstruction MSE：各chunk向量的重建质量
"""

import pickle, json, numpy as np, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os, time

BASE  = r"C:\Users\hwm20\OneDrive\桌面\Research\embedding"
DIM   = 384     # 向量维度
KMAX  = 7       # 最大chunk数
DEVICE = "cpu"

# ────────────────────────────────────────────────────────────────────
# 1. 数据加载
# ────────────────────────────────────────────────────────────────────
def load_chunk_data(pkl_path, token_method="T_mean"):
    """
    从 results_data.pkl 提取各题的 chunk 向量序列。
    返回：
        chunk_seqs  : List[np.ndarray shape (K, 384)]   每题的chunk向量矩阵
        doc_vecs    : np.ndarray (N, 384)               C_mean参考向量（基线）
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # 用C_concat推算各题chunk数，再拆出各chunk
    concat_key = (token_method, "C_concat")
    mean_key   = (token_method, "C_mean")

    concat_data = d[concat_key]["vectors"]
    mean_data   = np.array(d[mean_key]["vectors"])   # (N, 384)

    chunk_seqs = []
    for raw in concat_data:
        v = np.array(raw, dtype=np.float32)
        K = len(v) // DIM
        if K == 0:
            chunk_seqs.append(np.zeros((1, DIM), dtype=np.float32))
        else:
            chunk_seqs.append(v.reshape(K, DIM))

    return chunk_seqs, mean_data.astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# 2. Dataset
# ────────────────────────────────────────────────────────────────────
class ChunkDataset(Dataset):
    def __init__(self, chunk_seqs):
        self.data = chunk_seqs   # List[np.ndarray (K, 384)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)  # (K, 384)


def collate_pad(batch):
    """pad到同一K，返回 (B, KMAX, 384) + 实际K列表"""
    Ks = [x.shape[0] for x in batch]
    padded = torch.zeros(len(batch), KMAX, DIM)
    for i, (x, k) in enumerate(zip(batch, Ks)):
        padded[i, :k] = x
    return padded, torch.tensor(Ks, dtype=torch.long)


# ────────────────────────────────────────────────────────────────────
# 3. 模型
# ────────────────────────────────────────────────────────────────────
class MultiHeadAttentionPooling(nn.Module):
    """用多头注意力将变长chunk序列压缩为1个向量（类比CALM的summary token）"""
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
        # x: (B, K, dim), mask: (B, K) True=padding
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)   # (B, 1, dim)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        out = self.norm(out.squeeze(1))    # (B, dim)
        return self.proj(out)              # (B, dim)


class ChunkReconstructor(nn.Module):
    """从压缩向量重建K个chunk向量"""
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
        # z: (B, dim) -> (B, KMAX, dim)
        B = z.shape[0]
        out = self.expand(z).reshape(B, self.kmax, -1)
        return self.norm(out)


class CALMAdapter(nn.Module):
    def __init__(self, dim=DIM, heads=4, kmax=KMAX):
        super().__init__()
        self.compressor    = MultiHeadAttentionPooling(dim, heads)
        self.reconstructor = ChunkReconstructor(dim, kmax)

    def forward(self, x, Ks):
        """
        x  : (B, KMAX, dim) padded chunk vectors
        Ks : (B,)           actual K per sample
        returns: compressed (B, dim), reconstructed (B, KMAX, dim)
        """
        # 构建padding mask (True = padding position)
        B, K, D = x.shape
        mask = torch.arange(K, device=x.device).unsqueeze(0) >= Ks.unsqueeze(1)  # (B, K)
        z = self.compressor(x, key_padding_mask=mask)      # (B, dim)
        recon = self.reconstructor(z)                       # (B, KMAX, dim)
        return z, recon


# ────────────────────────────────────────────────────────────────────
# 4. 损失函数
# ────────────────────────────────────────────────────────────────────
def recon_loss(recon, x, Ks):
    """只在有效位置计算重建MSE"""
    B, K, D = x.shape
    loss = 0.0
    cnt  = 0
    for i in range(B):
        k = Ks[i].item()
        loss += ((recon[i, :k] - x[i, :k]) ** 2).mean()
        cnt  += 1
    return loss / max(cnt, 1)


def info_nce_loss(z, temperature=0.07):
    """简化版InfoNCE：batch内其他样本作为负例"""
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0)
    z_norm = nn.functional.normalize(z, dim=-1)
    sim = z_norm @ z_norm.T / temperature          # (B, B)
    labels = torch.arange(B, device=z.device)
    # 对角线为正例
    sim.fill_diagonal_(-1e9)
    # 使用最大相似负例（简化：取每行最大非对角值）
    loss = -torch.log(
        torch.exp(torch.diagonal(
            (z_norm @ z_norm.T) / temperature
        )) / torch.exp(sim).sum(dim=-1).clamp(min=1e-8)
    ).mean()
    return loss.clamp(min=0.0)


# ────────────────────────────────────────────────────────────────────
# 5. VecBrierLM 评估（直接复用 brier_eval.py）
# ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, BASE)
from brier_eval import compute_brierlm as _compute_brierlm

def compute_vecbrierlm(vecs):
    """复用 brier_eval.py 的实现，保持与已有结果一致"""
    score, detail = _compute_brierlm(vecs, temperature=0.1, subsample=len(vecs))
    return score, detail


# ────────────────────────────────────────────────────────────────────
# 6. 主训练流程
# ────────────────────────────────────────────────────────────────────
def train(epochs=60, batch_size=64, lr=3e-4, lam=0.1, patience=10):
    print("="*55)
    print("CALM-inspired Lightweight Adapter Training")
    print("="*55)

    # 加载数据
    pkl_path = os.path.join(BASE, "results_data.pkl")
    chunk_seqs, cmean_vecs = load_chunk_data(pkl_path)
    N = len(chunk_seqs)
    print(f"数据：{N} 条，平均K={np.mean([c.shape[0] for c in chunk_seqs]):.2f}")

    # 划分训练/验证集 (80/20)
    idx = list(range(N))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)
    tr_seqs = [chunk_seqs[i] for i in tr_idx]
    va_seqs = [chunk_seqs[i] for i in va_idx]

    tr_ds = ChunkDataset(tr_seqs)
    va_ds = ChunkDataset(va_seqs)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       collate_fn=collate_pad, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       collate_fn=collate_pad)

    # 模型 & 优化器
    model = CALMAdapter().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{n_params:,}  (~{n_params*4/1024:.0f} KB)")

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, Ks in tr_dl:
            xb, Ks = xb.to(DEVICE), Ks.to(DEVICE)
            z, recon = model(xb, Ks)
            l_r = recon_loss(recon, xb, Ks)
            l_c = info_nce_loss(z)
            loss = l_r + lam * l_c
            opt.zero_grad(); loss.backward(); opt.step()
            tr_losses.append(loss.item())
        sched.step()

        # 验证
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, Ks in va_dl:
                xb, Ks = xb.to(DEVICE), Ks.to(DEVICE)
                z, recon = model(xb, Ks)
                l_r = recon_loss(recon, xb, Ks)
                l_c = info_nce_loss(z)
                va_losses.append((l_r + lam * l_c).item())

        tr_l = np.mean(tr_losses)
        va_l = np.mean(va_losses)

        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"Ep {ep:3d}/{epochs}  train={tr_l:.4f}  val={va_l:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  [{elapsed:.0f}s]")

        if va_l < best_val - 1e-5:
            best_val = va_l
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {ep}")
                break

    # 加载最佳权重
    model.load_state_dict(best_state)
    print(f"\n训练完成，最佳 val_loss = {best_val:.4f}")

    # ── 全量推理，提取压缩向量 ──────────────────────────────────────
    print("\n全量推理中...")
    full_ds = ChunkDataset(chunk_seqs)
    full_dl = DataLoader(full_ds, batch_size=128, shuffle=False,
                         collate_fn=collate_pad)
    model.eval()
    all_z = []
    with torch.no_grad():
        for xb, Ks in full_dl:
            z, _ = model(xb.to(DEVICE), Ks.to(DEVICE))
            all_z.append(z.cpu().numpy())
    compressed_vecs = np.concatenate(all_z, axis=0)   # (N, 384)
    print(f"压缩向量 shape: {compressed_vecs.shape}")

    # ── VecBrierLM 评估 ─────────────────────────────────────────────
    print("\n计算 VecBrierLM...")
    score_adapter, detail_adapter = compute_vecbrierlm(compressed_vecs)
    score_cmean,   detail_cmean   = compute_vecbrierlm(cmean_vecs)
    # 读取已有的最优分数 (T_mean × C_combined = 98.5)
    brier_mat = np.load(os.path.join(BASE, "brier_results.npy"))
    best_existing = brier_mat.max()

    print(f"\n{'='*45}")
    print(f"  VecBrierLM 对比")
    print(f"  C_mean (基线均值压缩)       : {score_cmean:.2f}")
    print(f"  CALM Adapter (注意力压缩)   : {score_adapter:.2f}")
    print(f"  现有最优 (T_mean×C_combined): {best_existing:.2f}")
    print(f"  Adapter 细节: {detail_adapter}")
    print(f"{'='*45}")

    # ── 保存结果 ────────────────────────────────────────────────────
    out = {
        "compressed_vecs":  compressed_vecs,
        "score_adapter":    float(score_adapter),
        "score_cmean":      float(score_cmean),
        "best_existing":    float(best_existing),
        "n_params":         n_params,
        "epochs_trained":   ep,
        "best_val_loss":    float(best_val),
    }
    np.save(os.path.join(BASE, "calm_adapter_vecs.npy"), compressed_vecs)
    import json
    with open(os.path.join(BASE, "calm_adapter_results.json"), "w") as f:
        json.dump({k: v for k, v in out.items()
                   if not isinstance(v, np.ndarray)}, f, indent=2)
    torch.save(best_state, os.path.join(BASE, "calm_adapter_weights.pt"))
    print(f"\n结果已保存：calm_adapter_vecs.npy / calm_adapter_results.json / calm_adapter_weights.pt")
    return out


if __name__ == "__main__":
    results = train(epochs=80, batch_size=64, lr=3e-4, lam=0.05, patience=15)
