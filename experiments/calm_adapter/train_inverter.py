"""
基于 Transformer Decoder 的向量反演（Embedding Inversion）模型。
读取 results_data.pkl，对每种拼接组合训练一个生成模型，用重建质量作为第五指标。
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ---------------------------- 配置 ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 384                     # 你的目标维度（需与投影一致）
TOKENIZER_NAME = 'bert-base-uncased' # 使用 BERT tokenizer（也可换其他）
HIDDEN_DIM = 512
NUM_LAYERS = 3
NHEAD = 8
MAX_LEN = 32
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
SAMPLE_SIZE = 50                    # 每种组合最多使用的样本数（快速训练）
BLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

# 拼接方法列表（与你的实验一致）
t_names = ['T_mean', 'T_max', 'T_min', 'T_concat', 'T_weighted', 'T_product', 'T_first', 'T_last', 'T_attention']
c_names = ['C_mean', 'C_max', 'C_min', 'C_concat', 'C_weighted', 'C_product', 'C_diff', 'C_combined', 'C_attention']

# ---------------------------- 1. 模型定义 ----------------------------
class EmbeddingToTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=512, num_layers=4, nhead=8, max_len=50):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 投影层：将输入向量映射到 Decoder 隐藏层维度
        self.projector = nn.Linear(embed_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # 词嵌入 + 位置编码
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim))

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, cond_vector, tgt_tokens, tgt_mask=None):
        """
        cond_vector: (batch, embed_dim)
        tgt_tokens: (batch, seq_len)
        tgt_mask: (seq_len, seq_len) 因果掩码
        """
        # 1. 将条件向量投影并扩展为 memory (batch, 1, hidden_dim)
        memory = self.projector(cond_vector).unsqueeze(1)          # (B, 1, H)
        memory = self.proj_norm(memory)

        # 2. 目标序列嵌入 + 位置编码
        tgt_emb = self.token_embed(tgt_tokens)                    # (B, seq, H)
        seq_len = tgt_tokens.size(1)
        tgt_emb = tgt_emb + self.pos_embed[:, :seq_len, :]

        # 3. 生成因果掩码（如果未提供）
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_tokens.device)

        # 4. Decoder 前向
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)   # (B, seq, H)
        logits = self.output_layer(output)                          # (B, seq, vocab)
        return logits

    @torch.no_grad()
    def generate(self, cond_vector, tokenizer, max_len=30, beam_size=1):
        """
        贪心解码（beam_size=1）
        """
        self.eval()
        batch_size = cond_vector.size(0)
        device = cond_vector.device

        # 初始 token: [CLS] 或 [SOS]
        start_token = torch.tensor([[tokenizer.cls_token_id]] * batch_size, device=device)  # (B,1)

        for _ in range(max_len):
            # 构建当前输入
            tgt = start_token
            # 计算 logits
            logits = self.forward(cond_vector, tgt)                     # (B, seq_len, vocab)
            last_logits = logits[:, -1, :]                              # (B, vocab)
            next_token = last_logits.argmax(dim=-1, keepdim=True)       # (B,1)

            # 终止条件：如果所有序列都生成 [SEP]，可提前结束（简化）
            start_token = torch.cat([start_token, next_token], dim=1)

        # 解码
        result = tokenizer.decode(start_token[0], skip_special_tokens=True)
        return result

# ---------------------------- 2. 数据准备 ----------------------------
def build_dataset(vectors, texts, tokenizer, max_len=MAX_LEN):
    """
    vectors: list of np.array (each shape (d,))
    texts: list of str
    """
    # 将向量统一投影到 EMBED_DIM（如果维度不同）
    from evaluation import project_to_d   # 假设已有此函数
    proj_vectors = []
    for v in vectors:
        if v.shape[0] != EMBED_DIM:
            # 尝试投影
            try:
                v = project_to_d(v, EMBED_DIM)
            except ValueError:
                # 如果投影失败，跳过该样本
                continue
        proj_vectors.append(v.astype(np.float32))

    # 对文本进行 tokenize
    input_ids = []
    for t in texts:
        tokens = tokenizer.encode(t, add_special_tokens=True, max_length=max_len, truncation=True)
        input_ids.append(tokens)

    # 保证长度一致
    max_seq_len = max(len(ids) for ids in input_ids)
    padded = []
    for ids in input_ids:
        pad_len = max_seq_len - len(ids)
        padded.append(ids + [tokenizer.pad_token_id] * pad_len)

    return torch.tensor(proj_vectors, dtype=torch.float32), torch.tensor(padded, dtype=torch.long)

# ---------------------------- 3. 训练函数 ----------------------------
def train_one_combination(vectors, texts, tokenizer):
    """
    对一个拼接组合进行训练，返回平均 BLEU 分数（作为重建质量指标）
    """
    # 数据准备
    X, Y = build_dataset(vectors, texts, tokenizer, max_len=MAX_LEN)
    if X.size(0) == 0:
        return np.nan

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    vocab_size = tokenizer.vocab_size
    model = EmbeddingToTextModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        max_len=MAX_LEN
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_Y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_Y = batch_Y.to(DEVICE)

            # 构建 decoder 输入（去掉最后一个 token）和目标（去掉第一个 token）
            decoder_input = batch_Y[:, :-1]
            target = batch_Y[:, 1:]

            # 生成因果掩码
            seq_len = decoder_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X, decoder_input, tgt_mask)   # (B, seq, vocab)
            loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))

            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

    # 评估：计算 BLEU 分数
    model.eval()
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    with torch.no_grad():
        for idx in range(min(50, len(X))):   # 最多测50个样本
            vec = X[idx:idx+1].to(DEVICE)
            orig = texts[idx]
            gen = model.generate(vec, tokenizer, max_len=MAX_LEN)
            # 计算 BLEU
            reference = [orig.split()]
            candidate = gen.split()
            if candidate:
                bleu = sentence_bleu(reference, candidate, weights=BLEU_WEIGHTS, smoothing_function=smoothie)
                bleu_scores.append(bleu)
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    return avg_bleu

# ---------------------------- 4. 主程序 ----------------------------
def main():
    # 加载 results_data.pkl
    with open('results_data.pkl', 'rb') as f:
        results_data = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    # 结果矩阵 (9x9)
    bleu_matrix = np.full((len(t_names), len(c_names)), np.nan)

    for ti, t_name in enumerate(tqdm(t_names, desc="Token Methods")):
        for ci, c_name in enumerate(c_names):
            key = (t_name, c_name)
            if key not in results_data:
                continue
            data = results_data[key]
            vectors = data['vectors']
            texts = data['texts']

            if len(vectors) == 0:
                continue

            # 随机抽样（如果样本过多）
            if len(vectors) > SAMPLE_SIZE:
                indices = random.sample(range(len(vectors)), SAMPLE_SIZE)
                vectors = [vectors[i] for i in indices]
                texts = [texts[i] for i in indices]

            print(f"\n训练组合: {key}, 样本数: {len(vectors)}")
            bleu_score = train_one_combination(vectors, texts, tokenizer)
            bleu_matrix[ti, ci] = bleu_score
            print(f"  平均 BLEU: {bleu_score:.4f}")

    # 保存结果矩阵
    np.save('reconstruction_bleu.npy', bleu_matrix)

    # 绘制热力图
    plt.figure(figsize=(12, 10))
    data_plot = np.nan_to_num(bleu_matrix, nan=0.0)
    sns.heatmap(data_plot, annot=True, fmt='.3f',
                xticklabels=c_names, yticklabels=t_names,
                cmap='viridis', cbar_kws={'label': 'BLEU Score'})
    plt.xlabel('Chunk-level Methods')
    plt.ylabel('Token-level Methods')
    plt.title('Reconstruction Quality (BLEU)')
    plt.tight_layout()
    plt.savefig('reconstruction_bleu.png', dpi=150)
    plt.show()
    print("热力图已保存为 reconstruction_bleu.png")

if __name__ == '__main__':
    main()