import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from config import *
from data_loader import load_questions_from_jsonl, prepare_token_embeddings

# ==================== 数据集构建 ====================
class AttentionTrainDataset(Dataset):
    def __init__(self, questions, token_data, max_kp=MAX_KNOWLEDGE_POINTS, max_tokens=MAX_TOKENS_PER_POINT):
        self.questions = questions
        self.token_embs = token_data['token_embs']
        self.kp_token_counts = token_data['kp_token_counts']
        self.max_kp = max_kp
        self.max_tokens = max_tokens
        # 假设使用题目类型作为标签（如果有 type 字段）
        self.labels = [q.get('type', 'unknown') for q in questions]
        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder()
        self.label_ids = self.le.fit_transform(self.labels)
        self.num_classes = len(self.le.classes_)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        qid = idx
        # 构建张量: (max_kp, max_tokens, embed_dim)
        kp_token_tensor = torch.zeros((self.max_kp, self.max_tokens, EMBED_DIM))
        kp_mask = torch.zeros(self.max_kp, dtype=torch.bool)  # 哪些知识点有效
        token_mask = torch.zeros((self.max_kp, self.max_tokens), dtype=torch.bool)  # 每个知识点内哪些token有效

        for kid in range(self.max_kp):
            token_count = self.kp_token_counts.get((qid, kid), 0)
            if token_count == 0:
                continue
            kp_mask[kid] = True
            for tid in range(min(token_count, self.max_tokens)):
                key = (qid, kid, tid)
                if key in self.token_embs:
                    kp_token_tensor[kid, tid] = torch.tensor(self.token_embs[key])
                    token_mask[kid, tid] = True
        return kp_token_tensor, kp_mask, token_mask, torch.tensor(self.label_ids[idx], dtype=torch.long)

# ==================== 可训练注意力层 ====================
class TrainableAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x, mask):
        # x: (batch, seq_len, embed_dim)
        scores = self.score(x).squeeze(-1)  # (batch, seq_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weighted = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return weighted

class TwoLayerAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes, max_kp, max_tokens):
        super().__init__()
        self.token_attention = TrainableAttention(embed_dim)
        self.kp_attention = TrainableAttention(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        self.max_kp = max_kp
        self.max_tokens = max_tokens

    def forward(self, kp_token_tensor, kp_mask, token_mask):
        # kp_token_tensor: (batch, max_kp, max_tokens, embed_dim)
        batch_size = kp_token_tensor.size(0)
        # token 级注意力：得到每个知识点的向量
        kp_vectors = []
        for b in range(batch_size):
            kp_vecs = []
            for kid in range(self.max_kp):
                if not kp_mask[b, kid]:
                    kp_vecs.append(torch.zeros(EMBED_DIM, device=kp_token_tensor.device))
                else:
                    token_embs = kp_token_tensor[b, kid]  # (max_tokens, embed_dim)
                    token_mask_b = token_mask[b, kid]      # (max_tokens,)
                    vec = self.token_attention(token_embs.unsqueeze(0), token_mask_b.unsqueeze(0))
                    kp_vecs.append(vec.squeeze(0))
            kp_vectors.append(torch.stack(kp_vecs))
        kp_vectors = torch.stack(kp_vectors)  # (batch, max_kp, embed_dim)
        # 知识点级注意力
        pooled = self.kp_attention(kp_vectors, kp_mask)  # (batch, embed_dim)
        logits = self.classifier(pooled)
        return logits

# ==================== 训练函数 ====================
def train_attention_model():
    questions = load_questions_from_jsonl(JSONL_PATH, SAMPLE_SIZE)
    token_data = prepare_token_embeddings(questions)
    dataset = AttentionTrainDataset(questions, token_data)
    loader = DataLoader(dataset, batch_size=ATTENTION_BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoLayerAttentionClassifier(EMBED_DIM, ATTENTION_HIDDEN_DIM, dataset.num_classes,
                                        MAX_KNOWLEDGE_POINTS, MAX_TOKENS_PER_POINT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=ATTENTION_LR)

    for epoch in range(1, ATTENTION_TRAIN_EPOCHS+1):
        model.train()
        total_loss = 0
        for kp_token_tensor, kp_mask, token_mask, labels in loader:
            kp_token_tensor = kp_token_tensor.to(device)
            kp_mask = kp_mask.to(device)
            token_mask = token_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(kp_token_tensor, kp_mask, token_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss: {total_loss/len(loader):.4f}")

    # 保存模型（包含两个注意力层）
    torch.save(model.state_dict(), "two_layer_attention.pth")
    print("注意力模型已保存为 two_layer_attention.pth")
    return model.token_attention, model.kp_attention

if __name__ == "__main__":
    train_attention_model()