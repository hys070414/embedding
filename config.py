# ==================== 配置参数 ====================
import random
import numpy as np
import torch

# 文件路径
JSONL_PATH = "dataset_without_preference.jsonl"   # 题库文件路径
CACHE_FILE = "token_embeddings_cache.npy"         # token embedding缓存文件

# 模型选择
USE_LLM_EXTRACTION = True                     # 是否使用大模型提取知识点
QWEN_API_KEY = "sk-83da80fde3fd412a97a5668b8e1f8104"  
GENERATION_MODEL = "qwen-plus"                     # 用于知识点提取的生成模型

# 本地token embedding模型（使用 sentence-transformers）
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"             # 维度384
EMBED_DIM = 384                                     # 嵌入维度

# 实验参数
MAX_KNOWLEDGE_POINTS =7                         # 每道题最多保留的知识点数（不足则填充）
MAX_TOKENS_PER_POINT = 50                           # 每个知识点最多 token 数（用于 token级拼接）
RETRIEVAL_K =10                                # 检索时的 k 值
SAMPLE_SIZE=None   # None = 全量（1787道题）
TOP_K_EOSK = 2                                       # EOSk 中去除的主成分数
N_SUB_EOSK =5                                   # EOSk 子空间维度

# 训练参数（用于可学习注意力）
ATTENTION_TRAIN_EPOCHS = 5
ATTENTION_BATCH_SIZE = 16
ATTENTION_LR = 1e-3
ATTENTION_HIDDEN_DIM = 128

# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)