import numpy as np

def token_mean(token_embs):
    return np.mean(token_embs, axis=0)

def token_max(token_embs):
    return np.max(token_embs, axis=0)

def token_min(token_embs):
    return np.min(token_embs, axis=0)

def token_concat(token_embs):
    return np.concatenate(token_embs)

def token_weighted_avg(token_embs, weights=None):
    if weights is None:
        weights = np.ones(len(token_embs)) / len(token_embs)
    return np.average(token_embs, axis=0, weights=weights)

def token_elementwise_product(token_embs):
    result = np.ones_like(token_embs[0])
    for emb in token_embs:
        result *= emb
    return result

def token_first(token_embs):
    return token_embs[0]

def token_last(token_embs):
    return token_embs[-1]

def token_attention(token_embs):
    """可学习注意力需要外部训练，此处返回平均作为 placeholder（实际应加载训练好的注意力层）"""
    return np.mean(token_embs, axis=0)

TOKEN_METHODS_NO_ATTN = {
    "T_mean": token_mean,
    "T_max": token_max,
    "T_min": token_min,
    "T_concat": token_concat,
    "T_weighted": lambda x: token_weighted_avg(x, weights=None),
    "T_product": token_elementwise_product,
    "T_first": token_first,
    "T_last": token_last,
}

def get_token_methods_with_attention(attention_layer=None):
    methods = TOKEN_METHODS_NO_ATTN.copy()
    if attention_layer is not None:
        def attn_func(token_embs):
            import torch
            tokens_tensor = torch.tensor(np.stack(token_embs)).unsqueeze(0)  # (1, seq, d)
            with torch.no_grad():
                weighted = attention_layer(tokens_tensor, torch.ones(1, len(token_embs)).bool())
            return weighted.squeeze(0).numpy()
        methods["T_attention"] = attn_func
    return methods