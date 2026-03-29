import numpy as np

def chunk_mean(emb_list):
    return np.mean(emb_list, axis=0)

def chunk_max(emb_list):
    return np.max(emb_list, axis=0)

def chunk_min(emb_list):
    return np.min(emb_list, axis=0)

def chunk_concat(emb_list):
    return np.concatenate(emb_list)

def chunk_weighted_avg(emb_list, weights=None):
    if weights is None:
        weights = np.ones(len(emb_list)) / len(emb_list)
    return np.average(emb_list, axis=0, weights=weights)

def chunk_elementwise_product(emb_list):
    result = np.ones_like(emb_list[0])
    for emb in emb_list:
        result *= emb
    return result

def chunk_diff(emb_list):
    if len(emb_list) > 1:
        return emb_list[0] - emb_list[-1]
    else:
        return emb_list[0]

def chunk_combined(emb_list):
    concat = np.concatenate(emb_list)
    avg = np.mean(emb_list, axis=0)
    prod = np.ones_like(emb_list[0])
    for emb in emb_list:
        prod *= emb
    return np.concatenate([concat, avg, prod])

def chunk_attention(emb_list):
    return np.mean(emb_list, axis=0)  # placeholder

CHUNK_METHODS_NO_ATTN = {
    "C_mean": chunk_mean,
    "C_max": chunk_max,
    "C_min": chunk_min,
    "C_concat": chunk_concat,
    "C_weighted": lambda x: chunk_weighted_avg(x, weights=None),
    "C_product": chunk_elementwise_product,
    "C_diff": chunk_diff,
    "C_combined": chunk_combined,
}

def get_chunk_methods_with_attention(attention_layer=None):
    methods = CHUNK_METHODS_NO_ATTN.copy()
    if attention_layer is not None:
        def attn_func(chunk_embs):
            import torch
            chunk_embs_float32 = [v.astype(np.float32) for v in chunk_embs]
            chunks_tensor = torch.tensor(np.stack(chunk_embs_float32)).unsqueeze(0)
            with torch.no_grad():
                weighted = attention_layer(chunks_tensor, torch.ones(1, len(chunk_embs)).bool())
            return weighted.squeeze(0).numpy()
        methods["C_attention"] = attn_func
    return methods