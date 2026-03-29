import json, numpy as np

with open('weighted_full_metrics.json', 'r', encoding='utf-8') as f:
    r = json.load(f)

# 查看所有 keys
keys = list(r.keys())
print("Total keys:", len(keys))
print("Sample keys:", keys[:10])

t_names = ['T_mean','T_max','T_min','T_concat','T_weighted','T_product','T_first','T_last']
metric_names = ['eosk','recall_10','separation','silhouette','brierlm']

# 找 key 分隔符
for t in ['T_mean']:
    for c in ['C_sim_weighted']:
        for sep in ['x', chr(0x00D7), '_', '×']:
            k2 = t + sep + c
            if k2 in r:
                print("Found key format:", repr(k2))
                break

# 打印结果表格
print()
print("方法 | EOSk | Rec@10 | Sep | Sil | BrierLM")
print("-"*80)
for t in t_names:
    for c in ['C_sim_weighted', 'C_idf_weighted']:
        rv = None
        for sep in ['×', 'x', chr(0x00D7)]:
            k = t + sep + c
            if k in r:
                rv = r[k]
                key_used = k
                break
        if rv is None:
            print(f"{t}x{c}: NOT FOUND")
            continue
        print(f"{key_used:44s}  {rv.get('eosk',float('nan')):.4f}  {rv.get('recall_10',float('nan')):.4f}  "
              f"{rv.get('separation',float('nan')):.4f}  {rv.get('silhouette',float('nan')):.4f}  "
              f"{rv.get('brierlm',float('nan')):.2f}")

print()
print("--- 8 Token 均值 ---")
for c in ['C_sim_weighted', 'C_idf_weighted']:
    print(f"{c}:")
    for mn in metric_names:
        v = r.get(f'AVG_{c}_{mn}', float('nan'))
        print(f"  {mn:15s} = {v:.4f}")

print()
print("--- 原始基线均值 ---")
for ref in ['C_mean','C_weighted','C_concat','C_diff','C_combined']:
    print(f"{ref}:")
    for mn in metric_names:
        v = r.get(f'BASELINE_{ref}_{mn}', float('nan'))
        print(f"  {mn:15s} = {v:.4f}")
