from evaluation import compute_brierlm
import numpy as np, pickle
np.random.seed(42)
data = pickle.load(open('results_data.pkl', 'rb'))
tests = [('T_mean','C_mean'), ('T_mean','C_concat'),
         ('T_product','C_mean'), ('T_product','C_product'),
         ('T_weighted','C_combined'), ('T_concat','C_diff')]
for k in tests:
    if k in data:
        vecs = data[k]['vectors'][:300]
        bl, det = compute_brierlm(vecs)
        print(f"{k[0]:12s} x {k[1]:12s}: BrierLM={bl:.4f}  full={det['vec_brier_full']:.4f}  k10={det['brier_k10']:.4f}  k50={det['brier_k50']:.4f}")
