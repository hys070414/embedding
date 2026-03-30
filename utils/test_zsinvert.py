import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, torch, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ENCODER_NAME = 'all-MiniLM-L6-v2'
LM_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
BEAM_WIDTH, MAX_STEPS, TOP_K = 8, 20, 15
device = torch.device('cpu')

print('加载编码器...')
encoder = SentenceTransformer(ENCODER_NAME)
print('加载生成模型...')
lm_tok = AutoTokenizer.from_pretrained(LM_NAME)
lm_model = AutoModelForCausalLM.from_pretrained(LM_NAME, torch_dtype=torch.float32).eval()
print('模型加载完成')

orig = 'What is the number of peaks in the 1H NMR spectrum of a molecule with SMILES CC(=O)N1CCC1=O?'
target_vec = encoder.encode([orig], convert_to_numpy=True, normalize_embeddings=True)[0]
target_tensor = torch.tensor(target_vec, dtype=torch.float32).unsqueeze(0).to(device)

prompt = 'write a chemistry question:'
prompt_ids = lm_tok.encode(prompt, return_tensors='pt').to(device)[0].tolist()
beams = [(prompt_ids, 0.0)]

with torch.no_grad():
    for step in range(MAX_STEPS):
        new_beams = []
        for beam_ids, beam_score in beams:
            input_ids = torch.tensor([beam_ids], dtype=torch.long).to(device)
            logits = lm_model(input_ids=input_ids).logits[0, -1, :]
            topk_ids = torch.topk(logits, TOP_K).indices.tolist()
            for tid in topk_ids:
                if tid == lm_tok.eos_token_id: continue
                new_ids = beam_ids + [tid]
                gen_text = lm_tok.decode(new_ids[len(prompt_ids):], skip_special_tokens=True).strip()
                if not gen_text: continue
                gen_emb = encoder.encode([gen_text], convert_to_tensor=True, normalize_embeddings=True).to(device)
                cos_sim = torch.matmul(gen_emb, target_tensor.t()).item()
                new_beams.append((new_ids, cos_sim))
        if not new_beams: break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:BEAM_WIDTH]
        if step % 5 == 4:
            best_text = lm_tok.decode(beams[0][0][len(prompt_ids):], skip_special_tokens=True).strip()
            print(f'step {step+1}: cos={beams[0][1]:.3f} text={repr(best_text[:60])}')

best_text = lm_tok.decode(beams[0][0][len(prompt_ids):], skip_special_tokens=True).strip()
print('原文:', orig)
print('重建:', best_text)
print('cos_sim:', round(beams[0][1], 4))
