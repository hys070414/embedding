"""测试修复后的KP提取"""
import sys, io, json, re, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import dashscope
from dashscope import Generation

dashscope.api_key = 'sk-83da80fde3fd412a97a5668b8e1f8104'

def extract_kps(text, max_retries=3):
    prompt = (
        "List the chemistry knowledge points in this question as a JSON array of short strings (2-5 words each).\n"
        "IMPORTANT: Return ONLY a flat JSON array like [\"term1\", \"term2\", \"term3\"].\n"
        "Do NOT use nested objects. No explanation.\n\n"
        f"Question: {text[:400]}\n\n"
        "Output (JSON array only):"
    )

    def _parse_kps(content):
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r'\[.*?\]', content, re.DOTALL)
            if not m:
                return None
            try:
                parsed = json.loads(m.group())
            except Exception:
                return None

        if not isinstance(parsed, list) or not parsed:
            return None

        result = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                result.append(item.lower().strip())
            elif isinstance(item, dict):
                for field in ('topic', 'concept', 'name', 'term', 'key', 'point'):
                    if field in item and isinstance(item[field], str):
                        result.append(item[field].lower().strip())
                        break
        return result if result else None

    for _ in range(max_retries):
        try:
            resp = Generation.call(
                model="qwen-plus",
                prompt=prompt,
                temperature=0.0,
                max_tokens=200,
                result_format='message'
            )
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content.strip()
                kps = _parse_kps(content)
                if kps:
                    return kps, content
            time.sleep(0.3)
        except Exception:
            time.sleep(0.5)
    return [], ""

# 测试2道题
texts = [
    "Which of the following groups of substances has primarily oxidizing properties?",
    "What is the number of peaks in the 1H NMR spectrum of a molecule with SMILES CC(=O)N1CCC1=O?",
]

results = []
for t in texts:
    kps, raw = extract_kps(t)
    results.append({'text': t[:80], 'kps': kps, 'raw_len': len(raw)})
    print(f"TEXT: {t[:60]}")
    print(f"  KPs ({len(kps)}): {kps}")
    print()

with open('test_kp_fixed_result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("DONE - results saved to test_kp_fixed_result.json")
