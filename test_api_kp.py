"""验证API正常工作 + KP提取 + 写入文件"""
import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import dashscope
from dashscope import Generation

dashscope.api_key = 'sk-83da80fde3fd412a97a5668b8e1f8104'

text = 'Which of the following groups of substances has primarily oxidizing properties?'
prompt = f'Extract chemistry knowledge points as JSON array. Return only JSON. Question: {text}'
resp = Generation.call(model='qwen-plus', prompt=prompt, temperature=0, max_tokens=150, result_format='message')

result = {'status': resp.status_code}
if resp.status_code == 200:
    content = resp.output.choices[0].message.content.strip()
    result['content'] = content
    m = re.search(r'\[.*?\]', content, re.DOTALL)
    if m:
        try:
            kps = json.loads(m.group())
            result['kps'] = kps
            result['parsed_ok'] = True
        except:
            result['parsed_ok'] = False
else:
    result['error'] = resp.message

with open('test_api_result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"STATUS: {resp.status_code}")
print(f"KPs: {result.get('kps', 'parse failed')}")
print("DONE")
