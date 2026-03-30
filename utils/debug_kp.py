import dashscope, json, re
from dashscope import Generation

QWEN_API_KEY = 'sk-83da80fde3fd412a97a5668b8e1f8104'
dashscope.api_key = QWEN_API_KEY

text = 'Which of the following groups of substances has primarily oxidizing properties?'

prompt = f'Extract chemistry knowledge points as JSON array. Return only JSON. Question: {text}'
resp = Generation.call(model='qwen-plus', prompt=prompt, temperature=0, max_tokens=150, result_format='message')
print('status:', resp.status_code)
content = resp.output.choices[0].message.content
print('raw content:', repr(content))

# 搜索 JSON
m = re.search(r'\[.*?\]', content, re.DOTALL)
if m:
    print('found json:', m.group())
    try:
        kps = json.loads(m.group())
        print('parsed:', kps)
    except Exception as e:
        print('parse error:', e)
else:
    print('no JSON found')
    # 尝试更宽松的匹配
    m2 = re.search(r'\[.*\]', content, re.DOTALL)
    if m2:
        print('wider match:', m2.group()[:200])
