"""MODEL-RESUME-A-024 reproduction: real /api/ask conversation path (router + Qwen3-1.7B Q8_0 on) against an isolated,
tunnel-less instance of a checkout — never the Mac mini service. Start the instance first, e.g.
  python scripts/shared_demo.py --secret-file S --port 51990 --versions V   (V: deploy/macmini-site/versions.json, port changed)
usage: reproduce_conversation.py <port> <secret file> <out.json>"""
import http.client
import json
import re
import sys
import time

PORT = int(sys.argv[1]); SECRET = open(sys.argv[2]).read().strip(); OUT = sys.argv[3]
BASE = {'X-Trid-Proxy': SECRET, 'X-Trid-Host': 'test.tridatum.co'}


def call(method, path, data=None, headers=None):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=600)
    c.request(method, path, json.dumps(data).encode() if data is not None else None, {**BASE, **(headers or {})})
    r = c.getresponse(); body = r.read(); h = dict(r.getheaders()); c.close()
    return r.status, body, h


def visitor():
    _, body, h = call('GET', '/')
    token = re.search(rb'name="trid-token" content="([^"]+)"', body).group(1).decode()
    return {'Cookie': h['Set-Cookie'].split(';')[0], 'X-Trid-Token': token, 'Origin': 'https://test.tridatum.co'}


def ask(v, q):
    t = time.monotonic()
    status, body, _ = call('POST', '/api/ask', {'question': q, 'scope': 'workspace', 'version': 'trid-assistant-qwen3-1.7b'}, v)
    r = json.loads(body)
    ws = r.get('workspace') or {}
    applied = ws.get('applied') or {}
    rec = {'q': q, 'http': status, 'seconds': round(time.monotonic() - t, 1), 'status': r.get('status'), 'intent': r.get('intent'),
           'separate_snapshot': ws.get('separate_snapshot'), 'file': r.get('file'),
           'applied': (applied.get('institution'), applied.get('title')) if isinstance(applied, dict) else applied,
           'table_condition': ws.get('table_condition') or r.get('table_condition'),
           'applicability': r.get('applicability'), 'population_scope': r.get('population_scope'),
           'continued': ws.get('continued'), 'scope': r.get('scope'), 'warnings': r.get('warnings'),
           'quotes': len(r.get('quotes') or r.get('verified_quotes') or []), 'answer': (r.get('answer') or '')[:600]}
    print(json.dumps({k: rec[k] for k in ('q', 'http', 'seconds', 'status', 'intent', 'separate_snapshot', 'file', 'applied')},
                     ensure_ascii=False), flush=True)
    return rec, r


log = {}
a, b = visitor(), visitor()
log['A'] = []
for q in ['한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', '그럼 일비는?', '공무원 여비 규정에서 국내 출장 식비는 어떻게 지급하나요?',
          '그럼 숙박비는?', '지방공무원 여비 규정에서 숙박비는?']:
    rec, raw = ask(a, q); log['A'].append(rec)
log['A_raw_keys'] = sorted(raw)
log['B_isolated'] = [ask(b, '그럼 일비는?')[0]]  # another visitor's session: no context from A
call('POST', '/api/clear', {}, a)
log['A_after_clear'] = [ask(a, '그럼 일비는?')[0]]
log['C'] = [ask(visitor(), q)[0] for q in ['공공기관 출장비 지급 기준', '공공기관 여비', '출장비']]
open(OUT, 'w').write(json.dumps(log, ensure_ascii=False, indent=1, default=str))
print('written', OUT)
