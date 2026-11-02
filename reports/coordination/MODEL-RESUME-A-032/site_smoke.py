"""MODEL-RESUME-A-032: one real-path smoke of the deployed site origin through an SSH port forward to the Mac mini
(127.0.0.1:<port> → mini 127.0.0.1:51873), public questions only: version list, deployed answer path, explicit switch,
follow-up condition carry-over, new chat, a removed version id. Prints no secret.
usage: site_smoke.py <port> <proxy secret file> <out.json>"""
import http.client
import json
import re
import sys
import time

PORT, OUT = int(sys.argv[1]), sys.argv[3]
SECRET = open(sys.argv[2]).read().strip()
BASE = {'X-Trid-Proxy': SECRET, 'X-Trid-Host': 'test.tridatum.co'}


def call(method, path, data=None, headers=None):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=300)
    c.request(method, path, json.dumps(data).encode() if data is not None else None, {**BASE, **(headers or {})})
    r = c.getresponse()
    body, h = r.read(), dict(r.getheaders())
    c.close()
    return r.status, body, h


def visitor():
    _, body, h = call('GET', '/')
    token = re.search(rb'name="trid-token" content="([^"]+)"', body).group(1).decode()
    ids = re.findall(rb'<option value="([^"]+)"', body)
    return {'Cookie': h['Set-Cookie'].split(';')[0], 'X-Trid-Token': token, 'Origin': 'https://test.tridatum.co'}, \
        [i.decode() for i in ids]


def ask(v, q, tag, version='trid-assistant-qwen3-1.7b'):
    t = time.monotonic()
    status, body, _ = call('POST', '/api/ask', {'question': q, 'scope': 'workspace', 'version': version}, v)
    r = json.loads(body) if body.strip().startswith(b'{') else {'raw': body[:200].decode(errors='replace')}
    ws = r.get('workspace') or {}
    applied = ws.get('applied') or {}
    rec = {'tag': tag, 'q': q, 'http': status, 'sec': round(time.monotonic() - t, 1), 'status': r.get('status'),
           'reason': r.get('reason'), 'continued': ws.get('continued'),
           'applied': [applied.get('institution'), applied.get('title')] if isinstance(applied, dict) else applied,
           'conditions_carried': r.get('conditions_carried'), 'tables_set_aside': r.get('tables_set_aside'),
           'reviewed_rows_annex': re.findall(r'별표 (\S+)의 표 행', r.get('scope') or ''), 'error': r.get('error'),
           'answer': (r.get('answer') or '')[:300]}
    print(json.dumps({k: rec[k] for k in ('tag', 'http', 'sec', 'status', 'continued', 'applied', 'conditions_carried',
                                          'reviewed_rows_annex', 'error')}, ensure_ascii=False), flush=True)
    return rec


rec = {}
status, _, _ = call('GET', '/healthz')
rec['healthz'] = status
s, rec['version_ids'] = visitor()
print('versions', rec['version_ids'])
rec['turns'] = [
    ask(s, '한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', 'T1 institution'),
    ask(s, '공무원 여비 규정에서 국내 출장 식비는 얼마인가요?', 'T2 explicit rule switch, 국내 stated'),
    ask(s, '그럼 숙박비는?', 'T3 follow-up: 국내 carried'),
]
call('POST', '/api/clear', {}, s)
rec['turns'] += [ask(s, '그럼 숙박비는?', 'T4 after new chat: nothing carried'),
                 ask(s, '그럼 숙박비는?', 'T5 removed version id', version='baseline-qwen3-1.7b')]
open(OUT, 'w').write(json.dumps(rec, ensure_ascii=False, indent=1))
print('written', OUT)
