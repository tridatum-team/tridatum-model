"""C-025 independent real-path check (public questions only, C's own scenarios). One isolated instance at a time on
this MacBook (no tunnel, ports 51990 web / 51991 llama, Metal), never the Mac mini. Usage: run_c025.py <worktree> <out>"""
import http.client, json, os, re, signal, socket, subprocess, sys, time
from pathlib import Path

WT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PY = '/Users/eomgyuhyeon/사업/trid/.venv/bin/python'
PORT, SECRET = 51990, Path('/tmp/c031/secret').read_text().strip()
BASE = {'X-Trid-Proxy': SECRET, 'X-Trid-Host': 'test.tridatum.co'}


def free(p):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', p)) != 0


def call(method, path, data=None, headers=None, timeout=900):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    c.request(method, path, json.dumps(data).encode() if data is not None else None, {**BASE, **(headers or {})})
    r = c.getresponse(); body = r.read(); h = dict(r.getheaders()); c.close()
    return r.status, body, h


def visitor():
    _, body, h = call('GET', '/')
    token = re.search(rb'name="trid-token" content="([^"]+)"', body).group(1).decode()
    return {'Cookie': h['Set-Cookie'].split(';')[0], 'X-Trid-Token': token, 'Origin': 'https://test.tridatum.co'}


def ask(v, q, tag):
    t = time.monotonic()
    status, body, _ = call('POST', '/api/ask', {'question': q, 'scope': 'workspace', 'version': 'trid-assistant-qwen3-1.7b'}, v)
    r = json.loads(body); ws = r.get('workspace') or {}
    applied = ws.get('applied') or {}
    scope = r.get('scope') or ''
    rec = {'tag': tag, 'q': q, 'http': status, 'sec': round(time.monotonic() - t, 1), 'status': r.get('status'),
           'intent': r.get('intent'), 'separate_snapshot': ws.get('separate_snapshot'), 'pinned': ws.get('pinned'),
           'continued': ws.get('continued'), 'selection': ws.get('selection'),
           'applied': [applied.get('institution'), applied.get('title')] if isinstance(applied, dict) else applied,
           'others': [((o.get('institution') or ''), (o.get('title') or '')[:60]) for o in (ws.get('others') or [])],
           'table_condition': r.get('table_condition') or ws.get('table_condition'),
           'applicability': r.get('applicability'), 'population_scope': r.get('population_scope'),
           'reviewed_rows_annex': re.findall(r'별표 (\S+)의 표 행', scope),
           'evidence': [((c.get('source_label') or '')[:90], (c.get('text') or '')[:80]) for c in (r.get('candidates') or [])],
           'reason': r.get('reason'), 'rule_name': r.get('rule_name'), 'tables_set_aside': r.get('tables_set_aside'), 'conditions_carried': r.get('conditions_carried'), 'commentary_notes': r.get('commentary_notes'), 'quotes_n': len(r.get('candidates') or []), 'scope': scope[:900], 'warnings': r.get('warnings'), 'answer': (r.get('answer') or '')[:1200]}
    print(json.dumps({k: rec[k] for k in ('tag', 'http', 'sec', 'status', 'continued', 'conditions_carried', 'applied', 'reviewed_rows_annex')},
                     ensure_ascii=False), flush=True)
    RAW.append({'tag': tag, 'q': q, 'status': r.get('status'), 'commentary': r.get('commentary'), 'raw_output': r.get('raw_output'), 'quotes': [(c.get('source_label'), c.get('text')) for c in (r.get('candidates') or [])], 'answer': r.get('answer')})
    return rec


RAW = []


def pin(v, choice, tag):
    s, body, _ = call('POST', '/api/pin', {'choice': choice}, v)
    out = {'tag': tag, 'pin_choice': choice, 'http': s, 'result': json.loads(body)}
    print(json.dumps(out, ensure_ascii=False)[:300], flush=True)
    return out


def clear(v, tag):
    s, body, _ = call('POST', '/api/clear', {}, v)
    return {'tag': tag, 'clear_http': s, 'result': json.loads(body)}


assert free(51990) and free(51991), 'ports busy'
versions = json.load(open('/Users/eomgyuhyeon/사업/trid/deploy/macmini-site/versions.json'))
for v in versions:
    if v.get('port'):
        v['port'] = 51991
Path('/tmp/c031/versions.json').write_text(json.dumps(versions, ensure_ascii=False))
env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRID_DEMO_TUNNEL': '0', 'LLAMA_ARG_DEVICE': 'none', 'LLAMA_ARG_N_GPU_LAYERS': '0'}
log = open(OUT.with_suffix('.server.log'), 'w')
proc = subprocess.Popen([PY, 'scripts/shared_demo.py', '--secret-file', '/tmp/c031/secret', '--port', str(PORT),
                         '--versions', '/tmp/c031/versions.json'], cwd=WT, stdout=log, stderr=log, start_new_session=True, env=env)
rec = {'worktree': str(WT), 'sha': subprocess.run(['git', '-C', str(WT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
       'pressure_before': subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True).stdout.strip()}
try:
    for _ in range(600):
        if proc.poll() is not None:
            raise SystemExit('server exited; see log')
        try:
            if call('GET', '/', timeout=5)[0] == 200:
                break
        except OSError:
            time.sleep(1)
    a, b, c, d = visitor(), visitor(), visitor(), visitor()
    rec['A'] = [ask(a, '공무원 여비 규정상 국내 출장 일비는 하루 얼마인가요?', 'A1 domestic stated'),
                ask(a, '숙박비 상한은 어떻게 돼요?', 'A2 follow-up without 그럼 (carry 국내)'),
                ask(a, '국외 출장의 경우 숙박비는 어떻게 정하나요?', 'A3 explicit 국외 overrides'),
                ask(a, '식비는요?', 'A4 follow-up after 국외 (carry 국외)')]
    rec['D'] = [ask(d, '숙박비 상한은 어떻게 돼요?', 'D1 other session (no context)')]
    rec['B'] = [ask(b, '공무원 여비 규정에서 국내 출장 식비 기준은?', 'B1 domestic stated'),
                ask(b, '한국부동산원은 어떻게 되나요?', 'B2 institution switch (no carry)'),
                ask(b, '다시 공무원 여비 규정으로 돌아가서 숙박비는?', 'B3 rule switch back (no carry)')]
    rec['C'] = [ask(c, '공무원 여비 규정 국내 출장 숙박비 상한은?', 'C1 domestic stated'), clear(c, 'C2 clear'),
                ask(c, '식비는요?', 'C3 after clear (no carry)')]
finally:
    os.killpg(proc.pid, signal.SIGTERM)
    for _ in range(60):
        if proc.poll() is not None and free(51990) and free(51991):
            break
        time.sleep(1)
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    rec['ports_free_after'] = free(51990) and free(51991)
    rec['pressure_after'] = subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True).stdout.strip()
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    OUT.with_suffix('.raw-commentary.json').write_text(json.dumps(RAW, ensure_ascii=False, indent=1))
    print('written', OUT, 'ports_free_after', rec['ports_free_after'])
