"""C-025 independent real-path check (public questions only, C's own scenarios). One isolated instance at a time on
this MacBook (no tunnel, ports 51990 web / 51991 llama, Metal), never the Mac mini. Usage: run_c025.py <worktree> <out>"""
import http.client, json, os, re, signal, socket, subprocess, sys, time
from pathlib import Path

WT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PY = '/Users/eomgyuhyeon/사업/trid/.venv/bin/python'
PORT, SECRET = 51990, Path('/tmp/c025/secret').read_text().strip()
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
           'scope': scope[:900], 'warnings': r.get('warnings'), 'answer': (r.get('answer') or '')[:1200]}
    print(json.dumps({k: rec[k] for k in ('tag', 'http', 'sec', 'status', 'intent', 'pinned', 'continued', 'applied', 'reviewed_rows_annex')},
                     ensure_ascii=False), flush=True)
    return rec


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
Path('/tmp/c025/versions.json').write_text(json.dumps(versions, ensure_ascii=False))
env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRID_DEMO_TUNNEL': '0'}
log = open(OUT.with_suffix('.server.log'), 'w')
proc = subprocess.Popen([PY, 'scripts/shared_demo.py', '--secret-file', '/tmp/c025/secret', '--port', str(PORT),
                         '--versions', '/tmp/c025/versions.json'], cwd=WT, stdout=log, stderr=log, start_new_session=True, env=env)
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
    s1, s2, s3 = visitor(), visitor(), visitor()
    rec['S1'] = [ask(s1, '한국부동산원 여비규정에서 국내 출장 일비는 얼마인가요?', 'S1.1 REB'),
                 ask(s1, '그 규정에서 숙박비는요?', 'S1.2 true follow-up'),
                 ask(s1, '공무원 여비 규정에서는 숙박비가 어떻게 되나요?', 'S1.3 explicit rule switch'),
                 ask(s1, '그럼 식비는?', 'S1.4 follow-up after switch')]
    rec['S2'] = [ask(s2, '그럼 식비는?', 'S2.1 other session no context')]
    rec['S1'] += [ask(s1, '한국부동산원 기준으로 다시, 식비는요?', 'S1.5 explicit switch back'),
                  ask(s1, '지방공무원 여비 규정에서 일비는?', 'S1.6 population switch (rule absent)')]
    rec['S3'] = [ask(s3, '출장비 기준', 'S3.1 general (expect ask-back + others)')]
    rec['S3'].append(pin(s3, 1, 'S3.2 pin 1'))
    rec['S3'].append(ask(s3, '일비는 얼마예요?', 'S3.3 ask under pin'))
    rec['S3'].append(pin(s3, None, 'S3.4 unpin'))
    rec['S3'].append(ask(s3, '일비는 얼마예요?', 'S3.5 ask after unpin'))
    rec['S3'].append(ask(s3, '출장비 기준', 'S3.6 general again'))
    rec['S3'].append(pin(s3, 1, 'S3.7 pin 1 again'))
    rec['S3'].append(clear(s3, 'S3.8 clear'))
    rec['S3'].append(ask(s3, '일비는 얼마예요?', 'S3.9 ask after clear'))
    rec['S4'] = [ask(visitor(), q, t) for q, t in [
        ('상시 출장 공무원의 일비는 어떻게 지급되나요?', 'S4.a condition stated (별표8 expected)'),
        ('공무원 국외 출장 숙박비 기준은?', 'S4.b 국외 stated (별표4 expected)'),
        ('공무원 국내 출장 숙박비는 얼마인가요?', 'S4.c 국내 stated (별표2; 별표4 not expected)'),
        ('공공기관 출장비 지급 기준', 'S4.d A judgment: 공공기관 general'),
        ('그러면 공공기관의 출장비', 'S4.e A judgment item'),
        ('공무원이 아닌 위원의 출장 여비는 어떻게 주나요?', 'S4.f condition stated (별표9 expected)'),
        ('공기업 직원 출장비 기준', 'S4.g class question')]]
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
    print('written', OUT, 'ports_free_after', rec['ports_free_after'])
