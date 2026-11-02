"""C-025 independent real-path check (public questions only, C's own scenarios). One isolated instance at a time on
this MacBook (no tunnel, ports 51990 web / 51991 llama, Metal), never the Mac mini. Usage: run_c025.py <worktree> <out>"""
import http.client, json, os, re, signal, socket, subprocess, sys, time
from pathlib import Path

WT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PY = '/Users/eomgyuhyeon/사업/trid/.venv/bin/python'
PORT, SECRET = 51990, Path('/tmp/c028/secret').read_text().strip()
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
           'reason': r.get('reason'), 'rule_name': r.get('rule_name'), 'tables_set_aside': r.get('tables_set_aside'), 'scope': scope[:900], 'warnings': r.get('warnings'), 'answer': (r.get('answer') or '')[:1200]}
    print(json.dumps({k: rec[k] for k in ('tag', 'http', 'sec', 'status', 'reason', 'applied', 'reviewed_rows_annex')},
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
Path('/tmp/c028/versions.json').write_text(json.dumps(versions, ensure_ascii=False))
env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRID_DEMO_TUNNEL': '0', 'LLAMA_ARG_DEVICE': 'none', 'LLAMA_ARG_N_GPU_LAYERS': '0'}
log = open(OUT.with_suffix('.server.log'), 'w')
proc = subprocess.Popen([PY, 'scripts/shared_demo.py', '--secret-file', '/tmp/c028/secret', '--port', str(PORT),
                         '--versions', '/tmp/c028/versions.json'], cwd=WT, stdout=log, stderr=log, start_new_session=True, env=env)
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
    rec['X'] = [ask(visitor(), q, t) for q, t in [
        ('국가공무원 보수 규정에서 호봉 획정은 어떻게 하나요?', 'X1 qualifier + held rule with positive exact scope (central)'),
        ('국가공무원 근속승진 운영규칙에서 근속승진 기간은?', 'X2 qualifier + held rule with unknown scope (must not assert)'),
        ('지방공무원 보수 규정에서 승급은 언제 하나요?', 'X3 local held title; central rule must not intrude')]]
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
