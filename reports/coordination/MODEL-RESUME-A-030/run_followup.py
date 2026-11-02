"""MODEL-RESUME-A-030 real-path smoke of the follow-up condition carry-over (branch model-resume-030-followup): the
/api/ask path with the router and Qwen3-1.7B Q8_0 on, public questions only — stated case carried to follow-ups, the
current case wins, explicit switch and new chat release it. One isolated instance on this MacBook (no
tunnel, ports 51990 web / 51991 llama), never the Mac mini. `cpu` runs llama.cpp as the Mac mini does
(LLAMA_ARG_DEVICE=none, LLAMA_ARG_N_GPU_LAYERS=0) and records the llama.cpp log lines that show where the model ran.
usage: run_realpath.py <checkout> <out.json> cpu|metal"""
import http.client
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

WT, OUT, DEVICE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
PY = '/Users/eomgyuhyeon/사업/trid/.venv/bin/python'
PORT, LLAMA = 51990, 51991
RUN = Path('/tmp/a030-realpath')
RUN.mkdir(mode=0o700, exist_ok=True)
(RUN / 'secret').write_text(secrets.token_urlsafe(48))
os.chmod(RUN / 'secret', 0o600)
SECRET = (RUN / 'secret').read_text().strip()
BASE = {'X-Trid-Proxy': SECRET, 'X-Trid-Host': 'test.tridatum.co'}


def free(p):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', p)) != 0


def call(method, path, data=None, headers=None, timeout=900):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    c.request(method, path, json.dumps(data).encode() if data is not None else None, {**BASE, **(headers or {})})
    r = c.getresponse()
    body = r.read()
    h = dict(r.getheaders())
    c.close()
    return r.status, body, h


def visitor():
    _, body, h = call('GET', '/')
    token = re.search(rb'name="trid-token" content="([^"]+)"', body).group(1).decode()
    return {'Cookie': h['Set-Cookie'].split(';')[0], 'X-Trid-Token': token, 'Origin': 'https://test.tridatum.co'}


def ask(v, q, tag):
    t = time.monotonic()
    status, body, _ = call('POST', '/api/ask', {'question': q, 'scope': 'workspace', 'version': 'trid-assistant-qwen3-1.7b'}, v)
    r = json.loads(body)
    ws = r.get('workspace') or {}
    applied = ws.get('applied') or {}
    scope = r.get('scope') or ''
    rec = {'tag': tag, 'q': q, 'http': status, 'sec': round(time.monotonic() - t, 1), 'status': r.get('status'),
           'reason': r.get('reason'), 'intent': r.get('intent'), 'separate_snapshot': ws.get('separate_snapshot'),
           'pinned': ws.get('pinned'), 'continued': ws.get('continued'), 'selection': ws.get('selection'),
           'applied': [applied.get('institution'), applied.get('title')] if isinstance(applied, dict) else applied,
           'others': [((o.get('institution') or ''), (o.get('title') or '')[:60]) for o in (ws.get('others') or [])],
           'table_condition': r.get('table_condition'), 'tables_set_aside': r.get('tables_set_aside'),
           'conditions_carried': r.get('conditions_carried'),
           'applicability': r.get('applicability'), 'population_scope': r.get('population_scope'),
           'reviewed_rows_annex': re.findall(r'별표 (\S+)의 표 행', scope), 'scope': scope[:900],
           'answer': (r.get('answer') or '')[:1200]}
    print(json.dumps({k: rec[k] for k in ('tag', 'http', 'sec', 'status', 'reason', 'pinned', 'continued', 'applied',
                                          'reviewed_rows_annex')}, ensure_ascii=False), flush=True)
    return rec


def pin(v, choice, tag):
    s, body, _ = call('POST', '/api/pin', {'choice': choice}, v)
    return {'tag': tag, 'pin_choice': choice, 'http': s, 'result': json.loads(body)}


def clear(v, tag):
    s, body, _ = call('POST', '/api/clear', {}, v)
    return {'tag': tag, 'clear_http': s, 'result': json.loads(body)}


sysctl = lambda: subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True).stdout.strip()
assert free(PORT) and free(LLAMA), 'ports busy'
versions = json.load(open(WT / 'deploy/macmini-site/versions.json')) if (WT / 'deploy/macmini-site/versions.json').exists() \
    else json.load(open('/Users/eomgyuhyeon/사업/trid-memory-022/deploy/macmini-site/versions.json'))
for v in versions:
    if v.get('port'):
        v['port'] = LLAMA
(RUN / 'versions.json').write_text(json.dumps(versions, ensure_ascii=False))
runtime = RUN / f'runtime-{WT.name}-{DEVICE}'
runtime.mkdir(mode=0o700, exist_ok=True)
env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRID_DEMO_TUNNEL': '0', 'TRID_DEMO_RUNTIME': str(runtime)}
if DEVICE == 'cpu':
    env.update(LLAMA_ARG_DEVICE='none', LLAMA_ARG_N_GPU_LAYERS='0')
log = open(OUT.with_suffix('.server.txt'), 'w')
proc = subprocess.Popen([PY, 'scripts/shared_demo.py', '--secret-file', str(RUN / 'secret'), '--port', str(PORT),
                         '--versions', str(RUN / 'versions.json')], cwd=WT, stdout=log, stderr=log, start_new_session=True, env=env)
rec = {'checkout': str(WT), 'sha': subprocess.run(['git', '-C', str(WT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
       'device': DEVICE, 'env': {k: env.get(k) for k in ('LLAMA_ARG_DEVICE', 'LLAMA_ARG_N_GPU_LAYERS')}, 'pressure_before': sysctl()}
try:
    for _ in range(600):
        if proc.poll() is not None:
            raise SystemExit('server exited; see log')
        try:
            if call('GET', '/', timeout=5)[0] == 200:
                break
        except OSError:
            time.sleep(1)
    llama_log = (runtime / 'llama-trid-assistant-qwen3-1.7b.log').read_text(errors='replace')
    rec['llama_device_lines'] = [l for l in llama_log.splitlines()
                                 if re.search(r'offload|device|Metal|GPU|CPU_Mapped|using', l, re.I)][:20]
    s = visitor()
    rec['F'] = [ask(s, '공무원 여비 규정에서 국내 출장 식비는 얼마인가요?', 'F1 국내 stated'),
                ask(s, '그럼 숙박비는?', 'F2 follow-up: 국내 carried'),
                ask(s, '그럼 일비는?', 'F3 follow-up of a follow-up: 국내 still carried'),
                ask(s, '그럼 국외는?', 'F4 current case wins: 국외, nothing carried'),
                ask(s, '한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', 'F5 explicit institution switch: released'),
                ask(s, '그럼 일비는?', 'F6 follow-up of the institution turn')]
    rec['F'].append(clear(s, 'F7 new chat'))
    rec['F'].append(ask(s, '그럼 숙박비는?', 'F8 after new chat: nothing carried'))
    rec['F'].append(ask(visitor(), '공무원 여비 규정에서는 숙박비가 어떻게 되나요?', 'F9 fresh, case unstated (A-025 P1 unchanged)'))
    rss = subprocess.run(['ps', '-o', 'rss=,command=', '-g', str(proc.pid)], capture_output=True, text=True).stdout
    rec['rss_kb'] = [l.strip()[:160] for l in rss.splitlines() if 'llama-server' in l or 'shared_demo' in l]
finally:
    os.killpg(proc.pid, signal.SIGTERM)
    for _ in range(60):
        if proc.poll() is not None and free(PORT) and free(LLAMA):
            break
        time.sleep(1)
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    rec['ports_free_after'] = free(PORT) and free(LLAMA)
    rec['pressure_after'] = sysctl()
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    print('written', OUT, 'ports_free_after', rec['ports_free_after'])
