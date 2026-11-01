"""MODEL-RESUME-A-026 real-path check: the /api/ask path with the router and Qwen3-1.7B Q8_0 on, public questions only
(C-025's public scenarios S1–S4 and P1–P4, pin/clear/session isolation, and rule names R: abbreviated held, truly not
held, qualified forms; C-026 3). One isolated instance on this MacBook (no
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
RUN = Path('/tmp/a026-realpath')
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
    s1, s2, s3 = visitor(), visitor(), visitor()
    rec['S1'] = [ask(s1, '한국부동산원 여비규정에서 국내 출장 일비는 얼마인가요?', 'S1.1 REB'),
                 ask(s1, '그 규정에서 숙박비는요?', 'S1.2 true follow-up'),
                 ask(s1, '공무원 여비 규정에서는 숙박비가 어떻게 되나요?', 'S1.3 explicit rule switch (국내/국외 unstated)'),
                 ask(s1, '그럼 식비는?', 'S1.4 follow-up after switch')]
    rec['S2'] = [ask(s2, '그럼 식비는?', 'S2.1 other session no context')]
    rec['S1'] += [ask(s1, '한국부동산원 기준으로 다시, 식비는요?', 'S1.5 explicit switch back'),
                  ask(s1, '지방공무원 여비 규정에서 일비는?', 'S1.6 rule not in corpus')]
    rec['S3'] = [ask(s3, '출장비 기준', 'S3.1 general'), pin(s3, 1, 'S3.2 pin 1'),
                 ask(s3, '일비는 얼마예요?', 'S3.3 ask under pin'), pin(s3, None, 'S3.4 unpin'),
                 ask(s3, '일비는 얼마예요?', 'S3.5 ask after unpin'), ask(s3, '출장비 기준', 'S3.6 general again'),
                 pin(s3, 1, 'S3.7 pin 1 again'), clear(s3, 'S3.8 clear'), ask(s3, '일비는 얼마예요?', 'S3.9 ask after clear')]
    rec['S4'] = [ask(visitor(), q, t) for q, t in [
        ('상시 출장 공무원의 일비는 어떻게 지급되나요?', 'S4.a 상시 stated'),
        ('공무원 국외 출장 숙박비 기준은?', 'S4.b 국외 stated'),
        ('공무원 국내 출장 숙박비는 얼마인가요?', 'S4.c 국내 stated'),
        ('공공기관 출장비 지급 기준', 'S4.d class'),
        ('그러면 공공기관의 출장비', 'S4.e class'),
        ('공무원이 아닌 위원의 출장 여비는 어떻게 주나요?', 'S4.f 공무원 아닌 stated'),
        ('공기업 직원 출장비 기준', 'S4.g class')]]
    rec['P'] = [ask(visitor(), q, t) for q, t in [
        ('공무원 여비 규정에서는 숙박비가 어떻게 되나요?', 'P1 국내/국외 unstated'),
        ('공무원 여비 규정에서 국내 출장 숙박비는 얼마인가요?', 'P2 국내'),
        ('공무원 여비 규정에서 국외 출장 숙박비는 어떻게 정하나요?', 'P3 국외'),
        ('공무원 여비 규정에서 공무원이 아닌 사람의 여비는 어떻게 지급하나요?', 'P4 공무원 아닌')]]
    s5 = visitor()
    rec['R'] = [ask(visitor(), q, t) for q, t in [
        ('공무원수당 규정에서 정근수당은 어떻게 주나요?', 'R1 abbreviated held (C-026 V6)'),
        ('공무원 복무규정 연가 일수', 'R2 abbreviated held (V8)'),
        ('지방공무원 수당 규정에서 정근수당은?', 'R3 abbreviated held, scope fits'),
        ('지방공무원 여비 규정에서 일비는?', 'R4 truly not held'),
        ('국가공무원 여비 규정 숙박비', 'R5 qualified form of a held rule')]]
    rec['R'] += [ask(s5, '한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?', 'R6.1 REB'),
                 ask(s5, '지방공무원 여비 규정에서 숙박비는?', 'R6.2 truly not held after an institution turn'),
                 ask(s5, '공무원수당 규정에서 정근수당은?', 'R6.3 abbreviated held after it')]
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
