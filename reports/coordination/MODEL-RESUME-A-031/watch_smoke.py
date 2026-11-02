"""MODEL-RESUME-A-031: read-only monitor of the smoke run through `aws_eval_006.py status` (operator profile): appends
each phase change, instance state and result-object listing to <log>; exits at the first phase in --stop-at, when the
instance is terminated, or at --minutes. Never downloads outputs.
usage: AWS_PROFILE=trid-aws-eval-006-operator python watch_smoke.py <run_id> <log> [--stop-at awaiting-input,done,failed]"""
import argparse
import datetime as dt
import json
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('run_id')
p.add_argument('log')
p.add_argument('--stop-at', default='awaiting-input,done,failed')
p.add_argument('--minutes', type=int, default=80)
a = p.parse_args()
assert os.environ.get('AWS_PROFILE') == 'trid-aws-eval-006-operator'
stop = set(a.stop_at.split(','))
last, end = None, time.time() + a.minutes * 60
while time.time() < end:
    r = subprocess.run(['.venv/bin/python', 'scripts/aws_eval/aws_eval_006.py', 'status', a.run_id], capture_output=True, text=True)
    now = f'{dt.datetime.now(dt.timezone.utc):%FT%TZ}'
    try:
        s = json.loads(r.stdout)
    except ValueError:
        s = {'error': (r.stderr or r.stdout).strip()[:300]}
    phase = (s.get('phase') or {}).get('phase')
    states = [i[1] for i in s.get('instances', [])]
    key = (phase, tuple(states), len(s.get('objects', [])))
    if key != last:
        with open(a.log, 'a') as f:
            f.write(json.dumps({'checked': now, **s}, ensure_ascii=False) + '\n')
        print(now, phase, states, len(s.get('objects', [])), flush=True)
        last = key
    if phase in stop or (states and all(x == 'terminated' for x in states)):
        print('stop:', phase, states)
        break
    time.sleep(60)
