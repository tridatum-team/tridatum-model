# Appended after the launcher's FIRST-command independent shutdown timer.
set -Eeuo pipefail
umask 077
export AWS_DEFAULT_REGION=ap-northeast-2 AWS_MAX_ATTEMPTS=1 AWS_RETRY_MODE=standard AWS_PAGER=''
BASE=/var/lib/qwen013
ROOT='/Users/eomgyuhyeon/사업/trid'
if [ "${1:-}" != worker ]; then
  trap 'shutdown -h now || poweroff -f' EXIT
  BOOT_STARTED=$(date +%s)
  mkdir -p "$BASE"
  chmod 0711 "$BASE"
  printf '%s' "$QWEN_CONFIG" | base64 -d | gzip -d > "$BASE/config.json"
  finish() {
    rc=$?
    trap - EXIT TERM INT
    set +e
    deadline=$(timeout -k 1 5 python3 -c 'import json; print(json.load(open("/var/lib/qwen013/config.json"))["worker_deadline_unix"])')
    if ! [[ "$deadline" =~ ^[0-9]+$ ]]; then shutdown -h now || poweroff -f; return; fi
    FINISH_DEADLINE=$((BOOT_STARTED + 115 * 60))
    if [ "$deadline" -lt "$FINISH_DEADLINE" ]; then FINISH_DEADLINE=$deadline; fi
    FINISH_DEADLINE=$((FINISH_DEADLINE - 10))
    bounded() {
      cap=$1; shift
      remaining=$((FINISH_DEADLINE - $(date +%s) - 5))
      if [ "$remaining" -le 0 ]; then return 124; fi
      if [ "$cap" -gt "$remaining" ]; then cap=$remaining; fi
      timeout -k 5 "$cap" "$@"
    }
    bounded 60 python3 - "$BASE" <<'PY'
import json, pathlib, subprocess, sys
base = pathlib.Path(sys.argv[1])
run = json.loads((base / 'config.json').read_text())['run_id']
if run != 'qwen013-cuda-20261111015752':
    raise RuntimeError('Exact cleanup ownership required')
names = ['guard-unittest', 'outside', 'inside', 'deadline', 'descendants', 'environment',
         'packed-compatibility', 'cuda-inspect', 'cuda-generated-smoke', 'cuda-cache-validation', 'model-smoke',
         'full-training', 'native-convert-smoke', 'native-probe-train']
names += [f'cuda-cache-{n:03d}' for n in range(1, 7)]
names += [f'native-convert-{step}' for step in ('000008', '000016', '000023')]
names += [f'native-case-{step}-{group}-{n:03d}' for step in ('000008', '000016', '000023')
          for group in ('domain', 'uncertainty', 'retention') for n in range(1, 13)]
units = ['qwen013-bootstrap.service'] + [f'{run}-{n}.service' for n in names]
path = base / 'owned-unit-cleanup.json'
path.write_text(json.dumps(dict(completed=False, cleanup_state='cleanup_unresolved', units=units)))
stopped = subprocess.run(['systemctl', 'stop', '--', *units], capture_output=True, text=True, timeout=25)
records = []
for unit in units:
    result = subprocess.run(['systemctl', 'show', '--property=LoadState', '--property=ActiveState',
        '--property=MainPID', '--property=ControlPID', '--', unit], capture_output=True, text=True, timeout=3)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    collected = state.get('LoadState') == 'not-found' and state.get('ActiveState') == 'inactive'
    inactive = result.returncode == 0 and state.get('ActiveState') in ('inactive', 'failed') and state.get('MainPID') == state.get('ControlPID') == '0'
    if not (collected or inactive):
        raise RuntimeError(f'Owned unit not proven stopped: {unit}: {state}')
    records.append(dict(unit=unit, state=state, collected=collected))
path.write_text(json.dumps(dict(completed=True, cleanup_state='complete', stop_returncode=stopped.returncode, units=records)))
PY
    if [ "$?" -ne 0 ]; then shutdown -h now || poweroff -f; return; fi
    bounded 50 python3 - "$BASE" "$ROOT" "$rc" <<'PY'
import hashlib, json, pathlib, stat, subprocess, sys, tarfile
base, root, rc = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
c = json.loads((base / 'config.json').read_text())
if (base / 'verified-manifest.json').is_file():
    subprocess.run(['/bin/bash', str(root / 'scripts/aws_eval/qwen_native_setup.sh'), 'export', str(root), str(rc)], check=True, timeout=45)
    sys.exit(0)
included, errors, used = [], ['Authenticated exporter unavailable; partial bootstrap evidence only'], 0
with tarfile.open(base / 'results.tar', 'w', format=tarfile.USTAR_FORMAT) as tar:
    for name in ('bootstrap.log', 'owned-unit-cleanup.json'):
        p = base / name
        try:
            s = p.lstat()
            if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or any(x.is_symlink() for x in p.parents):
                raise ValueError('unsafe evidence')
            if s.st_size > 8 * 1024**2:
                raise ValueError('oversize evidence; not silently truncated')
            item = tarfile.TarInfo(name); item.size = s.st_size; item.mode = 0o600
            with p.open('rb') as stream: tar.addfile(item, stream)
            included.append(dict(path=name, bytes=s.st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            used += s.st_size
        except (OSError, ValueError) as e:
            errors.append(name + ': ' + str(e))
(base / 'completion.json').write_text(json.dumps(dict(run_id=c['run_id'], exit_code=rc,
    included=included, archived_payload_bytes=used, export_complete=False, errors=errors,
    completion_scope='bootstrap_exit_and_export_only', no_model_retry=True)))
(base / 'output-uri').write_text('s3://trid-training-148761639313/eval/aws-eval-006/runs/' + c['run_id'])
PY
    if [ "$?" -eq 0 ] && [ -f "$BASE/output-uri" ]; then
      S3=$(cat "$BASE/output-uri")
      bounded 120 aws s3 cp --only-show-errors --sse AES256 "$BASE/results.tar" "$S3/results.tar"
      bounded 30 aws s3 cp --only-show-errors --sse AES256 "$BASE/completion.json" "$S3/completion.json"
    fi
    shutdown -h now || poweroff -f
  }
  trap finish EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  deadline=$(timeout -k 1 5 python3 -c 'import json; print(json.load(open("/var/lib/qwen013/config.json"))["worker_deadline_unix"])')
  stop_at=$((BOOT_STARTED + 115 * 60))
  if [ "$deadline" -lt "$stop_at" ]; then stop_at=$deadline; fi
  worker_seconds=$((stop_at - $(date +%s) - 5))
  if [ "$worker_seconds" -le 0 ]; then exit 124; fi
  systemd-run --quiet --wait --pipe --unit=qwen013-bootstrap \
    --property=RuntimeMaxSec="$worker_seconds" --property=TimeoutStopSec=5 --property=KillMode=control-group \
    --property=LimitFSIZE=2147483648 \
    /bin/bash /var/lib/cloud/instance/scripts/part-001 worker > "$BASE/bootstrap.log" 2>&1
  exit $?
fi
# No model, cloud, package-download or resolver retries. The authenticated AMI
# provides its NVIDIA driver; never replace it or install a fallback driver.
export DEBIAN_FRONTEND=noninteractive
. /etc/os-release
[ "$ID" = ubuntu ] && [ "$VERSION_ID" = 24.04 ]
[ "$(uname -m)" = x86_64 ]
# Use the authenticated DLAMI's existing CLI; Ubuntu APT is not its installer.
# Fail before package work if that prerequisite is absent or cannot execute.
command -v aws
timeout -k 1 10 aws --version
timeout -k 5 120 apt-get -o Acquire::Retries=0 update
timeout -k 5 300 apt-get -y -o Acquire::Retries=0 install python3.12-venv gcc g++ cmake make util-linux nvidia-modprobe
command -v nvidia-smi
python3.12 -m venv /opt/qwen013-venv
# The wheel URL/hash and all package pins are authenticated user-data admission.
timeout -k 5 600 python3.12 - "$BASE" <<'PY'
import json, pathlib, subprocess, sys
c = json.loads((pathlib.Path(sys.argv[1]) / 'config.json').read_text())
r = c['admission']['runtime']
py = '/opt/qwen013-venv/bin/python'
flags = [py, '-m', 'pip', 'install', '--retries', '0', '--timeout', '30', '--only-binary=:all:', '--disable-pip-version-check', '--no-cache-dir']
subprocess.run(flags + [r['torch_wheel_url'] + '#sha256=' + r['torch_wheel_sha256']], check=True, timeout=420)
subprocess.run(flags + [f'{name}=={version}' for name, version in r['packages'].items()], check=True, timeout=150)
PY
chmod -R a+rX /opt/qwen013-venv
python3.12 - "$BASE" <<'PY'
import json, pathlib, subprocess, sys
base = pathlib.Path(sys.argv[1]); c = json.loads((base / 'config.json').read_text())
subprocess.run(['aws', 's3', 'cp', '--only-show-errors', c['bundle_s3_uri'], str(base / 'inputs.tar')], check=True, timeout=120)
PY
# Authenticate the complete tar and exact allowlist before writing or importing.
python3.12 - "$BASE" "$ROOT" <<'PY'
import hashlib, json, os, pathlib, re, shutil, sys, tarfile
base, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
c = json.loads((base / 'config.json').read_text())
def check(ok, why):
    if not ok: raise RuntimeError(why)
def sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
with (base / 'inputs.tar').open('rb') as stream: check(sha(stream) == c['bundle_sha256'], 'Archive SHA mismatch')
check(not root.exists() and (base / 'inputs.tar').stat().st_size <= 2 * 1024**3, 'Fresh root/input cap required')
run = 'qwen013-cuda-20261111015752'; prefix = 'runs/qwen-specialization-013/' + run
check(c['run_id'] == run and all(c[k] == prefix + '/' + n + '.json' for k, n in
      (('plan','execution-plan'),('allowlist','input-allowlist'),('approval','approval'),('provenance','provenance'))), 'Metadata identity mismatch')
required = {'scripts/aws_eval/qwen_cuda_smoke.py', 'scripts/aws_eval/qwen_cuda_bootstrap.sh',
    'scripts/aws_eval/qwen_cpu_smoke.py', 'scripts/aws_eval/qwen_cpu_pipeline.py',
    'scripts/aws_eval/qwen_cpu_native.py', 'scripts/aws_eval/qwen_cpu_native_harness.py',
    'scripts/aws_eval/qwen_native_setup.sh', 'scripts/qwen_linux_guard.py',
    'scripts/train_qwen_preferences_cuda.py', 'scripts/train_qwen_preferences.py',
    'scripts/train_open_weight_lora.py', 'scripts/q8_gguf_base.py',
    c['plan'], c['allowlist'], c['approval'], c['provenance']}
with tarfile.open(base / 'inputs.tar', 'r:') as tar:
    members = tar.getmembers(); names = [m.name for m in members]
    check(len(names) == len(set(names)) <= 129, 'Duplicate/excess archive members')
    for m in members:
        p = pathlib.PurePosixPath(m.name)
        check(m.isfile() and not m.issym() and not m.islnk() and not m.pax_headers and not m.sparse
              and not p.is_absolute() and '..' not in p.parts and str(p) == m.name
              and not m.name.startswith('./') and m.size <= 2 * 1024**3, 'Unsafe/noncanonical member')
    mm = tar.getmember('remote-bundle-manifest.json')
    check(mm.size <= 100000, 'Manifest too large')
    manifest = json.load(tar.extractfile(mm)); entries = manifest['files']; lookup = {r['path']:r for r in entries}
    check(manifest['required_root'] == str(root) and manifest['run_id'] == run, 'Root/run mismatch')
    check(len(lookup) == len(entries) <= 128 and set(names) == set(lookup) | {'remote-bundle-manifest.json'}, 'Exact manifest membership required')
    check(required <= set(lookup) and sum(r['bytes'] for r in entries) <= 2 * 1024**3, 'Required sources/input cap')
    check(tar.getmember(c['allowlist']).size <= 100000, 'Allowlist too large')
    allow = json.load(tar.extractfile(c['allowlist'])); rows = allow['files']; allowed = {r['path']:r for r in rows}
    check(allow['required_root'] == str(root) and allow['run_id'] == run and len(allowed) == len(rows) == allow['file_count']
          and allow['total_bytes'] == sum(r['bytes'] for r in rows), 'Allowlist identity/totals')
    check(c['allowlist'] not in allowed and set(lookup) == set(allowed) | {c['allowlist']}
          and all(lookup.get(n) == r for n,r in allowed.items()), 'Unapproved input/member')
    for name, row in lookup.items():
        m = tar.getmember(name)
        check('final24' not in name.lower(), 'Held-out final24 input prohibited')
        if name.startswith('runs/open-weight-base/'):
            assets = {'runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc/' + a for a in
                      ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')}
            check(name in assets, 'Unapproved HF base/weight asset')
        check(m.size == row['bytes'] and re.fullmatch('[0-9a-f]{64}', row['sha256'])
              and sha(tar.extractfile(m)) == row['sha256'], 'Member integrity: ' + name)
    check(lookup[c['approval']]['sha256'] == c['approval_sha256'], 'Approval digest mismatch')
    root.mkdir(parents=True, mode=0o755)
    for m in members:
        if m.name == 'remote-bundle-manifest.json': continue
        dest = root / m.name; dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as target, tar.extractfile(m) as source: shutil.copyfileobj(source, target, 1024*1024)
        dest.chmod(0o444)
    for directory, dirs, files in os.walk(root): pathlib.Path(directory).chmod(0o755)
    for parent in root.parents:
        if parent != pathlib.Path('/'): parent.chmod(0o755)
(base / 'verified-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False))
PY
useradd --system --no-create-home --shell /usr/sbin/nologin qwen-smoke
cd "$ROOT"
deadline=$(timeout -k 1 5 python3 -c 'import json; print(json.load(open("/var/lib/qwen013/config.json"))["worker_deadline_unix"])')
stage_seconds=$((deadline - $(date +%s) - 305))
if [ "$stage_seconds" -le 0 ]; then exit 124; fi
timeout -k 5 "$stage_seconds" /opt/qwen013-venv/bin/python -B scripts/aws_eval/qwen_cuda_smoke.py remote-stage
