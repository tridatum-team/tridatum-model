# Appended after launcher registers the 115-minute shutdown as its FIRST command.
# No xtrace, credential export, environment dump, model/launch retries or fallbacks.
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
  printf '%s' "$QWEN_CONFIG" | base64 -d > "$BASE/config.json"
  finish() {
    rc=$?
    trap - EXIT TERM INT
    set +e
    # All finalization shares one absolute remaining-time budget. Independently
    # stop and await every exact owned unit, even if bootstrap already failed.
    # BindsTo/After is defense in depth, not the evidence-collection barrier.
    deadline=$(timeout -k 1 5 python3 -c 'import json; print(json.load(open("/var/lib/qwen013/config.json"))["worker_deadline_unix"])')
    if ! [[ "$deadline" =~ ^[0-9]+$ ]]; then
      shutdown -h now || poweroff -f
      return
    fi
    FINISH_DEADLINE=$((BOOT_STARTED + 115 * 60))
    if [ "$deadline" -lt "$FINISH_DEADLINE" ]; then FINISH_DEADLINE=$deadline; fi
    FINISH_DEADLINE=$((FINISH_DEADLINE - 10))
    bounded() {
      cap=$1
      shift
      remaining=$((FINISH_DEADLINE - $(date +%s) - 5))
      if [ "$remaining" -le 0 ]; then return 124; fi
      if [ "$cap" -gt "$remaining" ]; then cap=$remaining; fi
      timeout -k 5 "$cap" "$@"
    }
    bounded 45 python3 - "$BASE" <<'PY'
import json, pathlib, re, subprocess, sys
base = pathlib.Path(sys.argv[1])
run_id = json.loads((base / 'config.json').read_text())['run_id']
if not re.fullmatch(r'qwen013-[a-z0-9-]{1,40}', run_id):
    raise RuntimeError('Invalid cleanup run identity')
# Fixed dispatch names only: never trust child-writable outputs as unit names.
names = ['guard-unittest', 'outside', 'inside', 'deadline', 'descendants', 'environment',
         'packed-compatibility']
names += ['cache-validation', 'model-smoke']
names += ['full-training', 'native-convert-smoke', 'native-probe-train']
names += [f'native-convert-{step}' for step in ('000008', '000016', '000023')]
names += [f'native-case-{step}-{group}-{n:03d}'
          for step in ('000008', '000016', '000023')
          for group in ('domain', 'uncertainty', 'retention') for n in range(1, 13)]
units = ['qwen013-bootstrap.service'] + [f'{run_id}-{name}.service' for name in names]
(base / 'owned-unit-cleanup.json').write_text(json.dumps({
    'completed': False, 'cleanup_state': 'cleanup_unresolved', 'units': units}))
# Missing/collected units may make stop nonzero; authoritative readback below
# must still prove bootstrap cannot dispatch and every child is fully stopped.
stopped = subprocess.run(['systemctl', 'stop', '--', *units],
                         capture_output=True, text=True, timeout=25)
records = []
for unit in units:
    result = subprocess.run(['systemctl', 'show', '--property=LoadState',
                             '--property=ActiveState', '--property=MainPID',
                             '--property=ControlPID', '--', unit],
                            capture_output=True, text=True, timeout=3)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    collected = state.get('LoadState') == 'not-found' and state.get('ActiveState') == 'inactive'
    inactive = (result.returncode == 0 and state.get('ActiveState') in ('inactive', 'failed')
                and state.get('MainPID') == state.get('ControlPID') == '0')
    if not (collected or inactive):
        raise RuntimeError(f'Owned unit not proven stopped: {unit}: {state}')
    records.append({'unit': unit, 'state': state, 'collected': collected})
(base / 'owned-unit-cleanup.json').write_text(json.dumps({
    'completed': True, 'cleanup_state': 'complete',
    'stop_returncode': stopped.returncode, 'units': records}, indent=2))
PY
    if [ "$?" -ne 0 ]; then
      shutdown -h now || poweroff -f
      return
    fi
    # Logs/results only: exact fresh subtree, regular files, bounded byte total.
    # Missing/unavailable AWS CLI never prevents shutdown.
    bounded 45 python3 - "$BASE" "$ROOT" "$rc" <<'PY'
import json, pathlib, stat, subprocess, sys, tarfile
base, root, rc = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
config = json.loads((base / 'config.json').read_text())
helper = root / 'scripts/aws_eval/qwen_native_setup.sh'
if (base / 'verified-manifest.json').is_file():
    subprocess.run(['/bin/bash', str(helper), 'export', str(root), str(rc)], check=True, timeout=40)
    sys.exit(0)
used = 0
included, skipped = [], []
with tarfile.open(base / 'results.tar', 'w', format=tarfile.USTAR_FORMAT) as archive:
    for name in ('bootstrap.log', 'owned-unit-cleanup.json'):
        path = base / name
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                continue
            if any(p.is_symlink() for p in path.parents):
                skipped.append(name)
                continue
            size = min(metadata.st_size, 8 * 1024**2)
            item = tarfile.TarInfo(name)
            item.size = size
            item.mode = 0o600
            with path.open('rb') as stream:
                archive.addfile(item, stream)
            used += size
            included.append({'path': name, 'bytes': size, 'log_truncated': size != metadata.st_size})
        except (OSError, ValueError):
            skipped.append(name)
(base / 'completion.json').write_text(json.dumps({'run_id': config['run_id'], 'exit_code': rc,
    'included': included, 'skipped': skipped, 'archived_payload_bytes': used,
    'export_complete': False, 'errors': ['Authenticated exporter unavailable; partial bootstrap evidence only'],
    'completion_scope': 'bootstrap_exit_and_export_only', 'no_model_retry': True}, indent=2))
(base / 'output-uri').write_text('s3://trid-training-148761639313/eval/aws-eval-006/runs/' + config['run_id'])
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
  # Absolute worker cutoff includes systemd's five-second stop grace.
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

# Worker runs only under the bounded bootstrap service; no downloaded code yet.
# Approved dependency-transport exception: at most one additional DNF package-
# download retry, within the 300-second outer timeout. Zero means infinite retries.
# No package/version substitution; AWS CLI stays one attempt and pip zero retries.
timeout -k 5 300 dnf -y --setopt=retries=1 install python3.12 python3.12-pip awscli-2 gcc gcc-c++ cmake make util-linux
command -v python3.12
command -v aws
python3.12 -m venv /opt/qwen013-venv
timeout -k 5 420 /opt/qwen013-venv/bin/python -m pip install --retries 0 --timeout 30 \
  --only-binary=:all: --disable-pip-version-check --no-cache-dir \
  torch==2.9.1 transformers==4.57.6 peft==0.17.1 numpy==2.5.4 tokenizers==0.22.2 \
  safetensors==0.8.0 gguf==0.19.0 accelerate==1.15.0 huggingface_hub==0.36.2 \
  psutil==7.2.2 jinja2==3.1.6 sympy==1.14.0
chmod -R a+rX /opt/qwen013-venv
python3.12 - "$BASE" <<'PY'
import json, pathlib, subprocess, sys
base = pathlib.Path(sys.argv[1])
c = json.loads((base / 'config.json').read_text())
subprocess.run(['aws', 's3', 'cp', '--only-show-errors', c['bundle_s3_uri'], str(base / 'inputs.tar')],
               check=True, timeout=120)
PY
# Authenticate the whole archive BEFORE trusting its manifest, then verify every
# member and the complete set BEFORE any extracted input/source can be used.
python3.12 - "$BASE" "$ROOT" <<'PY'
import hashlib, json, os, pathlib, re, shutil, sys, tarfile
base, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
config = json.loads((base / 'config.json').read_text())
def check(ok, why):
    if not ok:
        raise RuntimeError(why)
def sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()
with (base / 'inputs.tar').open('rb') as stream:
    check(sha(stream) == config['bundle_sha256'], 'Archive SHA mismatch')
check(not root.exists(), 'Remote project root must be absent/fresh')
run = 'qwen013-traced-20261109203337'
run_path = 'runs/qwen-specialization-013/' + run
check(config['run_id'] == run and config['plan'] == run_path + '/execution-plan.json'
      and config['allowlist'] == run_path + '/input-allowlist.json'
      and config['provenance'] == run_path + '/provenance.json'
      and config['approval'] == run_path + '/approval.json', 'Exact metadata identity mismatch')
check((base / 'inputs.tar').stat().st_size <= 2 * 1024**3, 'Archive size cap exceeded')
required = {'scripts/qwen_linux_guard.py', 'tests/test_qwen_linux_guard.py',
    'scripts/aws_eval/qwen_cpu_smoke.py', 'scripts/aws_eval/qwen_cpu_bootstrap.sh',
    'scripts/aws_eval/qwen_native_setup.sh',
    'scripts/train_qwen_preferences.py', 'scripts/train_open_weight_lora.py',
    'scripts/q8_gguf_base.py', 'tests/test_train_qwen_preferences.py',
    'tests/test_train_open_weight_lora.py', 'tests/test_qwen_cpu_smoke.py',
    'scripts/trace_qwen_preferences.py', 'tests/test_trace_qwen_preferences.py',
    config['plan'], config['allowlist'], config['approval'], config['provenance']}
with tarfile.open(base / 'inputs.tar', 'r:') as archive:
    members = archive.getmembers()
    names = [m.name for m in members]
    check(len(names) == len(set(names)) and len(names) <= 129, 'Duplicate/excess archive members')
    for m in members:
        p = pathlib.PurePosixPath(m.name)
        check(m.isfile() and not m.issym() and not m.islnk() and not m.pax_headers
              and not m.sparse and not p.is_absolute() and '..' not in p.parts
              and str(p) == m.name and not m.name.startswith('./')
              and m.size <= 2 * 1024 ** 3, 'Unsafe/noncanonical member')
    check('remote-bundle-manifest.json' in names, 'Missing bundle manifest')
    mm = archive.getmember('remote-bundle-manifest.json')
    check(mm.size <= 100000, 'Oversize manifest')
    manifest = json.load(archive.extractfile(mm))
    check(manifest['required_root'] == str(root) and manifest['run_id'] == run, 'Lexical root/run mismatch')
    entries = manifest['files']
    lookup = {r['path']: r for r in entries}
    check(len(lookup) == len(entries) <= 128, 'Duplicate/excess input manifest rows')
    check(set(names) == set(lookup) | {'remote-bundle-manifest.json'}, 'Extra/missing archive member')
    check(required <= set(lookup), 'Required runtime verification source missing')
    check(sum(r['bytes'] for r in entries) <= 2 * 1024 ** 3, 'Input size cap exceeded')
    check(archive.getmember(config['allowlist']).size <= 100000, 'Oversize allowlist')
    allowlist = json.load(archive.extractfile(config['allowlist']))
    rows = allowlist['files']
    allowed = {r['path']: r for r in rows}
    check(allowlist['required_root'] == str(root) and allowlist['run_id'] == run
          and len(allowed) == len(rows) == allowlist['file_count']
          and allowlist['total_bytes'] == sum(r['bytes'] for r in rows), 'Bad current allowlist')
    check(config['allowlist'] not in allowed and set(lookup) == set(allowed) | {config['allowlist']},
          'Bundle contains an unapproved input (no arbitrary repository tree/HF weights)')
    for name, row in allowed.items():
        check(lookup.get(name) == row, 'Current input allowlist binding changed')
    for name, row in lookup.items():
        m = archive.getmember(name)
        if name.startswith('runs/open-weight-base/'):
            allowed_model_assets = {
                'runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc/' + asset
                for asset in ('config.json', 'generation_config.json', 'tokenizer.json',
                              'tokenizer_config.json', 'vocab.json', 'merges.txt')}
            check(name in allowed_model_assets, 'HF weight/unapproved base asset prohibited')
        check(m.size == row['bytes'] and re.fullmatch('[0-9a-f]{64}', row['sha256'])
              and sha(archive.extractfile(m)) == row['sha256'], 'Member integrity mismatch: ' + name)
    # No tar extraction APIs: exclusive regular-file writes into fresh root.
    root.mkdir(parents=True, mode=0o755)
    for m in members:
        if m.name == 'remote-bundle-manifest.json':
            continue
        dest = root / m.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as target, archive.extractfile(m) as source:
            shutil.copyfileobj(source, target, 1024 * 1024)
        dest.chmod(0o444)
    for directory, dirs, files in os.walk(root):
        pathlib.Path(directory).chmod(0o755)
    # Parent path traversal permissions; inputs remain root-owned read-only.
    for parent in root.parents:
        if parent != pathlib.Path('/'):
            parent.chmod(0o755)
(base / 'verified-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False))
PY
deadline=$(timeout -k 1 5 python3 -c 'import json; print(json.load(open("/var/lib/qwen013/config.json"))["worker_deadline_unix"])')
native_seconds=$((deadline - $(date +%s) - 305))
if [ "$native_seconds" -gt 1195 ]; then native_seconds=1195; fi
if [ "$native_seconds" -le 0 ]; then exit 124; fi
timeout -k 5 "$native_seconds" /bin/bash "$ROOT/scripts/aws_eval/qwen_native_setup.sh" setup "$ROOT"
useradd --system --no-create-home --shell /sbin/nologin qwen-smoke
cd "$ROOT"
# Orchestrator is root only to create constrained nonroot systemd child units.
# It never imports model engines itself and never passes AWS credentials to them.
stage_seconds=$((deadline - $(date +%s) - 305))
if [ "$stage_seconds" -le 0 ]; then exit 124; fi
timeout -k 5 "$stage_seconds" /opt/qwen013-venv/bin/python -B scripts/aws_eval/qwen_cpu_smoke.py remote-stage
