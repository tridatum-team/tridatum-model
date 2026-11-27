#!/bin/bash
# All explicitly registered profiles: fail-safe before any network or ticket.
/sbin/shutdown -h +5 || exit 1
set -euo pipefail
umask 077
trap '/sbin/shutdown -h now' EXIT
# INJECT_BOUND_VALUES
# Sampling's immutable absolute watchdog survives cancellation of shutdown(8)
# by the authenticated worker. Arm before network; never extend this timer.
if [[ -n "${ABSOLUTE_WORK_DEADLINE:-}" ]]; then
  [[ "$ABSOLUTE_WORK_DEADLINE" =~ ^[0-9]+$ ]]
  (( $(date -u +%s) < ABSOLUTE_WORK_DEADLINE ))
  systemd-run --unit="trid-${RUN}-absolute-stop" \
    --on-calendar="@${ABSOLUTE_WORK_DEADLINE}" --timer-property=AccuracySec=1s \
    /sbin/shutdown -h now
fi
export AWS_DEFAULT_REGION="$REGION" AWS_MAX_ATTEMPTS=1 AWS_RETRY_MODE=standard AWS_PAGER='' AWS_IGNORE_CONFIGURED_ENDPOINT_URLS=true
export PYTHONDONTWRITEBYTECODE=1
BASE="/opt/trid-$RUN"
mkdir -m 700 "$BASE"
cd "$BASE"
# Historical same-AMI venv evidence and base-tool expectations are sealed.
# Actual postboot capability checks precede the pinned helper download; an
# independent termination ticket is also required. Never apt or an unpinned download.
# Gate all bootstrap outbound bytes before the first network operation. This
# fixed 16MiB allocation is charged to the predeclared network reserve. nft is
# an admitted AMI prerequisite; never install it or replace a preexisting table.
nft -f - <<'NFT'
create table inet trid_bootstrap
add chain inet trid_bootstrap output { type filter hook output priority -150; policy accept; }
add rule inet trid_bootstrap output ip daddr 127.0.0.0/8 accept
add rule inet trid_bootstrap output ip6 daddr ::1 accept
add rule inet trid_bootstrap output meta nfproto ipv6 drop
add rule inet trid_bootstrap output quota over 16777216 bytes drop
NFT
# Record ownership only after the atomic quota transaction succeeds. The
# authenticated worker removes this table only after its full gate is active.
printf '%s\n' "$RUN" "$PLAN_SHA" >bootstrap-network-owner
timeout 45 aws s3api get-object --endpoint-url "https://s3.${REGION}.amazonaws.com" --bucket "$BUCKET" --key "${CODE_PREFIX}plan.json" \
  --range bytes=0-2097151 --cli-connect-timeout 5 --cli-read-timeout 30 plan.json >/dev/null
printf '%s  %s\n' "$PLAN_SHA" plan.json | sha256sum --check --status
timeout 150 aws s3api get-object --endpoint-url "https://s3.${REGION}.amazonaws.com" --bucket "$BUCKET" --key "${CODE_PREFIX}bundle.tar" \
  --range "bytes=0-$((BUNDLE_BYTES-1))" --cli-connect-timeout 5 --cli-read-timeout 30 bundle.tar >/dev/null
printf '%s  %s\n' "$BUNDLE_SHA" bundle.tar | sha256sum --check --status
# This small stdlib verifier runs before any bundle import; no tar extraction API.
python3 -I -B - "$BUNDLE_BYTES" "${ABSOLUTE_WORK_DEADLINE:-}" "$BUNDLE_MAX_BYTES" <<'PY'
import datetime, hashlib, json, pathlib, sys, tarfile, time
p = json.loads(pathlib.Path('plan.json').read_text())
b = p['bundle']
assert p.get('schema') in ('trid-grounded-hf-v1', 'trid-ordered-hf-inference-v1',
    'trid-hf14-followup-v1', 'trid-eight-sft-experiment-v1',
    'trid-fresh-eight-training-experiment-v2', 'trid-fresh-eight-cuda-evaluation-v1',
    'trid-public-evidence-gpu-plan-v2', 'trid-public-work-gpu-plan-v1')
# The authenticated controller derives this immutable cap from the complete
# original comparator spec; its pinned worker revalidates that spec after import.
bundle_max_bytes = int(sys.argv[3])
assert bundle_max_bytes in (128 * 1024**2, 256 * 1024**2)
comparator = p.get('schema') == 'trid-public-work-gpu-plan-v1' and 'prior_adapter' in p.get('workload', {})
assert bundle_max_bytes == (256 if comparator else 128) * 1024**2
assert pathlib.Path('bundle.tar').stat().st_size == int(sys.argv[1]) == b['bytes'] <= bundle_max_bytes
if p.get('schema') == 'trid-fresh-eight-training-experiment-v2':
    lease, limits, cost = p['lifetime'], p['limits'], p['cost']
    assert lease == dict(total_seconds=2700, termination_grace_seconds=900, automatic_extension=False)
    assert cost['termination_grace_seconds'] == 900
    assert limits['lifetime_seconds'] == 1800 and limits['reserve_seconds'] == 300
    assert cost['cap_usd'] == 4.11
    assert p['cloud_evaluation'] is False
    assert p['workflow'] == 'cuda_capacity_train_export_only_then_local_mps_evaluation'
    assert b['bytes'] <= 32 * 1024**2 and limits['output_bytes'] == 512 * 1024**2
    assert 0 < cost['export_bytes'] <= limits['output_bytes']
    assert cost['tax_rate'] == 0.10 and cost['contingency_usd'] >= 1
    assert p['network']['egress_byte_cap'] <= 2 * 1024**3
if p.get('schema') == 'trid-fresh-eight-cuda-evaluation-v1':
    lease, limits, cost = p['lifetime'], p['limits'], p['cost']
    sampling = p.get('workload', {}).get('decoder_profile') == 'qwen3_nonthinking_sampling'
    assert lease == dict(total_seconds=14400 if sampling else 19800,
                         termination_grace_seconds=900, automatic_extension=False)
    assert limits['lifetime_seconds'] == (13500 if sampling else 18900) and limits['reserve_seconds'] == 300
    if sampling:
        assert p['absolute_termination_deadline'] == '2026-11-16T23:00:00Z'
        deadline = datetime.datetime.fromisoformat(p['absolute_termination_deadline'].replace('Z', '+00:00')).timestamp()
        assert int(sys.argv[2]) == deadline - 900 and time.time() < deadline - 900
    else:
        assert not sys.argv[2]
    assert cost['termination_grace_seconds'] == 900 and cost['cap_usd'] == 20
    assert limits['output_bytes'] == 512 * 1024**2
    assert 0 < cost['export_bytes'] <= limits['output_bytes']
    assert cost['tax_rate'] == 0.10 and cost['contingency_usd'] >= 1
    assert p['network']['egress_byte_cap'] <= 2 * 1024**3
    assert not {'recipe', 'data', 'aggregate_ledger', 'evaluation'} & set(p)
if p.get('schema') in ('trid-public-evidence-gpu-plan-v2', 'trid-public-work-gpu-plan-v1'):
    lease, limits, cost = p['lifetime'], p['limits'], p['cost']
    assert type(lease['total_seconds']) is int and 1200 < lease['total_seconds'] <= 28800
    assert lease == dict(total_seconds=limits['lifetime_seconds'] + 900,
                         termination_grace_seconds=900, automatic_extension=False)
    assert limits['reserve_seconds'] == 300 and cost['termination_grace_seconds'] == 900
    assert p['absolute_termination_deadline'].endswith('Z')
    deadline = datetime.datetime.fromisoformat(p['absolute_termination_deadline'].replace('Z', '+00:00')).timestamp()
    assert int(sys.argv[2]) == deadline - 900 and time.time() < deadline - 900
    assert limits['output_bytes'] == 512 * 1024**2
    assert 0 < cost['export_bytes'] <= limits['output_bytes']
    assert type(cost['cap_usd']) in (int, float) and 0 < cost['cap_usd'] <= 30
    assert cost['tax_rate'] == 0.10 and cost['contingency_usd'] >= 1
    assert p['network']['egress_byte_cap'] <= 2 * 1024**3
    assert not {'aggregate_ledger', 'evaluation'} & set(p)
assert len(b['files']) <= 4096
files = {r['path']: r for r in b['files']}
assert len(files) == len(b['files']) and sum(r['bytes'] for r in files.values()) <= b['unpacked_bytes'] <= 256 * 1024**2
with tarfile.open('bundle.tar', 'r:') as archive:
    members, seen = [], set()
    for m in archive:
        name = pathlib.PurePosixPath(m.name)
        assert m.isfile() and not m.pax_headers and not m.sparse
        assert not name.is_absolute() and '..' not in name.parts and str(name) == m.name and '\\' not in m.name
        assert m.name in files and m.name not in seen and m.size == files[m.name]['bytes']
        seen.add(m.name)
        with archive.extractfile(m) as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == files[m.name]['sha256']
        members.append(m)
    assert seen == set(files)
    root = pathlib.Path('code'); root.mkdir(mode=0o755)
    for m in members:
        path = root / m.name
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(m) as f, path.open('xb') as out:
            while block := f.read(1024 * 1024): out.write(block)
        path.chmod(0o444)
PY
python3 -I -B "$BASE/code/scripts/aws_eval/grounded_hf_worker.py" bootstrap \
  --plan "$BASE/plan.json" --plan-sha256 "$PLAN_SHA" --base "$BASE"
