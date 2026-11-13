#!/bin/bash
# Fail-safe first: this bootstrap must reach authenticated Python within 5 minutes.
/sbin/shutdown -h +5 || exit 1
set -euo pipefail
umask 077
trap '/sbin/shutdown -h now' EXIT
# INJECT_BOUND_VALUES
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
python3 -I -B - "$BUNDLE_BYTES" <<'PY'
import hashlib, json, pathlib, sys, tarfile
p = json.loads(pathlib.Path('plan.json').read_text())
b = p['bundle']
assert pathlib.Path('bundle.tar').stat().st_size == int(sys.argv[1]) == b['bytes'] <= 128 * 1024**2
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
