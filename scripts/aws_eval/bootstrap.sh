# AWS-EVAL-006 instance bootstrap (cloud-init user data body). scripts/aws_eval/aws_eval_006.py prepends "#!/bin/bash"
# and the run variables (RUN_ID, MODE, MAX_MINUTES, BUCKET, PREFIX, CODE_KEY, CODE_SHA, LLAMA_COMMIT, CUDA_ARCH,
# INPUT_SHA, WAIT_INPUT_MINUTES, CERT_SHA). The instance does the whole job by itself and then powers off; it was
# launched with InstanceInitiatedShutdownBehavior=terminate, so every power-off ends it:
#   - first command: an OS timer (shutdown -h +MAX_MINUTES) — the job's own hard limit;
#   - on success or any failure: the EXIT trap uploads the log and phase, then powers off;
#   - independently of this OS, an EventBridge Scheduler one-time schedule terminates the instance after the limit.
# Nothing listens on the network (no inbound rules); access is SSM only. Results go to the private bucket prefix.
# "smoke" runs the first two turns of the first public scenario; "full" waits for C to deliver the private set to this
# host over SSM (never through S3). Both write one checkpoint per finished dialogue, encrypted to C's certificate, and
# sync them to S3 every 30 s; at the end outputs.jsonl and every sealed/ file are encrypted the same way and the
# plaintext (and the delivered set) is shredded before upload. Only status.json, phase, GPU/CUDA/binary facts and
# this log stay readable to the operator.
set -uo pipefail
exec > >(tee -a /var/log/trid-eval.log) 2>&1
umask 077
export AWS_DEFAULT_REGION=ap-northeast-2 HOME=/root DEBIAN_FRONTEND=noninteractive
export AWS_RETRY_MODE=standard AWS_MAX_ATTEMPTS=3  # every upload is also wall-clock bounded (timeout) below
S3="s3://$BUCKET/$PREFIX/runs/$RUN_ID"
WORK=/opt/trid
shutdown -h +"$MAX_MINUTES" "trid AWS-EVAL-006 max runtime"
# Time limits, innermost first: (1) at MAX - DEADLINE_MARGIN the job's own deadline stops the running step so the EXIT
# trap below syncs the finished checkpoints (each upload bounded: 90 + 30 + 30 s < DEADLINE_MARGIN); (2) the OS
# timer powers off at MAX; (3) the EventBridge Scheduler terminates at MAX + 10 min whatever this OS does.
DEADLINE_MARGIN=300
( sleep $((MAX_MINUTES * 60 - DEADLINE_MARGIN)); touch /tmp/trid-deadline; echo "DEADLINE reached"
  pkill -TERM -f chat_blind_runner.py; pkill -TERM -f llama-server; pkill -TERM -f 'cmake --build'; pkill -TERM -f 'curl -fsSL'
  kill -TERM $$ ) &  # bash runs the TERM trap (→ exit → finish) as soon as the current foreground step returns

phase() {
  printf '{"run_id": "%s", "mode": "%s", "phase": "%s", "utc": "%s", "detail": "%s"}\n' \
    "$RUN_ID" "$MODE" "$1" "$(date -u +%FT%TZ)" "$(echo "${2:-}" | tr -d '"\\' | tr '\n' ' ')" > /tmp/phase.json
  timeout 30 aws s3 cp --only-show-errors --sse AES256 /tmp/phase.json "$S3/phase.json" || true
  echo "PHASE $1 ${2:-}"
}
sync_checkpoints() {  # only completed checkpoints (*.cms); partial ones are *.cms.tmp until renamed
  [ -d "$WORK/checkpoints" ] || return 0
  timeout "$1" aws s3 sync --only-show-errors --sse AES256 --exclude '*' --include '*.cms' "$WORK/checkpoints" "$S3/checkpoints/"
}
finish() {
  rc=$?
  trap - EXIT TERM
  sync_checkpoints 90 || echo "final checkpoint sync incomplete (bounded at 90 s)"
  if [ $rc -eq 0 ]; then phase done "exit 0"; elif [ -f /tmp/trid-deadline ]; then phase failed "deadline exit $rc"; else phase failed "exit $rc"; fi
  timeout 30 aws s3 cp --only-show-errors --sse AES256 /var/log/trid-eval.log "$S3/bootstrap.log" || true
  sync
  shutdown -h now
}
trap finish EXIT
trap 'exit 143' TERM
fail() { echo "FAIL: $*"; exit 1; }
fetch() {  # url path sha256
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 --retry-delay 5 -o "$2" "$1" || fail "download $2"
  echo "$3  $2" | sha256sum -c --quiet - || fail "sha256 mismatch $2"
}

phase start "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
mkdir -p "$WORK/out"
timeout 120 aws s3 cp --only-show-errors "s3://$BUCKET/$CODE_KEY" /tmp/code.tar.gz || fail "code download"
echo "$CODE_SHA  /tmp/code.tar.gz" | sha256sum -c --quiet - || fail "code sha256"
tar -xzf /tmp/code.tar.gz -C "$WORK" || fail "code extract"

phase build "llama.cpp $LLAMA_COMMIT CUDA arch $CUDA_ARCH"
command -v cmake >/dev/null || apt-get -y -q install cmake >/dev/null || fail "cmake install"
export PATH=/usr/local/cuda/bin:$PATH
git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp /opt/llama.cpp || fail "llama.cpp clone"
git -C /opt/llama.cpp checkout -q "$LLAMA_COMMIT" || fail "llama.cpp checkout"
[ "$(git -C /opt/llama.cpp rev-parse HEAD)" = "$LLAMA_COMMIT" ] || fail "llama.cpp commit mismatch"
cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > /tmp/cmake.log 2>&1 || { tail -40 /tmp/cmake.log; fail "cmake configure"; }
cmake --build /opt/llama.cpp/build --target llama-server -j "$(nproc)" > /tmp/build.log 2>&1 || { tail -60 /tmp/build.log; fail "build"; }
SERVER=/opt/llama.cpp/build/bin/llama-server
"$SERVER" --version > "$WORK/out/llama-server-version.txt" 2>&1 || true
{ nvcc --version; nvidia-smi; } > "$WORK/out/cuda.txt" 2>&1 || true

phase setup "python tokenizer env and pinned assets"
# The runner's tokenizer stack (transformers 5.21.0) needs Python >= 3.11; the pinned DLAMI ships 3.10 without venv
# support. Smoke attempts 1 and 2 failed here (MODEL-RESUME-A-031: no ensurepip, then no transformers 5.21.0 for 3.10).
# The env is built with a pinned, checksum-verified uv on the same CPython 3.12.13 as the local verified runs.
UV_URL=https://github.com/astral-sh/uv/releases/download/0.11.28/uv-x86_64-unknown-linux-gnu.tar.gz
fetch "$UV_URL" /tmp/uv.tar.gz e490a6464492183c5d4534a5527fb4440f7f2bb2f228162ad7e4afe076dc0224
tar -xzf /tmp/uv.tar.gz -C /tmp || fail "uv extract"
UV=/tmp/uv-x86_64-unknown-linux-gnu/uv
{ UV_PYTHON_INSTALL_DIR=/opt/uv-python timeout 600 "$UV" venv -q --python 3.12.13 /opt/venv && \
  timeout 600 "$UV" pip install -q --python /opt/venv/bin/python transformers==5.21.0 tokenizers==0.23.3 jinja2==3.1.6; } \
  > /tmp/pip.log 2>&1 || { tail -30 /tmp/pip.log; fail "python env"; }
while IFS=$'\t' read -r url path sha; do
  [ -z "$url" ] || fetch "$url" "$WORK/$path" "$sha"
done < "$WORK/assets.tsv"
for pk in P Q; do
  sha=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['expected_sha256'])" \
        "$WORK/runs/chat-blind-005/packages/$pk/package.json")
  [ -f "$WORK/models/$sha.gguf" ] && [ -d "$WORK/tokenizers/$sha" ] || fail "package $pk assets missing"
  ln -s "$WORK/models/$sha.gguf" "$WORK/runs/chat-blind-005/packages/$pk/model.gguf"
  ln -s "$WORK/tokenizers/$sha" "$WORK/runs/chat-blind-005/packages/$pk/tokenizer"
done

echo "$CERT_SHA  $WORK/c-cert.pem" | sha256sum -c --quiet - || fail "C certificate sha256"
# The dialogue set reaches this host only over SSM, encrypted to a key that is generated here and never leaves RAM
# (/run is tmpfs): SSM command history and CloudTrail then hold ciphertext only. Smoke uses the same path with the
# public 2-turn set, so the delivery itself is tested before the private run. Never through S3.
INBOX=/run/trid-input
mkdir -p "$INBOX" && chmod 700 "$INBOX"
openssl req -x509 -newkey rsa:3072 -nodes -keyout "$INBOX/key.pem" -out "$WORK/input-cert.pem" -days 2 \
  -subj "/CN=$RUN_ID set delivery" 2>/dev/null || fail "delivery key"
timeout 60 aws s3 cp --only-show-errors --sse AES256 "$WORK/input-cert.pem" "$S3/input-cert.pem" || fail "publish delivery certificate"
FPR=$(openssl x509 -in "$WORK/input-cert.pem" -noout -fingerprint -sha256 | cut -d= -f2)
if [ "$MODE" = smoke ]; then
  EXPECTED_SHA=$(python3 -c "import json,sys,hashlib; r=json.loads(open(sys.argv[1]).readline()); print(hashlib.sha256((json.dumps({'id': r['id'], 'turns': r['turns'][:2]}, ensure_ascii=False) + chr(10)).encode()).hexdigest())" \
    "$WORK/reports/coordination/CHAT-EVAL-004/public-20.jsonl") || fail "smoke set sha"
else
  EXPECTED_SHA=$INPUT_SHA
fi
phase awaiting-input "deliver the set encrypted to $S3/input-cert.pem (sha256 fingerprint $FPR) as $INBOX/set.cms.b64 + .ready within $WAIT_INPUT_MINUTES min; expected set sha256 $EXPECTED_SHA"
for _ in $(seq 1 $((WAIT_INPUT_MINUTES * 6))); do
  [ -f "$INBOX/set.cms.b64.ready" ] && break
  sleep 10
done
[ -f "$INBOX/set.cms.b64.ready" ] || fail "no delivery within $WAIT_INPUT_MINUTES min"
DIALOGUES=$INBOX/set.jsonl
base64 -d "$INBOX/set.cms.b64" > "$INBOX/set.cms" || fail "delivery base64"
openssl cms -decrypt -binary -inform DER -in "$INBOX/set.cms" -inkey "$INBOX/key.pem" -out "$DIALOGUES" || fail "delivery decrypt"
shred -u "$INBOX/key.pem" "$INBOX/set.cms" "$INBOX/set.cms.b64"
echo "$EXPECTED_SHA  $DIALOGUES" | sha256sum -c --quiet - || fail "delivered set sha256"

BACKEND="llama.cpp-8ca4060-cuda-sm$CUDA_ARCH-$(sha256sum "$SERVER" | cut -c1-12)"
sha256sum "$SERVER" > "$WORK/out/llama-server.sha256"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$WORK/out/gpu.csv" &
GPU_SAMPLER=$!
mkdir -p "$WORK/checkpoints"
( while sleep 30; do sync_checkpoints 60; done ) &
SYNCER=$!
cd "$WORK"
for pk in P Q; do
  phase "run-$pk"
  TRID_RUNNER_REMOTE=1 TRID_LLAMA_SERVER=$SERVER TRID_LLAMA_SERVER_ARGS="-ngl 999" TRID_BACKEND=$BACKEND \
  TRID_RUN_ID=$RUN_ID TRID_CODE_SHA=$CODE_SHA \
    /opt/venv/bin/python scripts/chat_blind_runner.py --package "$pk" --dialogues "$DIALOGUES" --out "$WORK/out/$pk" \
    --checkpoint-dir "$WORK/checkpoints" --checkpoint-cert "$WORK/c-cert.pem" || fail "runner $pk"
done
kill "$GPU_SAMPLER" "$SYNCER" 2>/dev/null
sync_checkpoints 120 || fail "checkpoint upload"

phase encrypt "outputs and every sealed file to C certificate"
for pk in P Q; do
  for f in "$WORK/out/$pk/outputs.jsonl" "$WORK/out/$pk"/sealed/*; do
    [ -f "$f" ] || continue
    openssl cms -encrypt -binary -aes256 -in "$f" -outform DER -out "$f.cms" "$WORK/c-cert.pem" || fail "encrypt $f"
    shred -u "$f"
  done
done
shred -u "$DIALOGUES"
phase upload
timeout 120 aws s3 cp --only-show-errors --recursive --sse AES256 "$WORK/out" "$S3/results/" || fail "upload"
