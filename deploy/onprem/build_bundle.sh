#!/usr/bin/env bash
# Build the offline bundle on an x86_64 Linux machine with Docker and internet.
# Output: deploy/onprem/dist/trid-onprem-<sha>/ (install folder incl. the image tar).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL=runs/open-weight-base/Qwen3-1.7B-Q8_0.gguf
MODEL_SHA=061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a
TOK=runs/reader-factorial-01-cpt-v1-v2-v4/tokenizer.json
TOK_SHA=73551e65acfa80834c235602900f31dc353ce230bfb69939ffdf44ca26ebc58d
CADDY_IMAGE=caddy:2.8.4

[ -f docs/ONPREM.md ] || { echo "docs/ONPREM.md is missing" >&2; exit 1; }
[ "$(uname -m)" = "x86_64" ] || { echo "build on x86_64 only" >&2; exit 1; }
echo "$MODEL_SHA  $MODEL" | sha256sum -c -
echo "$TOK_SHA  $TOK" | sha256sum -c -

SHA="$(git rev-parse --short HEAD)"
NAME="trid-onprem-$SHA"
OUT="deploy/onprem/dist/$NAME"
rm -rf "$OUT"
mkdir -p "$OUT/certs" "$OUT/data"

# Wheels for the app image (the image build itself installs with --no-index).
rm -rf deploy/onprem/wheels
python3 -m pip download --no-deps --require-hashes --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --abi cp312 \
  -d deploy/onprem/wheels -r deploy/onprem/requirements-app.txt

BASES="python:3.12.8-slim-bookworm debian:12.8-slim $CADDY_IMAGE"
for base in $BASES; do docker pull "$base"; done

docker build -f deploy/onprem/Dockerfile.app -t trid-onprem-app:latest .
docker build -f deploy/onprem/Dockerfile.llm -t trid-onprem-llm:latest .
docker save -o "$OUT/$NAME.tar" trid-onprem-app:latest trid-onprem-llm:latest "$CADDY_IMAGE"

{
  for img in trid-onprem-app:latest trid-onprem-llm:latest "$CADDY_IMAGE"; do
    echo "$img digests=$(docker image inspect --format '{{json .RepoDigests}}' "$img") id=$(docker image inspect --format '{{.Id}}' "$img")"
  done
  for base in $BASES; do
    echo "base $base digests=$(docker image inspect --format '{{json .RepoDigests}}' "$base")"
  done
  echo "llama.cpp commit 8ca4060b7e30e76baaa5670ee1ec1b760b7f8552"
  echo "model sha256 $MODEL_SHA"
  echo "tokenizer sha256 $TOK_SHA"
} > "$OUT/IMAGES.txt"

cp deploy/onprem/compose.yaml deploy/onprem/Caddyfile deploy/onprem/onprem.example.json \
   deploy/onprem/LICENSES.md deploy/onprem/make_self_signed.sh \
   deploy/onprem/release_check.py deploy/onprem/release-questions.json \
   deploy/onprem/release-document.txt docs/ONPREM.md "$OUT/"
chmod +x "$OUT/make_self_signed.sh"

(cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS && sha256sum -c SHA256SUMS)
echo "bundle ready: $OUT"
