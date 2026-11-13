#!/usr/bin/env bash
# Usage: make_self_signed.sh <hostname> [output-dir]   (default output: ./certs)
set -euo pipefail
host="${1:?usage: make_self_signed.sh <hostname> [output-dir]}"
out="${2:-certs}"
mkdir -p "$out"
if [[ "$host" =~ ^[0-9.]+$ ]]; then san="IP:$host"; else san="DNS:$host"; fi
openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
  -subj "/CN=$host" -addext "subjectAltName=$san" \
  -keyout "$out/key.pem" -out "$out/cert.pem"
chmod 644 "$out/key.pem"
echo "wrote $out/cert.pem and $out/key.pem (SAN=$san, 825 days)"
