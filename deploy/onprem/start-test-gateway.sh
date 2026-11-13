#!/bin/sh
set -eu
# The credential stays out of argv, source control, and Caddy's autosave.
TRID_PROXY_SECRET="$(cat "$1")"
[ -n "$TRID_PROXY_SECRET" ] || exit 1
export TRID_PROXY_SECRET
exec /opt/homebrew/bin/caddy run --config "$2" --adapter caddyfile
