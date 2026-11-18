#!/usr/bin/env bash
# Install kllm from this checkout: venv + CLI on PATH + (optionally) the background service.
# Idempotent. Does NOT launch any Kaggle GPU (that costs quota).
# Prefer a packaged install:   pipx install .     or     uv tool install .
set -euo pipefail
HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.local/bin"
mkdir -p "$BIN"

if [ ! -x "$HERE/.venv/bin/python" ]; then
  if command -v uv >/dev/null; then uv venv "$HERE/.venv"; else python3 -m venv "$HERE/.venv"; fi
fi
if command -v uv >/dev/null; then
  uv pip install --python "$HERE/.venv/bin/python" -e "$HERE"
else
  "$HERE/.venv/bin/pip" install -e "$HERE"
fi
ln -sf "$HERE/.venv/bin/kllm" "$BIN/kllm"
ln -sf "$HERE/.venv/bin/claude-kllm" "$BIN/claude-kllm"
echo "installed: kllm, claude-kllm -> $BIN   (root: $HERE)"

cat <<EOF2

kllm is installed.

Next (needs only ~/.kaggle/kaggle.json; the first command writes config.yaml for you):
  kllm up                 # start the default model (~8-10 min)
  claude-kllm             # Claude Code on it
  kllm doctor             # verify accounts, rendezvous, relay and backends
  kllm install-service    # optional: keep the relay running at every logon

Admin Panel: http://127.0.0.1:8080/admin
OpenAI base: http://127.0.0.1:8080/v1   (API key: in config.yaml)
EOF2
