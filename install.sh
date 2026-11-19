#!/usr/bin/env sh
# kllm one-line installer (Linux / macOS):
#   curl -fsSL https://raw.githubusercontent.com/SANTHAN-KUMAR/kllm/main/install.sh | sh
# Installs uv if needed (it brings its own Python), then kllm as an isolated tool: `kllm` and `claude-kllm`.
# Run from a checkout (./install.sh) it installs that checkout instead. Never launches a Kaggle GPU.
# Already have Python? `pipx install kllm-gateway` or `uv tool install kllm-gateway` does the same.
set -eu

say() { printf '%s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  say "kllm: installing uv (https://docs.astral.sh/uv/) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # make uv usable in this shell
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -x "$d/uv" ] && PATH="$d:$PATH" && export PATH
  done
fi

HERE=""
if [ -n "${0:-}" ] && [ -f "$0" ]; then
  HERE="$(cd "$(dirname "$0")" && pwd)"
fi

if [ -n "$HERE" ] && [ -f "$HERE/pyproject.toml" ] && grep -q '^name = "kllm-gateway"' "$HERE/pyproject.toml"; then
  say "kllm: installing from this checkout: $HERE"
  uv tool install --force --editable "$HERE"
else
  SRC="${KLLM_SOURCE:-kllm-gateway}"
  say "kllm: installing $SRC ..."
  uv tool install --force "$SRC" || uv tool install --force "git+https://github.com/SANTHAN-KUMAR/kllm"
fi
[ -n "${KLLM_NO_MODIFY_PATH:-}" ] || uv tool update-shell >/dev/null 2>&1 || true

# desktop icon (click: start kllm and open the panel); skipped on headless machines or with KLLM_NO_SHORTCUT=1
KLLM_BIN="$(uv tool dir --bin 2>/dev/null)/kllm"
[ -x "$KLLM_BIN" ] || KLLM_BIN="$(command -v kllm || true)"
if [ -z "${KLLM_NO_SHORTCUT:-}" ] && [ -n "$KLLM_BIN" ] && { [ "$(uname)" = Darwin ] || [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; }; then
  "$KLLM_BIN" shortcut >/dev/null 2>&1 && ICON=1 || ICON=
fi

cat <<'EOF'

kllm is installed.
EOF
if [ -n "${ICON:-}" ]; then
  echo "Click the kllm icon (desktop or app menu) to start kllm and open the panel."
fi
cat <<'EOF'
Or, in a new terminal:

  kllm             # starts kllm and opens the panel; paste your Kaggle token there on the first run
  claude-kllm      # Claude Code on it
  kllm update      # upgrade when a new version is out
EOF
