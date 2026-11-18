#!/usr/bin/env sh
# kllm one-line installer (Linux / macOS):
#   curl -fsSL https://raw.githubusercontent.com/SANTHAN-KUMAR/kllm/main/install.sh | sh
# Installs uv if needed (it brings its own Python), then kllm as an isolated tool: `kllm` and `claude-kllm`.
# Run from a checkout (./install.sh) it installs that checkout instead. Never launches a Kaggle GPU.
# Already have Python? `pipx install kllm` or `uv tool install kllm` does the same.
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

if [ -n "$HERE" ] && [ -f "$HERE/pyproject.toml" ] && grep -q '^name = "kllm"' "$HERE/pyproject.toml"; then
  say "kllm: installing from this checkout: $HERE"
  uv tool install --force --editable "$HERE"
else
  SRC="${KLLM_SOURCE:-kllm}"
  say "kllm: installing $SRC ..."
  uv tool install --force "$SRC" || uv tool install --force "git+https://github.com/SANTHAN-KUMAR/kllm"
fi
[ -n "${KLLM_NO_MODIFY_PATH:-}" ] || uv tool update-shell >/dev/null 2>&1 || true

cat <<'EOF'

kllm is installed. Open a new terminal, then:

  kllm up          # first run writes the config from ~/.kaggle/kaggle.json, then starts the default model
  claude-kllm      # Claude Code on it
  kllm doctor      # check everything end to end

Only prerequisite: a Kaggle API token saved as ~/.kaggle/kaggle.json
(kaggle.com -> Settings -> API -> Create New Token).
EOF
