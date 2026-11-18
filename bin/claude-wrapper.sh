#!/usr/bin/env bash
# Claude Code Wrapper for Antigravity IDE
# Switches between kllm backend and official Claude models based on ~/.config/kllm/claude_mode

CONFIG_DIR="${HOME}/.config/kllm"
MODE_FILE="${CONFIG_DIR}/claude_mode"

mkdir -p "${CONFIG_DIR}"

MODE="kllm"
if [ -f "${MODE_FILE}" ]; then
    MODE=$(cat "${MODE_FILE}" | tr -d '[:space:]')
fi

if [ "${MODE}" = "kllm" ]; then
    RELAY_PORT=8080
    RELAY_URL="http://127.0.0.1:${RELAY_PORT}"

    # Auto-start relay if not running
    if ! curl -s --max-time 1 "${RELAY_URL}/__relay/health" >/dev/null 2>&1; then
        if command -v kllm >/dev/null 2>&1; then
            kllm ensure-serve >/dev/null 2>&1 || true
        elif [ -n "${KLLM_DIR:-}" ] && [ -d "${KLLM_DIR}" ]; then
            (cd "${KLLM_DIR}" && python3 -m gateway.cli serve >/dev/null 2>&1 &)
            sleep 1
        fi
    fi

    # Detect active model and context window from relay state
    read -r AUTO_MODEL AUTO_CTX < <(python3 -c '
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8080/admin/api/state", timeout=1) as resp:
        d = json.load(resp)
        pool = d.get("pool", [])
        dflt = d.get("default_model", "qwen3.8-27b-vision-abliterated-256k")
        model = pool[0].get("model_key") if pool else dflt
        ctx = [b.get("ctx") for b in pool if b.get("ctx")]
        if not ctx:
            ctx = [m["ctx"] for m in d.get("models", []) if m.get("key") == model and m.get("ctx")]
        print(model, ctx[0] if ctx else 262144)
except Exception:
    print("qwen3.8-27b-vision-abliterated-256k 262144")
' 2>/dev/null || echo "qwen3.8-27b-vision-abliterated-256k 262144")

    CTX="${KLLM_CTX:-$AUTO_CTX}"
    ACTIVE_MODEL="${KLLM_MODEL:-$AUTO_MODEL}"
    # thinking shares max_tokens with the answer; the relay maps effort (high/xhigh/max) to a thinking budget inside it
    MAX_OUT=16384
    if [ "$CTX" -ge 65536 ] 2>/dev/null; then
        MAX_OUT=32768
    fi

    OPUS_MODEL="nanbeige4.3-33b-128k"
    SONNET_MODEL="$ACTIVE_MODEL"
    HAIKU_MODEL="$ACTIVE_MODEL"

    unset ANTHROPIC_API_KEY
    export ANTHROPIC_BASE_URL="${RELAY_URL}"
    export ANTHROPIC_AUTH_TOKEN="sk-kllm-local"
    export ANTHROPIC_MODEL="$ACTIVE_MODEL"

    export ANTHROPIC_DEFAULT_OPUS_MODEL="$OPUS_MODEL"
    export ANTHROPIC_DEFAULT_OPUS_MODEL_NAME="nanbeige4.3-33b-128k"
    export ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION="Nanbeige 4.3 33B 128K Dense Agentic Teacher"

    export ANTHROPIC_DEFAULT_SONNET_MODEL="$SONNET_MODEL"
    export ANTHROPIC_DEFAULT_SONNET_MODEL_NAME="$SONNET_MODEL"
    export ANTHROPIC_DEFAULT_SONNET_MODEL_DESCRIPTION="Active Foundation Model"

    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$HAIKU_MODEL"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME="$HAIKU_MODEL"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL_DESCRIPTION="Active Fast Model"

    export ANTHROPIC_CUSTOM_MODEL_OPTION="$SONNET_MODEL"
    export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="$SONNET_MODEL"
    export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Active KLLM Model"

    export CLAUDE_CODE_SUBAGENT_MODEL="$ACTIVE_MODEL"
    export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"
    export CLAUDE_CODE_AUTO_MODE_SERVER="0"
    # Auto-compact window (same rule as gateway/claude_launcher.py and bin/claude-kllm; see docs/long_run_stability_report.md)
case "${ACTIVE_MODEL}" in
  *256k*) if [ "${CTX}" -gt 220000 ]; then AUTO_COMPACT=150000; else AUTO_COMPACT=$(( CTX - 20000 )); fi ;;
  *)      if [ "${CTX}" -gt 130000 ]; then AUTO_COMPACT=90000; else AUTO_COMPACT=$(( CTX * 84 / 100 )); fi ;;
esac
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${KLLM_AUTO_COMPACT_CTX:-$AUTO_COMPACT}"
# cap one file read (12K tokens) so a single multi-file turn cannot jump past the compaction threshold and the window
export CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS:-12000}"
    export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${MAX_OUT}"
    export API_TIMEOUT_MS="3600000"
    export CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS="3600000"
    export CLAUDE_STREAM_IDLE_TIMEOUT_MS="3600000"
    export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS="3600000"
    export CLAUDE_ENABLE_BYTE_WATCHDOG="0"
    export CLAUDE_ENABLE_STREAM_WATCHDOG="0"
    export CLAUDE_CODE_ATTRIBUTION_HEADER="0"

    # MCP servers lean mode toggle (saves ~13k tokens when off)
    MCP_MODE_FILE="${CONFIG_DIR}/mcp_mode"
    MCP_MODE=$(cat "${MCP_MODE_FILE}" 2>/dev/null || echo "on")
    MCP_ARGS=()
    if [ "${MCP_MODE}" = "off" ] || [ "${KLLM_LEAN_MCP:-0}" = "1" ]; then
        case "${1:-}" in
            mcp|auth|doctor|update|upgrade|install|setup-token|purge|agents|attach|logs|respawn|rm|stop|kill)
                ;;
            *)
                MCP_ARGS=(--strict-mcp-config --mcp-config '{"mcpServers":{}}')
                ;;
        esac
    fi

    # Replace --permission-mode=auto with bypassPermissions so Claude Code does not
    # attempt to call the LLM auto-mode safety classifier on every single Bash tool call,
    # which causes "kllm is temporarily unavailable (timed out)" errors with self-hosted models.
    TARGET_PERM_MODE="${KLLM_PERMISSION_MODE:-bypassPermissions}"
    FINAL_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --permission-mode=auto)
                FINAL_ARGS+=("--permission-mode=${TARGET_PERM_MODE}")
                shift
                ;;
            --permission-mode)
                if [ "${2:-}" = "auto" ]; then
                    FINAL_ARGS+=("--permission-mode" "${TARGET_PERM_MODE}")
                    shift 2
                else
                    FINAL_ARGS+=("$1")
                    shift
                fi
                ;;
            *)
                FINAL_ARGS+=("$1")
                shift
                ;;
        esac
    done
    set -- "${FINAL_ARGS[@]}"
fi

exec "$@" "${MCP_ARGS[@]}"


