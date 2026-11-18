"""`claude-kllm`: run Claude Code against the Kaggle-hosted model via the kllm relay.

Python port of bin/claude-kllm (works on Linux, macOS and Windows). Your normal `claude` is
untouched: the settings below only apply to this process.
  usage: claude-kllm [any claude args]      (first run starts the relay itself; the GPU
      backend then starts on the first request and stops after 25 min idle)
Env: KLLM_RELAY (default http://127.0.0.1:<relay_port>), KLLM_MODEL, KLLM_CTX, KLLM_MAX_OUTPUT,
     KLLM_API_KEY (only for a remote relay), LEAN=0 (keep MCP servers / all tools).
"""
import json, os, shutil, subprocess, sys, urllib.request

LEAN_TOOLS = "Bash,Read,Edit,Write,WebFetch,NotebookEdit,Glob,Grep"


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def parse_model_arg(argv):
    """Extract --model <name> or --model=<name> from argv.
    Returns (model_name_or_None, remaining_argv)."""
    model = None
    remaining = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            i += 1
        else:
            remaining.append(arg)
            i += 1
    return model, remaining


def pick_model(state, want_env=None):
    """-> (model key, context window) from the relay's /admin/api/state, like the bash version."""
    pool = state.get("pool", [])
    default = state.get("default_model")
    if want_env:
        want = want_env
    elif any(b.get("model_key") == default for b in pool) or (not pool and default):
        # default running, or nothing running yet (autoscale will start the default on the
        # first request): ask for it by name so the window below is the right one
        want = default
    else:
        want = "kllm"
    ctx = [b.get("ctx") for b in pool if b.get("ctx") and (want == "kllm" or b.get("model_key") == want)]
    if not ctx:
        # not running yet: size for the model that will be started (its registry ctx)
        reg = {m.get("key"): m.get("ctx") for m in state.get("models", [])}
        if want in reg and reg[want]:
            ctx = [reg[want]]
        else:
            try:
                from . import common
                m_all = common.load_models()
                if want in m_all and m_all[want].get("ctx"):
                    ctx = [m_all[want]["ctx"]]
            except Exception:
                pass
    return want, (min(ctx) if ctx else 32768)


def build_env(relay, model, ctx, env=None, auto_compact=None):
    """Environment variables for the claude process."""
    e = dict(os.environ if env is None else env)
    max_out = e.get("KLLM_MAX_OUTPUT") or str(32768 if ctx >= 65536 else 16384)   # room for effort-mapped thinking (relay) + answer
    e.pop("ANTHROPIC_API_KEY", None)
    active_m = model or "qwen3.8-27b"
    e.update({
        "ANTHROPIC_BASE_URL": relay,
        "ANTHROPIC_AUTH_TOKEN": e.get("KLLM_API_KEY") or "sk-kllm-local",   # local relay: any value
        "ANTHROPIC_MODEL": active_m,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "nanbeige4.3-33b-128k",
        "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME": "nanbeige4.3-33b-128k",
        "ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION": "Nanbeige 4.3 33B 128K Dense Agentic Teacher",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": active_m,
        "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME": active_m,
        "ANTHROPIC_DEFAULT_SONNET_MODEL_DESCRIPTION": "Active KLLM Foundation Model",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": active_m,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME": active_m,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL_DESCRIPTION": "Active KLLM Fast Model",
        "ANTHROPIC_CUSTOM_MODEL_OPTION": active_m,
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": active_m,
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": active_m,
        "CLAUDE_CODE_SUBAGENT_MODEL": active_m,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # opt out of the classifier-billing auto-mode notice a gateway always triggers
        "CLAUDE_CODE_AUTO_MODE_SERVER": "0",
        "CLAUDE_CODE_AUTO_MODE_CLASSIFY_EDITS": "0",
        # context window = what the live backend serves, so Claude Code neither overflows nor compacts early
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(auto_compact if auto_compact else ctx),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": max_out,
        # bound how much one file read adds to the context: a single turn reading many whole files could otherwise jump
        # past the compaction threshold AND the backend window in one step (seen 2026-11-01: 6 files -> 130K tokens)
        "CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS": e.get("CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS") or "12000",
        # remote prefill/thinking can take minutes; don't give up early
        "API_TIMEOUT_MS": "3600000",
        "CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS": "3600000",
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "3600000",
        "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "3600000",
        # Disable Claude Code's internal 20s stream/byte watchdog so long remote prefill / thinking
        # doesn't trigger false "Waiting for API response · will retry in 29m · check your network"
        "CLAUDE_ENABLE_BYTE_WATCHDOG": "0",
        "CLAUDE_ENABLE_STREAM_WATCHDOG": "0",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    })
    return e


# NO modelPicker / behavesAs here. `behavesAs: claude-sonnet-4-6` makes Claude Code use Sonnet's 200K catalog window
# and silently ignore CLAUDE_CODE_MAX_CONTEXT_TOKENS (and AUTO_COMPACT_WINDOW / autoCompactWindow / PCT_OVERRIDE):
# verified 2026-11-02 with a mock server: with behavesAs no compaction at 100K-170K, without it compaction at 100K.
# A 128K backend then fills up (seen: 126K prompt + 4.8K output = 131072), llama-server stops with finish_reason
# "length", and Claude Code reports it as "response exceeded the 16384 output token maximum".
# Picker rows without behavesAs are hidden by Claude Code, so the picker is dropped; --model and the
# ANTHROPIC_DEFAULT_*_MODEL aliases still select kllm models.


def build_args(model, extra, lean=True):
    """claude argv (after the executable). The attribution header is dropped: it changes every
    request and would defeat the backend's prompt-prefix cache."""
    # Convert --permission-mode=auto / --permission-mode auto to bypassPermissions,
    # and pass --allow-dangerously-skip-permissions so Claude Code never blocks on classifier timeouts.
    filtered_extra = []
    i = 0
    extra_list = list(extra)
    while i < len(extra_list):
        arg = extra_list[i]
        if arg == "--permission-mode" and i + 1 < len(extra_list) and extra_list[i + 1] == "auto":
            filtered_extra.extend(["--permission-mode", "bypassPermissions"])
            i += 2
        elif arg.startswith("--permission-mode=auto"):
            filtered_extra.append("--permission-mode=bypassPermissions")
            i += 1
        else:
            filtered_extra.append(arg)
            i += 1

    settings_dict = {
        "env": {
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_AUTO_MODE_SERVER": "0",
            "CLAUDE_CODE_AUTO_MODE_CLASSIFY_EDITS": "0"
        }
    }
    args = ["--model", model, "--settings", json.dumps(settings_dict),
            "--exclude-dynamic-system-prompt-sections",
            "--allow-dangerously-skip-permissions"]
    if lean:   # core coding tools only: ~4K tokens of tool definitions instead of ~31K
        args += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--tools", LEAN_TOOLS]
    return args + filtered_extra


def _default_relay():
    try:
        from . import common
        return f"http://127.0.0.1:{common.load_config().get('relay_port', 8080)}"
    except Exception:
        return "http://127.0.0.1:8080"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli_model, extra_argv = parse_model_arg(argv)
    want_model = cli_model or os.environ.get("KLLM_MODEL")
    relay = (os.environ.get("KLLM_RELAY") or _default_relay()).rstrip("/")
    try:
        _get(relay + "/__relay/health")
    except Exception:
        # First use brings the stack up: start the relay here instead of failing.
        # GPU backends then start on demand (autoscale) and stop when idle.
        from . import common
        print("kllm relay not running — starting it now (`kllm serve` in the background)...",
              file=sys.stderr)
        if not common.ensure_serve(log=lambda *a: print(*a, file=sys.stderr)):
            print("kllm relay not running. Start it:  kllm serve   (or: kllm install-service "
                  "for at-logon start)", file=sys.stderr)
            return 1
    try:
        if '"url"' not in _get(relay + "/__relay/status"):
            print("Note: no Kaggle backend in the pool yet — the first request starts it "
                  "automatically (~8-10 min cold) and it stops after 25 min idle. "
                  "To pre-warm now: 'kllm up'. Falling back to local if available.", file=sys.stderr)
    except Exception:
        pass
    try:
        model, ctx = pick_model(json.loads(_get(relay + "/admin/api/state")), want_model)
    except Exception:
        model, ctx = want_model or "kllm", 32768
    ctx = int(os.environ.get("KLLM_CTX") or ctx)
    # Auto-compact window (measured in long-session runs): at 50% (65K for the 128K models) Claude Code's own
    # "autocompact is thrashing" guard aborted read-heavy tasks; 110K needs far fewer compactions. 256K models: 150K (a 167K-token compaction took 15 min, too close to Claude Code's patience).
    if "256k" in (model or "").lower():
        auto_compact = 150000 if ctx > 220000 else max(8192, ctx - 20000)
    else:
        auto_compact = 90000 if ctx > 130000 else int(ctx * 0.84)
    auto_compact = int(os.environ.get("KLLM_AUTO_COMPACT_CTX") or auto_compact)
    print(f"claude-kllm: model {model}, context {ctx}, auto-compact at {auto_compact}", file=sys.stderr)
    exe = shutil.which("claude")
    if not exe:
        print("Claude Code ('claude') not found on PATH. Install: https://docs.claude.com/claude-code", file=sys.stderr)
        return 1
    env = build_env(relay, model, ctx, auto_compact=auto_compact)
    cmd = [exe, *build_args(model, extra_argv, lean=os.environ.get("LEAN", "1") == "1")]
    if os.name == "nt":
        return subprocess.call(cmd, env=env)
    os.execvpe(exe, cmd, env)


if __name__ == "__main__":
    sys.exit(main())
