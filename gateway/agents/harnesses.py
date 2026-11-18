"""Which agent harnesses kllm can drive, and how.

driver "acp":    the harness speaks the Agent Client Protocol on stdio (opencode acp, agy acp) or through an adapter
                 binary (claude-agent-acp, codex-acp). Live thinking, tool calls, plans, permission requests
                 (Ask mode) and cancel all work.
driver "stream": Claude Code without the ACP adapter: `claude -p --output-format stream-json`, one process per turn,
                 steering through --resume. Permissions are bypassed, so only Skip mode is available.
"""
import os, shutil, subprocess, time
from .. import common

HARNESSES = {
    "opencode": {"name": "opencode", "bin": "opencode", "acp": ["opencode", "acp"],
                 "default_model": "kllm/kllm", "attach": "opencode -s {sid}",
                 "note": "Runs on kllm's own models through the kllm provider in opencode's config."},
    "agy": {"name": "Antigravity", "bin": "agy", "acp": ["agy", "acp"], "default_model": None,
            "attach": "agy --conversation {sid}", "note": "Gemini and other cloud models from your Google sign-in."},
    "claude-code": {"name": "Claude Code", "bin": "claude", "acp_bins": ["claude-agent-acp", "claude-code-acp"],
                    "stream": True, "kllm_env": False, "default_model": None, "attach": "claude --resume {sid}",
                    "note": "Your own Claude account.", "acp_install": "npm i -g @agentclientprotocol/claude-agent-acp"},
    "claude-kllm": {"name": "Claude Code on kllm", "bin": "claude", "stream": True, "kllm_env": True,
                    "default_model": None, "attach": "claude-kllm --resume {sid}",
                    "note": "Claude Code pointed at the kllm relay (Kaggle models)."},
    "codex": {"name": "Codex", "bin": "codex", "acp_bins": ["codex-acp"], "default_model": None,
              "attach": "codex resume {sid}", "note": "Needs the codex-acp adapter.",
              "acp_install": "npm i -g @zed-industries/codex-acp"},
}

_CACHE = {"at": 0, "rows": None}


def _version(binary):
    try:
        p = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=8)
        out = (p.stdout or p.stderr).strip().splitlines()
        return out[0][:80] if out else None
    except Exception:
        return None


def resolve(name):
    """-> (definition, driver, command) where driver is "acp", "stream" or None (not usable here)."""
    h = HARNESSES.get(name)
    if h is None:
        return None, None, None
    if not common.which(h["bin"]):
        return h, None, None
    if h.get("acp"):
        return h, "acp", [common.which(h["acp"][0]) or h["acp"][0], *h["acp"][1:]]
    for b in h.get("acp_bins") or []:
        if common.which(b):
            return h, "acp", [common.which(b)]
    if h.get("stream"):
        return h, "stream", [common.which(h["bin"])]
    return h, None, None


def detect(force=False):
    if not force and _CACHE["rows"] is not None and time.time() - _CACHE["at"] < 60:
        return _CACHE["rows"]
    rows = []
    for key, h in HARNESSES.items():
        d, driver, cmd = resolve(key)
        path = common.which(h["bin"])
        why = None
        if not path:
            why = f"{h['bin']} is not on PATH"
        elif driver is None:
            why = "needs an ACP adapter: " + h.get("acp_install", "")
        rows.append({"id": key, "name": h["name"], "installed": bool(path), "path": path,
                     "version": _version(path) if path else None, "driver": driver,
                     "acp": driver == "acp", "ask": driver == "acp", "mcp": driver is not None,
                     "thinking": driver is not None, "usable": driver is not None, "why_not": why,
                     "default_model": h.get("default_model"), "note": h.get("note"),
                     "acp_install": h.get("acp_install") if driver != "acp" else None})
    _CACHE.update(at=time.time(), rows=rows)
    return rows


def worker_env(base_env, relay, session_id, worker_id, access, kllm_env=False, model=None):
    env = dict(base_env or os.environ)
    env["PATH"] = common.user_path(env.get("PATH", ""))      # harness tools (node, npm -g bins) must resolve too
    env.update(KLLM_AGENT_SESSION=session_id, KLLM_AGENT_WORKER=worker_id, KLLM_ACCESS=access or "read",
               KLLM_RELAY=relay)
    if kllm_env:
        from .. import claude_launcher
        env = claude_launcher.build_env(relay, kllm_model(model), int(os.environ.get("KLLM_CTX") or 131072), env=env)
    return env


def kllm_model(model):
    """A model key Claude Code can be told about. Claude Code refuses names it does not know ("kllm" or an opencode
    style "kllm/<key>": unrecognized_model), so like the claude-kllm launcher: strip the prefix, and for a generic
    "kllm" pick a chat model that is serving, else the default model."""
    key = str(model or "").strip()
    if key.startswith("kllm/"):
        key = key[5:]
    if key and key != "kllm":
        return key
    try:
        from .. import media
        st, reg, cfg = common.read_state(), common.load_models(), common.load_config()
        serving = [b.get("model_key") for b in st.get("backends", []) if media.kind(reg.get(b.get("model_key")) or {}) == "chat"]
        return serving[0] if serving else (cfg.get("default_model") or "kllm")
    except Exception:
        return "kllm"


def permission_env(env, harness, permission_mode):
    """Make the harness's own permission policy match the session. opencode allows every tool by default, so an
    Ask session's worker edited and ran shell commands without asking (seen 2026-11-10)."""
    if harness == "opencode":
        import json
        perm = "ask" if permission_mode == "ask" else "allow"
        cfg = {}
        try:
            cfg = json.loads(env.get("OPENCODE_CONFIG_CONTENT") or "{}")
        except ValueError:
            cfg = {}
        cfg["permission"] = dict(cfg.get("permission") or {}, edit=perm, bash=perm, webfetch=perm)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(cfg)
    return env


# ---- each harness's own model list, for the model pickers ------------------------------------------------------
_MODELS = {}
CLAUDE_MODELS = [("fable", "Fable (latest)"), ("opus", "Opus (latest)"), ("sonnet", "Sonnet (latest)"), ("haiku", "Haiku (latest)"),
                 ("claude-fable-5-1", "Claude Fable 5.1"), ("claude-opus-5-5", "Claude Opus 5.5"),
                 ("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("claude-haiku-5-5", "Claude Haiku 5.5")]


def _run_lines(cmd, timeout=30):
    env = dict(os.environ, PATH=common.user_path(os.environ.get("PATH", "")))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return [l.rstrip() for l in (p.stdout or "").splitlines() if l.strip()]
    except Exception:
        return []


def models(name, kllm_keys=(), force=False):
    """-> [{id, label, group}] the harness can run, plus kllm pool models where the harness reaches the relay.
    Asked from the harness itself (opencode models, agy models) and cached for 10 minutes."""
    hit = _MODELS.get(name)
    if hit and not force and time.time() - hit[0] < 600:
        rows = hit[1]
    else:
        d, driver, cmd = resolve(name)
        rows = []
        if d is None:
            return []
        if name == "opencode" and driver:
            for line in _run_lines([cmd[0] if cmd else "opencode", "models"]):
                prov = line.split("/", 1)[0] if "/" in line else "other"
                rows.append({"id": line, "label": line.split("/", 1)[-1], "group": "kllm pool" if prov == "kllm" else prov})
        elif name == "agy" and driver:
            for line in _run_lines([cmd[0] if cmd else "agy", "models"]):
                if "\t" not in line:
                    continue                                          # "Fetching available models..."
                mid, label = line.split("\t", 1)
                rows.append({"id": mid.strip(), "label": label.strip(), "group": "Antigravity"})
        elif name == "claude-code":
            rows = [{"id": m, "label": l, "group": "Claude"} for m, l in CLAUDE_MODELS]
        _MODELS[name] = (time.time(), rows)
    rows = list(rows)
    have = {r["id"] for r in rows}
    if name in ("claude-kllm",) or (name == "opencode" and not any(r["group"] == "kllm pool" for r in rows)):
        prefix = "kllm/" if name == "opencode" else ""
        rows += [{"id": prefix + k, "label": k, "group": "kllm pool"} for k in kllm_keys if prefix + k not in have]
    return rows
