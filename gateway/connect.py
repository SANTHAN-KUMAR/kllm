"""Connect page: point every harness and app at kllm, from the control panel.

Each tool has up to two options:
  models  the tool's model provider points at the relay, so it runs on the Kaggle pool
  mcp     the tool gets kllm's MCP server, so it can start and steer kllm agents (spawn_harness_subagent and friends)

GET  /admin/api/connect                      -> {"tools": [...]} (detected, cached for 30 s)
POST /admin/api/connect/{tool}/setup {what}  -> writes the config (a timestamped backup first)
POST /admin/api/connect/{tool}/undo  {what}  -> restores the backup, or removes what setup added

Setup never loses what the user already has: JSON files are merged key by key, JSONC and TOML get a block inserted
next to their own content, and every write keeps `<file>.kllm-bak-<time>` so Undo puts the file back exactly.
"""
import json, os, pathlib, re, shutil, subprocess, time, sys
from . import common

_CACHE = {"at": 0, "rows": None}
BAK = ".kllm-bak-"


def _home():
    return pathlib.Path.home()


def _version(binary):
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=8)
        line = (out.stdout or out.stderr).strip().splitlines()
        return line[0][:80] if line else None
    except Exception:
        return None


class SetupError(Exception):
    pass


# ---- backups -----------------------------------------------------------------------------------------------------
def _backup(p):
    """Copy p aside before a write. A file that did not exist gets a '.new' marker, so Undo deletes it."""
    p = pathlib.Path(p)
    ms = int(time.time() * 1000)
    taken = {x.name for x in p.parent.glob(p.name + BAK + "*")} if p.parent.exists() else set()
    while p.name + f"{BAK}{ms}" in taken or p.name + f"{BAK}{ms}.new" in taken:
        ms += 1                                  # two writes in one millisecond must not share a backup
    stamp = f"{BAK}{ms}"
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + stamp))
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_name(p.name + stamp + ".new").write_text("", encoding="utf-8")


def _undo_file(p):
    p = pathlib.Path(p)
    baks = sorted(p.parent.glob(p.name + BAK + "*"))
    if not baks:
        raise SetupError(f"No kllm backup of {p} to restore.")
    last = baks[-1]
    if last.name.endswith(".new"):
        if p.exists():
            p.unlink()
        last.unlink()
        return f"Removed {p}, which kllm had created."
    shutil.copy2(last, p)
    last.unlink()
    return f"Restored {p} from before kllm changed it."


def _has_backup(p):
    p = pathlib.Path(p)
    return p.parent.exists() and any(p.parent.glob(p.name + BAK + "*"))


# ---- formats -------------------------------------------------------------------------------------------------------
def _strip_jsonc(text):
    """JSONC -> JSON: drop // and /* */ comments outside strings, then trailing commas."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1]); i += 2; continue
            if c == '"':
                in_str = False
            i += 1; continue
        if c == '"':
            in_str = True; out.append(c); i += 1; continue
        if text.startswith("//", i):
            j = text.find("\n", i); i = n if j < 0 else j; continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2); i = n if j < 0 else j + 2; continue
        out.append(c); i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _load_json(p, jsonc=False):
    p = pathlib.Path(p)
    if not p.exists():
        return {}
    raw = p.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return {}
    try:
        return json.loads(_strip_jsonc(raw) if jsonc else raw)
    except ValueError as e:
        raise SetupError(f"{p} is not valid JSON ({e}). Fix it, or add the block by hand.")


def _write_json(p, data):
    p = pathlib.Path(p)
    _backup(p)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _insert_jsonc(p, key, value):
    """Add "key": value at the top level of a JSONC file, or inside an existing object of that name, keeping every
    comment. Validated after the write; a bad result is rolled back."""
    p = pathlib.Path(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    _backup(p)
    m = re.search(r'"' + re.escape(key[0]) + r'"\s*:\s*\{', text)
    if len(key) == 2 and m:                          # merge inside the existing object
        body = json.dumps(value, indent=2).replace("\n", "\n    ")
        new = text[:m.end()] + f'\n    "{key[1]}": {body},' + text[m.end():]
    else:
        obj = {key[1]: value} if len(key) == 2 else value
        body = json.dumps(obj, indent=2).replace("\n", "\n  ")
        first = text.find("{")
        if first < 0:
            raise SetupError(f"{p} has no top-level object.")
        new = text[:first + 1] + f'\n  "{key[0]}": {body},' + text[first + 1:]
    try:
        json.loads(_strip_jsonc(new))
    except ValueError as e:
        _undo_file(p)
        raise SetupError(f"Could not merge into {p} safely ({e}). Add the block by hand.")
    p.write_text(new, encoding="utf-8")


def _append_text(p, block):
    p = pathlib.Path(p)
    _backup(p)
    cur = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    p.write_text(cur.rstrip("\n") + ("\n\n" if cur.strip() else "") + block.strip("\n") + "\n", encoding="utf-8")


def _toml_str(s):
    return json.dumps(str(s))                        # a TOML basic string accepts JSON string escapes


# ---- what kllm writes ----------------------------------------------------------------------------------------------
def _mcp_stdio():
    """kllm's MCP server as a stdio command (works with no relay key, same entry the IDE snippet uses)."""
    from .mcp_server import get_mcp_config_snippet
    return get_mcp_config_snippet()["mcpServers"]["kllm-spawner"]


def _opencode_provider(relay):
    return {"npm": "@ai-sdk/openai-compatible", "name": "kllm (Kaggle GPU)",
            "options": {"baseURL": f"{relay}/v1", "apiKey": "sk-kllm-local", "timeout": 1800000},
            "models": {"kllm": {"name": "kllm (whatever the pool serves)", "tool_call": True, "reasoning": True,
                                "limit": {"context": 262144, "output": 32768}}}}


# ---- per tool -------------------------------------------------------------------------------------------------------
def _paths():
    h = _home()
    return {"claude_json": h / ".claude.json", "agy_mcp": h / ".gemini" / "antigravity-cli" / "mcp.json",
            "gemini": h / ".gemini" / "settings.json", "codex": h / ".codex" / "config.toml",
            "cursor": h / ".cursor" / "mcp.json", "bin": h / ".local" / "bin"}


def _opencode_config():
    for name in ("opencode.jsonc", "opencode.json"):
        p = _home() / ".config" / "opencode" / name
        if p.exists():
            return p
    return None


_MARK = "rem kllm-launcher"


def _launcher(bin_dir):
    """Where the panel installs claude-kllm: a symlink on POSIX, a .cmd shim on Windows (symlinks need admin there)."""
    return bin_dir / ("claude-kllm.cmd" if os.name == "nt" else "claude-kllm")


def _install_launcher(dst):
    if os.name == "nt":
        lines = ["@echo off", f"{_MARK} {common.PKG_ROOT}",
                 f'set "PYTHONPATH={common.PKG_ROOT};%PYTHONPATH%"',
                 f'"{sys.executable}" -m gateway.claude_launcher %*']
        dst.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    else:
        os.symlink(common.PKG_ROOT / "bin" / "claude-kllm", dst)


def _ours(link):
    """True only for a claude-kllm launcher this install's panel created (symlink to bin/claude-kllm, or our .cmd shim)."""
    try:
        if link.is_symlink():
            return pathlib.Path(os.readlink(link)).resolve() == (common.PKG_ROOT / "bin" / "claude-kllm").resolve()
        if os.name == "nt" and link.suffix == ".cmd" and link.is_file():
            return f"{_MARK} {common.PKG_ROOT}" in link.read_text(encoding="utf-8", errors="replace")
        return False
    except OSError:
        return False


def _kllm_key(d):
    return any(str(k).lower().startswith("kllm") for k in (d or {}))


def _opt(what, label, state, detail, can_setup=False, can_undo=False):
    return {"what": what, "label": label, "state": state, "detail": detail, "can_setup": can_setup, "can_undo": can_undo}


def detect(force=False):
    if not force and _CACHE["rows"] is not None and time.time() - _CACHE["at"] < 30:
        return _CACHE["rows"]
    P, rows = _paths(), []

    claude = common.which("claude")
    launcher = common.which("claude-kllm")
    cj = {}
    try:
        cj = _load_json(P["claude_json"])
    except SetupError:
        pass
    mcp_on = _kllm_key(cj.get("mcpServers"))
    rows.append({"id": "claude-code", "name": "Claude Code", "installed": bool(claude), "version": _version(claude) if claude else None,
                 "command": "claude-kllm", "options": [
                     _opt("models", "Run on kllm's models", "on" if launcher else "off" if claude else "missing",
                          f"claude-kllm launcher at {launcher}" if launcher else "installs the claude-kllm launcher in ~/.local/bin",
                          can_setup=bool(claude) and not launcher, can_undo=_ours(_launcher(P["bin"]))),
                     _opt("mcp", "kllm tools over MCP", "on" if mcp_on else "off" if claude else "missing",
                          "kllm is in your Claude Code MCP servers" if mcp_on else "adds kllm with `claude mcp add --scope user`",
                          can_setup=bool(claude) and not mcp_on, can_undo=bool(claude) and "kllm" in (cj.get("mcpServers") or {}))]})

    oc = common.which("opencode")
    cfgp = _opencode_config()
    occ = {}
    if cfgp:
        try:
            occ = _load_json(cfgp, jsonc=True)
        except SetupError:
            occ = {}
    prov_on = "kllm" in (occ.get("provider") or {})
    ocmcp_on = _kllm_key(occ.get("mcp"))
    rows.append({"id": "opencode", "name": "opencode", "installed": bool(oc), "version": _version(oc) if oc else None,
                 "command": "opencode -m kllm/kllm", "options": [
                     _opt("models", "Run on kllm's models", "on" if prov_on else "off" if oc else "missing",
                          f"kllm provider in {cfgp}" if prov_on else f"adds the kllm provider to {cfgp or '~/.config/opencode/opencode.jsonc'}",
                          can_setup=bool(oc) and not prov_on, can_undo=bool(cfgp) and _has_backup(cfgp)),
                     _opt("mcp", "kllm tools over MCP", "on" if ocmcp_on else "off" if oc else "missing",
                          "kllm is in opencode's mcp servers" if ocmcp_on else "adds kllm as a remote MCP server",
                          can_setup=bool(oc) and not ocmcp_on, can_undo=bool(cfgp) and _has_backup(cfgp))]})

    agy = common.which("agy")
    am = {}
    try:
        am = _load_json(P["agy_mcp"]).get("mcpServers") or {}
    except SetupError:
        pass
    rows.append({"id": "agy", "name": "Antigravity", "installed": bool(agy), "version": _version(agy) if agy else None,
                 "command": "agy", "options": [
                     _opt("models", "Run on kllm's models", "unsupported", "Antigravity runs Google sign-in models. kllm joins it as tools over MCP."),
                     _opt("mcp", "kllm tools over MCP", "on" if _kllm_key(am) else "off" if agy else "missing",
                          f"kllm is in {P['agy_mcp']}" if _kllm_key(am) else f"adds kllm to {P['agy_mcp']}",
                          can_setup=bool(agy) and not _kllm_key(am), can_undo=_has_backup(P["agy_mcp"]))]})

    gem = common.which("gemini")
    gver = _version(gem) if gem else None
    if gver and "opencode" in gver.lower():          # some machines alias `gemini` to another tool
        gem = None
    gm = {}
    try:
        gm = _load_json(P["gemini"]).get("mcpServers") or {}
    except SetupError:
        pass
    rows.append({"id": "gemini", "name": "Gemini CLI", "installed": bool(gem), "version": gver if gem else None,
                 "command": "gemini", "options": [
                     _opt("models", "Run on kllm's models", "unsupported", "Gemini CLI runs Google models. kllm joins it as tools over MCP."),
                     _opt("mcp", "kllm tools over MCP", "on" if _kllm_key(gm) else "off" if gem else "missing",
                          f"kllm is in {P['gemini']}" if _kllm_key(gm) else f"adds kllm to {P['gemini']}",
                          can_setup=bool(gem) and not _kllm_key(gm), can_undo=_has_backup(P["gemini"]))]})

    codex = common.which("codex")
    ct = P["codex"].read_text(encoding="utf-8", errors="replace") if P["codex"].exists() else ""
    cprov = "[model_providers.kllm]" in ct
    cmcp = "[mcp_servers.kllm]" in ct
    rows.append({"id": "codex", "name": "Codex CLI", "installed": bool(codex), "version": _version(codex) if codex else None,
                 "command": "codex --profile kllm", "options": [
                     _opt("models", "Run on kllm's models", "on" if cprov else "off" if codex else "missing",
                          "kllm provider and profile in ~/.codex/config.toml" if cprov else "adds a kllm provider and a kllm profile",
                          can_setup=bool(codex) and not cprov, can_undo=_has_backup(P["codex"])),
                     _opt("mcp", "kllm tools over MCP", "on" if cmcp else "off" if codex else "missing",
                          "kllm is in Codex's mcp_servers" if cmcp else "adds kllm to ~/.codex/config.toml",
                          can_setup=bool(codex) and not cmcp, can_undo=_has_backup(P["codex"]))]})

    cursor_dir = P["cursor"].parent.exists()
    cm = {}
    try:
        cm = _load_json(P["cursor"]).get("mcpServers") or {}
    except SetupError:
        pass
    rows.append({"id": "cursor", "name": "Cursor", "installed": cursor_dir or None, "version": None, "command": None, "options": [
        _opt("models", "Run on kllm's models", "manual", "Settings, Models: override the OpenAI base URL with the relay address below."),
        _opt("mcp", "kllm tools over MCP", "on" if _kllm_key(cm) else "off" if cursor_dir else "missing",
             f"kllm is in {P['cursor']}" if _kllm_key(cm) else f"adds kllm to {P['cursor']}",
             can_setup=cursor_dir and not _kllm_key(cm), can_undo=_has_backup(P["cursor"]))]})

    for tid, name, detail in (("openai", "Other OpenAI-compatible apps", "Open WebUI, Continue, LibreChat: paste the base URL and key"),
                              ("code", "Python and JavaScript", "OpenAI SDK with base_url set to the relay"),
                              ("phone", "Phone through the tunnel", "needs the remote tunnel and a gateway key")):
        rows.append({"id": tid, "name": name, "installed": None, "version": None, "command": None,
                     "options": [_opt("models", "Run on kllm's models", "manual", detail)]})

    for r in rows:                                   # older fields the page used before options existed
        m = r["options"][0]
        r["configured"] = any(o["state"] == "on" for o in r["options"]) or None if r["installed"] is None else any(o["state"] == "on" for o in r["options"])
        r["can_setup"] = any(o["can_setup"] for o in r["options"])
        r["detail"] = m["detail"] if r["installed"] is not False else f"{r['name']} is not installed"
    _CACHE.update(at=time.time(), rows=rows)
    return rows


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise SetupError(((p.stderr or p.stdout) or f"{cmd[0]} failed").strip()[:400])
    return p.stdout


def setup(tool, what="models", relay="http://127.0.0.1:8080"):
    """-> message. Raises SetupError with the reason and the manual fix."""
    P = _paths()
    _CACHE["rows"] = None
    if tool == "claude-code":
        claude = common.which("claude")
        if not claude:
            raise SetupError("Claude Code is not installed: install it first, then press Set up again.")
        if what == "mcp":
            try:
                if _kllm_key(_load_json(P["claude_json"]).get("mcpServers")):
                    return "kllm is already in Claude Code's MCP servers."
            except SetupError:
                pass
            _run([claude, "mcp", "add", "--scope", "user", "--transport", "http", "kllm", f"{relay}/mcp"])
            return "Added kllm to Claude Code's MCP servers. New Claude Code sessions can start kllm agents."
        if common.which("claude-kllm"):
            return "claude-kllm is already installed."
        src = common.PKG_ROOT / "bin" / "claude-kllm"
        if os.name != "nt" and not src.exists():
            raise SetupError("This install has no bin/claude-kllm: run `pip install kllm`, which installs claude-kllm.")
        P["bin"].mkdir(parents=True, exist_ok=True)
        dst = _launcher(P["bin"])
        if dst.exists() or dst.is_symlink():
            raise SetupError(f"{dst} exists but is not on PATH: add {P['bin']} to PATH.")
        _install_launcher(dst)
        on_path = str(P["bin"]) in os.environ.get("PATH", "").split(os.pathsep)
        return f"Installed {dst}. Run claude-kllm." + ("" if on_path else f" Add {P['bin']} to PATH first.")
    if tool == "opencode":
        if not common.which("opencode"):
            raise SetupError("opencode is not installed: install it first, then press Set up again.")
        cur = _opencode_config() or (_home() / ".config" / "opencode" / "opencode.jsonc")
        if not cur.exists():
            _backup(cur)
            cur.write_text(json.dumps({"$schema": "https://opencode.ai/config.json"}, indent=2) + "\n", encoding="utf-8")
        have = _load_json(cur, jsonc=True)
        if what == "mcp":
            if _kllm_key(have.get("mcp")):
                return f"{cur} already has kllm's MCP server."
            _insert_jsonc(cur, ("mcp", "kllm"), {"type": "remote", "url": f"{relay}/mcp", "enabled": True})
            return f"Added kllm's MCP server to {cur}."
        if "kllm" in (have.get("provider") or {}):
            return f"{cur} already has the kllm provider."
        _insert_jsonc(cur, ("provider", "kllm"), _opencode_provider(relay))
        return f"Added the kllm provider to {cur}. Run `opencode -m kllm/kllm`."
    if tool in ("agy", "gemini", "cursor"):
        if what != "mcp":
            raise SetupError("This tool runs its own models; kllm joins it as tools over MCP only.")
        p = {"agy": P["agy_mcp"], "gemini": P["gemini"], "cursor": P["cursor"]}[tool]
        if tool == "cursor" and not p.parent.exists():
            raise SetupError("Cursor is not installed (no ~/.cursor folder).")
        if tool == "gemini" and common.which("gemini") and "opencode" in (_version(common.which("gemini")) or "").lower():
            raise SetupError("`gemini` on this machine runs opencode, not Gemini CLI. Install Gemini CLI to connect it.")
        if tool != "cursor" and not common.which("agy" if tool == "agy" else "gemini"):
            raise SetupError(f"{'Antigravity' if tool == 'agy' else 'Gemini CLI'} is not installed.")
        data = _load_json(p)
        servers = data.setdefault("mcpServers", {})
        if _kllm_key(servers):
            return f"kllm is already in {p}."
        servers["kllm"] = {"httpUrl": f"{relay}/mcp"} if tool == "gemini" else (
            {"url": f"{relay}/mcp"} if tool == "cursor" else _mcp_stdio())
        _write_json(p, data)
        return f"Added kllm to {p}. Restart the tool to load it."
    if tool == "codex":
        if not common.which("codex"):
            raise SetupError("Codex CLI is not installed: install it first, then press Set up again.")
        cur = P["codex"].read_text(encoding="utf-8", errors="replace") if P["codex"].exists() else ""
        if what == "mcp" and "[mcp_servers.kllm]" in cur:
            return "kllm is already in Codex's MCP servers."
        if what != "mcp" and "[model_providers.kllm]" in cur:
            return "Codex already has the kllm provider."
        if what == "mcp":
            m = _mcp_stdio()
            _append_text(P["codex"], "[mcp_servers.kllm]\n" + f"command = {_toml_str(m['command'])}\n"
                         + "args = [" + ", ".join(_toml_str(a) for a in m["args"]) + "]\n"
                         + f"cwd = {_toml_str(m.get('cwd', ''))}\n")
            return "Added kllm to Codex's MCP servers in ~/.codex/config.toml."
        _append_text(P["codex"], "[model_providers.kllm]\nname = \"kllm (Kaggle GPU)\"\n"
                     + f"base_url = {_toml_str(relay + '/v1')}\nwire_api = \"chat\"\n\n"
                     + "[profiles.kllm]\nmodel_provider = \"kllm\"\nmodel = \"kllm\"\n")
        return "Added a kllm provider and profile to ~/.codex/config.toml. Run `codex --profile kllm`."
    raise SetupError(f"'{tool}' has no automatic setup: follow the steps on the page.")


def undo(tool, what="models"):
    P = _paths()
    _CACHE["rows"] = None
    if tool == "claude-code":
        if what == "mcp":
            claude = common.which("claude")
            if not claude:
                raise SetupError("Claude Code is not installed.")
            _run([claude, "mcp", "remove", "--scope", "user", "kllm"])
            return "Removed kllm from Claude Code's MCP servers."
        dst = _launcher(P["bin"])
        if _ours(dst):
            dst.unlink()
            return f"Removed {dst}."
        raise SetupError("claude-kllm was not installed by the panel, so it is left alone.")
    p = {"opencode": _opencode_config(), "agy": P["agy_mcp"], "gemini": P["gemini"], "codex": P["codex"],
         "cursor": P["cursor"]}.get(tool)
    if not p:
        raise SetupError(f"Nothing to undo for {tool}.")
    return _undo_file(p)
