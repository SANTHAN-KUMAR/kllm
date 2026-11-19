"""Update checks: is a newer kllm on PyPI, and how does this install upgrade itself?

- The source of truth is PyPI (https://pypi.org/pypi/kllm-gateway/json), the same channel releases are published to
  through trusted publishing. Nothing else is downloaded: upgrades go through uv, pipx or pip, which verify the
  package hashes PyPI serves.
- Checks are cached in <root>/state/update_check.json for a day and run in the background, so no command ever waits
  on the network. Turn them off with `update_check: false` in config.yaml or KLLM_NO_UPDATE_CHECK=1.
- `kllm update` upgrades with the tool that installed kllm. On Windows the running kllm.exe cannot be replaced, so
  the upgrade runs in a separate console after this process exits.
"""
import json, os, pathlib, re, shutil, subprocess, sys, threading, time, urllib.request

from . import common

PACKAGE = "kllm-gateway"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE}/json"
RELEASES_URL = "https://github.com/SANTHAN-KUMAR/kllm/releases"
TTL_S = 24 * 3600
_lock = threading.Lock()


def current_version():
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(PACKAGE)
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    pp = common.PKG_ROOT / "pyproject.toml"           # running from a checkout without an install
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', pp.read_text(encoding="utf-8"), re.M)
        return m.group(1) if m else "0"
    except OSError:
        return "0"


def _key(v):
    """Comparable key for X.Y.Z[suffix]; pre-releases (a, b, rc, dev) sort before the release."""
    m = re.match(r"^(\d+(?:\.\d+)*)(.*)$", str(v).strip())
    if not m:
        return ((0,), 0)
    nums = tuple(int(x) for x in m.group(1).split("."))
    nums = nums + (0,) * (4 - len(nums))
    return (nums, 0 if m.group(2) else 1)


def newer(a, b):
    """True when version a is newer than version b."""
    return _key(a) > _key(b)


def _is_prerelease(v):
    return bool(re.search(r"(a|b|rc|dev)\d*", str(v)))


def enabled(cfg=None):
    if os.environ.get("KLLM_NO_UPDATE_CHECK") == "1":
        return False
    try:
        cfg = cfg if cfg is not None else common.load_config()
    except Exception:
        cfg = {}
    return cfg.get("update_check", True) is not False


def _cache_file():
    return common.ROOT / "state" / "update_check.json"


def _read_cache():
    try:
        return json.loads(_cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_cache(d):
    try:
        f = _cache_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d), encoding="utf-8")
    except OSError:
        pass


def fetch_latest(timeout=6):
    """Latest stable version on PyPI, or None when offline."""
    try:
        req = urllib.request.Request(PYPI_URL, headers={"User-Agent": f"kllm/{current_version()}", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    releases = [v for v, files in (d.get("releases") or {}).items()
                if files and not _is_prerelease(v) and not all(f.get("yanked") for f in files)]
    if not releases:
        return (d.get("info") or {}).get("version")
    return max(releases, key=_key)


def status(force=False, network=True):
    """{current, latest, update_available, checked_at, command, notes}. Uses the cache unless it is stale or force."""
    cur = current_version()
    with _lock:
        c = _read_cache()
        fresh = c.get("checked_at") and time.time() - c["checked_at"] < TTL_S and c.get("current") == cur
        if network and (force or not fresh):
            latest = fetch_latest()
            if latest:
                c = {"latest": latest, "checked_at": time.time(), "current": cur}
                _write_cache(c)
    latest = c.get("latest")
    return {"current": cur, "latest": latest, "update_available": bool(latest and newer(latest, cur)),
            "checked_at": c.get("checked_at"), "command": " ".join(upgrade_command()["display"]),
            "method": upgrade_command()["method"], "notes": RELEASES_URL}


def refresh_in_background(cfg=None):
    """Kick off a check if the cache is stale; never blocks and never raises."""
    if not enabled(cfg):
        return
    c = _read_cache()
    if c.get("checked_at") and time.time() - c["checked_at"] < TTL_S and c.get("current") == current_version():
        return
    threading.Thread(target=lambda: status(force=True), name="kllm-update-check", daemon=True).start()


def notice(cfg=None):
    """One line for the CLI when a newer version is known (from the cache only), else None."""
    if not enabled(cfg):
        return None
    s = status(network=False)
    if not s["update_available"]:
        return None
    return f"kllm {s['latest']} is available (you have {s['current']}). Update: kllm update"


# ---- how to upgrade this install ------------------------------------------------------------------------------------
def _install_method():
    root = common.PKG_ROOT
    if (root / ".git").exists() or ((root / "pyproject.toml").exists() and "site-packages" not in str(root).lower()):
        return "checkout"                       # a git clone (editable install): new code comes from git pull
    prefix = pathlib.Path(sys.prefix)
    if (prefix / "uv-receipt.toml").exists():   # uv tool environments carry a receipt
        return "uv"
    if (prefix / "pipx_metadata.json").exists():
        return "pipx"
    s = str(pathlib.Path(sys.executable)).lower().replace("\\", "/")
    if "/uv/tools/" in s:
        return "uv"
    if "/pipx/venvs/" in s:
        return "pipx"
    return "pip"


def find_tool(name):
    """Full path to uv / pipx: on PATH, or where their installers put them (a fresh window may not have PATH yet)."""
    found = shutil.which(name)
    if found:
        return found
    home = pathlib.Path(os.path.expanduser("~"))
    exe = name + (".exe" if os.name == "nt" else "")
    cands = [home / ".local" / "bin" / exe, home / ".cargo" / "bin" / exe]
    for var in ("UV_INSTALL_DIR", "XDG_BIN_HOME", "PIPX_BIN_DIR"):
        if os.environ.get(var):
            cands.insert(0, pathlib.Path(os.environ[var]) / exe)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cands.append(pathlib.Path(os.environ["LOCALAPPDATA"]) / "Programs" / name / exe)
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def _env_pip_steps(*pip_args):
    """Use this environment's own Python: install pip into it if it has none (uv environments do not), then pip."""
    py = sys.executable
    try:
        has_pip = subprocess.run([py, "-m", "pip", "--version"], capture_output=True, timeout=60,
                                 stdin=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        has_pip = False
    steps = [] if has_pip else [[py, "-m", "ensurepip", "--upgrade"]]
    return steps + [[py, "-m", "pip", *pip_args]]


def upgrade_command():
    """{method, steps: [argv, ...], display: [argv]} for upgrading this install."""
    method = _install_method()
    if method == "checkout":
        steps = [["git", "-C", str(common.PKG_ROOT), "pull", "--ff-only"]]
        return {"method": method, "steps": steps, "cmd": steps[0], "display": steps[0]}
    tool = find_tool(method) if method in ("uv", "pipx") else None
    if method == "uv" and tool:
        steps = [[tool, "tool", "upgrade", PACKAGE]]
        display = ["uv", "tool", "upgrade", PACKAGE]
    elif method == "pipx" and tool:
        steps = [[tool, "upgrade", PACKAGE]]
        display = ["pipx", "upgrade", PACKAGE]
    else:                                     # plain pip, or uv/pipx is no longer on this machine
        steps = _env_pip_steps("install", "--upgrade", PACKAGE)
        display = ["python", "-m", "pip", "install", "--upgrade", PACKAGE]
        if method in ("uv", "pipx"):
            method = method + " (not found: using this environment's pip)"
    return {"method": method, "steps": steps, "cmd": steps[-1], "display": display}


def launchers():
    """The `kllm` and `claude-kllm` launchers that belong to THIS environment (not another install's)."""
    prefix = str(pathlib.Path(sys.prefix).resolve())
    names = ["kllm", "claude-kllm"]
    exts = [".exe", ""] if os.name == "nt" else [""]
    dirs = {pathlib.Path(sys.executable).parent}
    for n in names:
        w = shutil.which(n)
        if w:
            dirs.add(pathlib.Path(w).parent)
    for var in ("UV_TOOL_BIN_DIR", "XDG_BIN_HOME", "PIPX_BIN_DIR"):
        if os.environ.get(var):
            dirs.add(pathlib.Path(os.environ[var]))
    dirs.add(pathlib.Path(os.path.expanduser("~")) / ".local" / "bin")
    out = []
    for d in dirs:
        for n in names:
            for e in exts:
                p = d / (n + e)
                if not p.is_file() and not p.is_symlink():
                    continue
                try:
                    if p.is_symlink() and str(p.resolve()).startswith(prefix):
                        out.append(p); continue
                    data = p.read_bytes()[:200000]
                    if prefix.encode("utf-8") in data or prefix.encode("utf-16-le") in data \
                            or str(pathlib.Path(sys.prefix)).encode("utf-8") in data:
                        out.append(p)
                except OSError:
                    pass
    return sorted(set(out))


def remove_command():
    """{method, steps: [argv, ...] | None, paths: [Path], display} for removing this install's package."""
    method = _install_method()
    if method == "checkout":
        return {"method": method, "steps": None, "paths": [], "display": None}
    tool = find_tool(method) if method in ("uv", "pipx") else None
    if method == "uv" and tool:
        return {"method": method, "steps": [[tool, "tool", "uninstall", PACKAGE]], "paths": [],
                "display": "uv tool uninstall " + PACKAGE}
    if method == "pipx" and tool:
        return {"method": method, "steps": [[tool, "uninstall", PACKAGE]], "paths": [],
                "display": "pipx uninstall " + PACKAGE}
    if method in ("uv", "pipx"):
        # the tool that made this environment is gone: the environment is self-contained, so delete it and the
        # launchers that point into it
        return {"method": method + " (not found)", "steps": None,
                "paths": [pathlib.Path(sys.prefix), *launchers()],
                "display": f"delete {sys.prefix} and the kllm launchers"}
    return {"method": method, "steps": _env_pip_steps("uninstall", "-y", PACKAGE), "paths": [],
            "display": "python -m pip uninstall -y " + PACKAGE}


def _batch(steps, paths, done, failed):
    """A cmd.exe script for Windows: waits for kllm to exit, runs the steps, deletes the paths, reports."""
    lines = ["@echo off", "timeout /t 3 /nobreak >nul"]
    for argv in steps or []:
        lines.append(subprocess.list2cmdline([str(a) for a in argv]) + " || goto fail")
    for p in paths or []:
        q = '"' + str(p) + '"'
        lines.append(f"if exist {q}\\* (rmdir /s /q {q}) else (if exist {q} del /f /q {q})")
    pause = [] if os.environ.get("KLLM_NO_PAUSE") == "1" else ["pause"]
    lines += ["echo.", f"echo {done}", *pause, "exit /b 0", ":fail", "echo.", f"echo {failed}", *pause, "exit /b 1"]
    return "\r\n".join(lines) + "\r\n"


def run_steps(steps, paths=(), what="update", log=print):
    """Run argv steps then delete paths. On Windows this happens after kllm exits, from cmd.exe (which lives outside
    the environment being changed), so nothing being replaced or deleted is still running."""
    import tempfile
    if os.name == "nt":
        bat = os.path.join(tempfile.gettempdir(), f"kllm-{what}.cmd")
        with open(bat, "w", encoding="utf-8") as f:
            f.write(_batch(steps, paths, f"kllm {what} finished.", f"kllm {what} failed: see the messages above."))
        subprocess.Popen(["cmd.exe", "/c", bat], creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10),
                         close_fds=True, cwd=os.path.expanduser("~"))
        log(f"kllm: the {what} continues in a new window once this command exits")
        return 0
    for argv in steps or []:
        try:
            r = subprocess.call(argv)
        except FileNotFoundError:
            log(f"kllm: {argv[0]} was not found")
            return 1
        if r != 0:
            return r
    for p in paths or []:
        p = pathlib.Path(p)
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    return 0


def run_upgrade(log=print):
    """Upgrade this install with the tool that installed it (or this environment's pip when that tool is gone)."""
    u = upgrade_command()
    log(f"kllm: {' '.join(u['display'])}" + (f"   [{u['method']}]" if "not found" in u["method"] else ""))
    return run_steps(u["steps"], what="update", log=log)
