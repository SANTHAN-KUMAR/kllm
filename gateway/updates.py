"""Update checks: is a newer kllm on PyPI, and how does this install upgrade itself?

- The source of truth is PyPI (https://pypi.org/pypi/kllm-gateway/json), the same channel releases are published to
  through trusted publishing. Nothing else is downloaded: upgrades go through uv, pipx or pip, which verify the
  package hashes PyPI serves.
- Checks are cached in <root>/state/update_check.json for a day and run in the background, so no command ever waits
  on the network. Turn them off with `update_check: false` in config.yaml or KLLM_NO_UPDATE_CHECK=1.
- `kllm update` upgrades with the tool that installed kllm. On Windows the running kllm.exe cannot be replaced, so
  the upgrade runs in a separate console after this process exits.
"""
import json, os, pathlib, re, subprocess, sys, threading, time, urllib.request

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


def upgrade_command():
    method = _install_method()
    if method == "uv":
        cmd = ["uv", "tool", "upgrade", PACKAGE]
    elif method == "pipx":
        cmd = ["pipx", "upgrade", PACKAGE]
    elif method == "checkout":
        cmd = ["git", "-C", str(common.PKG_ROOT), "pull", "--ff-only"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE]
    display = cmd if method != "pip" else ["python", "-m", "pip", "install", "--upgrade", PACKAGE]
    return {"method": method, "cmd": cmd, "display": display}


def run_upgrade(log=print):
    """Upgrade this install. Returns 0 on success (or when the upgrade was handed to a separate console on Windows)."""
    u = upgrade_command()
    if os.name == "nt" and u["method"] in ("uv", "pipx", "pip"):
        # kllm.exe is running right now and Windows will not let it be replaced: run the upgrade in a new console
        # that waits for this process to exit first.
        script = (f"import subprocess, time, os; time.sleep(2); "
                  f"r = subprocess.call({u['cmd']!r}); "
                  f"print(); print('kllm update finished' if r == 0 else 'kllm update failed (exit %d)' % r); "
                  f"input('Press Enter to close this window')")
        subprocess.Popen([sys.executable, "-c", script], creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10),
                         close_fds=True)
        log(f"kllm: upgrading in a new window: {' '.join(u['display'])}")
        return 0
    log(f"kllm: running {' '.join(u['display'])}")
    try:
        return subprocess.call(u["cmd"])
    except FileNotFoundError:
        log(f"kllm: {u['cmd'][0]} is not on PATH. Upgrade with: python -m pip install --upgrade {PACKAGE}")
        return 1
