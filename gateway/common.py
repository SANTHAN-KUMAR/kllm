"""Shared config, paths, state, and model-registry helpers for kllm."""
import os, sys, json, time, shutil, threading, pathlib, contextlib, yaml

try:                                   # POSIX
    import fcntl
except ImportError:                    # Windows: msvcrt byte-range lock gives the same exclusion
    fcntl = None
    import msvcrt

PKG_DIR = pathlib.Path(__file__).resolve().parent            # .../gateway
PKG_ROOT = PKG_DIR.parent                                    # repo checkout, or site-packages
DATA_DIR = PKG_DIR / "_data"                                 # kernel/, templates/, models.yaml in a wheel


def is_checkout():
    """True when running from a git/source checkout (models.yaml sits next to the package)."""
    return (PKG_ROOT / "models.yaml").exists()


def default_root():
    """Where config.yaml, models.yaml, state/ and logs/ live.
    $KLLM_ROOT, else the checkout itself, else the per-user config dir (pipx / uv tool install)."""
    env = os.environ.get("KLLM_ROOT")
    if env:
        return pathlib.Path(env).expanduser()
    if is_checkout():
        return PKG_ROOT
    if os.name == "nt":
        return pathlib.Path(os.environ.get("APPDATA") or pathlib.Path.home() / "AppData" / "Roaming") / "kllm"
    return pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or pathlib.Path.home() / ".config") / "kllm"


ROOT = default_root()
CONFIG_FILE = ROOT / "config.yaml"
MODELS_FILE = ROOT / "models.yaml"
STATE_FILE = ROOT / "state" / "state.json"
LOG_DIR = ROOT / "logs"


def data_path(rel):
    """Find a shipped data file/dir (kernel/, templates/, models.yaml, config.example.yaml).
    A copy under ROOT wins (user override), then the wheel's gateway/_data, then the checkout."""
    for base in (ROOT, DATA_DIR, PKG_ROOT):
        p = base / rel
        if p.exists():
            return p
    return ROOT / rel


def _seed_models():
    """An installed (non-checkout) root gets its own models.yaml so the panel can append to it."""
    if not MODELS_FILE.exists():
        src = DATA_DIR / "models.yaml"
        if src.exists():
            MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, MODELS_FILE)


LOG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "state").mkdir(parents=True, exist_ok=True)
_seed_models()

class ConfigMissing(FileNotFoundError):
    pass

def load_config():
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise ConfigMissing(f"no config at {CONFIG_FILE}. Run `kllm init` first "
                            f"(config lives in $KLLM_ROOT, default {ROOT}).") from None

def load_models():
    with open(MODELS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]

def resolve_model(key):
    models = load_models()
    if key not in models:
        raise KeyError(f"unknown model '{key}'. Known: {', '.join(models)}")
    m = dict(models[key])
    if m.get("alias_of"):
        target_key = m["alias_of"]
        target = resolve_model(target_key)
        merged = dict(target)
        merged.update({k: v for k, v in m.items() if v is not None})
        merged["key"] = key
        merged["target_key"] = target.get("target_key", target_key)
        return merged
    m["key"] = key
    return m

# state.json is read-modified-written by several processes (CLI, the relay's panel actions,
# the supervisor and its launch threads). One cross-process lock (flock) + a re-entrant
# thread lock makes every mutation atomic, so concurrent updates are never lost.
LOCK_FILE = ROOT / "state" / ".state.lock"
_tlock = threading.RLock()
_depth = threading.local()

def _lock_fd(fd, block=True):
    """Exclusive advisory lock on an open fd. Returns False only when block=False and it is held."""
    if fcntl:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
        except OSError:
            return False
        return True
    while True:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            if not block:
                return False
            time.sleep(0.05)

def _unlock_fd(fd):
    if fcntl:
        fcntl.flock(fd, fcntl.LOCK_UN)
    else:
        os.lseek(fd, 0, os.SEEK_SET)
        try: msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError: pass

def single_instance(name):
    """Hold a process-lifetime lock `state/.<name>.lock`. Returns a handle (keep a reference)
    or None when another process already holds it. Used so two supervisors never run at once."""
    fd = os.open(ROOT / "state" / f".{name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    if _lock_fd(fd, block=False):
        return fd
    os.close(fd)
    return None

@contextlib.contextmanager
def state_lock():
    with _tlock:
        d = getattr(_depth, "n", 0)
        if d == 0:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
            _lock_fd(fd)
            _depth.fd = fd
        _depth.n = d + 1
        try:
            yield
        finally:
            _depth.n -= 1
            if _depth.n == 0:
                _unlock_fd(_depth.fd); os.close(_depth.fd)

def read_state():
    for attempt in range(5):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                s = json.load(f)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            s = {}
            break
        except (PermissionError, OSError):
            if attempt == 4:
                s = {}
                break
            time.sleep(0.05)
    s.setdefault("backends", [])      # list of active Kaggle backends (one per account, parallel)
    s.setdefault("pending", {})       # account -> launching record
    s.setdefault("last_account_index", -1)
    s.setdefault("build_cache", {})   # account -> cached llama.cpp dataset ref
    s.setdefault("updated", 0)
    return s

def write_state(state):
    """Atomic write so the relay never reads a half-written file."""
    state["updated"] = time.time()
    with state_lock():
        tmp = STATE_FILE.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        for attempt in range(10):
            try:
                os.replace(tmp, STATE_FILE)
                break
            except (PermissionError, OSError):
                if attempt == 9:
                    raise
                time.sleep(0.05)

@contextlib.contextmanager
def mutate_state():
    """with mutate_state() as s: ...  — locked read-modify-write of state.json."""
    with state_lock():
        s = read_state()
        yield s
        write_state(s)

def update_state(**fields):
    with mutate_state() as s:
        s.update(fields)
    return s

def state_mtime():
    try:
        return STATE_FILE.stat().st_mtime
    except FileNotFoundError:
        return 0

# ---- backend-pool helpers (multiple parallel Kaggle backends) --------------
def upsert_backend(backend):
    """Add or replace the backend for backend['account'] in the pool."""
    with mutate_state() as s:
        s["backends"] = [b for b in s["backends"] if b["account"] != backend["account"]]
        s["backends"].append(backend)
        s["pending"].pop(backend["account"], None)
    return s

def remove_backend(account):
    with mutate_state() as s:
        s["backends"] = [b for b in s["backends"] if b["account"] != account]
        s["pending"].pop(account, None)
    return s

def set_pending(account, record):
    with mutate_state() as s:
        s["pending"][account] = record
    return s

# ---- GPU quota tracking ----------------------------------------------------------
WEEK = 7 * 24 * 3600
QUOTA_LIMIT = 30 * 3600                  # Kaggle weekly GPU quota per verified account: 30 h

def _week_start(now):
    return int(now - ((now - 2 * 86400) % WEEK))

def add_quota(account, seconds):
    """Accumulate GPU-seconds used by an account this week (auto-resets weekly)."""
    if seconds <= 0:
        return
    with mutate_state() as s:
        q = s.setdefault("quota", {})
        now = time.time()
        ws = _week_start(now)
        rec = q.get(account) or {}
        reset = False
        if rec.get("refresh_time"):
            try:
                import datetime
                dt = datetime.datetime.fromisoformat(rec["refresh_time"].replace("Z", "+00:00"))
                if now > dt.timestamp():
                    reset = True
            except Exception:
                pass
        elif rec.get("week") != ws and not (rec.get("synced_at") and now - rec.get("synced_at") < 6 * 86400):
            reset = True
        if reset:
            refresh_time = rec.get("refresh_time")
            if refresh_time:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(refresh_time.replace("Z", "+00:00"))
                    next_dt = dt + datetime.timedelta(days=7)
                    refresh_time = next_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                except Exception:
                    pass
            rec = {"week": ws, "used": 0.0, "refresh_time": refresh_time, "synced_at": now}
        rec["used"] = rec.get("used", 0.0) + seconds
        q[account] = rec

def quota_total(account, state=None):
    """Kaggle's published weekly GPU cap for a phone-verified account, in seconds.
    (Not to be confused with the Kaggle API field 'totalTimeAllowed' — that field has a
    constant value across the whole API and does not track the weekly 30 h cap; the
    authoritative weekly figure comes from 'timeUsed' plus the push-side error message.)"""
    return QUOTA_LIMIT

def mark_quota_exhausted(account):
    """Flag an account as weekly-GPU-quota-dead until the next Kaggle weekly reset.
    Call this when Kaggle itself rejects a push with 'Maximum weekly GPU quota ... reached';
    kllm's own usage tally can under-report those accounts, so the flag is authoritative."""
    until = _week_start(time.time()) + WEEK
    with mutate_state() as s:
        acct = (s.setdefault("accounts", {})).setdefault(account, {})
        acct["quota_exhausted_until"] = int(until)
    return until

def quota_left_h(account, state=None):
    """Hours of weekly GPU quota left. 0 while Kaggle itself reported the weekly cap reached
    (accounts.<name>.quota_exhausted_until), even when the local estimate still shows hours."""
    s = state or read_state()
    if ((s.get("accounts") or {}).get(account) or {}).get("quota_exhausted_until", 0) > time.time():
        return 0.0
    return max(0.0, (QUOTA_LIMIT - quota_used(account, s)) / 3600.0)


def quota_used(account, state=None):
    """Seconds used this week by an account (0 if stale week)."""
    s = state or read_state()
    rec = (s.get("quota") or {}).get(account) or {}
    now = time.time()
    if rec.get("refresh_time"):
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(rec["refresh_time"].replace("Z", "+00:00"))
            if now > dt.timestamp():
                return 0.0
            return rec.get("used", 0.0)
        except Exception:
            pass
    if rec.get("synced_at") and (now - rec.get("synced_at") < 6 * 86400):
        return rec.get("used", 0.0)
    if rec.get("week") != _week_start(now):
        return 0.0
    return rec.get("used", 0.0)



def set_config_value(key, value):
    """Set a top-level `key: value` line in config.yaml, keeping comments and layout."""
    import re
    with state_lock():
        txt = CONFIG_FILE.read_text(encoding="utf-8")
        if re.search(rf"^{key}:.*$", txt, flags=re.M):
            txt = re.sub(rf"^{key}:[^\n#]*", f"{key}: {value}  ", txt, count=1, flags=re.M)
        else:
            txt += f"\n{key}: {value}\n"
        CONFIG_FILE.write_text(txt, encoding="utf-8")


def set_section_value(section, key, value):
    """Set `key: value` inside a top-level `section:` block of config.yaml (adds it if missing)."""
    import re
    with state_lock():
        txt = CONFIG_FILE.read_text(encoding="utf-8")
        m = re.search(rf"^{section}:\n((?:[ \t]+.*\n|\s*\n)*)", txt, flags=re.M)
        if not m:
            txt += f"\n{section}:\n  {key}: {value}\n"
        else:
            body = m.group(1)
            if re.search(rf"^[ \t]+{key}:", body, flags=re.M):
                body = re.sub(rf"^([ \t]+){key}:[^\n#]*", rf"\g<1>{key}: {value}  ", body, count=1, flags=re.M)
            else:
                body = f"  {key}: {value}\n" + body
            txt = txt[:m.start(1)] + body + txt[m.end(1):]
        CONFIG_FILE.write_text(txt, encoding="utf-8")


# ---- relay auto-start (first use brings the stack up, no manual `kllm serve`) ----
def relay_base(cfg=None):
    """http://127.0.0.1:<relay_port> for this checkout's config."""
    try:
        c = cfg or load_config()
        return f"http://127.0.0.1:{c.get('relay_port', 8080)}"
    except ConfigMissing:
        return "http://127.0.0.1:8080"


def relay_up(base=None, timeout=4):
    """True when `kllm serve` already answers on this machine."""
    import urllib.request
    try:
        with urllib.request.urlopen((base or relay_base()) + "/__relay/health",
                                    timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_serve(wait_seconds=30, log=None):
    """Start `kllm serve` detached when it isn't running, then wait for it.

    This is what makes "select a kllm model and start a session" work with zero
    manual steps: the relay (~55 MB) comes up on first use, autoscale then starts
    the GPU backend on the first request (~8-10 min cold) and retires it after
    `idle_stop_minutes` idle. Safe to call when already running (no-op: True).
    Returns True when the relay answers, False otherwise."""
    import subprocess
    import urllib.request
    if relay_up():
        return True
    env = dict(os.environ, KLLM_ROOT=str(ROOT),
               PYTHONPATH=os.pathsep.join(filter(None, [str(PKG_ROOT),
                                                         os.environ.get("PYTHONPATH")])))
    cmd = [sys.executable, "-m", "gateway.cli", "serve"]
    try:
        if os.name == "nt":                       # detached, survives this process
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200) | \
                getattr(subprocess, "CREATE_NO_WINDOW", 0x8000000) | \
                getattr(subprocess, "DETACHED_PROCESS", 0x8)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            logf = open(LOG_DIR / "serve.log", "ab")
            try:
                try:
                    subprocess.Popen(cmd, cwd=str(ROOT), env=env, close_fds=False,
                                     stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     creationflags=flags | 0x01000000,
                                     startupinfo=startupinfo)
                except OSError:
                    subprocess.Popen(cmd, cwd=str(ROOT), env=env, close_fds=False,
                                     stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags,
                                     startupinfo=startupinfo)
            finally:
                logf.close()
        else:                                     # new session: no SIGHUP on shell exit
            logf = open(LOG_DIR / "serve.log", "ab")
            subprocess.Popen(cmd, cwd=str(ROOT), env=env, close_fds=True,
                             stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    except Exception as e:
        if log:
            log(f"could not start `kllm serve`: {e}")
        return False
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        time.sleep(1)
        if relay_up():
            return True
    if log:
        log(f"started `kllm serve` but the relay is not answering yet "
            f"(see {LOG_DIR / 'serve.log'})")
    return False


# Per-user install folders that a systemd/launchd service PATH usually lacks (opencode, npm -g, bun, cargo, go).
USER_BIN_DIRS = [".local/bin", ".opencode/bin", ".npm-global/bin", ".bun/bin", ".cargo/bin", ".deno/bin", "go/bin",
                 ".volta/bin", ".yarn/bin"]


def user_path(path=None):
    """PATH plus the per-user install folders that exist, without duplicates."""
    parts = (path if path is not None else os.environ.get("PATH", "")).split(os.pathsep)
    home = pathlib.Path.home()
    for d in [str(home / x) for x in USER_BIN_DIRS] + ["/usr/local/bin", "/opt/homebrew/bin"]:
        if d not in parts and os.path.isdir(d):
            parts.append(d)
    return os.pathsep.join(p for p in parts if p)


def which(name):
    """shutil.which over PATH and the per-user install folders: the relay runs as a service with a short PATH."""
    return shutil.which(name, path=user_path())
