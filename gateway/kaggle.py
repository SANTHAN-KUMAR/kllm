"""Thin wrapper over the kaggle CLI, one credential set per configured account.

How an account's credentials are found (first match wins):
  1. `kaggle_config_dir: DIR`   a directory containing kaggle.json  -> KAGGLE_CONFIG_DIR=DIR
  2. `kaggle_json: FILE`        a kaggle.json (any file name)       -> KAGGLE_USERNAME / KAGGLE_KEY
  3. `kaggle_home: DIR`         a home dir containing .kaggle/access_token (Bearer KGAT_ token)
                                 -> KAGGLE_API_TOKEN=<token> (works on Windows, no wrapper needed)
  4. `access_token: FILE`       a file containing a KGAT_ Bearer token -> KAGGLE_API_TOKEN=<token>
  5. legacy wrapper script `<account name>` (kg1..kgN) on PATH or in ~/.local/bin, which is
     expected to run the kaggle CLI with that account's credentials.
No wrapper scripts are needed with 1-4.
"""
import subprocess, re, os, sys, shutil, tempfile, json, pathlib, threading
from . import common

WINDOWS = os.name == "nt"


class NoCredentials(RuntimeError):
    pass


def _expand(p):
    s = os.path.expandvars(str(p))
    if s.startswith("~") and "HOME" in os.environ:
        s = os.environ["HOME"] + s[1:]
    path = pathlib.Path(s).expanduser()
    if not path.is_absolute():
        path = common.ROOT / path
    return path


def account_cfg(account, cfg=None):
    """The accounts: entry for `account` (dict), {} when it is not configured."""
    cfg = cfg if cfg is not None else common.load_config()
    for a in cfg.get("accounts") or []:
        if a.get("name") == account:
            return a
    return {}


def _wrapper_path(name):
    """Legacy kgN wrapper: ~/.local/bin first (what the old install used), then PATH."""
    p = pathlib.Path.home() / ".local" / "bin" / name
    if p.exists():
        return str(p)
    found = shutil.which(name)
    if found:
        return found
    if WINDOWS:
        for ext in ("", ".bat", ".cmd", ".exe", ".sh"):
            found = shutil.which(name + ext)
            if found:
                return found
        for p_dir in os.environ.get("PATH", "").split(os.pathsep):
            if p_dir:
                cand = pathlib.Path(p_dir) / name
                if cand.is_file():
                    return str(cand)
    return None


def kaggle_command(cfg=None):
    """argv prefix that runs the kaggle CLI: `kaggle_bin` from config, else the `kaggle` script
    installed next to this interpreter (pipx / venv), else PATH, else the package itself."""
    cfg = cfg if cfg is not None else common.load_config()
    configured = cfg.get("kaggle_bin")
    if configured and _expand(configured).exists():
        return [str(_expand(configured))]
    exe = "kaggle.exe" if WINDOWS else "kaggle"
    for cand in (pathlib.Path(sys.executable).parent / exe, pathlib.Path(sys.executable).parent / "Scripts" / exe):
        if cand.exists():
            return [str(cand)]
    found = shutil.which("kaggle")
    if found:
        return [found]
    return [sys.executable, "-c", "from kaggle.cli import main; main()"]


def resolve_credentials(acct):
    """How to authenticate one account: ("config_dir", dir) | ("json", file) |
    ("home", dir) | ("token", file) | ("wrapper", path).
    Raises NoCredentials with a message that says what to configure."""
    name = acct.get("name", "?")
    if acct.get("kaggle_config_dir"):
        d = _expand(acct["kaggle_config_dir"])
        if not (d / "kaggle.json").exists():
            raise NoCredentials(f"{name}: kaggle_config_dir {d} has no kaggle.json")
        return "config_dir", str(d)
    if acct.get("kaggle_json"):
        f = _expand(acct["kaggle_json"])
        if not f.is_file():
            raise NoCredentials(f"{name}: kaggle_json {f} not found")
        return "json", str(f)
    if acct.get("kaggle_home"):
        d = _expand(acct["kaggle_home"])
        for cand in (d / ".kaggle" / "access_token", d / "access_token"):
            if cand.is_file():
                return "home", str(d)
        raise NoCredentials(f"{name}: kaggle_home {d} has no .kaggle/access_token")
    if acct.get("access_token"):
        f = _expand(acct["access_token"])
        if not f.is_file():
            raise NoCredentials(f"{name}: access_token {f} not found")
        return "token", str(f)
    w = _wrapper_path(name)
    if w:
        return "wrapper", w
    raise NoCredentials(f"{name}: no credentials. Set `kaggle_json: /path/to/kaggle.json` (or "
                        f"`kaggle_config_dir:` / `kaggle_home:` / `access_token:`) on this account in config.yaml, "
                        f"or provide a '{name}' wrapper script")


def _read_token_file(path):
    """Read a KGAT_ Bearer token from a file (strips whitespace/newline)."""
    try:
        raw = pathlib.Path(path).read_bytes()
        tok = None
        for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                candidate = raw.decode(enc).strip()
                if candidate:
                    tok = candidate.split()[0]
                    if tok.startswith("KGAT_"):
                        break
            except Exception:
                continue
        if tok is None:
            tok = raw.decode("utf-8", errors="replace").strip().split()[0]
    except (OSError, IndexError) as e:
        raise NoCredentials(f"cannot read token from {path}: {e}")
    if not tok.startswith("KGAT_"):
        raise NoCredentials(f"token in {path} does not look like KGAT_<32 hex>")
    return tok



def _isolated_home():
    """Empty HOME for accounts that authenticate via KAGGLE_USERNAME/KEY or kaggle.json.
    The kaggle client prefers ~/.kaggle/access_token over those, so without this every such
    account silently acts as whichever account owns the global token (usage, pushes, all of it)."""
    d = common.ROOT / "state" / ".kaggle_empty_home"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def account_env(acct, base_env=None):
    """Environment for the kaggle CLI of one account (credentials never go on the command line)."""
    env = dict(os.environ if base_env is None else base_env)
    kind, where = resolve_credentials(acct)
    if kind in ("config_dir", "json"):
        env["HOME"] = _isolated_home()
        if WINDOWS:
            env["USERPROFILE"] = env["HOME"]
    if kind == "config_dir":
        env["KAGGLE_CONFIG_DIR"] = where
        env.pop("KAGGLE_USERNAME", None); env.pop("KAGGLE_KEY", None)
        env.pop("KAGGLE_API_TOKEN", None)
    elif kind == "json":
        try:
            j = json.loads(pathlib.Path(where).read_text(encoding="utf-8"))
            env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = j["username"], j["key"]
        except (ValueError, KeyError, OSError) as e:
            raise NoCredentials(f"{acct.get('name')}: cannot read username/key from {where}: {e}")
        env.pop("KAGGLE_API_TOKEN", None)
    elif kind in ("home", "token"):
        base = pathlib.Path(where)
        tok_file = base / ".kaggle" / "access_token" if kind == "home" else base
        if kind == "home" and not tok_file.is_file():
            tok_file = base / "access_token"
        env["KAGGLE_API_TOKEN"] = _read_token_file(tok_file)
        env.pop("KAGGLE_USERNAME", None); env.pop("KAGGLE_KEY", None); env.pop("KAGGLE_CONFIG_DIR", None)
    # Optional egress isolation: route this account's API traffic through a proxy so
    # the source IP Kaggle sees is not ours. Inert unless set (per-account or process-wide).
    # Accepts http(s):// proxies or socks5:// (Tor) — the latter needs `pip install socksio`.
    proxy = acct.get("egress_proxy") or os.environ.get("KLLM_EGRESS_PROXY")
    if proxy:
        p = str(proxy)
        vars_ = ("SOCKS_PROXY", "socks_proxy") if p.lower().startswith("socks") else \
                ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
        for _v in vars_:
            env[_v] = p
    return env


def build_command(acct, args, cfg=None):
    """(argv, env) to run `kaggle <args>` as this account."""
    kind, where = resolve_credentials(acct)
    if kind == "wrapper":
        return [where, *map(str, args)], dict(os.environ)
    return [*kaggle_command(cfg), *map(str, args)], account_env(acct)


def run(account, *args, timeout=120, check=False):
    """Run `kaggle <args>` as `account` and return (rc, stdout+stderr)."""
    cfg = common.load_config()
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if allow and not any(str(account).startswith(p) for p in allow):
        err = f"Account '{account}' is not permitted by account_allow_prefixes={allow}"
        if check:
            raise RuntimeError(err)
        return 127, err
    try:
        cmd, env = build_command(account_cfg(account, cfg) or {"name": account}, args, cfg)
    except NoCredentials as e:
        if check:
            raise RuntimeError(str(e))
        return 127, str(e)
    try:
        sub_env = dict(env)
        sub_env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, env=sub_env)
        out = (r.stdout or "") + (r.stderr or "")
        if check and r.returncode != 0:
            raise RuntimeError(f"{account} {' '.join(map(str,args))} failed:\n{out}")
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except FileNotFoundError as e:
        return 127, f"cannot run kaggle CLI: {e}. Install it: pip install kaggle"

STATUS_RE = re.compile(r"\b(RUNNING|COMPLETE|ERROR|CANCEL[A-Z]*|QUEUED)\b")

def kernel_status(account, ref):
    """Return RUNNING/QUEUED/COMPLETE/ERROR/CANCELED/UNKNOWN for user/slug ref."""
    rc, out = run(account, "kernels", "status", ref, timeout=40)
    m = STATUS_RE.search(out)
    return m.group(1) if m else "UNKNOWN"

def push_kernel(account, kernel_dir):
    """kaggle kernels push -p <dir>. Returns (ok, message)."""
    rc, out = run(account, "kernels", "push", "-p", kernel_dir, timeout=180)
    ok = "successfully pushed" in out.lower()
    return ok, out.strip().splitlines()[-1] if out.strip() else "no output"

def kernel_output(account, ref, dest, pattern=None):
    args = ["kernels", "output", ref, "-p", dest, "-q", "-o"]
    if pattern:
        args += ["--file-pattern", pattern]
    return run(account, *args, timeout=600)

def delete_kernel(account, ref):
    """Hard-stop: delete the kernel, which frees its running GPU session immediately.
    (Kaggle has no 'cancel'; delete is the only forced free. We re-push on next launch.)"""
    return run(account, "kernels", "delete", ref, "-y", timeout=60)

def whoami(account):
    rc, out = run(account, "kernels", "list", "-m", "--page-size", "1", timeout=40)
    # author column gives a hint; return raw-ish
    return rc == 0, out


def parse_duration_seconds(val):
    """Parse Kaggle duration strings like '61661.230886.0s', '21600s', '0s' into float seconds."""
    s = str(val or "0").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0


_KAGGLE_API_LOCK = threading.Lock()


def fetch_account_quota(acct, cfg=None):
    """Fetch live GPU quota usage from KaggleApi for one account.
    Returns dict: {'ok': True, 'used_s': ..., 'used_h': ..., 'refresh_time': ..., 'tpu_used_s': ...}
    or {'ok': False, 'error': ...}.
    NOTE: the raw response's 'totalTimeAllowed' is constant across accounts and is NOT the weekly
    cap (the authoritative weekly cap — 30 h for phone-verified accounts — is enforced by Kaggle's
    own kernel-launch rejection, not reported by this endpoint)."""
    cfg = cfg if cfg is not None else common.load_config()
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    acct_name = acct.get("name") if isinstance(acct, dict) else str(acct)
    if allow and not any(str(acct_name).startswith(p) for p in allow):
        return {"ok": False, "account": acct_name, "error": f"Account '{acct_name}' is not permitted by account_allow_prefixes={allow}"}
    if not isinstance(acct, dict):
        acct = account_cfg(str(acct), cfg) or {"name": str(acct)}
    with _KAGGLE_API_LOCK:
        base_env = os.environ.copy()
        try:
            kind, _ = resolve_credentials(acct)
            if kind == "wrapper":
                # the wrapper sets its own HOME for the CLI; in-process we would silently
                # fall through to the global ~/.kaggle token and report another account's usage
                return {"ok": False, "account": acct.get("name"),
                        "error": "wrapper-only account: live quota unavailable (configure kaggle_home/kaggle_json)"}
            env = account_env(acct)
            os.environ.clear()
            os.environ.update(env)
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            res = api.quota_view()
            d = res.to_dict() if hasattr(res, "to_dict") else res
            gpu = d.get("gpuQuota") or {}
            tpu = d.get("tpuQuota") or {}
            used_s = parse_duration_seconds(gpu.get("timeUsed"))
            total_s = parse_duration_seconds(gpu.get("totalTimeAllowed"))
            tpu_used_s = parse_duration_seconds(tpu.get("timeUsed"))
            refresh_time = d.get("quotaRefreshTime")
            return {
                "ok": True,
                "account": acct.get("name"),
                "used_s": used_s,
                "used_h": round(used_s / 3600, 2),
                "refresh_time": refresh_time,
            }
        except Exception as e:
            return {"ok": False, "account": acct.get("name"), "error": str(e)}
        finally:
            os.environ.clear()
            os.environ.update(base_env)


