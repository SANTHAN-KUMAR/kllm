"""Kaggle account management shared by `kllm account ...` and the panel API.

add_account() turns pasted Kaggle credentials into a working account: a private credentials dir (mode 600),
a check against Kaggle, and a line in config.yaml's accounts list that says where the credentials are. It accepts
either kind of Kaggle credential:
  - an API token, KGAT_<32 hex> (kaggle.com > Settings > API Tokens), stored as <home>/.kaggle/access_token
  - the contents of a classic kaggle.json ({"username": ..., "key": ...}), stored as <home>/.kaggle/kaggle.json
On Linux/macOS a `kgN` wrapper script is also written for using the kaggle CLI by hand (never needed by kllm).
Only files this module created are ever removed again (remove_account(delete_credentials=True)).
"""
import json, os, pathlib, re, stat, subprocess, sys
from . import common, kaggle, yamledit

TOKEN_RE = re.compile(r"KGAT_[0-9a-fA-F]{32}")
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,23}")
CRED_HELP = ("paste a Kaggle API token (KGAT_..., from kaggle.com > Settings > API Tokens) "
             "or the contents of your kaggle.json")

# Run with the account's credentials in the environment: prints the username, then proves the key works.
PROBE = ("from kaggle.api.kaggle_api_extended import KaggleApi as K\n"
         "a = K(); a.authenticate()\n"
         "print('username: ' + str(a.config_values.get('username') or 'None'), flush=True)\n"
         "a.kernels_list(mine=True, page_size=1)\n")


class AccountError(Exception):
    """Something the user can act on; the message is safe to show (never contains the token)."""


def next_name(cfg):
    names = {a.get("name") for a in cfg.get("accounts") or []}
    n = 1
    while f"kg{n}" in names:
        n += 1
    return f"kg{n}"


def credential_paths(name):
    """(home dir, token file, wrapper script) that `add_account` creates for an account name."""
    home = pathlib.Path.home() / f".kaggle_home{name[2:] if name.startswith('kg') else name}"
    return home, home / ".kaggle" / "access_token", pathlib.Path.home() / ".local" / "bin" / name


def parse_credential(text):
    """('token', 'KGAT_...') or ('json', {'username', 'key'}) from what the user pasted, else AccountError."""
    text = (text or "").lstrip("\ufeff").strip()
    if TOKEN_RE.fullmatch(text):
        return "token", text
    if text.startswith("{"):
        try:
            d = json.loads(text)
        except ValueError:
            d = None
        if isinstance(d, dict) and isinstance(d.get("username"), str) and isinstance(d.get("key"), str) \
                and d["username"].strip() and d["key"].strip():
            return "json", {"username": d["username"].strip(), "key": d["key"].strip()}
        raise AccountError("that kaggle.json is not valid: it needs \"username\" and \"key\"")
    raise AccountError("that is not a Kaggle credential: " + CRED_HELP + " (a token looks like KGAT_<32 hex>)")


def _private_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    old_umask = os.umask(0o077)
    try:
        path.write_text(text, encoding="utf-8")
    finally:
        os.umask(old_umask)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def add_account(token, name=None, cfg=None, run=None, log=lambda *a: None):
    """Create an account from pasted Kaggle credentials (a KGAT_ token or kaggle.json contents).
    Returns {name, user, home, wrapper}. `run` replaces subprocess.run (tests)."""
    run = run or subprocess.run
    cfg = cfg if cfg is not None else common.load_config()
    kind, cred = parse_credential(token)
    name = (name or "").strip() or next_name(cfg)
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if allow and not any(str(name).startswith(p) for p in allow):
        raise AccountError(f"account name '{name}' is not permitted by account_allow_prefixes={allow}")
    if not NAME_RE.fullmatch(name):
        raise AccountError("account name: letters, digits, - and _, starting with a letter (e.g. kg4)")
    if any(a.get("name") == name for a in cfg.get("accounts") or []):
        raise AccountError(f"an account named '{name}' already exists")
    home, tok, wrapper = credential_paths(name)
    if kind == "token":
        cred_file = tok
        _private_write(tok, cred + "\n")
        entry = {"name": name, "kaggle_home": str(home)}
    else:
        cred_file = home / ".kaggle" / "kaggle.json"
        _private_write(cred_file, json.dumps(cred))
        entry = {"name": name, "kaggle_config_dir": str(home / ".kaggle")}
    made = [cred_file]
    if os.name != "nt":
        kaggle_bin = os.path.expanduser(cfg.get("kaggle_bin") or "kaggle")
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(f'#!/bin/sh\nHOME=$HOME/{home.name} exec {kaggle_bin} "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
        made.append(wrapper)
    secret = cred if kind == "token" else cred["key"]

    def fail(msg):
        # rejected credentials must not stay behind
        for p in made:
            try: p.unlink()
            except OSError: pass
        raise AccountError(msg)

    try:
        probe_env = kaggle.account_env(entry)
        probe_env["PYTHONIOENCODING"] = "utf-8"
        res = run([sys.executable, "-c", PROBE], env=probe_env, capture_output=True, text=True,
                  encoding="utf-8", errors="replace", timeout=90)
    except Exception as e:
        fail(f"could not reach Kaggle to check the credentials: {type(e).__name__}")
    out = (res.stdout or "") + "\n" + (res.stderr or "")
    broken = re.search(r"(ModuleNotFoundError|ImportError|SyntaxError|AttributeError)[^\n]*", res.stderr or "")
    if broken and (res.returncode != 0 or not re.search(r"username:\s*(\S+)", res.stdout or "")):
        # kllm's own environment is broken (for example a kaggle release missing a dependency): say so, never
        # blame the user's token
        fail("kllm could not run its Kaggle check (" + broken.group(0)[:160] + "). Your token was not tested. "
             "Update kllm (kllm update) or report this at github.com/SANTHAN-KUMAR/kllm/issues.")
    m = re.search(r"username:\s*(\S+)", res.stdout or "")
    if not m or m.group(1) == "None":
        err_lines = [ln.strip() for ln in (res.stderr or "").splitlines() if ln.strip()]
        detail = err_lines[-1].replace(secret, "***")[:200] if err_lines else ""
        fail("Kaggle did not accept the credentials" + (f": {detail}." if detail else ".")
             + " Check the token is current at kaggle.com/settings (API).")
    user = m.group(1)
    if any(a.get("user") == user for a in cfg.get("accounts") or []):
        fail(f"account {user} is already configured")
    if res.returncode != 0:
        err_lines = [ln.strip() for ln in (res.stderr or out).splitlines() if ln.strip()]
        detail = err_lines[-1].replace(secret, "***")[:200] if err_lines else "unknown error"
        fail("credentials rejected by Kaggle: " + detail)
    entry = {"name": name, "user": user, **{k: v for k, v in entry.items() if k != "name"}}
    try:
        with common.state_lock():
            txt = common.CONFIG_FILE.read_text(encoding="utf-8")
            yamledit.atomic_write(common.CONFIG_FILE, yamledit.append_account(txt, entry))
    except yamledit.EditError as e:
        fail(str(e))
    log(f"added {name} = {user}")
    return {"name": name, "user": user, "home": str(home), "wrapper": str(wrapper) if wrapper in made else None}


def set_disabled(name, disabled):
    with common.state_lock():
        txt = common.CONFIG_FILE.read_text(encoding="utf-8")
        new = yamledit.edit_account(txt, name, set_fields={"disabled": True}) if disabled \
            else yamledit.edit_account(txt, name, drop_fields=("disabled",))
        if new != txt:
            yamledit.atomic_write(common.CONFIG_FILE, new)


def removable_credentials(name):
    """Credential files kllm itself created for this account (and that still exist)."""
    home, tok, wrapper = credential_paths(name)
    out = []
    if home.is_dir() and home.name.startswith(".kaggle_home"):
        out.append(home)
    if wrapper.is_file() and "exec" in wrapper.read_text(errors="ignore", encoding="utf-8")[:400] and home.name in wrapper.read_text(errors="ignore", encoding="utf-8"):
        out.append(wrapper)
    return out


def remove_account(name, delete_credentials=False):
    """Drop the account from config.yaml. Credential files are kept unless delete_credentials."""
    import shutil
    s = common.read_state()
    if any(b["account"] == name for b in s["backends"]) or name in s.get("pending", {}):
        raise AccountError(f"{name} is running a kernel: stop it first")
    deleted = []
    paths = removable_credentials(name) if delete_credentials else []
    with common.state_lock():
        txt = common.CONFIG_FILE.read_text(encoding="utf-8")
        try:
            new = yamledit.remove_account(txt, name)
        except yamledit.EditError as e:
            raise AccountError(str(e)) from None
        yamledit.atomic_write(common.CONFIG_FILE, new)
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try: p.unlink()
            except OSError: continue
        deleted.append(str(p))
    return deleted
