"""Kaggle account management shared by `kllm account ...` and the panel API.

add_account() turns a pasted Kaggle API token into a working account: a private credentials dir
(mode 600), a `kgN` wrapper script, a check against Kaggle, and a line in config.yaml's accounts list.
Only files this module created are ever removed again (remove_account(delete_credentials=True)).
"""
import os, pathlib, re, stat, subprocess
from . import common, yamledit

TOKEN_RE = re.compile(r"KGAT_[0-9a-fA-F]{32}")
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,23}")


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


def add_account(token, name=None, cfg=None, run=None, log=lambda *a: None):
    """Create an account from a Kaggle API token. Returns {name, user, home, wrapper}.
    `run` replaces subprocess.run (tests)."""
    run = run or subprocess.run
    cfg = cfg if cfg is not None else common.load_config()
    token = (token or "").strip()
    if not TOKEN_RE.fullmatch(token):
        raise AccountError("expected a Kaggle API token like KGAT_<32 hex> (Kaggle > Settings > API Tokens)")
    name = (name or "").strip() or next_name(cfg)
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if allow and not any(str(name).startswith(p) for p in allow):
        raise AccountError(f"account name '{name}' is not permitted by account_allow_prefixes={allow}")
    if not NAME_RE.fullmatch(name):
        raise AccountError("account name: letters, digits, - and _, starting with a letter (e.g. kg4)")
    if any(a.get("name") == name for a in cfg.get("accounts") or []):
        raise AccountError(f"an account named '{name}' already exists")
    home, tok, wrapper = credential_paths(name)
    (home / ".kaggle").mkdir(parents=True, exist_ok=True)
    old_umask = os.umask(0o077)
    try:
        tok.write_text(token + "\n", encoding="utf-8")
    finally:
        os.umask(old_umask)
    tok.chmod(stat.S_IRUSR | stat.S_IWUSR)
    kaggle_bin = os.path.expanduser(cfg.get("kaggle_bin") or "kaggle")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(f'#!/bin/sh\nHOME=$HOME/{home.name} exec {kaggle_bin} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)

    def fail(msg):
        # a rejected token must not leave credentials behind
        for p in (tok, wrapper):
            try: p.unlink()
            except OSError: pass
        raise AccountError(msg)

    out = run([str(wrapper), "config", "view"], capture_output=True, text=True, timeout=60).stdout
    m = re.search(r"username:\s*(\S+)", out or "")
    if not m or m.group(1) == "None":
        fail("Kaggle did not accept the token (no username).")
    user = m.group(1)
    if any(a.get("user") == user for a in cfg.get("accounts") or []):
        fail(f"account {user} is already configured")
    chk = run([str(wrapper), "kernels", "list", "-m", "--page-size", "1"], capture_output=True, text=True, timeout=60)
    if chk.returncode != 0:
        fail("token rejected by Kaggle: " + ((chk.stdout or "") + (chk.stderr or "")).strip()[-200:].replace(token, "***"))
    with common.state_lock():
        txt = common.CONFIG_FILE.read_text(encoding="utf-8")
        yamledit.atomic_write(common.CONFIG_FILE, yamledit.append_account(txt, {"name": name, "user": user}))
    log(f"added {name} = {user}")
    return {"name": name, "user": user, "home": str(home), "wrapper": str(wrapper)}


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
