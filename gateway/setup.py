"""`kllm init`: create config.yaml (from config.example.yaml), accounts, the rendezvous secret (optional gist) and a
gateway key, and check prerequisites. Never launches a Kaggle kernel (no GPU quota is used)."""
import getpass, importlib.util, json, re, secrets, shutil, sys, pathlib, urllib.request, urllib.error
from . import common, rendezvous

GIST_HELP = ("Create a token at https://github.com/settings/tokens/new (classic) with ONLY the "
             "'gist' scope, then:  kllm init --github-token <token>   (the private gist is created for you).\n"
             "Or by hand:  gh gist create --secret <file>  and put its id in config.yaml (gist_id).")


# ---- prerequisites ---------------------------------------------------------------------
def _reachable(url, timeout=5):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "kllm"}),
                               timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True            # the server answered: network is fine
    except Exception:
        return False


def prerequisites(network=True):
    """[(name, ok, detail, fix)] — python version, kaggle CLI, network, optional claude."""
    out = []
    v = sys.version_info
    out.append(("python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}", "install Python 3.10+"))
    cmd = None
    try:
        from . import kaggle
        cmd = kaggle.kaggle_command(common.load_config() if common.CONFIG_FILE.exists() else {})
    except Exception:
        pass
    have = bool(importlib.util.find_spec("kaggle")) or bool(cmd and len(cmd) == 1)
    out.append(("kaggle CLI", have, " ".join(cmd[:1]) if have and cmd else "not found",
                "pip install kaggle   (or: pipx install kllm-gateway / uv tool install kllm-gateway, which pulls it in)"))
    if network:
        out.append(("network: ntfy.sh", _reachable("https://ntfy.sh/v1/health"), "",
                    "kernels report their URL through ntfy.sh (set ntfy_server for your own); check your connection/proxy"))
        out.append(("network: kaggle.com", _reachable("https://www.kaggle.com"), "", "check your connection/proxy"))
    out.append(("claude (optional)", bool(shutil.which("claude")), shutil.which("claude") or "not installed",
                "only needed for claude-kllm"))
    return out


# ---- config rendering ------------------------------------------------------------------
def kaggle_json_user(path):
    """username from a kaggle.json (None if unreadable)."""
    try:
        return json.loads(pathlib.Path(path).expanduser().read_text(encoding="utf-8"))["username"]
    except Exception:
        return None


def _set(txt, key, value):
    pat = re.compile(rf"^{re.escape(key)}:[^\n#]*", re.M)
    if pat.search(txt):
        return pat.sub(lambda m: f"{key}: {value}  ", txt, count=1)
    return txt.rstrip("\n") + f"\n{key}: {value}\n"


def render_config(example, accounts, gist_id=None, gist_user=None, gist_token=None, api_key=None,
                  rendezvous_secret=None):
    """Fill config.example.yaml text. accounts: [{name, user, kaggle_json}], keeping all comments.
    accounts == [] writes an empty list (a starter config: the panel's welcome page adds the first account)."""
    txt = example
    if accounts == []:
        txt = re.sub(r"^accounts:\n(?:[ \t]+.*\n|[ \t]*\n)*?(?=^\S|\Z)", lambda m: "accounts: []\n\n", txt, count=1, flags=re.M)
    elif accounts:
        lines = []
        for a in accounts:
            lines.append("  - {" + ", ".join(f"{k}: {json.dumps(str(a[k]))}" for k in
                         ("name", "user", "kaggle_json", "access_token", "kaggle_home") if a.get(k)) + "}")
        block = "accounts:\n" + "\n".join(lines) + "\n"
        txt, n = re.subn(r"^accounts:\n(?:[ \t]+.*\n|[ \t]*\n)*?(?=^\S|\Z)", lambda m: block + "\n", txt, count=1, flags=re.M)
        if not n:
            txt += "\n" + block
    # the example's default kaggle_bin points at one person's venv: let kllm auto-detect it
    txt = re.sub(r"^kaggle_bin:.*$", "# kaggle_bin: /path/to/kaggle   # optional: auto-detected next to kllm / on PATH",
                 txt, flags=re.M)
    for k, v in (("gist_id", gist_id), ("gist_user", gist_user), ("gist_token", gist_token),
                 ("gateway_api_key", api_key), ("rendezvous_secret", rendezvous_secret)):
        if v:
            txt = _set(txt, k, v)
    if gist_id and gist_token:
        txt = _set(txt, "rendezvous", "gist")
    return txt


def _example_text():
    return common.data_path("config.example.yaml").read_text(encoding="utf-8")


def parse_account(spec, index):
    """'PATH' | 'NAME=PATH' -> account dict. PATH is a kaggle.json, or an access_token file (KGAT_ token)."""
    if isinstance(spec, dict):
        return dict(spec, name=spec.get("name") or f"kg{index}")
    name, _, path = spec.partition("=") if "=" in spec and not spec.startswith(("/", "~", ".")) else ("", "", spec)
    path = str(pathlib.Path(path).expanduser().resolve())
    if pathlib.Path(path).name == "access_token":
        user = token_user(path)
        if not user:
            raise ValueError(f"Kaggle did not accept the token in {path} (or it could not be reached)")
        return {"name": name or f"kg{index}", "user": user, "access_token": path}
    user = kaggle_json_user(path)
    if not user:
        raise ValueError(f"cannot read a username from {path} (expected the kaggle.json you downloaded "
                         "from kaggle.com -> Settings -> API -> Create New Token)")
    return {"name": name or f"kg{index}", "user": user, "kaggle_json": path}


# ---- zero-config: find the Kaggle credentials the user already has ---------------------------------------------
def token_user(token_file=None, token=None, timeout=60):
    """Kaggle username for a KGAT_ access token (asks Kaggle; None if rejected/offline)."""
    import os, subprocess
    try:
        tok = token or pathlib.Path(token_file).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    env = dict(os.environ, KAGGLE_API_TOKEN=tok)
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        env.pop(k, None)
    code = ("from kaggle.api.kaggle_api_extended import KaggleApi as K\n"
            "a = K(); a.authenticate(); print('KLLM_USER=' + str(a.config_values.get('username') or ''))")
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    m = re.search(r"KLLM_USER=(\S+)", r.stdout or "")
    return m.group(1) if m else None


def detect_account(root=None, log=print):
    """The Kaggle credentials already on this machine, as an account dict, or None. In order:
    $KAGGLE_CONFIG_DIR/kaggle.json, ~/.kaggle/kaggle.json, $KAGGLE_USERNAME+$KAGGLE_KEY,
    ~/.kaggle/access_token, $KAGGLE_API_TOKEN. Env-only credentials are copied into kllm's own config dir
    (mode 600) so the relay service, which does not see your shell's environment, can use them."""
    import os
    root = pathlib.Path(root or common.ROOT)
    home = pathlib.Path(os.path.expanduser("~"))
    cands = []
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        cands.append(pathlib.Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json")
    cands.append(home / ".kaggle" / "kaggle.json")
    for c in cands:
        if c.is_file() and kaggle_json_user(c):
            return {"name": "kg1", "user": kaggle_json_user(c), "kaggle_json": str(c)}
    store = root / "kaggle_accounts" / "kg1" / ".kaggle"

    def _private_write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        f = store / "kaggle.json"
        _private_write(f, json.dumps({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}))
        return {"name": "kg1", "user": os.environ["KAGGLE_USERNAME"], "kaggle_json": str(f)}
    tok_file = home / ".kaggle" / "access_token"
    if tok_file.is_file():
        user = token_user(tok_file)
        if user:
            return {"name": "kg1", "user": user, "access_token": str(tok_file)}
        log(f"  ! Kaggle did not accept {tok_file}")
    if os.environ.get("KAGGLE_API_TOKEN"):
        user = token_user(token=os.environ["KAGGLE_API_TOKEN"])
        if user:
            f = store / "access_token"
            _private_write(f, os.environ["KAGGLE_API_TOKEN"].strip())
            return {"name": "kg1", "user": user, "access_token": str(f)}
    return None


NO_KAGGLE_HELP = ("No Kaggle credentials found. Get them once (a phone-verified Kaggle account is needed for GPUs):\n"
                  "  kaggle.com -> Settings -> API -> Create New Token, save it as ~/.kaggle/kaggle.json\n"
                  "  (or ~/.kaggle/access_token for the newer token format), then run the command again.")


OLD_FALLBACK_TEMPLATE = ("# Last-resort fallback: your existing local llama-server (abliterated models/serve.sh).\n"
                         "# If every Kaggle backend is down, the relay falls back here so you are never fully offline.\n"
                         "local_fallback:\n  enabled: true")


def migrate_local_fallback(cfg_path):
    """Configs written from the 0.3.0/0.3.1 template had local_fallback switched on, pointing at a llama-server most
    machines do not run; requests then failed with 'unreachable (local: ConnectError)'. Switch it off, but only when
    the block is still exactly that template (a hand-written or deliberately enabled fallback is left alone)."""
    try:
        txt = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return False
    norm = txt.replace("\r\n", "\n")
    if OLD_FALLBACK_TEMPLATE not in norm:
        return False
    new = norm.replace(OLD_FALLBACK_TEMPLATE,
                       "# Optional last-resort fallback: a llama-server you run yourself on this machine. When enabled and no\n"
                       "# Kaggle backend is up, requests go there instead of failing.\n"
                       "local_fallback:\n  enabled: false                    # true only if you run your own llama-server at the url below",
                       1)
    cfg_path.write_text(new, encoding="utf-8")
    return True


def ensure_config(root=None, log=print, allow_empty=False):
    """Create config.yaml without questions when it is missing (first `kllm` command), and fill in
    rendezvous_secret when an older config lacks it. Returns True when a usable config exists.
    With no Kaggle credentials on this machine: allow_empty=True writes a starter config with no accounts
    (the panel's welcome page adds one); otherwise it explains the one step and returns False."""
    from . import sealbox
    root = pathlib.Path(root or common.ROOT)
    cfg_path = root / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return True
        if not rendezvous._real(cfg.get("rendezvous_secret")):
            txt = _set(cfg_path.read_text(encoding="utf-8"), "rendezvous_secret", sealbox.new_secret())
            cfg_path.write_text(txt, encoding="utf-8")
            log("kllm: generated rendezvous_secret in config.yaml")
        if migrate_local_fallback(cfg_path):
            log("kllm: turned off local_fallback in config.yaml (0.3.0/0.3.1 enabled it by default for a server "
                "most installs do not have)")
        return True
    acct = detect_account(root, log=log)
    if not acct:
        if allow_empty:
            run_init(root=root, accounts=[], network=False, log=lambda *a: None, allow_empty=True)
            log(f"kllm: first run: wrote {cfg_path} (no Kaggle account yet: add one on the setup page)")
            return True
        log(NO_KAGGLE_HELP)
        return False
    log(f"kllm: first run: using Kaggle account {acct['user']} ({acct.get('kaggle_json') or acct.get('access_token')})")
    run_init(root=root, accounts=[acct], network=False, log=lambda *a: None)
    log(f"kllm: wrote {cfg_path}")
    return True


# ---- the wizard ------------------------------------------------------------------------
def run_init(root=None, accounts=(), github_token=None, gist_id=None, gist_user=None, no_gist=False,
             api_key=None, force=False, interactive=False, network=True, log=print,
             ask=input, ask_secret=getpass.getpass, allow_empty=False):
    """Create <root>/config.yaml. Returns a summary dict. `accounts`: list of specs (see parse_account)."""
    root = pathlib.Path(root or common.ROOT)
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "config.yaml"
    log("== prerequisites ==")
    bad = 0
    for name, ok, detail, fix in prerequisites(network):
        req = "optional" not in name
        bad += (not ok) and req
        log(f"  {'OK  ' if ok else ('FAIL' if req else 'skip')} {name:24s} {detail}" + ("" if ok else f"\n       -> {fix}"))
    if cfg_path.exists() and not force:
        if interactive and ask(f"{cfg_path} exists. Overwrite (a backup is kept)? [y/N] ").strip().lower() == "y":
            force = True
        else:
            raise FileExistsError(f"{cfg_path} already exists (use --force to overwrite; a .bak is kept)")

    accts = [parse_account(s, i + 1) for i, s in enumerate(accounts)]
    if not accts:
        found = detect_account(root, log=log)
        if found:
            log(f"\n== Kaggle account ==\n  found {found['user']} ({found.get('kaggle_json') or found.get('access_token')})")
            accts.append(found)
    if interactive:
        log("\n== Kaggle accounts ==\nCredentials: kaggle.com -> Settings -> API -> Create New Token.\n"
            "Follow Kaggle's terms (one account per person); add more only for accounts you may use.")
        while True:
            p = ask(f"Path to kaggle.json for account {len(accts) + 1} (blank = done): ").strip()
            if not p:
                if accts: break
                log("  at least one account is needed"); continue
            try:
                accts.append(parse_account(p.strip("'\""), len(accts) + 1))
            except ValueError as e:
                log(f"  ! {e}")
    if not accts and not allow_empty:
        raise ValueError(NO_KAGGLE_HELP + "\n  (or pass --account /path/to/kaggle.json)")

    # rendezvous: the zero-setup ntfy channel unless a gist was asked for (--github-token / --gist-id)
    token = (github_token or "").strip() or None
    gist_token = None
    if token:
        gist_token = token
        if not gist_id and not no_gist:
            try:
                gist_id, login = rendezvous.create_gist({}, token)
                gist_user = gist_user or login
                log(f"  created private gist {gist_id} (owner {login})")
            except Exception as e:
                log(f"  ! {e}\n  {GIST_HELP}")
        elif not gist_user:
            gist_user = rendezvous.whoami({}, token)
    if token and not gist_id:
        log("  gist not configured; using the default ntfy channel.\n  " + GIST_HELP.replace("\n", "\n  "))

    key_existing = secret_existing = None
    if cfg_path.exists():
        try:
            import yaml
            from . import auth
            old = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            key_existing = auth.real_key(old)
            secret_existing = old.get("rendezvous_secret") if rendezvous._real(old.get("rendezvous_secret")) else None
        except Exception:
            pass
    api_key = api_key or key_existing or "sk-kllm-" + secrets.token_urlsafe(24)
    from . import sealbox
    rv_secret = secret_existing or sealbox.new_secret()      # kept across --force so running kernels still match

    if cfg_path.exists():
        shutil.copyfile(cfg_path, cfg_path.with_suffix(".yaml.bak"))
    txt = render_config(_example_text(), accts if accts else ([] if allow_empty else accts),
                        gist_id, gist_user, gist_token, api_key, rv_secret)
    cfg_path.write_text(txt, encoding="utf-8")
    try:
        cfg_path.chmod(0o600)          # holds tokens
    except OSError:
        pass
    if not (root / "models.yaml").exists():
        src = common.data_path("models.yaml")
        if src.exists() and src != root / "models.yaml":
            shutil.copyfile(src, root / "models.yaml")
    log(f"\nwrote {cfg_path}  ({len(accts)} account(s), gateway_api_key set)")
    log("next:  kllm install-service   (or: kllm serve)   then   kllm up   and   kllm doctor")
    return {"config": str(cfg_path), "accounts": [a["name"] for a in accts], "gist_id": gist_id,
            "gateway_api_key": api_key, "prereq_failures": bad}
