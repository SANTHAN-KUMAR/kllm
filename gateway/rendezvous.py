"""URL/status handoff between the Kaggle kernel and this PC.

Two backends (config `rendezvous`):
  ntfy (default, zero setup)  sealed messages on unguessable ntfy topics. Needs only `rendezvous_secret`, which
                              `kllm init` (or the first `kllm serve`) generates. See gateway/sealbox.py.
  gist                        a private GitHub Gist you own (needs a gist-scoped token). Details below.
Only kernel status and the tunnel URL travel here; prompts and replies never do (they go through the tunnel,
guarded by the per-kernel API key).

Gist backend:

The gist is the user's own, so this is just reading/writing your own GitHub data — no
third-party relay. The PC talks to the GitHub gist REST API directly (urllib, no extra
dependencies) with a token from config; the `gh` CLI is only a fallback when no token is set.
The kernel writes with a gist-scoped token (see config: gist_token).

PC-side token, first one found:  github_token, gist_token (config), $GH_TOKEN, $GITHUB_TOKEN.
(Placeholder values like ghp_YOUR_... count as unset.)  API base: $KLLM_GITHUB_API or
config `github_api` (default https://api.github.com) — mainly for tests/GitHub Enterprise.

Files in the gist:
  s-<account>.json   kernel -> PC   latest status/url for that account (current value)
  c-<account>.json   PC -> kernel   latest control command (e.g. stop)
"""
import json, os, subprocess, time, urllib.parse, urllib.request, urllib.error

from . import sealbox

API = "https://api.github.com"
_etag = {}      # gist_id -> (etag, parsed gist); a 304 costs no rate limit


def _real(tok):
    return bool(tok) and isinstance(tok, str) and "YOUR_" not in tok and "CHANGE_ME" not in tok and not tok.startswith("<")


def token(cfg):
    """The GitHub token the PC side should use, or None (then the `gh` CLI is tried)."""
    for t in (cfg.get("github_token"), cfg.get("gist_token"),
              os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_TOKEN")):
        if _real(t):
            return t.strip()
    return None


def _base(cfg):
    return (os.environ.get("KLLM_GITHUB_API") or cfg.get("github_api") or API).rstrip("/")


def _http(cfg, method, path, body=None, tok=None, timeout=25, etag=None):
    """-> (status, parsed-json | None, response headers). Never raises on HTTP errors."""
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "kllm"}
    data = None
    if body is not None:
        data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
    if etag:
        h["If-None-Match"] = etag
    req = urllib.request.Request(_base(cfg) + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw.strip() else {}), r.headers
    except urllib.error.HTTPError as e:
        return e.code, None, e.headers
    except Exception:
        return 0, None, {}


def _gh_json(args, timeout=25):
    """Fallback: the authenticated `gh` CLI."""
    try:
        r = subprocess.run(["gh", "api", *args], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout) if r.stdout.strip() else {}
    except Exception:
        return None


def _get_gist(cfg):
    gid = cfg.get("gist_id")
    if not _real(gid):
        return None
    tok = token(cfg)
    if not tok:
        return _gh_json([f"gists/{gid}"])
    prev = _etag.get(gid)
    st, d, hdr = _http(cfg, "GET", f"/gists/{gid}", tok=tok, etag=prev[0] if prev else None)
    if st == 304 and prev:
        return prev[1]
    if st == 200 and d:
        if hdr.get("ETag"):
            _etag[gid] = (hdr["ETag"], d)
        return d
    return None


# ---- backend selection ---------------------------------------------------------------------------------------------
def mode(cfg):
    """'gist' when a gist is configured and not overridden (`rendezvous: ntfy`); otherwise the zero-setup ntfy
    channel. A config with a real gist_id and no `rendezvous:` key predates ntfy and keeps using its gist."""
    want = cfg.get("rendezvous")
    if want != "ntfy" and _real(cfg.get("gist_id")):
        return "gist"
    return "ntfy"


def ntfy_server(cfg):
    return (os.environ.get("KLLM_NTFY_SERVER") or cfg.get("ntfy_server") or "https://ntfy.sh").rstrip("/")


# ---- ntfy backend --------------------------------------------------------------------------------------------------
# One status topic (kernel -> PC) and one control topic (PC -> kernel), both derived from rendezvous_secret.
# The PC keeps the newest event per account and polls incrementally (since=<last message id>), so a steady
# poll costs one small GET. ntfy keeps messages ~12 h, which covers any kernel session (11 h).
_NTFY = {}          # status topic -> {"last_id": str|None, "by_account": {acct: event}, "at": float, "ok": bool}
NTFY_MIN_POLL_S = 4.0
STATUS_MAX_AGE_S = 13 * 3600


def _ntfy_get(url, timeout=20):
    req = urllib.request.Request(url, headers={"Accept": "application/x-ndjson", "User-Agent": "kllm"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _ntfy_post(cfg, topic, body, timeout=20):
    req = urllib.request.Request(f"{ntfy_server(cfg)}/{topic}", data=body.encode("utf-8"), method="POST",
                                 headers={"User-Agent": "kllm", "Cache": "yes", "Firebase": "no"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return True


def _ntfy_poll(cfg):
    """(ok, {account: newest event}). ok=False when ntfy could not be read (offline, rate limited)."""
    secret = cfg.get("rendezvous_secret")
    if not _real(secret):
        return False, {}
    status, _ = sealbox.topics(secret)
    st = _NTFY.setdefault(status, {"last_id": None, "by_account": {}, "at": 0.0, "ok": False})
    if time.time() - st["at"] < NTFY_MIN_POLL_S:
        return st["ok"], st["by_account"]
    since = st["last_id"] or "12h"
    try:
        raw = _ntfy_get(f"{ntfy_server(cfg)}/{status}/json?poll=1&since={urllib.parse.quote(since)}")
        ok = True
    except Exception:
        raw, ok = "", False
    now = time.time()
    for line in raw.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("event") != "message":
            continue
        st["last_id"] = msg.get("id") or st["last_id"]
        ev = sealbox.unseal(secret, msg.get("message", ""))
        if not ev or not ev.get("account"):
            continue                                   # not ours, forged or altered: dropped
        if now - float(ev.get("ts") or 0) > STATUS_MAX_AGE_S:
            continue
        prev = st["by_account"].get(ev["account"])
        if not prev or float(ev.get("ts") or 0) >= float(prev.get("ts") or 0):
            st["by_account"][ev["account"]] = ev
    st["at"], st["ok"] = now, ok
    return ok, st["by_account"]


def _ntfy_events(cfg):
    ok, by = _ntfy_poll(cfg)
    return ok, sorted(by.values(), key=lambda e: e.get("ts", 0), reverse=True)


def _ntfy_control(cfg, account, cmd):
    secret = cfg.get("rendezvous_secret")
    if not _real(secret):
        return False
    _, control = sealbox.topics(secret)
    try:
        return _ntfy_post(cfg, control, sealbox.seal(secret, {"cmd": cmd, "account": account, "ts": int(time.time())}))
    except Exception:
        return False


def latest_events(cfg, limit=50):
    """Current status (one per account), newest first. Each s-*.json holds the latest dict."""
    if mode(cfg) == "ntfy":
        return _ntfy_events(cfg)[1]
    d = _get_gist(cfg)
    if not d:
        return []
    evs = []
    for name, f in (d.get("files") or {}).items():
        if not name.startswith("s-"):
            continue
        try:
            evs.append(json.loads(f.get("content", "")))
        except Exception:
            continue
    evs.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return evs

def latest_events_checked(cfg):
    """(ok, events): ok=False when the gist could not be read (GitHub down, rate limit, no
    network) — callers must then treat kernel progress as UNKNOWN, not as "nothing happened"."""
    if mode(cfg) == "ntfy":
        return _ntfy_events(cfg)
    d = _get_gist(cfg)
    if not d:
        return False, []
    return True, _parse_events(d)

def _parse_events(d):
    evs = []
    for name, f in (d.get("files") or {}).items():
        if not name.startswith("s-"):
            continue
        try:
            evs.append(json.loads(f.get("content", "")))
        except Exception:
            continue
    evs.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return evs

def latest_by_account(cfg, limit=80):
    m = {}
    for e in latest_events(cfg):
        a = e.get("account")
        if a and a not in m:
            m[a] = e
    return m

def publish_control(cfg, account, cmd="stop"):
    """Send a control command to the kernel (PC -> kernel): sealed on the control topic, or c-<account>.json."""
    if mode(cfg) == "ntfy":
        return _ntfy_control(cfg, account, cmd)
    content = json.dumps({"cmd": cmd, "account": account, "ts": int(time.time())})
    tok = token(cfg)
    if tok:
        st, _, _ = _http(cfg, "PATCH", f"/gists/{cfg['gist_id']}", tok=tok,
                         body={"files": {f"c-{account}.json": {"content": content}}})
        return 200 <= st < 300
    try:
        r = subprocess.run(
            ["gh", "api", "-X", "PATCH", f"gists/{cfg['gist_id']}",
             "-f", f"files[c-{account}.json][content]={content}"],
            capture_output=True, text=True, timeout=25)
        return r.returncode == 0
    except Exception:
        return False

def create_gist(cfg, tok, description="kllm rendezvous (private)"):
    """Create the private (secret) rendezvous gist. Returns (gist_id, github_login) or raises."""
    st, d, _ = _http(cfg, "POST", "/gists", tok=tok, body={
        "description": description, "public": False,
        "files": {"README.txt": {"content": "kllm rendezvous. Kernels write s-<account>.json here; "
                                            "the PC writes c-<account>.json. Do not share this gist."}}})
    if st != 201 or not d:
        raise RuntimeError(f"GitHub refused to create the gist (HTTP {st or 'no connection'}). "
                           "The token needs the 'gist' scope.")
    return d["id"], (d.get("owner") or {}).get("login", "")

def whoami(cfg, tok):
    """GitHub login for a token (None if the token is rejected / offline)."""
    st, d, _ = _http(cfg, "GET", "/user", tok=tok, timeout=15)
    return (d or {}).get("login") if st == 200 else None
