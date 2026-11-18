"""Control-center API behind the admin panel (/admin): metrics, sessions, backends, autoscale, accounts,
models, config, diagnostics. Kept out of relay.py: relay.include_router()s `make_router(relay_module)`
before its catch-all proxy route, and passes itself in so handlers can use the live ROUTER / state.

Everything here is local-only (gateway/auth.py refuses /admin to remote callers) and additionally
guarded against browsers on other sites: state-changing calls need the `X-Kllm-Panel` header (a
cross-site page cannot set it without a CORS preflight, which is never granted) and the Host header
must be a local name (DNS rebinding).
"""
import asyncio, json, os, re, secrets, shlex, subprocess, sys, threading, time, urllib.parse
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
import yaml
from . import common, metrics, scheduler, media, accounts as acct_mod, yamledit, kaggle, auth, tunnel, rendezvous, telemetry, media_store, connect
from . import inspect_model as im

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "testserver"}
SECRET_LINE = re.compile(r"^(\s*(?:-\s+)?(?:\w*_token|\w*_secret|\w*password|gateway_api_key|api_key)\s*:)[ \t]*([^#\n]*?)([ \t]+#.*)?$", re.M)

# autoscale settings the panel can edit: (type, min, max, label)
AUTOSCALE_SPEC = {
    "enabled": ("bool", None, None),
    "min_backends": ("int", 0, 10), "max_backends": ("int", 1, 10), "spare_backends": ("int", 0, 5),
    "sessions_per_backend": ("int", 1, 8), "slots_per_backend": ("int", 1, 8),"idle_stop_minutes": ("int", 1, 1440),
    "queue_timeout_seconds": ("int", 30, 3600), "max_wait_seconds": ("int", 30, 7200),
    "min_quota_hours": ("float", 0, 30),
    "session_active_seconds": ("int", 60, 7200), "load_timeout_minutes": ("int", 5, 120),
    "fail_cooldown_minutes": ("int", 1, 720), "queue_cooldown_minutes": ("int", 1, 720),
    "max_queued_per_model": ("int", 1, 5),
}
MODEL_FIELDS = {"ctx", "vram_gb", "desc", "args", "tags", "quant", "file", "template", "hf", "opts"}
KEEP_HEADER = "x-kllm-panel"


class Bad(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg); self.msg, self.code = msg, code


def mask_secrets(text):
    """config.yaml for display: the value of every token / key / secret line is replaced
    (empty and null values stay as they are, comments are kept)."""
    def sub(m):
        if m.group(2).strip() in ("", "null", "~", "''", '""'):
            return m.group(0)
        return f"{m.group(1)} ******** (set){m.group(3) or ''}"
    return SECRET_LINE.sub(sub, text)


def parse_value(spec, v, name):
    kind, lo, hi = spec
    if kind == "bool":
        if isinstance(v, bool):
            return v
        if str(v).lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            return str(v).lower() in ("true", "1", "yes", "on")
        raise Bad(f"{name}: expected on/off")
    try:
        x = int(v) if kind == "int" and not isinstance(v, bool) and float(v) == int(float(v)) else float(v)
    except (TypeError, ValueError):
        raise Bad(f"{name}: expected a number") from None
    if kind == "int" and isinstance(x, float):
        raise Bad(f"{name}: expected a whole number")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise Bad(f"{name}: must be between {lo} and {hi}")
    return x


# ---- doctor job -----------------------------------------------------------------------------------------
DOCTOR = {"running": False, "started": 0, "finished": 0, "lines": [], "summary": None, "rc": None}
_DOCTOR_LOCK = threading.Lock()
DOCTOR_LINE = re.compile(r"^\s{2}(OK|FAIL)\s+(.*?)(?:\s{2,}|\s*$)(.*)$")


def parse_doctor(lines):
    """`kllm doctor` output -> [{ok, name, detail, fix}] and the summary line."""
    checks, summary = [], None
    for raw in lines:
        raw = raw.rstrip("\n")
        m = DOCTOR_LINE.match(raw)
        if m:
            checks.append({"ok": m.group(1) == "OK", "name": m.group(2).strip(), "detail": m.group(3).strip(), "fix": ""})
        elif raw.strip().startswith("->") and checks:
            checks[-1]["fix"] = raw.strip()[2:].strip()
        elif raw.strip() and not raw.startswith("==="):
            summary = raw.strip()
    return checks, summary


def start_doctor(cmd=None, env=None):
    with _DOCTOR_LOCK:
        if DOCTOR["running"]:
            return False
        DOCTOR.update(running=True, started=time.time(), finished=0, lines=[], summary=None, rc=None)
    cmd = cmd or [sys.executable, "-m", "gateway.cli", "doctor"]
    env = env or dict(os.environ, KLLM_ROOT=str(common.ROOT), PYTHONUNBUFFERED="1",
                      PYTHONPATH=os.pathsep.join(filter(None, [str(common.PKG_ROOT), os.environ.get("PYTHONPATH")])))
    def run():
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                 cwd=str(common.PKG_ROOT))
            for line in p.stdout:
                with _DOCTOR_LOCK:
                    DOCTOR["lines"].append(line)
                    del DOCTOR["lines"][:-300]
            DOCTOR["rc"] = p.wait()
        except Exception as e:
            DOCTOR["lines"].append(f"  FAIL doctor  {e}\n")
            DOCTOR["rc"] = -1
        finally:
            DOCTOR["finished"] = time.time(); DOCTOR["running"] = False
    threading.Thread(target=run, daemon=True, name="doctor").start()
    return True


def doctor_status():
    checks, summary = parse_doctor(list(DOCTOR["lines"]))
    return {"running": DOCTOR["running"], "started": int(DOCTOR["started"]), "finished": int(DOCTOR["finished"]),
            "checks": checks, "summary": summary, "rc": DOCTOR["rc"],
            "failed": sum(1 for c in checks if not c["ok"])}


# ---- service log ----------------------------------------------------------------------------------------
LOG_UNITS = {"kllm", "kllm-relay", "kllm-supervisor", "kllm-litellm"}


def service_log(n=200, unit="kllm"):
    """Last n lines of the service log: journalctl --user (read only), else <root>/logs/serve.log."""
    n = max(10, min(int(n), 1000))
    unit = unit if unit in LOG_UNITS else "kllm"
    try:
        r = subprocess.run(["journalctl", "--user", "-u", unit, "-n", str(n), "--no-pager", "-o", "short-iso"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip() and "No entries" not in r.stdout[:40]:
            return {"source": f"journalctl --user -u {unit}", "lines": r.stdout.splitlines()[-n:]}
    except (OSError, subprocess.SubprocessError):
        pass
    f = common.LOG_DIR / "serve.log"
    try:
        with open(f, "rb") as fh:
            fh.seek(0, 2); fh.seek(max(0, fh.tell() - 400_000))
            lines = fh.read().decode(errors="replace").splitlines()
        return {"source": str(f), "lines": lines[-n:]}
    except OSError:
        return {"source": None, "lines": [], "note": "no journal entries for this unit and no logs/serve.log "
                "(is `kllm serve` running as a service? try: journalctl --user -u kllm)"}


# ---- alerts ---------------------------------------------------------------------------------------------
def compute_alerts(cfg, state, demand, supervisor_on, now=None, models=None):
    now = now or time.time()
    s = scheduler.settings(cfg)
    out = []
    def add(id_, level, text, tab, **kw):
        out.append({"id": id_, "level": level, "text": text, "tab": tab, **kw})
    accts = cfg.get("accounts") or []
    enabled = [a for a in accts if not a.get("disabled")]
    if not accts:
        add("no-accounts", "crit", "No Kaggle accounts configured: add one to run any model.", "accounts")
    elif not enabled:
        add("all-disabled", "crit", "Every account is disabled: nothing can be launched.", "accounts")
    if (s["enabled"] or s["min_backends"]) and not supervisor_on:
        add("supervisor", "crit", "Supervisor is not running: nothing starts, stops or fails over. "
            "Run `kllm serve` (or install the service).", "dashboard")
    pool_left = 0.0
    for a in enabled:
        left = common.quota_left_h(a["name"], state)
        pool_left += max(0, left)
        if left < s["min_quota_hours"]:
            add(f"quota-{a['name']}", "warn" if left > 0 else "crit",
                f"{a['name']} has {max(0, left):.1f} h of weekly GPU quota left"
                + (f" (below the {s['min_quota_hours']} h minimum: the planner skips it)" if left > 0 else " (exhausted)"),
                "accounts", account=a["name"])
    if enabled and pool_left < 3:
        add("pool-quota", "crit", f"Whole pool has only {pool_left:.1f} h of GPU quota left this week.", "usage")
    for m, mf in (state.get("models_failing") or {}).items():
        if mf.get("until", 0) > now:
            add(f"model-{m}", "crit", f"Model '{m}' is paused (circuit breaker) for {int((mf['until'] - now) // 60)} min: "
                f"{(mf.get('reason') or 'failed to load')[:160]}", "models", model=m)
    # a launch the planner gave up on vanished from Fleet without a word: say so for 30 minutes
    for e in ((state.get("scheduler") or {}).get("log") or [])[-40:]:
        if e.get("action") == "cancel" and e.get("why") == "too slow to load" and now - e.get("ts", 0) < 1800:
            add(f"cancel-{e.get('target')}-{e.get('ts')}", "warn",
                f"The launch on {e.get('target')} was stopped: {e.get('why')}. "
                "Check Logs, then launch again; a fresh account's first build can take 20 to 30 min.", "logs",
                account=e.get("target"))
    for b in state.get("backends", []):
        h = (state.get("health") or {}).get(b["account"])
        if h is not None and not h.get("ok", True):
            add(f"unhealthy-{b['account']}", "crit", f"Backend {b['account']} ({b.get('model_key')}) is not answering health checks.",
                "backends", account=b["account"])
        left = b.get("started", now) + b.get("session_seconds", 0) - now
        if 0 < left < 1800 and not b.get("replaced_by"):
            same = [x for x in state.get("backends", []) if x is not b and x.get("model_key") == b.get("model_key")]
            pend = [p for p in (state.get("pending") or {}).values() if p.get("model_key") == b.get("model_key")]
            if not same and not pend:
                add(f"expiring-{b['account']}", "info", f"Backend {b['account']} hits Kaggle's session cap in {int(left // 60)} min "
                    "and has no successor yet.", "backends", account=b["account"])
    for a, p in (state.get("pending") or {}).items():
        age = now - p.get("started", now)
        if p.get("phase") in ("error", "crash"):
            add(f"launch-failed-{a}", "crit", f"Launch of {p.get('model_key')} on {a} failed.", "backends", account=a)
        elif not p.get("boot_at") and age > s["queue_timeout_seconds"]:
            add(f"queued-{a}", "warn", f"Launch on {a} has sat in Kaggle's queue for {int(age // 60)} min.", "backends", account=a)
        elif age > s["load_timeout_minutes"] * 60:
            add(f"stuck-{a}", "crit", f"Launch on {a} has not become ready after {int(age // 60)} min.", "backends", account=a)
    waiting = {m: d.get("waiting", 0) for m, d in ((demand or {}).get("models") or {}).items() if d.get("waiting")}
    if waiting and not s["enabled"] and not s["min_backends"]:
        n = sum(waiting.values())
        add("autoscale-off", "crit", f"{n} request(s) are waiting for {', '.join(waiting)} but autoscale is off: "
            "start a model or turn autoscale on.", "autoscale")
    elif waiting and not state.get("backends") and not state.get("pending"):
        add("waiting-nothing", "warn", f"{sum(waiting.values())} request(s) waiting and nothing is starting "
            "(no free account or model paused?).", "backends")
    if cfg.get("remote_enabled") and not auth.real_key(cfg):
        add("remote-nokey", "warn", "Remote tunnel is on but gateway_api_key is not set: remote callers are refused.", "settings")
    return out


# ---- helpers --------------------------------------------------------------------------------------------
def account_allowed(cfg, name):
    """config.yaml account_allow_prefixes (for example [kg]): the only accounts kllm may launch on."""
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    return not allow or any(str(name).startswith(p) for p in allow)


def _json_body(body):
    return body if isinstance(body, dict) else {}


def _param(request, name, default=None, cast=str):
    v = request.query_params.get(name)
    if v in (None, ""):
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        raise Bad(f"bad value for {name}")


def explorer_filters(request):
    st = request.query_params.get("status") or ""
    status = st if st in ("ok", "error") else int(st) if st.isdigit() else None
    since = _param(request, "since", None, float)
    if since is not None and since < 10**8:             # "seconds ago"
        since = time.time() - since
    return {"q": request.query_params.get("q") or None, "model": request.query_params.get("model") or None,
            "backend": request.query_params.get("backend") or None, "status": status,
            "loop": request.query_params.get("loop") in ("1", "true"), "since": since,
            "kind": request.query_params.get("kind") or None,
            "min_total": _param(request, "min_total", None, float), "min_prompt": _param(request, "min_prompt", None, int)}


def guess_abliterated(*names):
    return any("abliterat" in (n or "").lower() or "uncensor" in (n or "").lower() or "huihui" in (n or "").lower()
               for n in names)


_BASE_STRIP = re.compile(r"(?i)(-?i1)?[-_.]?(gguf|abliterated|uncensored|instruct|imatrix|bf16|f16|q\d\w*|iq\d\w*|mxfp4\w*)\b")


def base_name(repo):
    """'mradermacher/Huihui-Qwen3-32B-abliterated-i1-GGUF' -> 'Qwen3-32B' (for the alternatives search)."""
    n = repo.split("/")[-1]
    n = re.sub(r"(?i)^(huihui[-_]?ai[_-]|huihui[-_]|bartowski[_-]|mradermacher[_-])", "", n)
    n = _BASE_STRIP.sub("", n)
    return re.sub(r"[-_.]+$", "", re.sub(r"[-_]{2,}", "-", n)).strip() or repo.split("/")[-1]


def hf_search(query, limit=8, token=None, gguf=True, abliterated=True):
    """On-demand HuggingFace search (public API, read only). Returns [{repo, downloads, likes, updated}]."""
    q = f"{query} abliterated" if abliterated and "abliterat" not in query.lower() else query
    url = f"{im.HF}/api/models?" + urllib.parse.urlencode(
        {"search": q, "limit": limit, "sort": "downloads", "direction": "-1", **({"filter": "gguf"} if gguf else {})})
    data, _ = im._json(url, token)
    out = []
    for m in data if isinstance(data, list) else []:
        rid = m.get("id") or m.get("modelId")
        if rid:
            out.append({"repo": rid, "downloads": m.get("downloads"), "likes": m.get("likes"),
                        "updated": m.get("lastModified") or m.get("createdAt"), "abliterated": guess_abliterated(rid)})
    return out


def model_row(key, m, cfg, state, now):
    running = [b["account"] for b in state.get("backends", []) if b.get("model_key") == key]
    pending = [a for a, p in (state.get("pending") or {}).items() if p.get("model_key") == key]
    mf = (state.get("models_failing") or {}).get(key) or {}
    tags = m.get("tags") or []
    kind = media.kind(m)
    return {"key": key, "hf": m.get("hf"), "quant": m.get("quant"), "file": m.get("file"), "template": m.get("template"),
            "vram_gb": m.get("vram_gb"), "ctx": m.get("ctx"), "args": m.get("args") or [], "desc": m.get("desc", ""),
            "tags": tags, "opts": m.get("opts"), "kind": kind, "runtime": media.runtime(m),
            "default": key == cfg.get("default_model"), "default_for": [k for k in ("image", "video")
                                                                         if cfg.get(f"default_{k}_model") == key],
            "running": running, "launching": pending,
            "failing": mf.get("reason") if mf.get("until", 0) > now else None,
            "verified": "verified" in tags, "experiment": key.startswith("x-") or "experiment" in tags,
            "abliterated": guess_abliterated(key, m.get("hf"), m.get("file")), "problems": media.problems(m)}


def clean_model_changes(changes, current):
    """Validate a model edit. Returns {field: value|None}."""
    if not isinstance(changes, dict) or not changes:
        raise Bad("nothing to change")
    out = {}
    for k, v in changes.items():
        if k not in MODEL_FIELDS:
            raise Bad(f"'{k}' cannot be edited here")
        if k == "ctx":
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise Bad("ctx must be a whole number") from None
            if not 1024 <= v <= 4_194_304:
                raise Bad("ctx must be between 1024 and 4194304 tokens")
        elif k == "vram_gb":
            if v in (None, ""):
                v = None
            else:
                try:
                    v = float(v); v = int(v) if v == int(v) else v
                except (TypeError, ValueError):
                    raise Bad("vram_gb must be a number") from None
                if not 0 <= v <= 64:
                    raise Bad("vram_gb must be between 0 and 64")
        elif k == "args":
            if isinstance(v, str):
                try:
                    v = shlex.split(v)
                except ValueError as e:
                    raise Bad(f"args: {e}") from None
            if not isinstance(v, list) or not all(isinstance(x, (str, int, float)) for x in v) or len(v) > 120:
                raise Bad("args must be a list of up to 120 strings")
            v = [str(x) for x in v]
            if any(len(x) > 300 or "\n" in x for x in v):
                raise Bad("args: an argument is too long or contains a newline")
        elif k == "tags":
            if isinstance(v, str):
                v = [t.strip() for t in v.split(",") if t.strip()]
            if not isinstance(v, list) or not all(isinstance(x, str) and re.fullmatch(r"[\w.+-]{1,32}", x) for x in v):
                raise Bad("tags: words made of letters, digits, - _ . +")
        elif k == "opts":
            if v in (None, "", {}):
                v = None
            elif not isinstance(v, dict):
                raise Bad("opts must be an object (diffusers options)")
            else:
                try:
                    json.dumps(v)
                except (TypeError, ValueError):
                    raise Bad("opts must be plain JSON") from None
        else:                                   # desc, quant, file, template, hf: single-line strings
            v = None if v in (None, "") else str(v).strip()
            if v is not None and ("\n" in v or len(v) > (500 if k == "desc" else 220)):
                raise Bad(f"{k}: must be one short line")
            if k == "hf" and not (v and re.fullmatch(r"[\w.-]+/[\w.-]+", v)):
                raise Bad("hf must look like org/name")
            if k == "template" and v and (v.startswith("/") or ".." in v):
                raise Bad("template must be a relative path under templates/")
        out[k] = v
    return out


# ---- the router -------------------------------------------------------------------------------------------
def make_router(R):
    """R is the relay module (ROUTER, state(), cfg(), _spawn(), _warm(), ...)."""
    ROLL = metrics.Rollup()

    async def guard(request: Request):
        try:
            allowed = LOCAL_HOSTS | set((R.cfg().get("admin_allowed_hosts") or []))
        except Exception:
            allowed = LOCAL_HOSTS
        host = (request.headers.get("host") or "").strip()
        hostname = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        if hostname.lower() not in allowed:
            raise HTTPException(403, "kllm panel: unexpected Host header")
        if request.method not in ("GET", "HEAD", "OPTIONS") and not request.headers.get(KEEP_HEADER):
            raise HTTPException(403, f"kllm panel: missing {KEEP_HEADER} header")
        agent = request.headers.get("x-kllm-agent-session")
        if agent:                                   # a call from inside a kllm Agents session: scope and log it
            from .agents.runtime import MANAGER
            level = MANAGER.access().get("api", "read")
            write = request.method not in ("GET", "HEAD", "OPTIONS")
            allowed = level == "full" or (level == "read" and not write)
            MANAGER.kllm_action(agent, request.headers.get("x-kllm-agent-worker"),
                                f"{request.method} {request.url.path}", allowed, via="api")
            if not allowed:
                raise HTTPException(403, f"kllm Agents access for the admin API is '{level}': "
                                         "change it on the Access page")

    r = APIRouter(dependencies=[Depends(guard)])

    def ok(**kw):
        return {"ok": True, **kw}

    def fail(msg, code=400, **kw):
        return JSONResponse({"ok": False, "msg": msg, **kw}, status_code=code)

    async def body_of(request):
        try:
            return _json_body(await request.json())
        except ValueError:
            raise Bad("request body must be JSON")

    def bust():
        R._cfg["mtime"] = 0

    def supervisor_on():
        return R.INPROC["supervisor"] or R._svc_active("kllm-supervisor")

    def guarded(fn):
        """Turn Bad(...) into a JSON error response."""
        import functools
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            try:
                return await fn(*a, **kw)
            except Bad as e:
                return fail(e.msg, e.code)
            except (yamledit.EditError, acct_mod.AccountError) as e:
                return fail(str(e), 400)
        return wrapper

    # ---- metrics / usage -------------------------------------------------------------------------------
    @r.get("/admin/api/metrics")
    @guarded
    async def api_metrics(request: Request, range: str = "24h", tz: int = 0):
        if range not in metrics.RANGES:
            raise Bad(f"range must be one of {', '.join(metrics.RANGES)}")
        v = await asyncio.to_thread(ROLL.query, range, None, max(-840, min(840, tz)))
        return v

    @r.get("/admin/api/compute")
    @guarded
    async def api_compute(request: Request):
        cfg, s = R.cfg_fresh(), common.read_state()
        week_start = common._week_start(time.time())
        busy = await asyncio.to_thread(ROLL.busy_hours, week_start)
        dec = await asyncio.to_thread(metrics.read_decisions, week_start - 7 * 86400, 2000)
        rep = metrics.compute_report(cfg, s, busy, dec)
        rep["log_bytes"] = sum(os.path.getsize(p) for p in metrics.log_files())
        return rep

    # ---- alerts / overview ------------------------------------------------------------------------------
    @r.get("/admin/api/alerts")
    @guarded
    async def api_alerts(request: Request):
        cfg, s = R.cfg(), common.read_state()
        demand = await R.relay_demand()
        alerts = compute_alerts(cfg, s, demand, supervisor_on()) + telemetry.HUB.alerts(s.get("backends", []))
        return {"alerts": alerts, "supervisor": bool(supervisor_on()),
                "backends": len(s.get("backends", [])), "pending": len(s.get("pending", {})),
                "waiting": sum(d.get("waiting", 0) for d in (demand.get("models") or {}).values()),
                "autoscale": scheduler.settings(cfg)["enabled"]}

    # ---- sessions ---------------------------------------------------------------------------------------
    def sid(key):
        return key[:14]

    def find_session(ident):
        hits = [(k, s) for k, s in R.ROUTER.sessions.items() if k == ident or sid(k) == ident]
        if not hits:
            raise Bad("session not found (it may have expired)", 404)
        return hits[0]

    @r.get("/admin/api/sessions")
    @guarded
    async def api_sessions(request: Request):
        now = time.time()
        window = scheduler.settings(R.cfg())["session_active_seconds"]
        rows = []
        for k, x in R.ROUTER.sessions.items():
            if now - x["last"] > 6 * 3600:
                continue
            lr = x.get("last_req")
            rows.append({"id": sid(k), "model": x.get("model"), "backend": x.get("backend"), "requests": x["n"],
                         "ctx_est": int(len(lr[1]) / 3.6) if lr else None, "big": bool(x.get("big")),
                         "client": x.get("client"),
                         "idle_s": int(now - x["last"]), "age_s": int(now - x["first"]),
                         "busy": now - x["last"] <= scheduler.BUSY_SECONDS,
                         "active": now - x["last"] <= window and R.ROUTER.counts(x),
                         "counts": bool(R.ROUTER.counts(x)), "probe": bool(x.get("probe")),
                         "moved": x.get("moved", 0), "warming": bool(x.get("warming")), "replayable": bool(lr)})
        rows.sort(key=lambda s: s["idle_s"])
        by_model = {}
        for b in R._backends():
            by_model.setdefault(b["model"], []).append(b["label"])
        return {"sessions": rows[:300], "total": len(rows), "handoffs": R.HANDOFFS,
                "targets": by_model, "per_backend": scheduler.settings(R.cfg())["sessions_per_backend"]}

    @r.post("/admin/api/sessions/move")
    @guarded
    async def api_session_move(request: Request):
        b = await body_of(request)
        key, sess = find_session(str(b.get("id") or ""))
        label = str(b.get("backend") or "")
        label = label if label.startswith("kaggle:") else "kaggle:" + label
        target = next((x for x in R._backends() if x["label"] == label), None)
        if target is None:
            raise Bad(f"{label} is not a running backend")
        if target["model"] != sess.get("model"):
            raise Bad(f"{label} serves {target['model']}, this session uses {sess.get('model')}")
        if sess.get("backend") == label:
            raise Bad("the session is already on that backend")
        if sess.get("warming"):
            raise Bad("a handoff for this session is already running", 409)
        t0 = time.time()
        if sess.get("last_req"):
            sess["warming"] = True
            R.ROUTER.begin(label)
            try:
                done = await asyncio.wait_for(R._warm(sess, target), timeout=900)
            except Exception as e:
                done = False
            finally:
                R.ROUTER.end(label); sess["warming"] = False
            if not done:
                if not b.get("cold"):
                    return fail("warm handoff failed (backend busy or unreachable); the session was not moved. "
                                "Retry, or move it cold (next turn re-reads its whole prompt).", 502, can_cold=True)
        sess["backend"] = label
        sess["moved"] = sess.get("moved", 0) + 1
        warm = bool(sess.get("last_req")) and not b.get("cold")
        return ok(msg=f"session moved to {label}" + (f" (cache warmed in {time.time() - t0:.0f} s)" if warm else " (cold)"))

    @r.post("/admin/api/sessions/forget")
    @guarded
    async def api_session_forget(request: Request):
        b = await body_of(request)
        key, _ = find_session(str(b.get("id") or ""))
        R.ROUTER.sessions.pop(key, None)
        return ok(msg="session forgotten: its next request is placed fresh")

    # ---- backends ---------------------------------------------------------------------------------------
    STEPS = [("push", "Pushed to Kaggle"), ("queue", "Kaggle queue"), ("boot", "Kernel booted"),
             ("build", "Preparing llama.cpp"), ("download", "Downloading model"), ("serve", "Loading model"),
             ("ready", "Ready")]
    PHASE_IDX = {"boot": 2, "binary": 3, "building": 3, "downloading": 4, "serving": 5, "ready": 6, "alive": 6}

    def timeline(a, p, now):
        started = p.get("started", now)
        failed = p.get("failed") or p.get("phase") in ("error", "crash")
        idx = PHASE_IDX.get(p.get("phase"), 2 if p.get("boot_at") else 1)
        steps = []
        for i, (k, label) in enumerate(STEPS):
            st = "done" if i < idx else "active" if i == idx else "todo"
            if failed and st == "active":
                st = "failed"
            row = {"key": k, "label": label, "state": st}
            if k == "queue" and p.get("boot_at"):
                row["secs"] = max(0, int(p["boot_at"] - started))
            if k == "push":
                row["secs"] = 0
            steps.append(row)
        if p.get("boot_at"):
            steps[3]["since"] = int(p["boot_at"])
        if p.get("bytes_done") is not None:                   # reported by the kernel while llama-server downloads
            steps[4].update(bytes_done=p["bytes_done"], bytes_total=p.get("bytes_total"), bytes_at=p.get("bytes_at"))
        return {"account": a, "model_key": p.get("model_key"), "origin": p.get("origin"), "started": int(started),
                "bytes_done": p.get("bytes_done"), "bytes_total": p.get("bytes_total"),
                "age_s": int(now - started), "phase": p.get("phase"), "boot_at": p.get("boot_at"),
                "queue_s": int(p["boot_at"] - started) if p.get("boot_at") else None,
                "failed": failed, "note": p.get("note"), "steps": steps}

    @r.get("/admin/api/backends")
    @guarded
    async def api_backends(request: Request):
        now = time.time(); cfg = R.cfg(); s = common.read_state()
        demand = await R.relay_demand()
        dback = demand.get("backends", {})
        lead = cfg.get("prelaunch_lead_seconds", 1500)
        pinned = s.get("pinned") or {}
        cool = {u: t - now for u, t in R._bad.items() if t > now}
        reg = R._registry()
        def spec_of(key):
            m = reg.get(key) or {}
            return {"kind": m.get("kind", "chat"), "quant": m.get("quant"), "vram_gb": m.get("vram_gb"),
                    "params": m.get("params"), "desc": m.get("desc")}
        rows = []
        for b in s.get("backends", []):
            label = "kaggle:" + b["account"]
            d = dback.get(label, {})
            left = b.get("started", now) + b.get("session_seconds", 0) - now
            url = (b.get("url") or "")
            rows.append({"account": b["account"], "user": b.get("user"), "model_key": b.get("model_key"),
                         "host": urllib.parse.urlparse(url).netloc, "ctx": b.get("ctx"),
                         "spec": spec_of(b.get("model_key")),
                         "healthy": (s.get("health") or {}).get(b["account"], {}).get("ok", True),
                         "up_s": int(now - (b.get("ready_at") or b.get("started", now))), "left_s": int(left),
                         "draining": left <= lead, "replaced_by": b.get("replaced_by"),
                         "sessions": d.get("sessions"), "inflight": d.get("inflight"), "idle_s": d.get("idle_s"),
                         "errors_5m": d.get("errors_5m", 0), "ok_age_s": d.get("ok_age_s"),
                         "pinned": b["account"] in pinned, "cooldown_s": int(cool.get(b.get("url", "").rstrip("/"), 0)),
                         "quota_used_h": round(common.quota_used(b["account"], s) / 3600, 1),
                         "quota_limit_h": round(common.quota_total(b["account"], s) / 3600, 1),
                         "rpm": R.ROUTER.rpm(label), "res": telemetry.HUB.summary(b["account"], now)})
        # client tiles for the fleet map: active sessions (this window) and req/min per client kind,
        # from the relay's in-memory request tracking — nothing here is synthesized
        window = scheduler.settings(cfg)["session_active_seconds"]
        client_sessions = {}
        for x in R.ROUTER.active_sessions(window):
            c = x.get("client") or "API"
            client_sessions[c] = client_sessions.get(c, 0) + 1
        client_rpm = R.ROUTER.client_rpm(60.0)
        client_kinds = sorted(set(client_sessions) | set(client_rpm))
        clients = [{"kind": k, "sessions": client_sessions.get(k, 0), "rpm": client_rpm.get(k, 0.0)}
                   for k in client_kinds]
        traffic = {"series": R.ROUTER.traffic_series(60, 60),
                   "rpm": sum(R.ROUTER.rpm("kaggle:" + b["account"]) for b in s.get("backends", [])),
                   "ttft_p50": R.ROUTER.ttft_p50(3600)}
        return {"backends": rows, "pending": [timeline(a, p, now) for a, p in (s.get("pending") or {}).items()],
                "clients": clients, "traffic": traffic,
                "supervisor": bool(supervisor_on()), "autoscale": scheduler.settings(cfg)["enabled"],
                "models": [k for k, m in R._registry().items() if not media.problems(m)],
                "accounts": [{"name": a["name"], "disabled": bool(a.get("disabled")), "allowed": account_allowed(cfg, a["name"]),
                              "busy": a["name"] in {x["account"] for x in s.get("backends", [])} | set(s.get("pending", {})),
                              "left_h": round(common.quota_left_h(a["name"], s), 1)}
                             for a in cfg.get("accounts") or []]}

    @r.post("/admin/api/backends/launch")
    @guarded
    async def api_launch(request: Request):
        b = await body_of(request)
        cfg, s = R.cfg_fresh(), common.read_state()
        reg = R._registry()
        model = (b.get("model") or cfg.get("default_model") or "").strip()
        if model not in reg:
            raise Bad(f"unknown model '{model}'")
        bad = media.problems(reg[model])
        if bad:
            raise Bad(f"'{model}' is not launchable: " + "; ".join(bad))
        force = bool(b.get("force"))
        mf = (s.get("models_failing") or {}).get(model) or {}
        if mf.get("until", 0) > time.time() and not force:
            return fail(f"'{model}' is paused by the circuit breaker: {mf.get('reason', 'failed to load')}", 409, can_force=True)
        busy = {x["account"] for x in s.get("backends", [])} | set(s.get("pending", {}))
        accts = {a["name"]: a for a in cfg.get("accounts") or []}
        name = (b.get("account") or "").strip()
        if name:
            if name not in accts:
                raise Bad(f"unknown account '{name}'")
            if not account_allowed(cfg, name):
                raise Bad(f"{name} is not permitted by account_allow_prefixes in config.yaml", 403)
            if name in busy:
                raise Bad(f"{name} is already running or launching a kernel", 409)
            if accts[name].get("disabled") and not force:
                return fail(f"{name} is disabled", 409, can_force=True)
            left = common.quota_left_h(name, s)
            if left < 0.25 and not force:
                return fail(f"{name} has only {left:.1f} h of GPU quota left", 409, can_force=True)
        elif not [a for a in accts if a not in busy and not accts[a].get("disabled") and account_allowed(cfg, a)]:
            raise Bad("every enabled account is busy: stop a backend first", 409)
        args = ["up", "--no-wait", "--model", model] + (["--account", name] if name else [])
        R._spawn(args)
        return ok(msg=f"launching {model} on {name or 'the next free account'}: ready in ~8-10 min "
                      "(first build on an account ~20 min)")

    @r.post("/admin/api/backends/stop")
    @guarded
    async def api_stop(request: Request):
        b = await body_of(request)
        s = common.read_state()
        name = (b.get("account") or "").strip()
        live = {x["account"] for x in s.get("backends", [])} | set(s.get("pending", {}))
        if name and name not in live:
            raise Bad(f"{name} has nothing running", 404)
        if not name and not live:
            try:
                cfg = common.load_config()
                rendezvous.publish_control(cfg, "all", "stop")
            except Exception:
                pass
            R._spawn(["down"])
            return ok(msg="broadcast stop sent to all accounts")
        R._spawn(["down"] + (["--account", name] if name else []))
        return ok(msg=f"stopping {name or 'everything'}")

    @r.post("/admin/api/backends/rotate")
    @guarded
    async def api_rotate(request: Request):
        b = await body_of(request)
        s = common.read_state()
        name = (b.get("account") or "").strip()
        cur = next((x for x in s.get("backends", []) if x["account"] == name), None)
        if cur is None:
            raise Bad(f"{name or 'that account'} is not a running backend", 404)
        R._spawn(["rotate", "--account", name, "--model", b.get("model") or cur.get("model_key")])
        return ok(msg=f"rotating {name}: a fresh backend starts, sessions are handed over, then {name} stops")

    @r.post("/admin/api/backends/pin")
    @guarded
    async def api_pin(request: Request):
        b = await body_of(request)
        name = (b.get("account") or "").strip()
        s = common.read_state()
        if name not in {x["account"] for x in s.get("backends", [])} | set(s.get("pending", {})):
            raise Bad(f"{name or 'that account'} has no backend", 404)
        want = bool(b.get("pinned", True))
        with common.mutate_state() as st:
            pins = st.setdefault("pinned", {})
            if want:
                pins[name] = int(time.time())
            else:
                pins.pop(name, None)
        return ok(msg=f"{name} pinned: autoscale will not stop it for being idle (it still rotates before Kaggle's 12 h cap)"
                  if want else f"{name} unpinned")

    # ---- live kernel resources (gateway/telemetry.py) -------------------------------------------------------
    def live_account(account):
        if account not in {b["account"] for b in common.read_state().get("backends", [])}:
            raise Bad(f"{account} is not a running backend", 404)

    @r.get("/admin/api/backends/{account}/telemetry")
    @guarded
    async def api_backend_telemetry(request: Request, account: str, since: float = 0.0, limit: int = 720):
        live_account(account)
        hub = telemetry.HUB
        return {"account": account, "status": hub.status.get(account, "waiting"), "age_s": hub.age(account),
                "interval_s": hub.interval, "samples": hub.since(account, since, max(1, min(limit, 720)))}

    @r.get("/admin/api/telemetry/stream")
    async def api_telemetry_stream(request: Request):
        """Server-sent events: one `sample` event per kernel with its newest sample, then again on every change."""
        hub = telemetry.HUB

        async def gen():
            sent = {}
            seq = -1
            while not await request.is_disconnected():
                seq = await hub.wait_new(seq, 15.0)
                live = common.read_state().get("backends", [])
                out = []
                for b in live:
                    a = b["account"]
                    s = hub.latest(a)
                    key = s["ts"] if s else None
                    if sent.get(a, "x") != key:
                        sent[a] = key
                        out.append({"account": a, "status": hub.status.get(a, "waiting"), "age_s": hub.age(a),
                                    "sample": s})
                for row in out:
                    yield f"event: sample\ndata: {json.dumps(row)}\n\n"
                if not out:
                    yield ": keepalive\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    # ---- media history (gateway/media_store.py) ---------------------------------------------------------------
    @r.get("/admin/api/media")
    @guarded
    async def api_media(request: Request, kind: str = "", limit: int = 100):
        if kind not in ("", "image", "video"):
            raise Bad("kind must be image or video")
        rows = await asyncio.to_thread(media_store.list_media, kind or None, limit)
        for row in rows:
            row["url"] = f"/admin/api/media/{row['id']}/file"
        return {"media": rows}

    @r.get("/admin/api/media/{media_id}/file")
    async def api_media_file(media_id: str):
        from fastapi.responses import FileResponse
        p, _ = await asyncio.to_thread(media_store.path_of, media_id)
        if p is None:
            raise HTTPException(404, "no such output")
        return FileResponse(str(p), headers={"Cache-Control": "private, max-age=86400"})

    @r.post("/admin/api/media/{media_id}/delete")
    @guarded
    async def api_media_delete(request: Request, media_id: str):
        if not await asyncio.to_thread(media_store.delete, media_id):
            raise Bad("no such output", 404)
        return ok(msg="output deleted")

    # ---- connect a client (gateway/connect.py) -------------------------------------------------------------
    @r.get("/admin/api/connect")
    @guarded
    async def api_connect(request: Request, refresh: int = 0):
        return {"tools": await asyncio.to_thread(connect.detect, bool(refresh))}

    @r.post("/admin/api/connect/{tool}/setup")
    @guarded
    async def api_connect_setup(request: Request, tool: str):
        b = await body_of(request)
        relay = f"http://127.0.0.1:{R.cfg().get('relay_port', 8080)}"
        try:
            msg = await asyncio.to_thread(connect.setup, tool, str(b.get("what") or "models"), relay)
        except connect.SetupError as e:
            raise Bad(str(e), 409)
        return ok(msg=msg, tools=await asyncio.to_thread(connect.detect, True))

    @r.post("/admin/api/connect/{tool}/undo")
    @guarded
    async def api_connect_undo(request: Request, tool: str):
        b = await body_of(request)
        try:
            msg = await asyncio.to_thread(connect.undo, tool, str(b.get("what") or "models"))
        except connect.SetupError as e:
            raise Bad(str(e), 409)
        return ok(msg=msg, tools=await asyncio.to_thread(connect.detect, True))

    # ---- autoscale ----------------------------------------------------------------------------------------
    @r.get("/admin/api/autoscale/config")
    @guarded
    async def api_autoscale_get(request: Request):
        cfg = R.cfg_fresh()
        sset = scheduler.settings(cfg)
        s = common.read_state()
        dem = await R.relay_demand()
        dec = await asyncio.to_thread(metrics.read_decisions, 0, 150)
        return {"settings": {k: sset[k] for k in AUTOSCALE_SPEC},
                "spec": {k: {"type": v[0], "min": v[1], "max": v[2]} for k, v in AUTOSCALE_SPEC.items()},
                "defaults": {k: scheduler.DEFAULTS[k] for k in AUTOSCALE_SPEC},
                "decisions": list(reversed(dec)),
                "tick": (s.get("scheduler") or {}).get("tick"),
                "demand": dem.get("models"), "supervisor": bool(supervisor_on()),
                "desired_pool": cfg.get("desired_pool", 0)}

    @r.post("/admin/api/autoscale/config")
    @guarded
    async def api_autoscale_set(request: Request):
        b = _json_body(await request.json())
        vals = b.get("settings") if isinstance(b.get("settings"), dict) else b
        clean = {}
        for k, v in vals.items():
            if k not in AUTOSCALE_SPEC:
                raise Bad(f"unknown autoscale setting '{k}'")
            clean[k] = parse_value(AUTOSCALE_SPEC[k], v, k)
        if not clean:
            raise Bad("nothing to change")
        cur = scheduler.settings(R.cfg_fresh())
        eff_min = clean.get("min_backends", cur["min_backends"])
        eff_max = clean.get("max_backends", cur["max_backends"])
        if eff_min > eff_max:
            raise Bad("min_backends cannot exceed max_backends")
        set_autoscale(clean)
        bust()
        return ok(msg="saved: " + ", ".join(f"{k} = {v}" for k, v in clean.items()), settings=clean)

    def set_autoscale(clean):
        """Write autoscale settings into config.yaml (block or one-line flow style), verified."""
        with common.state_lock():
            txt = common.CONFIG_FILE.read_text(encoding="utf-8")
            sec = (yaml.safe_load(txt) or {}).get("autoscale")
            flow_line = re.search(r"^autoscale:\s*\{.*$", txt, re.M)
            if flow_line:
                merged = dict(sec or {}); merged.update(clean)
                new_line = "autoscale: " + yaml.safe_dump(merged, default_flow_style=True, width=10**6, sort_keys=False).strip()
                yamledit.atomic_write(common.CONFIG_FILE, txt[:flow_line.start()] + new_line + txt[flow_line.end():])
            else:
                for k, v in clean.items():
                    common.set_section_value("autoscale", k, "true" if v is True else "false" if v is False else v)
                chk = yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8")) or {}
                if any((chk.get("autoscale") or {}).get(k) != v for k, v in clean.items()):
                    common.CONFIG_FILE.write_text(txt, encoding="utf-8")
                    raise Bad("could not write autoscale settings safely; config.yaml unchanged")
            if "min_backends" in clean:                    # the legacy "always on" knob must agree
                common.set_config_value("desired_pool", clean["min_backends"])

    # ---- accounts -----------------------------------------------------------------------------------------
    @r.get("/admin/api/accounts")
    @guarded
    async def api_accounts(request: Request):
        cfg, s = R.cfg_fresh(), common.read_state()
        now = time.time()
        week_start = common._week_start(now)
        busy = await asyncio.to_thread(ROLL.busy_hours, week_start)
        dec = await asyncio.to_thread(metrics.read_decisions, week_start - 7 * 86400, 2000)
        rep = metrics.compute_report(cfg, s, busy, dec)
        by_name = {a["name"]: a for a in cfg.get("accounts") or []}
        running = {b["account"]: b for b in s.get("backends", [])}
        for row in rep["accounts"]:
            a = by_name[row["name"]]
            row["allowed"] = account_allowed(cfg, row["name"])
            try:
                kind, where = kaggle.resolve_credentials(a)
                row["credentials"] = {"ok": True, "kind": kind, "where": where}
            except kaggle.NoCredentials as e:
                row["credentials"] = {"ok": False, "error": str(e)}
            row["model_key"] = (running.get(row["name"]) or (s.get("pending") or {}).get(row["name"]) or {}).get("model_key")
            row["removable_credentials"] = [str(p) for p in acct_mod.removable_credentials(row["name"])]
            st = (s.get("accounts") or {}).get(row["name"]) or {}
            q_rec = (s.get("quota") or {}).get(row["name"]) or {}
            row["stats"] = {k: st.get(k) for k in ("fails", "last_error", "queue_ewma_s", "ready_ewma_s")}
            row["quota_info"] = {
                "refresh_time": q_rec.get("refresh_time"),
                "synced_at": q_rec.get("synced_at"),
                "is_live": bool(q_rec.get("live")),
            }
            row["error_active"] = bool(row.get("cooldown_left_s", 0) > 0)
            if row.get("cooldown_left_s", 0) == 0 and row.get("fails", 0) == 0:
                row["last_error"] = None
            if q_rec.get("refresh_time") and rep.get("quota_refresh_time") is None:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(q_rec["refresh_time"].replace("Z", "+00:00"))
                    rem_h = max(0.0, (dt.timestamp() - now) / 3600)
                    rep["week_resets_in_h"] = round(rem_h, 1)
                    rep["quota_refresh_time"] = q_rec["refresh_time"]
                except Exception:
                    pass
        rep["next_name"] = acct_mod.next_name(cfg)

        # Trigger auto-sync in background if quota has never been synced or is older than 30 minutes
        quotas = s.get("quota") or {}
        last_sync = max([q.get("synced_at", 0) for q in quotas.values()] or [0])
        acct_list = cfg.get("accounts") or []
        if (now - last_sync > 1800 or any(a["name"] not in quotas for a in acct_list)):
            async def _bg_sync():
                try:
                    for a in acct_list:
                        res = await asyncio.to_thread(kaggle.fetch_account_quota, a, cfg)
                        if res.get("ok"):
                            cur_now = time.time()
                            with common.mutate_state() as cur_s:
                                q = cur_s.setdefault("quota", {})
                                q[a["name"]] = {
                                    "used": res["used_s"],
                                    "refresh_time": res.get("refresh_time"),
                                    "synced_at": cur_now,
                                    "week": common._week_start(cur_now),
                                    "live": True,
                                }
                except Exception:
                    pass
            asyncio.create_task(_bg_sync())

        return rep

    @r.post("/admin/api/accounts/sync-quota")
    @guarded
    async def api_account_sync_quota(request: Request):
        b = await body_of(request)
        target = (b.get("account") or "").strip()
        cfg = R.cfg_fresh()
        accounts_to_sync = [a for a in cfg.get("accounts") or [] if not target or a.get("name") == target]
        if not accounts_to_sync:
            raise Bad(f"account '{target}' not found" if target else "no accounts configured")

        async def _sync_one(acct):
            res = await asyncio.to_thread(kaggle.fetch_account_quota, acct, cfg)
            return acct["name"], res

        results = await asyncio.gather(*[_sync_one(a) for a in accounts_to_sync])
        now = time.time()
        synced = []
        with common.mutate_state() as s:
            q = s.setdefault("quota", {})
            for name, r in results:
                if r.get("ok"):
                    q[name] = {
                        "used": r["used_s"],
                        "refresh_time": r.get("refresh_time"),
                        "synced_at": now,
                        "week": common._week_start(now),
                        "live": True,
                    }
                    synced.append(name)
        bust()
        return ok(msg=f"synced quota for {len(synced)} account(s) from Kaggle" if synced else "could not sync quota (check credentials / network)",
                  synced=synced, results={name: r for name, r in results})

    @r.post("/admin/api/accounts/clear-error")
    @guarded
    async def api_account_clear_error(request: Request):
        b = await body_of(request)
        target = (b.get("account") or "").strip()
        clear_all = bool(b.get("all"))
        now = time.time()
        with common.mutate_state() as s:
            accts = s.setdefault("accounts", {})
            if clear_all:
                for st in accts.values():
                    st["fails"] = 0
                    st["last_error"] = None
                    st["cooldown_until"] = 0
                msg = "cleared errors and cooldowns for all accounts"
            elif target in accts:
                accts[target]["fails"] = 0
                accts[target]["last_error"] = None
                accts[target]["cooldown_until"] = 0
                msg = f"cleared error and cooldown for {target}"
            elif target:
                msg = f"{target} has no error recorded"
            else:
                for st in accts.values():
                    if st.get("cooldown_until", 0) <= now:
                        st["last_error"] = None
                        st["fails"] = 0
                msg = "cleared stale errors"
        bust()
        return ok(msg=msg)


    @r.post("/admin/api/accounts/add")
    @guarded
    async def api_account_add(request: Request):
        b = await body_of(request)
        res = await asyncio.to_thread(acct_mod.add_account, str(b.get("token") or ""), b.get("name"), R.cfg_fresh())
        bust()
        return ok(msg=f"added {res['name']} ({res['user']}). Kaggle only gives GPUs to phone-verified accounts.", **res)

    @r.post("/admin/api/accounts/disable")
    @guarded
    async def api_account_disable(request: Request):
        b = await body_of(request)
        name = str(b.get("name") or "")
        want = bool(b.get("disabled", True))
        await asyncio.to_thread(acct_mod.set_disabled, name, want)
        bust()
        return ok(msg=f"{name} {'disabled: the planner will not launch on it' if want else 'enabled'}")

    @r.post("/admin/api/accounts/remove")
    @guarded
    async def api_account_remove(request: Request):
        b = await body_of(request)
        name = str(b.get("name") or "")
        want_creds = bool(b.get("delete_credentials"))
        if want_creds and b.get("confirm_credentials") != name:
            raise Bad("deleting credential files needs confirm_credentials set to the account name")
        deleted = await asyncio.to_thread(acct_mod.remove_account, name, want_creds)
        bust()
        return ok(msg=f"{name} removed from config.yaml" + (f"; deleted {', '.join(deleted)}" if deleted else
                  ("; credential files were kept" if acct_mod.credential_paths(name)[0].exists() else "")),
                  deleted=deleted)

    # ---- models -------------------------------------------------------------------------------------------
    @r.get("/admin/api/models")
    @guarded
    async def api_models(request: Request):
        cfg, s, now = R.cfg_fresh(), common.read_state(), time.time()
        reg = await asyncio.to_thread(common.load_models)
        return {"models": [model_row(k, m, cfg, s, now) for k, m in reg.items()],
                "default_model": cfg.get("default_model"),
                "default_image_model": cfg.get("default_image_model"),
                "default_video_model": cfg.get("default_video_model")}

    @r.post("/admin/api/models/update")
    @guarded
    async def api_model_update(request: Request):
        b = await body_of(request)
        key = str(b.get("key") or "")
        with common.state_lock():
            txt = common.MODELS_FILE.read_text(encoding="utf-8")
            reg = (yaml.safe_load(txt) or {}).get("models") or {}
            if key not in reg:
                raise Bad(f"unknown model '{key}'", 404)
            changes = clean_model_changes(b.get("changes"), reg[key])
            merged = dict(reg[key])
            for k, v in changes.items():
                merged.pop(k, None) if v is None else merged.__setitem__(k, v)
            bad = media.problems(merged)
            if bad:
                raise Bad("; ".join(bad))
            new = yamledit.update_model(txt, key, changes)
            if new == txt:
                return ok(msg="nothing changed")
            yamledit.atomic_write(common.MODELS_FILE, new)
        note = ""
        s = common.read_state()
        if any(x.get("model_key") == key for x in s.get("backends", [])):
            note = " Running backends keep the old settings until they are relaunched."
        try:
            R._refresh_litellm()
        except Exception as e:
            note += f" (LiteLLM refresh failed: {e})"
        return ok(msg=f"'{key}' updated: {', '.join(changes)}.{note}")

    @r.post("/admin/api/models/delete")
    @guarded
    async def api_model_delete(request: Request):
        b = await body_of(request)
        key = str(b.get("key") or "")
        cfg, s = R.cfg_fresh(), common.read_state()
        if key == cfg.get("default_model"):
            raise Bad("this is the default model: make another one the default first")
        for kind in ("image", "video"):
            if cfg.get(f"default_{kind}_model") == key:
                raise Bad(f"this is the default {kind} model: change that in Settings first")
        if any(x.get("model_key") == key for x in s.get("backends", [])) or \
                any(p.get("model_key") == key for p in (s.get("pending") or {}).values()):
            raise Bad("this model is running or launching: stop it first")
        with common.state_lock():
            txt = common.MODELS_FILE.read_text(encoding="utf-8")
            yamledit.atomic_write(common.MODELS_FILE, yamledit.delete_model(txt, key))
        try:
            R._refresh_litellm()
        except Exception:
            pass
        return ok(msg=f"'{key}' removed from models.yaml (previous file kept as models.yaml.bak)")

    @r.post("/admin/api/models/default")
    @guarded
    async def api_model_default(request: Request):
        b = await body_of(request)
        key = str(b.get("key") or "")
        reg = R._registry()
        if key not in reg:
            raise Bad(f"unknown model '{key}'")
        if media.kind(reg[key]) != "chat":
            raise Bad(f"'{key}' is a {media.kind(reg[key])} model: the default must be a chat model")
        common.set_config_value("default_model", key)
        bust()
        return ok(msg=f"default model is now {key}. A running backend keeps serving what it has: rotate it to switch.")

    @r.post("/admin/api/models/suggest")
    @guarded
    async def api_model_suggest(request: Request):
        """Abliterated alternatives for a model (HuggingFace search, only when asked)."""
        b = await body_of(request)
        src = str(b.get("repo") or b.get("name") or "").strip()
        if not src:
            raise Bad("give a repo or a model name")
        try:
            cfg = R.cfg()
        except Exception:
            cfg = {}
        base = base_name(src)
        try:
            found = await asyncio.to_thread(hf_search, base, 8, cfg.get("hf_token") or None)
        except im.InspectError as e:
            raise Bad(str(e), 502)
        found = [f for f in found if f["repo"].lower() != src.lower()]
        return ok(base=base, results=found,
                  search_url=f"{im.HF}/models?search={urllib.parse.quote(base + ' abliterated')}&library=gguf")

    @r.post("/admin/api/models/search")
    @guarded
    async def api_model_search(request: Request):
        b = await body_of(request)
        q = str(b.get("q") or "").strip()
        if len(q) < 2:
            raise Bad("type at least 2 characters")
        try:
            cfg = R.cfg()
        except Exception:
            cfg = {}
        try:
            found = await asyncio.to_thread(hf_search, q, 12, cfg.get("hf_token") or None, True,
                                            b.get("abliterated", True))
        except im.InspectError as e:
            raise Bad(str(e), 502)
        return ok(results=found)

    # ---- config -------------------------------------------------------------------------------------------
    def config_view():
        cfg = R.cfg_fresh()
        key = cfg.get("gateway_api_key")
        real = auth.real_key(cfg)
        try:
            remote_url = json.loads((common.ROOT / "state" / "remote.json").read_text(encoding="utf-8")).get("url")
        except Exception:
            remote_url = None
        lf = cfg.get("local_fallback") or {}
        return {"gateway_key_set": bool(real), "gateway_key_preview": (str(key)[:8] + "..." + str(key)[-3:]) if real else None,
                "remote_enabled": bool(cfg.get("remote_enabled")), "remote_url": remote_url,
                "tunnel_running": tunnel.CTL.running, "default_model": cfg.get("default_model"),
                "default_image_model": cfg.get("default_image_model"), "default_video_model": cfg.get("default_video_model"),
                "health_interval_seconds": cfg.get("health_interval_seconds", 30), "health_fail_limit": cfg.get("health_fail_limit", 3),
                "session_seconds": cfg.get("session_seconds", 39600), "prelaunch_lead_seconds": cfg.get("prelaunch_lead_seconds", 1500),
                "desired_pool": cfg.get("desired_pool", 0), "local_fallback_enabled": bool(lf.get("enabled")),
                "local_fallback_url": lf.get("url"), "relay": f"{cfg.get('relay_host', '127.0.0.1')}:{cfg.get('relay_port', 8080)}",
                "litellm_enabled": bool(cfg.get("litellm_enabled")), "kernel_slug": cfg.get("kernel_slug"),
                "models_file": str(common.MODELS_FILE), "config_file": str(common.CONFIG_FILE), "root": str(common.ROOT)}

    @r.get("/admin/api/config")
    @guarded
    async def api_config(request: Request):
        return config_view()

    @r.get("/admin/api/config/key")
    @guarded
    async def api_config_key(request: Request):
        return {"key": R.cfg_fresh().get("gateway_api_key") or ""}

    @r.post("/admin/api/config/rotate-key")
    @guarded
    async def api_rotate_key(request: Request):
        key = "sk-kllm-" + secrets.token_urlsafe(24)
        R._set_config_value("gateway_api_key", key)
        note = ""
        try:
            R._refresh_litellm()
        except Exception as e:
            note = f" (restart LiteLLM to apply: {e})"
        return ok(key=key, msg="new gateway key saved: update every remote client." + note)

    def ensure_model_kind(v, kind):
        reg = R._registry()
        if v in (None, ""):
            return "null"
        if v not in reg or media.kind(reg[v]) != kind:
            raise Bad(f"'{v}' is not a {kind} model in models.yaml")
        return v

    @r.post("/admin/api/config/set")
    @guarded
    async def api_config_set(request: Request):
        b = await body_of(request)
        key, v = str(b.get("key") or ""), b.get("value")
        msg = None
        if key == "remote_enabled":
            on = parse_value(("bool", None, None), v, key)
            R._set_config_value("remote_enabled", "true" if on else "false")
            if on:
                tunnel.CTL.start(lambda *a: print(*a, flush=True))
            else:
                tunnel.CTL.stop()
            msg = "remote tunnel " + ("starting: the URL shows up in a few seconds" if on else "stopped")
        elif key in ("default_image_model", "default_video_model"):
            val = ensure_model_kind(v, key.split("_")[1])
            R._set_config_value(key, val)
            msg = f"{key} = {v or 'none'}"
        elif key == "default_model":
            reg = R._registry()
            if v not in reg or media.kind(reg[v]) != "chat":
                raise Bad(f"'{v}' is not a chat model")
            R._set_config_value("default_model", v)
            msg = f"default model = {v}"
        elif key == "local_fallback_enabled":
            on = parse_value(("bool", None, None), v, key)
            lf = R.cfg().get("local_fallback")
            if isinstance(lf, dict) and re.search(r"^local_fallback:\s*\{", common.CONFIG_FILE.read_text(encoding="utf-8"), re.M):
                raise Bad("local_fallback is written in one-line style: edit config.yaml by hand")
            common.set_section_value("local_fallback", "enabled", "true" if on else "false")
            msg = "local fallback " + ("on" if on else "off")
        else:
            spec = {"health_interval_seconds": ("int", 20, 600), "health_fail_limit": ("int", 1, 10),
                    "session_seconds": ("int", 3600, 41400), "prelaunch_lead_seconds": ("int", 300, 7200),
                    "desired_pool": ("int", 0, 10)}.get(key)
            if spec is None:
                raise Bad(f"'{key}' cannot be changed from the panel")
            n = parse_value(spec, v, key)
            R._set_config_value(key, n)
            msg = f"{key} = {n}"
        bust()
        return ok(msg=msg, config=config_view())

    @r.get("/admin/api/config/raw")
    @guarded
    async def api_config_raw(request: Request, file: str = "config"):
        if file == "models":
            return {"file": str(common.MODELS_FILE), "text": common.MODELS_FILE.read_text(encoding="utf-8"), "masked": False}
        if file != "config":
            raise Bad(f"unknown file '{file}' (allowed: config, models)")
        return {"file": str(common.CONFIG_FILE), "text": mask_secrets(common.CONFIG_FILE.read_text(encoding="utf-8")), "masked": True}

    # ---- diagnostics --------------------------------------------------------------------------------------
    @r.post("/admin/api/doctor/run")
    @guarded
    async def api_doctor_run(request: Request):
        started = start_doctor()
        return ok(started=started, msg="doctor started: it sends a few real requests (about a minute)" if started
                  else "doctor is already running")

    @r.get("/admin/api/doctor")
    @guarded
    async def api_doctor(request: Request):
        return doctor_status()

    @r.get("/admin/api/logs/service")
    @guarded
    async def api_service_log(request: Request, n: int = 200, unit: str = "kllm"):
        return await asyncio.to_thread(service_log, n, unit)

    @r.get("/admin/api/logs/requests")
    @guarded
    async def api_log_requests(request: Request, limit: int = 100):
        f = explorer_filters(request)
        recs, scanned = await asyncio.to_thread(metrics.search, f, max(1, min(limit, 1000)))
        return {"records": recs, "scanned": scanned}

    @r.get("/admin/api/logs/requests.csv")
    @guarded
    async def api_log_csv(request: Request):
        f = explorer_filters(request)
        name = "kllm-requests-" + time.strftime("%Y%m%d-%H%M") + ".csv"
        return StreamingResponse(metrics.csv_rows(f), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # ---- housekeeping: rotation, decision history, stale pins ----------------------------------------------
    def housekeeping_once():
        try:
            metrics.rotate_if_needed()
        except OSError:
            pass
        try:
            s = common.read_state()
            metrics.sync_decisions(s)
            live = {b["account"] for b in s.get("backends", [])} | set(s.get("pending", {}))
            if any(a not in live for a in (s.get("pinned") or {})):
                with common.mutate_state() as st:
                    st["pinned"] = {a: t for a, t in (st.get("pinned") or {}).items() if a in live}
        except Exception:
            pass

    async def housekeeping():
        n = 0
        while True:
            try:
                await asyncio.to_thread(housekeeping_once)
                if n % 8 == 0:                       # keep the metrics rollup warm (and saved) every ~2 min
                    await asyncio.to_thread(ROLL.update)
            except Exception as e:
                print("housekeeping error:", repr(e), flush=True)
            n += 1
            await asyncio.sleep(15)

    @r.on_event("startup")
    async def _start_housekeeping():
        asyncio.ensure_future(housekeeping())
        if (R.cfg() or {}).get("panel_telemetry", True):
            telemetry.HUB.start(lambda: R.state().get("backends", []))

    r.housekeeping_once = housekeeping_once
    r.rollup = ROLL
    return r
