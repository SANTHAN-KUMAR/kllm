"""
kllm resource manager: Kaggle GPU kernels are the servers, conversations are the clients.

Two halves, deliberately separated:

  Router  (lives in the relay, per request)
    * resolves the requested model ("kllm" = whatever runs, a registry key = that model)
    * keeps a sticky session table: a conversation is placed ONCE on the least-loaded healthy
      backend of its model and stays there (its prompt cache lives there). It only moves when
      that backend dies or is retired. Adding a backend never moves existing conversations.
    * tracks live load per backend and the demand per model (waiting requests, active
      sessions, in-flight), which the supervisor reads from /__relay/demand.

  plan()  (lives in the supervisor; a PURE function, unit-tested)
    state + demand + config  ->  actions [launch / cancel / retire]
    * desired backends per model from demand (waiting, active sessions / sessions_per_backend)
      and the always-on minimum, capped by usable accounts and max_backends
    * a launch stuck in Kaggle's queue (no boot within queue_timeout) or loading too long stops
      counting -> the planner launches on the next-best account (hedge); surplus launches are
      cancelled as soon as demand is met
    * idle backends are retired (scale to zero if allowed), expiring ones replaced first
    * accounts ranked by remaining quota, measured boot/ready speed, and failure cooldowns
"""
import collections, math, time
from . import media

# ----------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------
DEFAULTS = {
    "enabled": True,                 # demand-driven scaling; False = manual (kllm up/down only)
    "min_backends": 0,               # always keep this many of the default model (panel "Always on")
    "max_backends": 3,               # never run more GPU kernels than this at once
    "sessions_per_backend": 2,       # conversations placed on one backend (placement, rebalance, autoscale demand)
    "slots_per_backend": 2,          # concurrent REQUESTS one backend accepts (= llama-server -np). Kept separate so a
                                     # session's own overlapping requests (subagent, side call) stay on its warm backend
                                     # even with sessions_per_backend: 1
    "session_active_seconds": 900,   # a conversation counts as active this long after its last request
    "idle_stop_minutes": 45,         # retire a backend nobody used for this long (if above the minimum)
    "queue_timeout_seconds": 240,    # kernel not running after this -> treat account as queued, hedge
    "load_timeout_minutes": 25,      # booted but not ready after this -> give up on it
    "manual_queue_timeout_minutes": 120,   # an explicitly launched kernel still waiting in KAGGLE's queue keeps its place this long
    "max_wait_seconds": 1200,        # how long the relay holds a request while a backend starts
    "min_quota_hours": 1.5,          # skip accounts with less weekly GPU quota than this left
    "fail_cooldown_minutes": 20,     # account cooldown after a failed launch (doubles on repeats)
    "queue_cooldown_minutes": 30,    # account cooldown after it sat in the queue
    "max_queued_per_model": 2,       # stop hedging when this many launches already sit in queues
    "spare_backends": 1,             # hot standby while someone is actively working: a backend
                                     # dying (or hitting the 12 h cap) then costs a warm handoff,
                                     # not a ~10 min cold start. 0 = cheapest.
}

def settings(cfg):
    s = dict(DEFAULTS)
    s.update((cfg or {}).get("autoscale") or {})
    # legacy knob from the panel: desired_pool = always-on minimum
    s["min_backends"] = max(int(s.get("min_backends") or 0), int((cfg or {}).get("desired_pool") or 0))
    return s

def disabled_accounts(cfg):
    """Names of accounts switched off in config (`accounts: - {name: kg3, ..., disabled: true}`)."""
    out = set()
    for a in (cfg or {}).get("accounts") or []:
        if isinstance(a, dict) and a.get("disabled") in (True, "true", "True", "yes", 1):
            out.add(a.get("name"))
    return out

# ----------------------------------------------------------------------------------------
# relay side: model resolution, sticky placement, load and demand
# ----------------------------------------------------------------------------------------
# Standard client model aliases (Anthropic, OpenAI, shorthands) mapped to canonical KLLM models
UNIVERSAL_ALIASES = {
    # Shorthands
    "nanbeige": "nanbeige4.3-33b-128k",
    "nanbeige-33b": "nanbeige4.3-33b-128k",
    "nanbeige-128k": "nanbeige4.3-33b-128k",
    "nanbeige4.3": "nanbeige4.3-33b-128k",
    "fable": "nanbeige4.3-33b-128k",
    "opus": "nanbeige4.3-33b-128k",
    "sonnet": "qwen3.8-27b",
    "haiku": "qwen3.8-27b",
    "qwen": "qwen3.8-27b",
    "qwen3.8": "qwen3.8-27b",
    "coder": "qwen3.8-27b",
    "256k": "qwen3.8-27b-256k",
    "long": "qwen3.8-27b-256k",
    "vision": "qwen3.8-27b-vision",
    "abliterated": "qwen3.8-27b-abliterated",
    "qwen3.8-abliterated-vision-256k": "qwen3.8-27b-vision-abliterated-256k",
    "qwen3.8-vision-abliterated-256k": "qwen3.8-27b-vision-abliterated-256k",
    "abliterated-vision-256k": "qwen3.8-27b-vision-abliterated-256k",
    "qwen3.8-27b-abliterated-vision-256k": "qwen3.8-27b-vision-abliterated-256k",   # old alias entry (removed from models.yaml)
    "vision-abliterated-256k": "qwen3.8-27b-vision-abliterated-256k",
    "qwen3.8-27b-abliterated-vision": "qwen3.8-27b-vision-abliterated-256k",
    "qwen3.8-27b-vision-abliterated": "qwen3.8-27b-vision-abliterated-256k",
    # Anthropic Opus & Fable tier
    "claude-3-opus": "nanbeige4.3-33b-128k",
    "claude-3-opus-20240229": "nanbeige4.3-33b-128k",
    "claude-opus-4-0": "nanbeige4.3-33b-128k",
    "claude-opus-4-1": "nanbeige4.3-33b-128k",
    "claude-opus-4-5": "nanbeige4.3-33b-128k",
    "claude-opus-4-6": "nanbeige4.3-33b-128k",
    "claude-opus-4-7": "nanbeige4.3-33b-128k",
    "claude-opus-4-8": "nanbeige4.3-33b-128k",
    "claude-opus-5": "nanbeige4.3-33b-128k",
    "claude-opus-5-5": "nanbeige4.3-33b-128k",
    "claude-fable-5": "nanbeige4.3-33b-128k",
    "claude-fable-5-1": "nanbeige4.3-33b-128k",
    # Anthropic Sonnet tier
    "claude-3-5-sonnet": "qwen3.8-27b",
    "claude-3-5-sonnet-20241022": "qwen3.8-27b",
    "claude-3-5-sonnet-20240620": "qwen3.8-27b",
    "claude-3-7-sonnet": "qwen3.8-27b",
    "claude-3-7-sonnet-20250219": "qwen3.8-27b",
    "claude-sonnet-4": "qwen3.8-27b",
    "claude-sonnet-4-5": "qwen3.8-27b",
    "claude-sonnet-4-6": "qwen3.8-27b",
    "claude-sonnet-5": "qwen3.8-27b",
    "claude-sonnet-5-5": "qwen3.8-27b",
    # Anthropic Haiku tier
    "claude-3-haiku": "qwen3.8-27b",
    "claude-3-haiku-20240307": "qwen3.8-27b",
    "claude-3-5-haiku": "qwen3.8-27b",
    "claude-3-5-haiku-20241022": "qwen3.8-27b",
    "claude-haiku-4-5": "qwen3.8-27b",
    "claude-haiku-5-5": "qwen3.8-27b",
    # OpenAI models
    "gpt-4o": "qwen3.8-27b-vision",
    "gpt-4o-mini": "qwen3.8-27b",
    "o1": "nanbeige4.3-33b-128k",
    "o3-mini": "nanbeige4.3-33b-128k",
}

def resolve_model(requested, registry, default_model, pool_models):
    """Which registry model a request wants. Handles aliases, exact names, and semantic tier matching."""
    if requested in UNIVERSAL_ALIASES and UNIVERSAL_ALIASES[requested] != requested:
        return resolve_model(UNIVERSAL_ALIASES[requested], registry, default_model, pool_models)
    entry = registry.get(requested)
    if isinstance(entry, dict) and entry.get("alias_of"):
        return resolve_model(entry["alias_of"], registry, default_model, pool_models)
    if requested in registry:
        if not pool_models or requested in pool_models:
            return requested
        # If requested model is not currently in pool, but default_model is running in pool and matches family:
        if default_model in pool_models and (
            (requested.startswith("qwen") and default_model.startswith("qwen"))
            or requested.split("-")[0] == default_model.split("-")[0]
        ):
            return default_model
        # If neither requested nor default_model is in pool, but other chat models ARE running:
        chat_pool = [m for m in pool_models if not (media.is_media(registry.get(m)) if registry.get(m) else False)]
        if chat_pool:
            family_match = [m for m in chat_pool if (requested.startswith("qwen") and m.startswith("qwen")) or requested.split("-")[0] == m.split("-")[0]]
            if family_match:
                return max(sorted(set(family_match)), key=lambda m: family_match.count(m))
            return max(sorted(set(chat_pool)), key=lambda m: chat_pool.count(m))
        return requested

    # Semantic family matching for Anthropic / Claude tiers
    req_lower = (requested or "").lower()
    if "opus" in req_lower or "fable" in req_lower:
        for c in ("nanbeige4.3-33b-128k", "nanbeige4.3-33b", "nanbeige"):
            if c in registry:
                return c
    elif "haiku" in req_lower:
        for c in ("qwen3.8-27b", "qwen3-coder-30b-abliterated"):
            if c in registry:
                return c
    elif "sonnet" in req_lower:
        for c in ("qwen3.8-27b", "qwen3.8-27b-256k"):
            if c in registry:
                return c

    pool_models = [m for m in pool_models if not media.is_media(registry.get(m))]   # chat only
    if default_model in pool_models:
        return default_model
    if pool_models:
        # the model with the most backends serving
        return max(sorted(set(pool_models)), key=lambda m: pool_models.count(m))
    return default_model


BUSY_SECONDS = 1800   # a session with a request this recent is "busy": prevents flapping during long prefills
SUSTAINED_SECONDS = 600   # busy AND working for this long: only such sessions justify a hot standby

class Router:
    """Per-request placement and live load. Not thread-safe by itself: the relay is a single
    asyncio loop, so mutations never interleave."""

    def __init__(self):
        self.sessions = {}     # key -> {"backend", "model", "first", "last", "n", "big"}
        self.inflight = {}     # backend label -> requests in flight
        self.last_used = {}    # backend label -> last request time
        self.waiting = {}      # model -> requests currently waiting for a backend
        self.last_ok = {}      # backend label -> last time it answered a request successfully
        self.recent = {}       # backend label -> deque of recent request start times (requests per minute)
        self.client_recent = collections.deque(maxlen=4000)   # (ts, client) of recent request arrivals
        self.ttft_recent = collections.deque(maxlen=2000)     # (ts, ttft seconds) for the fleet traffic card
        self.started = time.time()

    # -- bookkeeping -------------------------------------------------------------------
    def begin(self, label):
        now = time.time()
        self.inflight[label] = self.inflight.get(label, 0) + 1
        self.last_used[label] = now
        q = self.recent.get(label)
        if q is None:
            q = self.recent[label] = collections.deque(maxlen=4000)
        q.append(now)

    def rpm(self, label, window=60.0):
        """Requests started on `label` in the last `window` seconds, scaled to a per-minute rate."""
        q = self.recent.get(label)
        if not q:
            return 0
        cut = time.time() - window
        while q and q[0] < cut:
            q.popleft()
        return round(len(q) * 60.0 / window, 1)

    def client_rpm(self, window=60.0):
        """{client: requests a minute over `window`}."""
        now = time.time()
        cut = now - window
        while self.client_recent and self.client_recent[0][0] < cut - 3599:  # keep a little over an hour
            self.client_recent.popleft()
        counts = {}
        for ts, c in self.client_recent:
            if ts >= cut:
                counts[c] = counts.get(c, 0) + 1
        return {c: round(n * 60.0 / window, 1) for c, n in counts.items()}

    def traffic_series(self, buckets=60, bucket_s=60):
        """Per-`bucket_s` request counts, oldest first, `buckets` entries covering the last
        buckets*bucket_s seconds. Uses the per-backend recent deques (no log scan)."""
        now = time.time()
        cut = now - buckets * bucket_s
        total = [0] * buckets
        for q in self.recent.values():
            for ts in q:
                if ts >= cut:
                    total[int((ts - cut) // bucket_s)] += 1
        return total

    def ttft_p50(self, window=3600):
        """Median time to first token over `window` seconds, from in-memory samples."""
        now = time.time()
        cut = now - window
        vals = [t for ts, t in self.ttft_recent if ts >= cut]
        if not vals:
            return None
        vals.sort()
        m = len(vals) // 2
        return round(vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2, 2)

    def end(self, label):
        self.inflight[label] = max(0, self.inflight.get(label, 0) - 1)
        self.last_used[label] = time.time()

    def note_client(self, client):
        if client:
            self.client_recent.append((time.time(), client))

    def touch_session(self, key, model, body_len, client=None):
        self.note_client(client)
        if not key:
            return None
        now = time.time()
        s = self.sessions.get(key)
        if s is None:
            s = self.sessions[key] = {"backend": None, "model": model, "first": now, "last": now,
                                      "n": 0, "big": False}
        s["last"] = now; s["n"] += 1; s["model"] = model
        if client:
            s["client"] = client
        s["big"] = s["big"] or body_len >= 24000          # ~6K+ tokens: a real agent conversation
        if len(self.sessions) > 5000:                     # bound memory: drop the stalest
            for k, _ in sorted(self.sessions.items(), key=lambda kv: kv[1]["last"])[:1000]:
                self.sessions.pop(k, None)
        return s

    @staticmethod
    def counts(s):
        """Does this conversation count as real demand? Side requests (titles, summaries, a
        one-off probe) and diagnostics (pinned / probe requests) must not make the pool grow."""
        return not s.get("probe") and (s["n"] >= 3 or s["big"])

    def active_sessions(self, window, model=None, backend=None):
        now = time.time()
        return [s for s in self.sessions.values()
                if now - s["last"] <= window and self.counts(s)
                and (model is None or s["model"] == model)
                and (backend is None or s["backend"] == backend)]

    # -- placement ----------------------------------------------------------------------
    def order(self, session, backends, window, bad=(), per_backend=None):
        """Order candidate backends (dicts with label/url/key/model/age_left/lead) for one
        request: the session's own backend first, otherwise least-loaded first.

        Rebalance: if the session's backend holds more active sessions than `per_backend`
        (its slots) while another backend has spare capacity, the NEWEST session there moves
        (smallest context -> cheapest re-prefill); older, cache-heavy sessions stay put."""
        if not backends:
            return []
        healthy = [b for b in backends if b["url"] not in bad] or list(backends)

        def load(b):
            sess = len(self.active_sessions(window, backend=b["label"]))
            soon = 1 if b.get("age_left", 1e9) <= b.get("lead", 0) else 0   # expiring: avoid for NEW sessions
            return (soon, sess + self.inflight.get(b["label"], 0), b["label"])

        ranked = sorted(healthy, key=load)
        if session is not None:
            mine = next((b for b in ranked if b["label"] == session.get("backend")), None)
            if mine is not None and per_backend:
                here = self.active_sessions(window, backend=mine["label"])
                spare = [b for b in ranked if b is not mine and
                         len(self.active_sessions(window, backend=b["label"])) < per_backend]
                newest = max(here, key=lambda x: x["first"]) if here else None
                if len(here) > per_backend and spare and newest is session:
                    session["backend"] = spare[0]["label"]
                    session["moved"] = session.get("moved", 0) + 1
                    return [spare[0]] + [b for b in ranked if b is not spare[0]]
            if mine is not None:
                return [mine] + [b for b in ranked if b is not mine]
            session["backend"] = ranked[0]["label"]       # first placement or re-placement
        return ranked

    def placed(self, session, label):
        if session is not None:
            session["backend"] = label

    # -- demand snapshot for the supervisor --------------------------------------------
    def demand(self, window, pool):
        """pool: {backend label: model key}. Snapshot the supervisor plans from."""
        now = time.time()
        models = {}
        def mrow(m):
            return models.setdefault(m, {"sessions": 0, "waiting": 0, "inflight": 0})
        for sess in self.active_sessions(window):
            mrow(sess["model"])["sessions"] += 1
            is_busy = (now - sess["last"] <= BUSY_SECONDS) or (self.inflight.get(sess.get("backend"), 0) > 0)
            if is_busy:
                mrow(sess["model"])["busy"] = mrow(sess["model"]).get("busy", 0) + 1
                if sess["last"] - sess["first"] >= SUSTAINED_SECONDS:
                    mrow(sess["model"])["sustained"] = mrow(sess["model"]).get("sustained", 0) + 1
        for model, n in self.waiting.items():
            if n:
                mrow(model)["waiting"] += n
        backends = {}
        for label, model in pool.items():
            inflight = self.inflight.get(label, 0)
            if inflight:
                mrow(model)["inflight"] += inflight
            backends[label] = {
                "model": model, "inflight": inflight,
                "sessions": len(self.active_sessions(window, backend=label)),
                # unknown (relay restarted) counts as "just used": never retire on missing data
                "idle_s": round(now - self.last_used.get(label, self.started)),
            }
        return {"ts": now, "models": models, "backends": backends}

# ----------------------------------------------------------------------------------------
# supervisor side: account ranking + the planner (pure)
# ----------------------------------------------------------------------------------------
def account_score(name, stats, quota_left_h):
    """Higher is better. Prefer accounts with more quota left and that historically start fast.
    Accounts on repeat failure are demoted below every account that has never failed — but never
    hard-blocked: one successful boot flips them back to the front."""
    st = stats.get(name) or {}
    ready_min = (st.get("ready_ewma_s") or 600) / 60          # unknown -> assume 10 min
    queue_min = (st.get("queue_ewma_s") or 30) / 60
    fails = st.get("fails", 0)
    score = quota_left_h - 0.5 * ready_min - 2.0 * queue_min - 3.0 * min(fails, 5)
    if fails >= 2:
        score -= 30                                           # repeat failure -> below any unfailed account
    return score


def plan(now, cfg, registry, default_model, backends, pending, accounts, stats, quota_left_h,
         demand, healthy=None, failing_models=None):
    """Decide what to change. Pure: no I/O, returns a list of actions.

    backends : [{account, model_key, started, session_seconds, ready_at?, replaced_by?}]
    pending  : {account: {model_key, started, boot_at?, phase?}}
    accounts : [names] in config order
    stats    : {account: {cooldown_until, ready_ewma_s, queue_ewma_s, fails}}
    quota_left_h : {account: hours of weekly GPU quota left}
    demand   : relay snapshot {"models": {mk: {sessions, waiting}}, "backends": {"kaggle:acct": {inflight, sessions, idle_s}}}
               or None when the relay is unreachable (then nothing is retired for idleness)
    healthy  : {account: bool}; unhealthy backends are not counted as capacity
    failing_models : {model: until_ts} — models that keep failing to load; never launched until then
    Actions  : ("launch", model, account, reason) | ("cancel", account, reason) | ("retire", account, reason)
    """
    s = settings(cfg)
    lead = cfg.get("prelaunch_lead_seconds", 1500)
    healthy = healthy or {}
    # alias keys (alias_of) are the SAME model: a backend launched as an alias must count as capacity for its target
    def _canon(k, depth=0):
        if depth > 5:
            return k
        e = registry.get(k) if isinstance(registry, dict) else None
        if isinstance(e, dict) and e.get("alias_of") and e["alias_of"] != k:
            return _canon(e["alias_of"], depth + 1)
        if k not in (registry or {}) and k in UNIVERSAL_ALIASES and UNIVERSAL_ALIASES[k] != k:
            return _canon(UNIVERSAL_ALIASES[k], depth + 1)
        return k
    backends = [dict(b, model_key=_canon(b.get("model_key"))) for b in backends]
    pending = {a: dict(p, model_key=_canon(p.get("model_key"))) for a, p in (pending or {}).items()}
    # Account allowlist: when set, NEW launches may only use accounts whose name starts with one of
    # these prefixes (e.g. ["kg"]). Already-running backends on other accounts finish normally; they
    # are just never chosen for a new/replacement launch. Empty/absent => all accounts eligible.
    allow = [p for p in (cfg.get("account_allow_prefixes") or []) if p]
    if allow:
        accounts = [a for a in accounts if any(str(a).startswith(p) for p in allow)]
    actions = []
    if not s["enabled"] and not s["min_backends"]:
        return actions

    dmodels = (demand or {}).get("models", {})
    dback = (demand or {}).get("backends", {})
    busy = {b["account"] for b in backends} | set(pending)
    off = disabled_accounts(cfg)      # `disabled: true` on an account: never launch there (running ones finish)

    def age_left(b):
        # A backend's remaining life is the SOONER of its Kaggle session-runtime cap and its
        # weekly GPU quota running out. Folding quota in here makes every consumer below
        # (hard expiry, `serving`, `expiring`, retire) quota-aware, so a backend about to exhaust
        # its weekly quota is replaced and handed off exactly like one nearing the 12 h session cap.
        runtime = b.get("started", now) + b.get("session_seconds", 0) - now
        q = quota_left_h.get(b["account"])
        return min(runtime, q * 3600) if q is not None else runtime

    # 1. hard expiry
    for b in backends:
        if age_left(b) <= 0:
            actions.append(("retire", b["account"], "session budget used up"))

    # candidate accounts for new launches, best first
    def free_accounts():
        out = []
        for a in accounts:
            st = stats.get(a) or {}
            if a in busy or a in off or now < st.get("cooldown_until", 0) \
                    or quota_left_h.get(a, 0) < s["min_quota_hours"]:
                continue
            out.append(a)
        return sorted(out, key=lambda a: -account_score(a, stats, quota_left_h.get(a, 0)))

    models = set(dmodels) | {b["model_key"] for b in backends} | {p["model_key"] for p in pending.values()}
    if s["min_backends"]:
        models.add(default_model)
    # Swarm models require multi-node cluster orchestration; never autoscale as single backends
    models = [m for m in sorted(models) if m in registry and registry[m].get("runtime") != "swarm"]

    total_budget = s["max_backends"]
    running_total = len([b for b in backends if age_left(b) > 0]) + len(pending)

    for m in models:
        mine = [b for b in backends if b["model_key"] == m and age_left(b) > 0]
        serving = [b for b in mine if healthy.get(b["account"], True) and age_left(b) > lead]
        expiring = [b for b in mine if age_left(b) <= lead]
        pend = {a: p for a, p in pending.items() if p["model_key"] == m}

        d = dmodels.get(m, {})
        want = 0
        if s["enabled"]:
            if d.get("waiting") or d.get("sessions"):
                # any active session keeps one backend; only BUSY sessions (and requests already
                # queued for a slot) justify more, so an idle conversation never buys a GPU
                n_busy = d.get("busy", d.get("sessions", 0)) + d.get("waiting", 0)
                # image/video backends run one request at a time: one backend per waiting request
                per = media.MEDIA_SLOTS if media.is_media(registry.get(m)) else s["sessions_per_backend"]
                want = max(1, math.ceil(n_busy / max(1, per)))
                if d.get("sustained") and not media.is_media(registry.get(m)):
                    # hot standby only for sustained chat work (>= 10 min and still busy): a quick
                    # question never buys a second GPU, so there's no launch/cancel churn
                    want += s["spare_backends"]
            # an expiring backend that is still in use needs a successor (its sessions then
            # count towards `want` like any other demand)
            for b in expiring:
                bd = dback.get("kaggle:" + b["account"], {})
                if bd.get("sessions") or bd.get("inflight") or bd.get("idle_s", 0) < s["idle_stop_minutes"] * 60:
                    want = max(want, 1)
        if not mine and not pend and not d.get("waiting"):
            # A recently active session with NOTHING waiting may keep a running backend, but never LAUNCH one from zero:
            # a stopped/closed session counted as "active" for 15 min made the autoscaler relaunch a backend the user had
            # just stopped (and with idle_stop disabled it then ran, unused, until the 11 h cap). A live agent's next
            # request shows up as `waiting` within seconds and launches normally.
            want = 0
        if not serving and want > 1 and not media.is_media(registry.get(m)):
            # cold start: until one backend serves, queued requests cannot be spread over several, and each Kaggle launch
            # takes 8 to 20 min. One launch (plus the hedge below for a stuck one) is the most that helps. A retrying
            # agent looked like 6 busy sessions and bought 6 GPUs for one model in 4 minutes (2026-11-10). Chat only: an
            # image or video kernel runs one job at a time, so each waiting render still gets its own backend.
            want = 1
        if m == default_model:
            want = max(want, s["min_backends"])
        want = min(want, total_budget)

        def stuck(p):
            age = now - p.get("started", now)
            if not p.get("boot_at") and age > s["queue_timeout_seconds"]:
                return "queued"
            # loading is timed from boot: minutes spent in Kaggle's queue are not loading time (a 10 min queue plus a
            # 20 min first build on a fresh account was killed at 25 min, a minute before it registered: 2026-11-10)
            if p.get("boot_at") and now - p["boot_at"] > s["load_timeout_minutes"] * 60:
                return "too slow to load"
            return None

        live_pend = [a for a, p in pend.items() if not stuck(p)]
        stuck_pend = [a for a, p in pend.items() if stuck(p)]

        # 2. launch to cover the deficit (stuck launches don't count -> this is the hedge),
        #    but don't hop endlessly when Kaggle queues everywhere, and never relaunch a model
        #    that keeps failing to load (circuit breaker)
        deficit = want - len(serving) - len(live_pend)
        if (failing_models or {}).get(m, 0) > now:
            deficit = 0
        if len(stuck_pend) >= s["max_queued_per_model"]:
            deficit = 0
        if deficit > 0:
            for a in free_accounts():
                if deficit <= 0 or running_total >= total_budget + len(stuck_pend):
                    break
                why = "hedge: " + ",".join(stuck_pend) + " queued/slow" if stuck_pend else \
                      ("demand" if (d.get("waiting") or d.get("sessions")) else "always-on minimum")
                actions.append(("launch", m, a, why))
                busy.add(a); deficit -= 1; running_total += 1

        # 3. cancel surplus launches once serving covers demand; stuck ones whenever a live
        #    launch or a server already covers their place, and any stuck past load timeout
        manual = {a for a, p in pend.items() if p.get("origin") == "manual"}

        def manual_expired(p):
            # A launch the user asked for is never "not needed". If it booted but is not ready after load_timeout it is
            # broken; if Kaggle has not even started it (busy days: 25+ min queues, seen 2026-11-07) it keeps its place
            # in the queue much longer - cancelling it only re-queues the user's request at the back.
            if p.get("boot_at"):
                return now - p["boot_at"] > s["load_timeout_minutes"] * 60
            return now - p.get("started", now) > s.get("manual_queue_timeout_minutes", 120) * 60

        if len(serving) >= want:
            for a in pend:
                if a not in manual or manual_expired(pend[a]):
                    actions.append(("cancel", a, "not needed any more" if a not in manual else "too slow to load"))
        else:
            room = want - len(serving)
            # keep the most advanced launches: booted before queued, then the oldest
            keep = sorted(live_pend, key=lambda a: (0 if pend[a].get("boot_at") else 1,
                                                    pend[a].get("boot_at") or pend[a].get("started", now)))
            for a in keep[room:]:
                if a not in manual:
                    actions.append(("cancel", a, "surplus launch"))
            # a queued launch keeps its place in Kaggle's queue until something else actually
            # RUNS for that slot (booted or serving) — abandoning it earlier just hops queues
            booted = [a for a in live_pend if pend[a].get("boot_at")]
            covered = len(booted) >= room
            for a in stuck_pend:
                p = pend[a]
                too_slow = manual_expired(p) if a in manual else (now - p.get("started", now)) > s["load_timeout_minutes"] * 60
                if too_slow or (covered and a not in manual):
                    actions.append(("cancel", a, stuck(p)))

        # 4. retire: expiring backends once a successor serves; idle backends above `want`
        for b in expiring:
            if age_left(b) > 0 and len(serving) >= max(1, want):
                bd = dback.get("kaggle:" + b["account"], {})
                # the relay warm-hands its sessions to the successor first; retire once empty
                if (not bd.get("inflight") and not bd.get("sessions")) or age_left(b) < 120:
                    actions.append(("retire", b["account"], "replaced before session cap"))
        if demand is not None and s["enabled"]:
            extra = len(serving) - want
            idle = sorted((b for b in serving
                           if dback.get("kaggle:" + b["account"], {}).get("idle_s", 0) >= s["idle_stop_minutes"] * 60
                           and not dback.get("kaggle:" + b["account"], {}).get("inflight")),
                          key=lambda b: -dback.get("kaggle:" + b["account"], {}).get("idle_s", 0))
            for b in idle[:max(0, extra)]:
                actions.append(("retire", b["account"], f"idle {dback['kaggle:' + b['account']]['idle_s'] // 60} min"))

    # de-duplicate (an account can be named by two rules)
    seen, out = set(), []
    for act in actions:
        k = (act[0], act[1] if act[0] != "launch" else act[2])
        if k not in seen:
            seen.add(k); out.append(act)
    return out
