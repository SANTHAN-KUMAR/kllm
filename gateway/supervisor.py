"""
kllm supervisor — the control loop that runs the Kaggle GPU pool like a server fleet
(`kllm watch`, or the kllm-supervisor systemd service).

Every tick (health_interval_seconds):
  1. observe   read every kernel's latest status from the gist once; update each pending launch
               (booted? which phase?), record failures (with account cooldown + backoff)
  2. promote   pending launches that report a ready tunnel and accept our key -> pool
               (measured queue and ready times feed the account ranking)
  3. health    authenticated check of every serving backend, in parallel; a backend failing
               `health_fail_limit` times in a row is dropped (its sessions re-place themselves)
  4. demand    ask the relay what clients want (waiting requests, active sessions, load)
  5. plan      scheduler.plan(): pure policy -> launch / cancel / retire actions
  6. act       launches and stops run in background threads; the loop never blocks on Kaggle

All policy lives in scheduler.plan() (unit-tested). This file is mechanism only.
"""
import json, random, time, threading, urllib.request, urllib.error

TUNNEL_GRACE_S = 600             # keep an unreachable-but-heartbeating kernel this long before replacing it
KERNEL_HEARTBEAT_FRESH_S = 480   # gist: kernel 'alive' beats every ~5-6 min; fresh = seen within this window
KERNEL_HEARTBEAT_FRESH_NTFY_S = 1500   # ntfy: beats every ~10-11 min (ntfy.sh per-IP message budget)
from concurrent.futures import ThreadPoolExecutor
from . import common, orchestrator as orch, rendezvous, scheduler, kaggle


def _beat_window(cfg):
    return KERNEL_HEARTBEAT_FRESH_NTFY_S if rendezvous.mode(cfg) == "ntfy" else KERNEL_HEARTBEAT_FRESH_S

def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def _internet_ok():
    for url in ("https://api.github.com", "https://www.kaggle.com"):
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=6)
            return True
        except urllib.error.HTTPError:
            return True          # an HTTP answer still proves connectivity
        except Exception:
            continue
    return False

def apply_pins(demand, state):
    """Backends pinned in the panel look "just used" to the planner, so idle-stop never retires them."""
    for a in (state.get("pinned") or {}):
        row = ((demand or {}).get("backends") or {}).get("kaggle:" + a)
        if row is not None:
            row["idle_s"] = 0
    return demand

def _pinned_accounts(state):
    return set((state.get("pinned") or {}).keys())

def _ewma(old, new, alpha=0.4):
    return new if not old else round(alpha * new + (1 - alpha) * old, 1)

class Supervisor:
    def __init__(self):
        self.cfg = common.load_config()
        self.fails = {}            # account -> consecutive health failures
        self.first_fail = {}       # account -> time of the first failure in the current outage
        self._kstat = {}           # account -> (checked_at, kernel_running)
        self._pending_kstat = {}   # account -> (checked_at, kernel status string)
        self.pushing = {}          # account -> (model, started) for launches being pushed right now
        self.stopping = set()      # accounts with a stop in progress
        self.surplus_at = {}       # model -> when the planner last cancelled surplus hedges for it
        self.lock = threading.Lock()
        self.last_tick = time.time()
        self.pool_exec = ThreadPoolExecutor(max_workers=8)

    # ---- account statistics (persisted in state["accounts"]) ----------------------------
    def _stat(self, account, **upd):
        with common.mutate_state() as s:
            st = s.setdefault("accounts", {}).setdefault(account, {})
            for k, v in upd.items():
                if callable(v):
                    st[k] = v(st.get(k))
                else:
                    st[k] = v

    def _fail(self, account, why, minutes=None):
        st = (common.read_state().get("accounts") or {}).get(account, {})
        n = min(st.get("fails", 0) + 1, 6)
        sset = scheduler.settings(self.cfg)
        cool = (minutes or sset["fail_cooldown_minutes"]) * 60 * (2 ** (n - 1) if minutes is None else 1)
        self._stat(account, fails=n, cooldown_until=time.time() + cool, last_error=why[:200])
        _log(f"[{account}] {why} -> cooldown {int(cool // 60)} min")

    def _model_failed(self, model, account, why):
        """Circuit breaker: a model that fails to load on 2 different accounts within 2 h is
        broken (bad repo, OOM, HF outage) — stop burning accounts on it for an hour and let the
        relay fail its waiting requests fast with the reason."""
        if not model:
            return
        now = time.time()
        with common.mutate_state() as s:
            mf = s.setdefault("models_failing", {}).setdefault(model, {"accounts": {}, "until": 0})
            mf["accounts"] = {a: t for a, t in mf["accounts"].items() if now - t < 7200}
            mf["accounts"][account] = now
            mf["reason"] = why[:300]
            if len(mf["accounts"]) >= 2:
                mf["until"] = now + 3600
                _log(f"model {model} failed on {sorted(mf['accounts'])}: paused for 1 h ({why})")

    # ---- actions ------------------------------------------------------------------------
    PUSH_MIN_GAP = 20      # seconds between Kaggle pushes (all accounts)
    PUSH_MAX_HOUR = 12     # pushes per rolling hour: hedging can never turn into a burst

    def _launch(self, model, account, reason):
        reg = common.load_models()
        if reg.get(model, {}).get("runtime") == "swarm":
            _log(f"supervisor: ignoring autoscale request for swarm model '{model}' (requires cluster launch)")
            return
        with self.lock:
            if account in self.pushing:
                return
            now = time.time()
            self.push_log = [t for t in getattr(self, "push_log", []) if now - t < 3600]
            min_gap = int((self.cfg or {}).get("push_min_gap", 5 if (self.cfg or {}).get("swarm", {}).get("enabled") else self.PUSH_MIN_GAP))
            if self.push_log and now - self.push_log[-1] < min_gap:
                return                       # next tick
            if len(self.push_log) >= self.PUSH_MAX_HOUR:
                _log(f"push budget ({self.PUSH_MAX_HOUR}/h) reached; holding launch of {model}")
                return
            self.push_log.append(now)
            self.pushing[account] = (model, now)
        def run():
            seen = []
            def lg(*a):
                seen.append(" ".join(str(x) for x in a))
                _log(f"  [{account}]", *a)
            try:
                _log(f"[{account}] launch {model} ({reason})")
                r = orch.launch(model_key=model, account=account, wait=False, origin="auto", log=lg)
                if not r and any("weekly GPU quota" in m for m in seen):
                    # Kaggle's own count beats kllm's estimate: retrying every 20 min only fails again
                    until = common.mark_quota_exhausted(account)
                    self._fail(account, "Kaggle says this account's weekly GPU quota is used up",
                               minutes=max(1, int((until - time.time()) // 60)))
                elif not r:
                    self._fail(account, "push failed")
            except Exception as e:
                self._fail(account, f"launch error: {e}")
            finally:
                with self.lock:
                    self.pushing.pop(account, None)
        threading.Thread(target=run, daemon=True).start()

    def _kernel_running(self, b):
        """Is this backend's Kaggle kernel still RUNNING? Asked of Kaggle itself, so it does not depend on the tunnel
        or on the (rate-limited, shared) gist heartbeat. Cached for 20 s per account."""
        a = b["account"]
        c = self._kstat.get(a)
        if c and time.time() - c[0] < 20:
            return c[1]
        try:
            st = kaggle.kernel_status(a, b.get("kernel_ref") or "")
        except Exception:
            st = "UNKNOWN"
        ok = st in ("RUNNING", "UNKNOWN")      # UNKNOWN (API hiccup): do not kill a possibly healthy kernel on missing data
        self._kstat[a] = (time.time(), ok)
        return ok

    def _stop(self, account, reason, queued=False):
        with self.lock:
            if account in self.stopping:
                return
            self.stopping.add(account)
        def run():
            try:
                _log(f"[{account}] stop ({reason})")
                orch.stop(account, log=lambda *a: _log(" ", *a))
                if queued:
                    self._fail(account, "sat in Kaggle's queue",
                               minutes=scheduler.settings(self.cfg)["queue_cooldown_minutes"])
            finally:
                with self.lock:
                    self.stopping.discard(account)
        threading.Thread(target=run, daemon=True).start()

    DEAD_STATUS = ("ERROR", "CANCELED", "CANCELLED", "COMPLETE")   # kernel finished (or died) without announcing ready
    DEAD_AFTER_S = 180                                             # a young kernel may take minutes to die; don't judge it
    DEAD_CHECK_INTERVAL_S = 60                                     # one Kaggle status call per account per minute at most

    def _pending_kernel_status(self, p):
        """Kaggle kernel status of a pending launch, rate-limited per account. UNKNOWN on any
        API trouble: a missing answer never fails a launch."""
        acct = p.get("account")
        c = self._pending_kstat.get(acct)
        if c and time.time() - c[0] < self.DEAD_CHECK_INTERVAL_S:
            return c[1]
        try:
            st = kaggle.kernel_status(acct, p.get("kernel_ref") or "") or "UNKNOWN"
        except Exception:
            st = "UNKNOWN"
        st = st or "UNKNOWN"
        self._pending_kstat[acct] = (time.time(), st)
        return st

    def _check_dead_pending_launches(self, state, by_acct, now, skew=120):
        """A launch whose kernel died before announcing anything is not a 'queued' launch; ask
        Kaggle directly so the panel shows it as failed instead of 'queue active' forever."""
        for acct, p in list((state.get("pending") or {}).items()):
            if p.get("failed"):                           # an immediate push-time failure — no kernel exists
                if now - p.get("started", now) > 600:     # let the panel show it for a while
                    with common.mutate_state() as s:
                        s["pending"].pop(acct, None)
                continue
            e = by_acct.get(acct)
            if e and e.get("ts", 0) >= p.get("since", 0) - skew:
                continue                               # a fresh gist signal handles this pending
            if now - p.get("started", now) < self.DEAD_AFTER_S:
                continue
            kstat = self._pending_kernel_status(p)
            if kstat not in self.DEAD_STATUS:
                continue
            why = f"kernel {kstat.lower()} on Kaggle (dead launch; GPU not allocated or kernel crashed)"
            self._fail(acct, why)
            self._stop(acct, "failed launch")

    # ---- the loop -----------------------------------------------------------------------
    def tick(self):
        self.cfg = common.load_config()
        now = time.time()
        state = common.read_state()

        # 1. observe kernels (one gist read per tick). If GitHub can't be read, kernel progress
        #    is UNKNOWN: nothing is hedged, cancelled or promoted on missing data this tick.
        # Read the gist only when there is something to learn from it: a launch in progress
        # (every tick) or serving backends (every ~3rd tick, to notice tunnel URL changes).
        self.n_tick = getattr(self, "n_tick", 0) + 1
        failing_now = any(self.fails.get(b["account"], 0) > 0 for b in state.get("backends", []))
        need_gist = bool(state.get("pending")) or bool(self.pushing) or failing_now or \
            (state.get("backends") and self.n_tick % 3 == 0)      # failing backend: read EVERY tick so a re-announced tunnel URL is adopted fast
        if need_gist:
            gist_ok, events = rendezvous.latest_events_checked(self.cfg)
        else:
            gist_ok, events = True, []
        by_acct = {}
        for e in events:
            by_acct.setdefault(e.get("account"), e)
        SKEW = 120   # tolerate a Kaggle/PC clock difference; promotion still requires OUR api key
        for acct, p in list(state.get("pending", {}).items()):
            e = by_acct.get(acct)
            if not e or e.get("ts", 0) < p.get("since", 0) - SKEW:
                continue
            if e.get("event") in ("error", "crash"):
                stage = e.get("stage") or ""
                why = f"kernel {e['event']}: {stage or e.get('err') or ''}"
                if stage == "no_gpu":
                    # Kaggle gave no GPU: weekly quota used up (or account not phone-verified)
                    self._fail(acct, why + " (GPU quota exhausted or not verified)", minutes=12 * 60)
                else:
                    self._fail(acct, why)
                    if stage in ("server_health", "download", "tunnel") or e.get("event") == "crash":
                        self._model_failed(p.get("model_key"), acct, why)
                self._stop(acct, "failed launch")
                continue
            boot = e.get("ts", now) - e.get("t", 0)
            if not p.get("boot_at") or p.get("phase") != e.get("event") or \
                    (e.get("bytes_done") is not None and e.get("bytes_done") != p.get("bytes_done")):
                first_boot = False
                with common.mutate_state() as s:
                    q = s["pending"].get(acct)
                    if q is not None:
                        if not q.get("boot_at"):
                            q["boot_at"] = boot; first_boot = True
                        q["phase"] = e.get("event")
                        if e.get("bytes_done") is not None:           # download progress from the kernel
                            q["bytes_done"] = int(e["bytes_done"])
                            if e.get("bytes_total"):
                                q["bytes_total"] = int(e["bytes_total"])
                            q["bytes_at"] = int(e.get("ts") or now)
                # separate write: a nested state write here would be overwritten by the outer one
                if first_boot:
                    self._stat(acct, queue_ewma_s=lambda o, b=boot - p["started"]: _ewma(o, max(0, b)))

        # no gist signal for these pending launches? ask Kaggle directly — a kernel that died
        # (no GPU allocated, instant crash) must not read as 'queue active' in the panel forever
        self._check_dead_pending_launches(state, by_acct, now)

        # 1b. a kernel whose tunnel restarted announces a NEW url: adopt it instead of declaring
        #     a healthy kernel dead
        for b in state.get("backends", []):
            e = by_acct.get(b["account"])
            if e and e.get("url") and e.get("ts", 0) >= b.get("ready_at", 0) - SKEW \
                    and e.get("event") in ("ready", "alive") and e["url"] != b.get("url") \
                    and orch.health_check(e["url"], b["api_key"], timeout=15):
                with common.mutate_state() as s2:
                    for x in s2["backends"]:
                        if x["account"] == b["account"]:
                            x["url"] = e["url"]
                _log(f"[{b['account']}] tunnel moved -> {e['url']} (adopted)")

        # 2. promote
        for acct, p in list(common.read_state().get("pending", {}).items()):
            if not gist_ok:
                break
            try:
                if orch.try_promote_pending(self.cfg, p, log=_log, events=events):
                    self._stat(acct, fails=0, ready_ewma_s=lambda o, r=time.time() - p["started"]: _ewma(o, r))
            except Exception as e:
                _log(f"[{acct}] promote error: {e}")

        # quota accrual: serving backends and booted launches use GPU time
        state = common.read_state()
        dt = now - self.last_tick
        self.last_tick = now
        if 0 < dt < 600:
            for b in state["backends"]:
                common.add_quota(b["account"], dt)
            for a, p in state["pending"].items():
                if p.get("boot_at"):
                    common.add_quota(a, dt)

        # 3. demand (also carries the relay's live error signal per backend)
        demand = None
        try:
            port = self.cfg.get("relay_port", 8080)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/__relay/demand", timeout=5) as r:
                demand = json.load(r)
        except Exception:
            pass

        # a pinned backend (panel) is never retired for being idle
        apply_pins(demand, state)

        # 4. health, in parallel
        limit = self.cfg["health_fail_limit"]
        live = [b for b in state["backends"] if b.get("url") and
                (b.get("started", now) + b.get("session_seconds", 0)) > now]
        dback0 = (demand or {}).get("backends", {})
        def probe(b):
            # passive health: a backend that answered real traffic in the last minute is alive;
            # only probe the quiet ones (fewer requests through the tunnels, less bot-like)
            ok_age = dback0.get("kaggle:" + b["account"], {}).get("ok_age_s")
            if ok_age is not None and ok_age < 60:
                return True
            return orch.health_check(b["url"], b["api_key"], timeout=20)
        oks = dict(zip([b["account"] for b in live], self.pool_exec.map(probe, live)))
        if live and not any(oks.values()) and not _internet_ok():
            # every backend "failed" because THIS machine is offline (wifi drop, suspend):
            # never stop healthy Kaggle kernels for a local outage
            _log("network looks down from here; skipping health verdicts this tick")
            oks = {a: True for a in oks}
        with common.mutate_state() as s3:
            s3["health"] = {a: {"ok": bool(v), "ts": int(now)} for a, v in oks.items()}
        healthy = {}
        dback = (demand or {}).get("backends", {})
        for b in live:
            a = b["account"]
            relay_errs = dback.get("kaggle:" + a, {}).get("errors_5m", 0)
            if oks[a]:
                self.fails[a] = 0
                self.first_fail.pop(a, None)
                healthy[a] = relay_errs < 3          # flapping: don't count it as capacity
            else:
                self.fails[a] = self.fails.get(a, 0) + 1
                healthy[a] = False
                self.first_fail.setdefault(a, now)
                ev0 = by_acct.get(a)
                gist_fresh = bool(ev0 and now - ev0.get("ts", 0) < _beat_window(self.cfg)
                                  and ev0.get("event") in ("alive", "ready", "serving", "tunnel_recovering", "tunnel_drop"))
                if now - self.first_fail[a] < TUNNEL_GRACE_S and (gist_fresh or self._kernel_running(b)):
                    # kernel alive, tunnel (probably) down: keep counting it as capacity from the FIRST failed check so the
                    # planner does not launch a cold replacement while the kernel re-opens its tunnel (the warm prompt
                    # cache lives in this kernel's llama-server)
                    healthy[a] = True
                    self.fails[a] = min(self.fails[a], limit - 1)
                    if self.fails[a] == 1 or int(now - self.first_fail[a]) % 120 < 35:
                        _log(f"[{a}] unreachable for {int(now - self.first_fail[a])}s but its kernel is alive: waiting for tunnel recovery")
                    continue
                if self.fails[a] >= limit:
                    # Tunnel recovery grace: an unanswered health check usually means the TUNNEL is down, not the
                    # kernel. While the kernel keeps heartbeating (or announcing a new tunnel) its llama-server and
                    # warm prompt cache are alive: stopping it here is what created cold instances that had to
                    # re-prefill 100K+ tokens. Bounded (TUNNEL_GRACE_S) so a really dead kernel is still replaced.
                    ev = by_acct.get(a)
                    kernel_alive = (not gist_ok) or bool(
                        ev and now - ev.get("ts", 0) < _beat_window(self.cfg)
                        and ev.get("event") in ("alive", "ready", "serving", "tunnel_recovering", "tunnel_drop"))
                    if kernel_alive and now - self.first_fail[a] < TUNNEL_GRACE_S:
                        healthy[a] = True            # still capacity: do not plan a cold replacement
                        self.fails[a] = limit - 1    # keep probing every tick; do not reset the clock
                        if int(now - self.first_fail[a]) % 120 < 35:
                            _log(f"[{a}] unreachable for {int(now - self.first_fail[a])}s but kernel is alive "
                                 f"(last event: {ev.get('event') if ev else 'gist unreadable'}): waiting for tunnel recovery")
                        continue
                    self.fails[a] = 0
                    self.first_fail.pop(a, None)
                    if a in _pinned_accounts(state) or (b and b.get("model_key", "").endswith("-swarm")):
                        _log(f"[{a}] health fail {limit}/{limit} but account is pinned/swarm; skipping stop")
                    else:
                        self._fail(a, "backend stopped answering", minutes=10)
                        self._stop(a, "unhealthy")

        # 5. plan (launches being pushed right now count as pending)
        state = common.read_state()
        pending = dict(state.get("pending", {}))
        with self.lock:
            for a, (m, t0) in self.pushing.items():
                pending.setdefault(a, {"model_key": m, "started": t0, "origin": "auto"})
            stopping = set(self.stopping)
        backends = [b for b in state["backends"] if b["account"] not in stopping]
        pending = {a: p for a, p in pending.items() if a not in stopping}
        quota_left = {a["name"]: common.quota_left_h(a["name"], state) for a in self.cfg["accounts"]}
        if not gist_ok:
            # unknown kernel progress: present every pending launch as booted so the planner
            # neither hedges nor cancels on missing data
            pending = {a: dict(p, boot_at=p.get("boot_at") or p.get("started", now)) for a, p in pending.items()}
        failing = {m: v.get("until", 0) for m, v in (state.get("models_failing") or {}).items()}
        actions = scheduler.plan(now, self.cfg, common.load_models(), self.cfg["default_model"],
                                 backends, pending, [a["name"] for a in self.cfg["accounts"]],
                                 state.get("accounts") or {}, quota_left, demand, healthy, failing)

        # 6. act. Damp hedging: right after the planner cancelled surplus hedges for a model, do not hedge it again
        # for 15 min (seen 2026-11-10: launch, cancel as surplus, relaunch 6 min later, while a manual launch queued)
        for act in actions:
            if act[0] == "cancel" and act[2] == "surplus launch":
                m = (pending.get(act[1]) or {}).get("model_key")
                if m:
                    self.surplus_at[m] = now
        actions = [a for a in actions if not (a[0] == "launch" and str(a[3]).startswith("hedge")
                                              and now - self.surplus_at.get(a[1], 0) < 900)]
        for act in actions:
            if act[0] == "launch":
                self._launch(act[1], act[2], act[3])
            elif act[0] in ("cancel", "retire"):
                self._stop(act[1], act[2], queued=(act[0] == "cancel" and act[2] == "queued"))
        with common.mutate_state() as s:
            sch = s.setdefault("scheduler", {})
            sch["tick"] = now
            sch["demand"] = (demand or {}).get("models")
            if actions:
                log = sch.setdefault("log", [])
                log.extend([{"ts": int(now), "action": a[0], "target": a[2] if a[0] == "launch" else a[1],
                             "model": a[1] if a[0] == "launch" else None, "why": a[-1]} for a in actions])
                del log[:-40]

    def run(self):
        sset = scheduler.settings(self.cfg)
        _log(f"supervisor up: autoscale={'on' if sset['enabled'] else 'off'}, min={sset['min_backends']}, "
             f"max={sset['max_backends']}, tick {self.cfg['health_interval_seconds']}s")
        while True:
            try:
                self.tick()
            except Exception as e:
                _log("tick error:", repr(e))
            base = max(20, self.cfg.get("health_interval_seconds", 30))
            time.sleep(base * random.uniform(0.75, 1.25))      # jitter: no fixed-period pattern

def supervise(desired=None):
    if desired:
        # legacy `kllm watch --size N`: an always-on minimum for this run
        cfg = common.load_config()
        if int(cfg.get("desired_pool") or 0) != desired:
            common.set_config_value("desired_pool", desired)
    Supervisor().run()
