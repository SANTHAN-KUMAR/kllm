"""Live kernel resources for the panel: poll each backend's /__kllm/telemetry, keep an hour per account in memory.

Each kernel samples itself every `telemetry_interval` seconds (kernel/server_kernel.tmpl.py, _telemetry_sample) and
serves the samples behind its api key. The relay polls every live backend with `?since=<newest ts>`, so a poll
carries only new samples (about 600 bytes each). Runs inside the relay process, because the panel reads it from there
(the supervisor may run as a separate process).

Read by:
  GET /admin/api/backends/{account}/telemetry?since=   samples for one kernel
  GET /admin/api/telemetry/stream                      server-sent events, newest sample per kernel
  /admin/api/state and /admin/api/backends              a small `res` summary per backend
  /admin/api/alerts                                     memory, OOM kills, GPU heat, disk, silence
"""
import asyncio, collections, time
import httpx

KEEP_SECONDS = 3600
POLL_SECONDS = 5.0
STALE_SECONDS = 30          # no new sample for this long while the backend is up: the kernel or tunnel is stuck
MEM_ALERT = 0.90
GPU_TEMP_ALERT = 85
DISK_ALERT = 0.90


class TelemetryHub:
    def __init__(self, keep_s=KEEP_SECONDS, interval=POLL_SECONDS):
        self.keep_s, self.interval = keep_s, interval
        self.buf = {}           # account -> deque of samples (oldest first)
        self.got_at = {}        # account -> local time a NEW sample last arrived
        self.status = {}        # account -> "ok" | "unsupported" | "error: ..."
        self.first_oom = {}     # account -> cg_ev_oom_kill of the first sample seen (alert on any increase)
        self.seq = 0            # bumped whenever any account gets new samples
        self._cond = None
        self._task = None

    # ---- intake -------------------------------------------------------------------------------------------
    def add(self, account, samples, now=None):
        """Append samples (already ordered by ts) for one account. Returns how many were new."""
        now = now or time.time()
        q = self.buf.setdefault(account, collections.deque(maxlen=int(self.keep_s / 2) + 64))
        last = q[-1]["ts"] if q else 0
        n = 0
        for s in samples:
            if not isinstance(s, dict) or not isinstance(s.get("ts"), (int, float)) or s["ts"] <= last:
                continue
            q.append(s); last = s["ts"]; n += 1
            if account not in self.first_oom and s.get("cg_ev_oom_kill") is not None:
                self.first_oom[account] = s["cg_ev_oom_kill"]
        cut = last - self.keep_s
        while q and q[0]["ts"] < cut:
            q.popleft()
        if n:
            self.got_at[account] = now
            self.status[account] = "ok"
            self.seq += 1
        return n

    def forget_missing(self, live):
        for a in [a for a in self.buf if a not in live]:
            for d in (self.buf, self.got_at, self.status, self.first_oom):
                d.pop(a, None)

    async def poll_once(self, client, backends):
        """backends: list of state backends ({account, url, api_key}). Polls them concurrently."""
        live = {b["account"] for b in backends if b.get("url")}
        self.forget_missing(live)

        async def one(b):
            a = b["account"]
            q = self.buf.get(a)
            since = q[-1]["ts"] if q else 0
            url = b["url"].rstrip("/") + "/__kllm/telemetry"
            try:
                r = await client.get(url, params={"since": since}, timeout=8.0,
                                     headers={"Authorization": "Bearer " + (b.get("api_key") or "")})
            except Exception as e:
                self.status[a] = f"error: {type(e).__name__}"
                return 0
            if r.status_code in (401, 404):
                self.status[a] = "unsupported"      # an older kernel, or telemetry turned off in its config
                return 0
            if r.status_code != 200:
                self.status[a] = f"error: HTTP {r.status_code}"
                return 0
            try:
                rows = r.json().get("samples") or []
            except ValueError:
                self.status[a] = "error: bad JSON"
                return 0
            if not rows and a not in self.status:
                self.status[a] = "ok"
            return self.add(a, rows)

        got = await asyncio.gather(*(one(b) for b in backends if b.get("url")))
        if any(got):
            await self._notify()
        return sum(got)

    async def _notify(self):
        if self._cond is None:
            self._cond = asyncio.Condition()
        async with self._cond:
            self._cond.notify_all()

    async def wait_new(self, seq, timeout):
        """Wait until seq changes (or timeout). Returns the current seq."""
        if self.seq != seq:
            return self.seq
        if self._cond is None:
            self._cond = asyncio.Condition()
        try:
            async with self._cond:
                await asyncio.wait_for(self._cond.wait_for(lambda: self.seq != seq), timeout)
        except asyncio.TimeoutError:
            pass
        return self.seq

    async def run(self, get_backends):
        """Poll loop. get_backends() -> current state backends. Sleeps longer while nothing runs."""
        async with httpx.AsyncClient(follow_redirects=False) as client:
            while True:
                try:
                    bs = get_backends()
                    if bs:
                        await self.poll_once(client, bs)
                    else:
                        self.forget_missing(set())
                except Exception as e:                       # never let the loop die
                    print("telemetry poll error:", repr(e), flush=True)
                await asyncio.sleep(self.interval)

    def start(self, get_backends):
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.run(get_backends))
        return self._task

    # ---- reads ----------------------------------------------------------------------------------------------
    def since(self, account, ts=0.0, limit=720):
        q = self.buf.get(account) or ()
        rows = [s for s in q if s["ts"] > ts]
        return rows[-limit:]

    def latest(self, account):
        q = self.buf.get(account)
        return q[-1] if q else None

    def age(self, account, now=None):
        t = self.got_at.get(account)
        return None if t is None else round((now or time.time()) - t, 1)

    def summary(self, account, now=None):
        """Small per-backend view for Fleet tiles: GPU memory and busy %, memory share, freshness."""
        s = self.latest(account)
        out = {"status": self.status.get(account, "waiting"), "age_s": self.age(account, now)}
        if not s:
            return out
        gpus = s.get("gpu") or []
        out.update(ts=s["ts"],
                   gpu=[{"i": g.get("i"), "used_mib": g.get("used_mib"), "total_mib": g.get("total_mib"),
                         "util": g.get("util"), "temp": g.get("temp")} for g in gpus],
                   gpu_used_mib=sum(g.get("used_mib") or 0 for g in gpus),
                   gpu_total_mib=sum(g.get("total_mib") or 0 for g in gpus),
                   gpu_util=round(sum(g.get("util") or 0 for g in gpus) / len(gpus)) if gpus else None)
        if s.get("cg_used") and s.get("cg_limit"):
            out["mem_used"], out["mem_limit"] = s["cg_used"], s["cg_limit"]
        out["stale"] = out["age_s"] is not None and out["age_s"] > STALE_SECONDS
        return out

    def alerts(self, backends, now=None):
        """Alert rows in the compute_alerts() shape for live backends (BACKEND_SPEC: live kernel resources)."""
        now = now or time.time()
        out = []

        def add(id_, level, text, account):
            out.append({"id": id_, "level": level, "text": text, "tab": "backends", "account": account})

        for b in backends:
            a = b["account"]
            st = self.status.get(a)
            if st in (None, "unsupported"):
                continue
            s = self.latest(a)
            up_s = now - (b.get("ready_at") or b.get("started") or now)
            age = self.age(a, now)
            if (age is None and up_s > 60 and st != "ok") or (age is not None and age > STALE_SECONDS):
                add(f"telemetry-silent-{a}", "warn",
                    f"{a} has sent no resource sample for {int(age if age is not None else up_s)} s: "
                    "the kernel or its tunnel may be stuck.", a)
            if not s:
                continue
            if s.get("cg_used") and s.get("cg_limit") and s["cg_used"] > MEM_ALERT * s["cg_limit"]:
                add(f"mem-{a}", "crit", f"{a} uses {100 * s['cg_used'] / s['cg_limit']:.0f}% of its container memory: "
                    "Kaggle kills the kernel at 100%.", a)
            oom0, oom = self.first_oom.get(a), s.get("cg_ev_oom_kill")
            if oom0 is not None and oom is not None and oom > oom0:
                add(f"oom-{a}", "crit", f"{a} had {oom - oom0} out-of-memory kill(s) in the last hour.", a)
            for g in s.get("gpu") or []:
                if (g.get("temp") or 0) > GPU_TEMP_ALERT:
                    add(f"gpu-temp-{a}-{g.get('i')}", "warn", f"{a} GPU {g.get('i')} is at {g['temp']} °C.", a)
            for path, d in (s.get("disk") or {}).items():
                if d.get("total") and d["used"] > DISK_ALERT * d["total"]:
                    add(f"disk-{a}-{path}", "warn", f"{a} disk {path} is {100 * d['used'] / d['total']:.0f}% full.", a)
        return out


HUB = TelemetryHub()
