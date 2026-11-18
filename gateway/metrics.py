"""Usage and compute metrics for the panel, from logs/requests.jsonl and state.json.

The request log grows with every model request (and is rotated, see `rotate_if_needed`), so it is
never re-read as a whole: a small hourly *rollup* (state/metrics_rollup.json) is kept and brought up
to date by reading only the bytes appended since the last call (the file's inode + offset are stored in
the rollup, rotation is followed). Every view (24 h, 7 d, by model / backend / hour / day, percentiles,
speed vs context size) is computed from the rollup, so a call costs O(new bytes), not O(log size).

Memory: one pass, line by line; per hour only sums plus small bounded sample lists for percentiles.
"""
import json, os, random, threading, time
from . import common

HOUR = 3600
RETAIN_HOURS = 31 * 24            # rollup keeps a month of hourly buckets (backs the 30d view)
SAMPLE_CAP = 300                  # percentile samples kept per hour (reservoir)
SAVE_EVERY = 30                   # seconds between rollup writes when new data keeps arriving
VERSION = 2
RANGES = {"1h": HOUR, "6h": 6 * HOUR, "24h": 24 * HOUR, "7d": 7 * 24 * HOUR, "14d": 14 * 24 * HOUR,
          "30d": 30 * 24 * HOUR}
CTX_BUCKETS = [(0, 4096, "<4K"), (4096, 16384, "4-16K"), (16384, 32768, "16-32K"),
               (32768, 65536, "32-64K"), (65536, 131072, "64-128K"), (131072, None, "128K+")]
SUM_FIELDS = ("n", "err", "loop", "new_in", "cached_in", "out", "think_c", "text_c", "think_tok", "text_tok",
              "total_s", "ttft_sum", "ttft_n", "gen_sum", "gen_n")

REQ_LOG = common.LOG_DIR / "requests.jsonl"
ROLLUP_FILE = common.ROOT / "state" / "metrics_rollup.json"
DECISIONS_FILE = common.ROOT / "state" / "decisions.jsonl"


# ---- log rotation --------------------------------------------------------------------------------------
def rotate_if_needed(path=None, max_bytes=50 * 1024 * 1024, keep=3):
    """requests.jsonl -> .1 -> .2 -> .3 (the oldest is dropped) once it exceeds max_bytes.
    The relay opens the file in append mode for every record, so renaming it is safe: the next
    record simply starts a fresh file. Returns True when it rotated."""
    path = str(path or REQ_LOG)
    try:
        if os.path.getsize(path) <= max_bytes:
            return False
    except OSError:
        return False
    for i in range(keep, 0, -1):
        src = path if i == 1 else f"{path}.{i - 1}"
        dst = f"{path}.{i}"
        if os.path.exists(src):
            os.replace(src, dst)
    return True


def log_files(path=None, keep=3):
    """Existing log files, oldest first (requests.jsonl.3 ... requests.jsonl)."""
    path = str(path or REQ_LOG)
    out = [f"{path}.{i}" for i in range(keep, 0, -1)] + [path]
    return [p for p in out if os.path.exists(p)]


# ---- helpers -------------------------------------------------------------------------------------------
def _num(v, default=0):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _ctx_bucket(tokens):
    for i, (lo, hi, _) in enumerate(CTX_BUCKETS):
        if tokens >= lo and (hi is None or tokens < hi):
            return i
    return 0


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _reservoir_add(lst, value, seen, rng):
    """Bounded uniform sample of everything `seen` so far."""
    if len(lst) < SAMPLE_CAP:
        lst.append(value)
    else:
        j = rng.randrange(seen)
        if j < SAMPLE_CAP:
            lst[j] = value


def _blank_row():
    return {f: 0 for f in SUM_FIELDS}


def _ts_fast(line):
    """Leading `{"ts": 1759...,` without a full JSON parse (None when the line is shaped differently)."""
    if line.startswith(b'{"ts": ') and len(line) > 17 and line[7:17].isdigit():
        return int(line[7:17])
    return None


# ---- the rollup ----------------------------------------------------------------------------------------
class Rollup:
    def __init__(self, path=None, cache=None):
        self.path = str(path or REQ_LOG)
        self.cache = str(cache or ROLLUP_FILE)
        self.lock = threading.Lock()
        self.rng = random.Random(7)
        self.hours = {}          # str(hour start) -> bucket
        self.inode = None
        self.offset = 0
        self.dirty = False
        self.last_save = 0.0
        self.parsed_bytes = 0    # for tests / diagnostics: bytes parsed since this object was created
        self._load()

    # -- persistence
    def _load(self):
        try:
            with open(self.cache, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("v") == VERSION and d.get("path") == self.path:
                self.hours, self.inode, self.offset = d["hours"], d.get("inode"), d.get("offset", 0)
        except (OSError, ValueError, KeyError):
            pass

    def _save(self, force=False):
        if not self.dirty or (not force and time.time() - self.last_save < SAVE_EVERY):
            return
        tmp = f"{self.cache}.tmp{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"v": VERSION, "path": self.path, "inode": self.inode, "offset": self.offset,
                           "hours": self.hours}, f, separators=(",", ":"))
            os.replace(tmp, self.cache)
            self.dirty = False
            self.last_save = time.time()
        except OSError:
            pass

    # -- ingest
    def _bucket(self, hour):
        b = self.hours.get(hour)
        if b is None:
            b = self.hours[hour] = {"rows": {}, "ttft": [], "tps": [], "ctx": {}, "seen": {"ttft": 0, "tps": 0}}
        return b

    def add(self, r):
        ts = _num(r.get("ts"), None)
        if ts is None:
            return
        b = self._bucket(str(int(ts // HOUR * HOUR)))
        key = f"{r.get('model') or '?'}|{r.get('backend') or '?'}|{r.get('kind') or 'chat'}"
        row = b["rows"].setdefault(key, _blank_row())
        row["n"] += 1
        status = r.get("status")
        ok = isinstance(status, int) and 200 <= status < 300
        if not ok:
            row["err"] += 1
        if r.get("loop_cut"):
            row["loop"] += 1
        new_in, cached = _num(r.get("new_in")), _num(r.get("cached_in"))
        out = _num(r.get("out"))
        row["new_in"] += new_in
        row["cached_in"] += cached
        row["out"] += out
        tc, xc = _num(r.get("think_chars")), _num(r.get("text_chars"))
        row["think_c"] += tc
        row["text_c"] += xc
        if out and tc + xc:
            t = out * tc / (tc + xc)
            row["think_tok"] += t
            row["text_tok"] += out - t
        row["total_s"] += _num(r.get("total_s"))
        if not ok:
            return
        ttft, tps = r.get("ttft_s"), _num(r.get("gen_tps"))
        ctx = new_in + cached or _num(r.get("prompt"))
        ci = str(_ctx_bucket(ctx))
        cb = b["ctx"].setdefault(ci, {"n": 0, "tps_sum": 0.0, "tps_n": 0, "ttft_sum": 0.0, "ttft_n": 0, "tps": []})
        cb["n"] += 1
        if isinstance(ttft, (int, float)):
            row["ttft_sum"] += ttft; row["ttft_n"] += 1
            b["seen"]["ttft"] += 1
            _reservoir_add(b["ttft"], ttft, b["seen"]["ttft"], self.rng)
            cb["ttft_sum"] += ttft; cb["ttft_n"] += 1
        if tps > 0:
            row["gen_sum"] += tps; row["gen_n"] += 1
            b["seen"]["tps"] += 1
            _reservoir_add(b["tps"], tps, b["seen"]["tps"], self.rng)
            cb["tps_sum"] += tps; cb["tps_n"] += 1
            if len(cb["tps"]) < 60:
                cb["tps"].append(tps)

    def _read(self, path, start):
        """Feed complete lines of `path` from byte `start`; returns the new offset."""
        cutoff = time.time() - RETAIN_HOURS * HOUR
        pos = start
        with open(path, "rb") as f:
            f.seek(start)
            for line in f:
                if not line.endswith(b"\n"):
                    break                       # a record still being written: pick it up next time
                pos += len(line)
                self.parsed_bytes += len(line)
                t = _ts_fast(line)
                if t is not None and t < cutoff:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    self.add(r)
        self.dirty = self.dirty or pos != start
        return pos

    def _rebuild(self):
        self.hours, self.inode, self.offset = {}, None, 0
        for p in log_files(self.path):
            if p == self.path:
                break
            self._read(p, 0)
        self.dirty = True

    def update(self):
        """Bring the rollup up to date with the log files (cheap when nothing new was written)."""
        with self.lock:
            try:
                st = os.stat(self.path)
            except OSError:
                if self.inode is not None or self.hours:
                    pass                          # log gone (deleted): keep what we have
                return
            if self.inode is None and not self.hours:
                self._rebuild()
            if self.inode == st.st_ino and st.st_size >= self.offset:
                pass
            elif self.inode is None:
                self.offset = 0
            elif self.inode != st.st_ino:
                # rotated: finish the old file (now .1/.2/.3) from our offset, then everything newer
                files = log_files(self.path)
                idx = next((i for i, p in enumerate(files[:-1]) if os.stat(p).st_ino == self.inode), None)
                if idx is None or os.stat(files[idx]).st_size < self.offset:
                    self._rebuild()
                else:
                    self._read(files[idx], self.offset)
                    for p in files[idx + 1:-1]:
                        self._read(p, 0)
                    self.offset = 0
            else:                                  # truncated in place
                self._rebuild()
            self.inode = st.st_ino
            self.offset = self._read(self.path, self.offset)
            cut = str(int(time.time() // HOUR * HOUR) - RETAIN_HOURS * HOUR)
            for h in [h for h in self.hours if h < cut]:
                del self.hours[h]
                self.dirty = True
            self._save()

    # -- views
    def query(self, range_="24h", now=None, tz_minutes=0):
        now = now or time.time()
        span = RANGES.get(range_, RANGES["24h"])
        self.update()
        with self.lock:
            return _view(self.hours, range_ if range_ in RANGES else "24h", span, now, tz_minutes)

    def busy_hours(self, since, now=None):
        """{backend label: number of distinct hours with at least one request since `since`}."""
        self.update()
        out = {}
        with self.lock:
            for h, b in self.hours.items():
                if int(h) + HOUR <= since:
                    continue
                for key in b["rows"]:
                    backend = key.split("|")[1]
                    out.setdefault(backend, set()).add(h)
        return {k: len(v) for k, v in out.items()}


def _fold(dst, row):
    for f in SUM_FIELDS:
        dst[f] += row[f]


def _finish(row):
    """Derived numbers for one aggregated row."""
    prompt = row["new_in"] + row["cached_in"]
    row["prompt"] = prompt
    row["cache_hit"] = round(row["cached_in"] / prompt, 4) if prompt else None
    row["ttft_avg"] = round(row["ttft_sum"] / row["ttft_n"], 2) if row["ttft_n"] else None
    row["gen_tps_avg"] = round(row["gen_sum"] / row["gen_n"], 1) if row["gen_n"] else None
    row["avg_total_s"] = round(row["total_s"] / row["n"], 2) if row["n"] else None
    for f in ("think_tok", "text_tok", "total_s", "ttft_sum", "gen_sum"):
        row[f] = round(row[f], 1)
    return row


def _view(hours, range_, span, now, tz_minutes):
    cur_hour = int(now // HOUR * HOUR)
    since = cur_hour - (span // HOUR - 1) * HOUR       # whole hour buckets, current hour included
    tot = _blank_row()
    by_model, by_backend, by_kind = {}, {}, {}
    ttft, tps, ctxb = [], [], {}
    hourly = {cur_hour - i * HOUR: _blank_row() for i in range(23, -1, -1)}
    tz = int(tz_minutes) * 60
    day_of = lambda t: int((t + tz) // 86400 * 86400 - tz)           # local midnight, as UTC seconds
    today = day_of(now)
    daily_span = max(7, min(31, span // 86400))                       # cover the range for long views
    daily = {today - i * 86400: _blank_row() for i in range(daily_span - 1, -1, -1)}
    daily_min = min(daily)
    for h, b in hours.items():
        hs = int(h)
        in_range = hs >= since
        for key, row in b["rows"].items():
            model, backend, kind = key.split("|")
            if in_range:
                _fold(tot, row)
                _fold(by_model.setdefault(model, _blank_row()), row)
                _fold(by_backend.setdefault(backend, _blank_row()), row)
                _fold(by_kind.setdefault(kind, _blank_row()), row)
            if hs in hourly:
                _fold(hourly[hs], row)
            if hs >= daily_min - 86400:
                d = day_of(hs)
                if d in daily:
                    _fold(daily[d], row)
        if in_range:
            ttft += b["ttft"]; tps += b["tps"]
            for ci, cb in b["ctx"].items():
                acc = ctxb.setdefault(ci, {"n": 0, "tps_sum": 0.0, "tps_n": 0, "ttft_sum": 0.0, "ttft_n": 0, "tps": []})
                for f in ("n", "tps_sum", "tps_n", "ttft_sum", "ttft_n"):
                    acc[f] += cb[f]
                acc["tps"] += cb["tps"]
    def table(d):
        rows = []
        for k, v in d.items():
            rows.append({"key": k, **_finish(dict(v))})
        return sorted(rows, key=lambda r: -r["n"])
    totals = _finish(dict(tot))
    totals["req_per_hour"] = round(tot["n"] / (span / HOUR), 2)
    totals["think_share"] = round(tot["think_c"] / (tot["think_c"] + tot["text_c"]), 3) \
        if tot["think_c"] + tot["text_c"] else None
    return {
        "range": range_, "since": int(since), "generated": int(now), "tz": int(tz_minutes),
        "totals": totals,
        "by_model": table(by_model), "by_backend": table(by_backend), "by_kind": table(by_kind),
        "hourly": [{"t": t, **{f: r[f] for f in ("n", "err", "loop", "new_in", "cached_in", "out", "think_tok", "text_tok")}}
                   for t, r in sorted(hourly.items())],
        "daily": [{"t": t, **{f: r[f] for f in ("n", "err", "loop", "new_in", "cached_in", "out", "think_tok", "text_tok")}}
                  for t, r in sorted(daily.items())],
        "latency": {
            "ttft": {"n": len(ttft), "p50": _r(pct(ttft, 50)), "p90": _r(pct(ttft, 90)), "p99": _r(pct(ttft, 99))},
            "gen_tps": {"n": len(tps), "p10": _r(pct(tps, 10), 1), "p50": _r(pct(tps, 50), 1), "p90": _r(pct(tps, 90), 1)},
        },
        "ctx_buckets": [{"label": CTX_BUCKETS[int(i)][2], "n": c["n"],
                         "tps_mean": round(c["tps_sum"] / c["tps_n"], 1) if c["tps_n"] else None,
                         "tps_p50": _r(pct(c["tps"], 50), 1),
                         "ttft_mean": round(c["ttft_sum"] / c["ttft_n"], 2) if c["ttft_n"] else None}
                        for i, c in sorted(ctxb.items(), key=lambda kv: int(kv[0]))],
    }


def _r(v, nd=2):
    return None if v is None else round(v, nd)


# ---- request explorer (tail scan, no rollup) ------------------------------------------------------------
def matches(r, f):
    """Filter one record. f: kind (image|video|chat), model, backend, status (200 | 'error' | int), loop (bool), min_total, since, q."""
    if f.get("kind") and (r.get("kind") or "chat") != f["kind"]:
        return False
    if f.get("model") and f["model"] not in str(r.get("model") or ""):
        return False
    if f.get("backend") and f["backend"] not in str(r.get("backend") or ""):
        return False
    st = f.get("status")
    if st == "error" and (r.get("status") == 200):
        return False
    if st == "ok" and r.get("status") != 200:
        return False
    if isinstance(st, int) and r.get("status") != st:
        return False
    if f.get("loop") and not r.get("loop_cut"):
        return False
    if f.get("min_total") and _num(r.get("total_s")) < f["min_total"]:
        return False
    if f.get("min_prompt") and _num(r.get("new_in")) + _num(r.get("cached_in")) < f["min_prompt"]:
        return False
    if f.get("since") and _num(r.get("ts")) < f["since"]:
        return False
    if f.get("until") and _num(r.get("ts")) > f["until"]:
        return False
    if f.get("q") and f["q"].lower() not in json.dumps(r).lower():
        return False
    return True


def iter_records(path=None, newest_first=True, max_bytes=None):
    """Parsed records, newest first (reads the file backwards in blocks), across rotated files."""
    files = log_files(path)
    budget = max_bytes
    for p in reversed(files) if newest_first else files:
        with open(p, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if not newest_first:
                f.seek(0)
                for line in f:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        pass
                continue
            carry = b""
            while end > 0:
                if budget is not None and budget <= 0:
                    return
                size = min(1 << 20, end)
                end -= size
                f.seek(end)
                block = f.read(size) + carry
                parts = block.split(b"\n")
                carry = parts[0] if end > 0 else b""
                for line in reversed(parts[1:] if end > 0 else parts):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
                if budget is not None:
                    budget -= size
            if carry.strip():
                try:
                    yield json.loads(carry)
                except ValueError:
                    pass


def search(filters, limit=200, path=None, max_bytes=64 * 1024 * 1024):
    """Newest-first matching records (at most `limit`) plus how many were scanned."""
    out, scanned = [], 0
    for r in iter_records(path, max_bytes=max_bytes):
        scanned += 1
        if matches(r, filters):
            out.append(r)
            if len(out) >= limit:
                break
    return out, scanned


CSV_COLUMNS = ["time", "ts", "backend", "model", "path", "status", "new_in", "cached_in", "out", "think_chars",
               "text_chars", "ttft_s", "gen_tps", "prompt_tps", "total_s", "n_tools", "loop_cut"]


def csv_rows(filters, path=None, limit=100000):
    """Generator of CSV text chunks (header first), oldest first, streamed from disk."""
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    yield buf.getvalue(); buf.seek(0); buf.truncate()
    n = 0
    for r in iter_records(path, newest_first=False):
        if not matches(r, filters):
            continue
        row = dict(r)
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_num(r.get("ts"))))
        row["loop_cut"] = "yes" if r.get("loop_cut") else ""
        w.writerow([row.get(c, "") if row.get(c) is not None else "" for c in CSV_COLUMNS])
        n += 1
        if n % 500 == 0:
            yield buf.getvalue(); buf.seek(0); buf.truncate()
        if n >= limit:
            break
    yield buf.getvalue()


# ---- scheduler decision history --------------------------------------------------------------------------
def sync_decisions(state=None, path=None, keep=2000):
    """Append scheduler decisions that state.json's short log (last 40) has and the history lacks.
    Called periodically from the relay process; the supervisor keeps writing state as before."""
    path = str(path or DECISIONS_FILE)
    log = ((state or common.read_state()).get("scheduler") or {}).get("log") or []
    if not log:
        return 0
    seen, last_ts = set(), 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f.readlines()[-120:]:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                seen.add((e["ts"], e["action"], e["target"], e.get("model")))
                last_ts = max(last_ts, e["ts"])
    except OSError:
        pass
    new = [e for e in log if e.get("ts", 0) >= last_ts and
           (e["ts"], e["action"], e["target"], e.get("model")) not in seen]
    if new:
        with open(path, "a", encoding="utf-8") as f:
            for e in new:
                f.write(json.dumps(e) + "\n")
        _trim(path, keep)
    return len(new)


def _trim(path, keep):
    try:
        if os.path.getsize(path) < keep * 400:
            return
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > keep * 1.5:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(lines[-keep:])
            os.replace(tmp, path)
    except OSError:
        pass


def read_decisions(since=0, limit=200, path=None):
    out = []
    try:
        with open(str(path or DECISIONS_FILE), encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ts", 0) >= since:
                    out.append(e)
    except OSError:
        pass
    return out[-limit:]


# ---- compute / quota report ---------------------------------------------------------------------------------
def compute_report(cfg, state, busy_hours, decisions, now=None):
    """Per-account GPU usage this week, quota left, burn rate and projected exhaustion, standby estimate,
    launch / cancel / retire counts, measured queue and ready times, cooldown and last error."""
    now = now or time.time()
    week_start = common._week_start(now)
    week_h = max(0.25, (now - week_start) / 3600)
    running = {b["account"] for b in state.get("backends", [])}
    booted = {a for a, p in (state.get("pending") or {}).items() if p.get("boot_at")}
    counts = {}
    for e in decisions:
        if e.get("ts", 0) >= week_start:
            c = counts.setdefault(e["target"], {"launch": 0, "cancel": 0, "retire": 0})
            c[e["action"]] = c.get(e["action"], 0) + 1
    rows, pool_left, burn_now, pool_limit, num_rows = [], 0.0, 0, 0.0, 0
    for a in cfg.get("accounts") or []:
        name = a["name"]
        num_rows += 1
        st = (state.get("accounts") or {}).get(name) or {}
        used_h = common.quota_used(name, state) / 3600
        total_h = common.quota_total(name, state) / 3600
        # Kaggle-reported weekly cap overrides a hard-coded local constant — same rule as common.quota_left_h
        exhausted = (st.get("quota_exhausted_until", 0) or 0) > now
        left_h = 0.0 if exhausted else max(0.0, total_h - used_h)
        on = name in running or name in booted
        disabled = bool(a.get("disabled"))
        avg_burn = used_h / week_h
        busy_h = min(used_h, float(busy_hours.get("kaggle:" + name, 0)))
        row = {
            "name": name, "user": a.get("user"), "disabled": disabled, "running": name in running,
            "launching": name in (state.get("pending") or {}) and name not in running,
            "used_h": round(used_h, 2), "left_h": round(left_h, 2), "limit_h": round(total_h, 1),
            "burn_h_per_h": 1.0 if on else 0.0, "avg_burn_h_per_h": round(avg_burn, 3),
            "exhaust_in_h": round(left_h, 2) if on else None,
            "exhaust_at": int(now + left_h * 3600) if on else None,
            "exhaust_at_avg": int(now + left_h / avg_burn * 3600) if avg_burn > 0.01 else None,
            "busy_h": round(busy_h, 1), "standby_h": round(max(0.0, used_h - busy_h), 1),
            "cooldown_left_s": max(0, int(st.get("cooldown_until", 0) - now)),
            "fails": st.get("fails", 0), "last_error": st.get("last_error"),
            "queue_s": st.get("queue_ewma_s"), "ready_s": st.get("ready_ewma_s"),
            "launches": counts.get(name, {}).get("launch", 0), "cancels": counts.get(name, {}).get("cancel", 0),
            "retires": counts.get(name, {}).get("retire", 0),
        }
        rows.append(row)
        if not disabled:
            pool_left += left_h
            pool_limit += total_h
        burn_now += 1 if on else 0
    total_used = sum(r["used_h"] for r in rows)
    return {
        "week_start": week_start, "week_ends": week_start + common.WEEK,
        "week_resets_in_h": round((week_start + common.WEEK - now) / 3600, 1),
        "accounts": rows,
        "pool": {"used_h": round(total_used, 1), "left_h": round(pool_left, 1),
                 "limit_h": round(pool_limit, 1), "burn_now": burn_now,
                 "avg_burn_h_per_h": round(total_used / week_h, 3),
                 "exhaust_in_h": round(pool_left / burn_now, 1) if burn_now else None,
                 "exhaust_in_h_avg": round(pool_left / (total_used / week_h), 1) if total_used / week_h > 0.01 else None,
                 "launches": sum(r["launches"] for r in rows), "cancels": sum(r["cancels"] for r in rows),
                 "retires": sum(r["retires"] for r in rows),
                 "standby_h": round(sum(r["standby_h"] for r in rows), 1)},
    }
