"""SQLite store for kllm Agents (BACKEND_SPEC.md, "What data does the Agents workspace need?").

One file, <root>/state/agents.db, WAL mode. Events are append-only and their integer id is the stream cursor
(Last-Event-ID). JSON-shaped fields are stored as TEXT and decoded on read. All access goes through one
connection guarded by a lock: every statement is small, so callers use it directly from the event loop.
"""
import json, sqlite3, threading, time, uuid
from .. import common

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, title TEXT, goal TEXT, mode TEXT, lead TEXT, permission_mode TEXT, status TEXT,
  cwd TEXT, worktree_path TEXT, base_commit TEXT, attempts INTEGER DEFAULT 1, attempt_of TEXT, attempt_n INTEGER,
  chosen INTEGER DEFAULT 0, guardrails TEXT, playbook_id TEXT, schedule_id TEXT, summary TEXT,
  created_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS workers (
  id TEXT, session_id TEXT, harness TEXT, model TEXT, account TEXT, role TEXT, driver TEXT, acp_session_id TEXT,
  contract TEXT, status TEXT, turns INTEGER DEFAULT 0, tool_calls INTEGER DEFAULT 0, tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0, context_used INTEGER, context_max INTEGER, last_text TEXT, error TEXT,
  moved_to TEXT, created_at REAL, finished_at REAL, PRIMARY KEY (session_id, id));
CREATE TABLE IF NOT EXISTS plan_steps (
  session_id TEXT, n INTEGER, text TEXT, owner_worker_id TEXT, files TEXT, status TEXT, PRIMARY KEY (session_id, n));
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, worker_id TEXT, at REAL, type TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS events_session ON events (session_id, id);
CREATE TABLE IF NOT EXISTS checkpoints (
  id TEXT PRIMARY KEY, session_id TEXT, worker_id TEXT, n INTEGER, at REAL, commit_sha TEXT, label TEXT, files TEXT);
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY, session_id TEXT, worker_id TEXT, kind TEXT, summary TEXT, diff TEXT, options TEXT,
  status TEXT, decided_by TEXT, created_at REAL, decided_at REAL);
CREATE TABLE IF NOT EXISTS playbooks (
  id TEXT PRIMARY KEY, name TEXT, description TEXT, mode TEXT, lead TEXT, workers TEXT, guardrails TEXT,
  goal TEXT, cwd TEXT, permission_mode TEXT, uses INTEGER DEFAULT 0, created_at REAL);
CREATE TABLE IF NOT EXISTS schedules (
  id TEXT PRIMARY KEY, playbook_id TEXT, cron TEXT, conditions TEXT, enabled INTEGER DEFAULT 1, last_run REAL,
  last_result TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY, text TEXT, source TEXT, pinned INTEGER DEFAULT 0, created_at REAL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""

JSON_COLS = {"lead", "guardrails", "contract", "files", "options", "workers", "conditions", "flow"}
# columns added after the first release: ALTER TABLE on open (sqlite has no ADD COLUMN IF NOT EXISTS)
MIGRATIONS = [("sessions", "flow", "TEXT"), ("workers", "step", "TEXT"), ("playbooks", "flow", "TEXT")]
BOOL_COLS = {"pinned", "enabled", "chosen"}


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class Store:
    def __init__(self, path=None):
        self.path = path or (common.ROOT / "state" / "agents.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript(SCHEMA)
            for table, col, typ in MIGRATIONS:
                have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
                if col not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    # ---- generic helpers ------------------------------------------------------------------------------------
    @staticmethod
    def _enc(k, v):
        if k in JSON_COLS:
            return None if v is None else json.dumps(v)
        if k in BOOL_COLS:
            return 1 if v else 0
        return v

    @staticmethod
    def _dec(row):
        if row is None:
            return None
        d = dict(row)
        for k in list(d):
            if k in JSON_COLS and d[k] is not None:
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
            elif k in BOOL_COLS:
                d[k] = bool(d[k])
        return d

    def q(self, sql, args=()):
        with self.lock:
            return [self._dec(r) for r in self.db.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        with self.lock:
            return self._dec(self.db.execute(sql, args).fetchone())

    def insert(self, table, row):
        cols = list(row)
        with self.lock:
            cur = self.db.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                  [self._enc(k, row[k]) for k in cols])
            return cur.lastrowid

    def update(self, table, where, **fields):
        if not fields:
            return 0
        sets = ", ".join(f"{k} = ?" for k in fields)
        conds = " AND ".join(f"{k} = ?" for k in where)
        with self.lock:
            return self.db.execute(f"UPDATE {table} SET {sets} WHERE {conds}",
                                   [self._enc(k, v) for k, v in fields.items()] + list(where.values())).rowcount

    def delete(self, table, **where):
        conds = " AND ".join(f"{k} = ?" for k in where)
        with self.lock:
            return self.db.execute(f"DELETE FROM {table} WHERE {conds}", list(where.values())).rowcount

    # ---- sessions -------------------------------------------------------------------------------------------
    def session(self, sid):
        return self.one("SELECT * FROM sessions WHERE id = ?", (sid,))

    def sessions(self, status=None, limit=200, include_attempts=False):
        sql = "SELECT * FROM sessions"
        conds, args = [], []
        if status:
            conds.append("status IN (%s)" % ",".join("?" * len(status)))
            args += list(status)
        if not include_attempts:
            conds.append("attempt_of IS NULL")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return self.q(sql + " ORDER BY created_at DESC LIMIT ?", args + [limit])

    def workers(self, sid=None, running=False):
        sql, args = "SELECT * FROM workers", []
        conds = []
        if sid:
            conds.append("session_id = ?"); args.append(sid)
        if running:
            conds.append("status IN ('starting', 'working', 'needs_you', 'idle', 'paused')")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return self.q(sql + " ORDER BY created_at", args)

    def worker(self, sid, wid):
        return self.one("SELECT * FROM workers WHERE session_id = ? AND id = ?", (sid, wid))

    def plan(self, sid):
        return self.q("SELECT * FROM plan_steps WHERE session_id = ? ORDER BY n", (sid,))

    def set_plan(self, sid, steps):
        with self.lock:
            self.db.execute("DELETE FROM plan_steps WHERE session_id = ?", (sid,))
            for i, s in enumerate(steps, 1):
                self.db.execute("INSERT INTO plan_steps VALUES (?, ?, ?, ?, ?, ?)",
                                (sid, i, s.get("text", ""), s.get("owner"), json.dumps(s.get("files") or []),
                                 s.get("status") or "todo"))

    # ---- events ---------------------------------------------------------------------------------------------
    def add_event(self, sid, type_, worker=None, data=None, at=None):
        at = at or time.time()
        eid = self.insert("events", {"session_id": sid, "worker_id": worker, "at": at, "type": type_,
                                     "data": json.dumps(data or {})})
        return {"id": eid, "type": type_, "worker": worker, "at": round(at, 3), "data": data or {}}

    def events(self, sid, after=0, limit=2000, types=None):
        sql = "SELECT * FROM events WHERE session_id = ? AND id > ?"
        args = [sid, after]
        if types:
            sql += " AND type IN (%s)" % ",".join("?" * len(types))
            args += list(types)
        rows = self.q(sql + " ORDER BY id LIMIT ?", args + [limit])
        return [{"id": r["id"], "type": r["type"], "worker": r["worker_id"], "at": round(r["at"], 3),
                 "data": json.loads(r["data"] or "{}")} for r in rows]

    def event_bytes(self, sid, type_):
        r = self.one("SELECT COALESCE(SUM(LENGTH(data)), 0) AS n FROM events WHERE session_id = ? AND type = ?",
                     (sid, type_))
        return int(r["n"]) if r else 0

    # ---- settings -------------------------------------------------------------------------------------------
    def get_setting(self, key, default=None):
        r = self.one("SELECT value FROM settings WHERE key = ?", (key,))
        if r is None:
            return default
        try:
            return json.loads(r["value"])
        except ValueError:
            return default

    def set_setting(self, key, value):
        with self.lock:
            self.db.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, json.dumps(value)))
