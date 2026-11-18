"""kllm Agents runtime: sessions, workers, the event bus, permissions, checkpoints, approvals, schedules.

Lives in the relay process (MANAGER). Everything the UI shows is persisted in store.py; worker processes are
in memory only, so a relay restart marks unfinished sessions "stopped" (their checkpoints and worktrees stay).
"""
import asyncio, collections, fnmatch, json, os, pathlib, re, shlex, sys, time
from types import SimpleNamespace
from .. import common
from . import gitops, harnesses
from .store import Store, new_id
from .workers import AcpWorker, StreamWorker, tool_paths, tool_command

GUARDRAIL_DEFAULTS = {
    "never_touch": ["config.yaml", "config.yaml.*", "kaggle_accounts/**", ".env", ".env.*", "**/*.pem",
                    "state/**", ".git/**"],
    "delete_breaker": True,          # refuse rm -rf / git reset --hard / force pushes in Skip; restore mass deletes
    "delete_limit": 5,               # a turn that deletes more files than this is undone
    "max_workers": 4,                # per session
    "worker_limit": 6,               # across all sessions
    "max_turns": 60,                 # per worker
    "thought_cap_mb": 2,             # stored thinking per session
    "verify_command": None,          # run in each attempt's worktree when it finishes (best of N)
    "worktree": True,
}
ACCESS_DEFAULTS = {"cli": "read", "api": "read"}
ACCESS_LEVELS = ("off", "read", "full")
ACTIVE = ("queued", "running", "needs_you", "waiting")
LIVE = ("queued", "running", "needs_you")          # a turn is running or about to: never close its processes
FLOW_MAX = 16
PEER_TOOLS = {"get_session", "message_worker", "list_models"}   # what a worker (not the lead) may call
DANGEROUS = re.compile(r"\brm\s+-[a-zA-Z]*[rf]|\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|push\s+(-f\b|--force))"
                       r"|\bmkfs\b|\bdd\s+if=|>\s*/dev/sd")
EPHEMERAL = {"message_chunk"}


class AgentError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.msg, self.code = msg, code


def _rel(p):
    p = str(p).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def match_any(path, patterns):
    p = _rel(path)
    for pat in patterns or []:
        pat = _rel(pat)
        if fnmatch.fnmatch(p, pat) or (pat.endswith("/**") and (p == pat[:-3] or p.startswith(pat[:-2]))):
            return True
        if "/" not in pat and fnmatch.fnmatch(pathlib.PurePosixPath(p).name, pat):
            return True
    return False


class Manager:
    def __init__(self, store=None):
        self._store = store
        self.relay = None                 # the relay module, set by api.make_router (ROUTER, cfg, state)
        self.workers = {}                 # (sid, wid) -> worker object
        self.futures = {}                 # approval id -> Future resolved by decide()
        self.live = collections.defaultdict(lambda: collections.deque(maxlen=3000))   # sid -> (seq, event)
        self.seq = 0
        self.cond = None
        self.thought_bytes = {}
        self.last_snapshot = {}           # sid -> tree sha at the last checkpoint
        self._sched_task = None
        self._recovered = False

    @property
    def store(self):
        if self._store is None:
            self._store = Store()
        return self._store

    # ---- settings ---------------------------------------------------------------------------------------------
    def guardrails(self):
        g = dict(GUARDRAIL_DEFAULTS)
        g.update(self.store.get_setting("guardrails", {}) or {})
        return g

    def access(self):
        a = dict(ACCESS_DEFAULTS)
        a.update(self.store.get_setting("access", {}) or {})
        return a

    def relay_base(self):
        try:
            return common.relay_base(self.relay.cfg() if self.relay else common.load_config())
        except Exception:
            return "http://127.0.0.1:8080"

    # ---- event bus ----------------------------------------------------------------------------------------------
    def emit(self, sid, type_, worker=None, data=None):
        data = data or {}
        store_it = type_ not in EPHEMERAL
        if type_ == "thought_chunk":
            capmb = self.guardrails().get("thought_cap_mb")
            cap = int(float(2 if capmb is None else capmb) * 1024 * 1024)
            n = self.thought_bytes.get(sid)
            if n is None:
                n = self.store.event_bytes(sid, "thought_chunk")
            n += len(data.get("text", ""))
            self.thought_bytes[sid] = n
            store_it = n <= cap
        if store_it:
            ev = self.store.add_event(sid, type_, worker, data)
        else:
            ev = {"id": None, "type": type_, "worker": worker, "at": round(time.time(), 3), "data": data}
        self.seq += 1
        self.live[sid].append((self.seq, ev))
        self.live["*"].append((self.seq, dict(ev, session=sid)))
        asyncio.ensure_future(self._notify())
        return ev

    async def _notify(self):
        if self.cond is None:
            self.cond = asyncio.Condition()
        async with self.cond:
            self.cond.notify_all()

    async def subscribe(self, sid, after=0, is_gone=None, follow=True):
        """Yield events: stored ones after `after`, then live ones (stored and ephemeral) as they happen."""
        if self.cond is None:
            self.cond = asyncio.Condition()
        mark = self.seq
        last_id = after
        if sid != "*":
            for ev in self.store.events(sid, after, limit=20000):
                last_id = ev["id"]
                yield ev
        while follow:
            if is_gone and await is_gone():
                return
            async with self.cond:
                try:
                    await asyncio.wait_for(self.cond.wait_for(lambda: self.seq != mark), 15)
                except asyncio.TimeoutError:
                    yield None                     # keepalive
                    continue
            for s, ev in list(self.live[sid]):
                if s <= mark:
                    continue
                mark = s
                if ev["id"] is not None and ev["id"] <= last_id:
                    continue
                if ev["id"] is not None:
                    last_id = ev["id"]
                yield ev

    # ---- recovery and views ---------------------------------------------------------------------------------------
    def recover(self):
        """At startup: no worker process survives a relay restart."""
        if self._recovered:
            return
        self._recovered = True
        now = time.time()
        # the harness processes are gone, but their conversations are not: the session waits, and a message to any
        # worker reopens it (ACP session/load or --resume), so restarting the relay no longer ends your agents
        for s in self.store.sessions(status=ACTIVE, limit=10000, include_attempts=True):
            self.store.update("sessions", {"id": s["id"]}, status="waiting")
            self.store.add_event(s["id"], "status", None, {"status": "waiting",
                                                           "why": "the relay restarted; write to an agent to carry on"})
        with self.store.lock:
            self.store.db.execute("UPDATE workers SET status = 'stopped' WHERE status IN "
                                  "('starting', 'working', 'needs_you', 'idle', 'paused')")
            self.store.db.execute("UPDATE approvals SET status = 'expired' WHERE status = 'pending' AND kind = 'tool'")

    def session_view(self, sid):
        s = self.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        workers = []
        for w in self.store.workers(sid):
            live = self.workers.get((sid, w["id"]))
            w["queued"] = len(live.queue) if live else 0
            w["commands"] = live.commands if live else []
            w["command_info"] = getattr(live, "command_info", []) if live else []
            workers.append(w)
        s["workers"] = workers
        s["plan"] = self.store.plan(sid)
        s["counts"] = {
            "events": (self.store.one("SELECT COUNT(*) AS n FROM events WHERE session_id = ?", (sid,)) or {}).get("n", 0),
            "approvals_pending": len(self.store.q("SELECT id FROM approvals WHERE session_id = ? AND status = 'pending'",
                                                  (sid,))),
            "checkpoints": len(self.store.q("SELECT id FROM checkpoints WHERE session_id = ?", (sid,))),
            "workers_running": sum(1 for w in workers if w["status"] in ("starting", "working", "needs_you")),
        }
        s["last_event_id"] = (self.store.one("SELECT MAX(id) AS m FROM events WHERE session_id = ?", (sid,)) or {}).get("m")
        if s.get("attempts", 1) > 1 and not s.get("attempt_of"):
            s["attempt_ids"] = [x["id"] for x in self.store.q(
                "SELECT id FROM sessions WHERE attempt_of = ? ORDER BY attempt_n", (sid,))]
        return s

    def list_sessions(self, status=None):
        rows = self.store.sessions(status=status)
        counts = {}
        for r in self.store.q("SELECT session_id, COUNT(*) AS n FROM approvals WHERE status = 'pending' "
                              "GROUP BY session_id"):
            counts[r["session_id"]] = r["n"]
        wc = collections.Counter()
        for r in self.store.q("SELECT session_id, COUNT(*) AS n FROM workers GROUP BY session_id"):
            wc[r["session_id"]] = r["n"]
        for r in rows:
            r["approvals_pending"] = counts.get(r["id"], 0)
            r["workers"] = wc.get(r["id"], 0)
        return rows

    # ---- session lifecycle -----------------------------------------------------------------------------------------
    def _validate_harness(self, harness, permission_mode):
        d, driver, cmd = harnesses.resolve(harness)
        if d is None:
            raise AgentError(f"unknown harness '{harness}': use one of {', '.join(harnesses.HARNESSES)}")
        if driver is None:
            raise AgentError(f"{d['name']} cannot run here: " + next(
                (r["why_not"] for r in harnesses.detect() if r["id"] == harness), "not installed"), 409)
        if permission_mode == "ask" and driver != "acp":
            raise AgentError(f"Ask mode needs an ACP harness; {d['name']} runs without one here. "
                             "Use Skip, or pick opencode or Antigravity.", 409)
        return d, driver, cmd

    async def create_session(self, body, attempt_of=None, attempt_n=None, base=None):
        goal = str(body.get("goal") or "").strip()
        if not goal:
            raise AgentError("describe the goal")
        mode = body.get("mode") or "one"
        if mode not in ("one", "team", "playbook", "flow"):
            raise AgentError("mode must be one, team, playbook or flow")
        perm = body.get("permission_mode") or "skip"
        if perm not in ("ask", "skip"):
            raise AgentError("permission_mode must be ask or skip")
        g = self.guardrails()
        g.update(body.get("guardrails") or {})
        playbook = None
        if mode == "playbook":
            playbook = self.store.one("SELECT * FROM playbooks WHERE id = ?", (body.get("playbook_id"),))
            if not playbook:
                raise AgentError("no such playbook", 404)
            g.update(playbook.get("guardrails") or {})
        lead = body.get("lead") or (playbook or {}).get("lead")
        steps = body.get("steps") or ((playbook or {}).get("flow") or {}).get("steps") or []
        pb_workers = (playbook or {}).get("workers") or []
        # The session's default harness: the one asked for, else the one its own workers/steps use, else opencode.
        # (Never require opencode when every worker or step names another harness.)
        harness = (body.get("harness") or (lead or {}).get("harness")
                   or next((x.get("harness") for x in [*pb_workers, *steps] if isinstance(x, dict) and x.get("harness")), None)
                   or "opencode")
        flow = None
        if mode == "flow" or (playbook or {}).get("flow"):
            flow = self.validate_flow(steps, perm)          # validates each step's own harness
            mode, lead = "flow", None
        if mode == "team" and not lead:
            lead = {"harness": harness, "model": body.get("model")}
        if not flow and (lead or not pb_workers):
            self._validate_harness((lead or {}).get("harness") or harness, perm)
        for w in pb_workers:
            self._validate_harness(w.get("harness") or harness, perm)
        cwd = os.path.expanduser(str(body.get("cwd") or common.ROOT))
        if not os.path.isdir(cwd):
            raise AgentError(f"{cwd} is not a folder")
        repo = gitops.repo_root(cwd)
        use_wt = bool(g.get("worktree", True))
        if perm == "skip" and not repo:
            raise AgentError("Skip mode needs a git repository so every change has a checkpoint: pick a folder in a "
                             "git checkout, or use Ask mode.", 409)
        attempts = max(1, min(int(body.get("attempts") or 1), 4))
        sid = new_id("s")
        title = str(body.get("title") or goal.splitlines()[0])[:80]
        row = {"id": sid, "title": title, "goal": goal, "mode": mode, "lead": lead if mode != "one" else None,
               "permission_mode": perm, "status": "queued", "cwd": cwd, "attempts": attempts,
               "attempt_of": attempt_of, "attempt_n": attempt_n, "guardrails": g,
               "playbook_id": (playbook or {}).get("id"), "schedule_id": body.get("schedule_id"),
               "flow": flow, "created_at": time.time()}
        self.store.insert("sessions", row)
        if playbook:
            with self.store.lock:
                self.store.db.execute("UPDATE playbooks SET uses = uses + 1 WHERE id = ?", (playbook["id"],))
        self.emit(sid, "status", None, {"status": "queued"})
        if attempts > 1 and attempt_of is None:
            # best of N: one child session per attempt, each in its own worktree; this one only aggregates
            kids = []
            for n in range(1, attempts + 1):
                child = dict(body, attempts=1, title=f"{title} (attempt {n})")
                kids.append(await self.create_session(child, attempt_of=sid, attempt_n=n))
            self.store.update("sessions", {"id": sid}, status="running", worktree_path=None)
            self.emit(sid, "status", None, {"status": "running", "attempts": [k["id"] for k in kids]})
            return {"id": sid, "status": "running", "attempts": [k["id"] for k in kids]}
        wt_path, base_commit = cwd, None
        if repo and use_wt:
            base_commit = base or gitops.head(repo)
            wt_path = str(common.ROOT / "state" / "worktrees" / f"{sid}-{gitops.slug(title, 24)}")
            try:
                await asyncio.to_thread(gitops.add_worktree, repo, wt_path, base_commit)
            except gitops.GitError as e:
                self.store.update("sessions", {"id": sid}, status="failed", finished_at=time.time())
                raise AgentError(f"could not create a worktree: {e}", 500)
            sub = os.path.relpath(cwd, repo)
            if sub != ".":
                wt_path = os.path.join(wt_path, sub)
            self.store.update("sessions", {"id": sid}, worktree_path=wt_path, base_commit=base_commit)
            self.last_snapshot[sid] = await asyncio.to_thread(gitops.tree_of, wt_path, base_commit)
            self.store.insert("checkpoints", {"id": new_id("c"), "session_id": sid, "worker_id": None, "n": 0,
                                              "at": time.time(), "commit_sha": base_commit, "label": "Start",
                                              "files": []})
        elif repo:
            base_commit = gitops.head(repo)
            self.store.update("sessions", {"id": sid}, worktree_path=None, base_commit=base_commit)
        self.store.update("sessions", {"id": sid}, status="running")
        self.emit(sid, "status", None, {"status": "running", "worktree": wt_path if use_wt and repo else None})
        try:
            if mode == "flow":
                self.emit(sid, "flow", None, {"steps": flow["steps"]})
                await self._flow_advance(sid)
            elif mode == "one":
                await self.spawn_worker(sid, harness, body.get("model"), {"objective": goal})
            elif mode == "team":
                await self.spawn_worker(sid, lead["harness"], lead.get("model"), {"objective": goal}, role="lead")
            else:
                if lead:
                    await self.spawn_worker(sid, lead["harness"], lead.get("model"), {"objective": goal}, role="lead")
                for w in playbook.get("workers") or []:
                    await self.spawn_worker(sid, w.get("harness") or harness, w.get("model"),
                                            {"objective": w.get("objective") or goal,
                                             "may_edit": w.get("may_edit") or [],
                                             "must_not_touch": w.get("must_not_touch") or []})
        except AgentError as e:
            self.store.update("sessions", {"id": sid}, status="failed", finished_at=time.time())
            self.emit(sid, "status", None, {"status": "failed", "why": e.msg})
            raise
        return {"id": sid, "status": "running", "worktree": wt_path if repo and use_wt else None}

    def _workdir(self, s):
        return s.get("worktree_path") or s["cwd"]

    def _notes(self):
        rows = self.store.q("SELECT text FROM notes WHERE pinned = 1 ORDER BY created_at")
        return "\n".join(f"- {r['text']}" for r in rows)

    def _prompt(self, s, wid, contract, role):
        c = contract or {}
        notes = self._notes()
        lines = [f"You are {'the lead' if role == 'lead' else 'worker ' + wid} in a kllm Agents session.",
                 f"Session goal: {s['goal']}"]
        if role == "lead":
            names = ", ".join(r["id"] for r in harnesses.detect() if r["usable"])
            pool = self.pool_models()
            lines += ["You coordinate. Use only the tools of the MCP server named `kllm` (ignore any other kllm or "
                      "kllm-spawner server you may see; its tools are for a different purpose):",
                      "- post_plan(steps): publish the plan as a list of {text, owner, files}.",
                      "- spawn_worker(objective, harness, model, may_edit, must_not_touch): start a worker. "
                      f"Harnesses available here: {names}. Workers can mix harnesses.",
                      "- list_models(): the models on the human's kllm pool and which are serving now.",
                      "- message_worker(to, text, mode): steer a worker (mode next_turn or interrupt; to may be all).",
                      "- stop_worker(to): stop a worker that is done or off track.",
                      "- get_session(): workers, their status and last message.",
                      "- request_approval(summary): ask the human before a risky step.",
                      "- finish_session(summary): end the session when the goal is met.",
                      "To run work on a model from the kllm pool, start a worker with harness opencode and model "
                      "kllm/<model key> (for example kllm/" + (pool[0]["key"] if pool else "<key>") + "). "
                      "Never call the relay with curl and never read kllm's own source to learn these tools: "
                      "the descriptions above are complete.",
                      "Serving now on the pool: " + (", ".join(p["key"] for p in pool if p["serving"]) or "nothing") + ".",
                      "kllm itself (relay, kernels, pool, this session's machinery) is not your task. If a model is "
                      "slow, missing or answers badly, tell the human in one sentence and continue; never debug, "
                      "tune or edit kllm unless the goal explicitly asks for it.",
                      "When the human writes to you without a coding task, answer briefly and wait for the next "
                      "message; do not invent work. Call finish_session only when the goal is met.",
                      "Workers report back to you when they finish a turn. Keep each worker to one part of the code."]
        else:
            lines.append(f"Your objective: {c.get('objective') or s['goal']}")
            if s["mode"] in ("team", "playbook", "flow"):
                lines.append("Other agents work in this session. The `kllm` MCP server gives you get_session() to see "
                             "them and their last messages, message_worker(to, text) to tell one of them something "
                             "(to: an id such as W2, L for the lead, or all), and list_models().")
            for inp in c.get("inputs") or []:
                lines.append(f"Earlier step \"{inp.get('title')}\" ({inp.get('worker')}, {inp.get('harness')}) reported:\n"
                             f"{str(inp.get('text') or '')[:3000]}")
            if c.get("may_edit"):
                lines.append("You may edit only: " + ", ".join(c["may_edit"]))
            if c.get("must_not_touch"):
                lines.append("Never touch: " + ", ".join(c["must_not_touch"]))
            if c.get("report_when"):
                lines.append("Report when: " + "; ".join(c["report_when"]))
        never = (s.get("guardrails") or {}).get("never_touch") or []
        if never:
            lines.append("Never edit these paths (kllm reverts them): " + ", ".join(never))
        if notes:
            lines += ["Project notes:", notes]
        lines += ["Work in the current directory and stay inside it: never read or change files outside it.",
                  "kllm (its relay, kernels, config, and any kllm or kllm-spawner MCP server or its files) runs this "
                  "session; it is not your task. Do not inspect, call or debug it unless the objective asks for it. "
                  "If the objective is only to wait or stand by, say you are ready in one line and stop.",
                  "Do not commit: kllm checkpoints every change.",
                  "When you finish, reply with a short summary of what changed and what is left."]
        return "\n".join(lines)

    def _mcp(self, sid, wid, driver, role="lead"):
        env = {"KLLM_ROOT": str(common.ROOT), "PYTHONPATH": str(common.PKG_ROOT), "KLLM_AGENT_SESSION": sid,
               "KLLM_AGENT_WORKER": wid, "KLLM_RELAY": self.relay_base(), "KLLM_AGENT_ROLE": role}
        if driver == "acp":
            return [{"name": "kllm", "command": sys.executable, "args": ["-m", "gateway.mcp_server"],
                     "env": [{"name": k, "value": v} for k, v in env.items()]}]
        return {"mcpServers": {"kllm": {"command": sys.executable, "args": ["-m", "gateway.mcp_server"], "env": env}}}

    def pool_note(self, harness, model):
        """A worker on kllm's own models with no chat kernel serving would sit at "working" with no output for the
        6 to 30 minutes a kernel needs (seen 2026-11-10: the lead went digging through kllm's files to find out why).
        -> a sentence for the timeline and the lead, or None when a chat model is serving or the worker uses its own."""
        d = harnesses.HARNESSES.get(harness) or {}
        if not (d.get("kllm_env") or str(model or "").startswith("kllm")):
            return None
        try:
            from .. import media
            reg = common.load_models()
            st = common.read_state()
        except Exception:
            return None
        chat = lambda key: media.kind(reg.get(key) or {}) == "chat"
        if any(chat(b.get("model_key")) for b in st.get("backends", [])):
            return None
        pend = [p for p in (st.get("pending") or {}).values() if chat(p.get("model_key"))]
        if pend:
            return (f"No kllm chat model is serving yet; {pend[0].get('model_key')} is starting. "
                    "This worker answers once it is ready, usually 6 to 30 minutes on a fresh account.")
        return ("No kllm chat model is serving or starting. Launch one on Fleet, or let autoscale start it on this "
                "worker's first request (6 to 30 minutes).")

    async def spawn_worker(self, sid, harness, model, contract, role="worker", resume=None, prompt=None):
        s = self.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        g = s.get("guardrails") or self.guardrails()
        live = [w for w in self.store.workers(sid) if w["status"] not in ("done", "stopped", "failed", "moved")
                and w["role"] != "lead"]
        if role != "lead" and len(live) >= int(g.get("max_workers") or 4):
            raise AgentError(f"this session already runs {len(live)} workers (limit {g.get('max_workers')})", 409)
        total = len(self.store.workers(running=True))
        if total >= int(self.guardrails().get("worker_limit") or 6):
            raise AgentError(f"{total} workers are running across sessions (limit "
                             f"{self.guardrails().get('worker_limit')}): stop one first", 409)
        d, driver, cmd = self._validate_harness(harness, s["permission_mode"])
        n = len(self.store.workers(sid))
        wid = "L" if role == "lead" and not any(w["id"] == "L" for w in self.store.workers(sid)) else f"W{n + 1}"
        while self.store.worker(sid, wid):
            n += 1
            wid = f"W{n + 1}"
        model = model or d.get("default_model")
        wd = self._workdir(s)
        env = harnesses.worker_env(None, self.relay_base(), sid, wid, self.access()["cli"],
                                   kllm_env=d.get("kllm_env", False), model=model)
        env = harnesses.permission_env(env, harness, s["permission_mode"])
        # the lead gets every session tool; in a team or a flow each worker gets the peer tools (see who else works,
        # message them, list models), so agents can coordinate instead of grepping kllm to find each other
        peers = role == "lead" or s["mode"] in ("team", "playbook", "flow")
        if driver == "acp":
            w = AcpWorker(self, sid, wid, harness, model, wd, env, cmd, role=role,
                          mcp_servers=self._mcp(sid, wid, "acp", role) if peers else [], acp_session_id=resume)
        else:
            w = StreamWorker(self, sid, wid, harness, model, wd, env, cmd[0], role=role,
                             mcp_config=self._mcp(sid, wid, "stream", role) if peers else None, resume=resume,
                             kllm=d.get("kllm_env", False))
        self.store.insert("workers", {"id": wid, "session_id": sid, "harness": harness, "model": model, "role": role,
                                      "driver": driver, "contract": contract, "status": "starting",
                                      "acp_session_id": resume, "created_at": time.time()})
        self.workers[(sid, wid)] = w
        self.emit(sid, "status", wid, {"status": "starting", "harness": harness, "model": model, "role": role,
                                       "objective": (contract or {}).get("objective")})
        note = self.pool_note(harness, model)
        if note:
            self.emit(sid, "message", wid, {"text": f"{wid} waits for a model. {note}", "from": "kllm"})
        w.send(prompt or self._prompt(s, wid, contract, role))
        return self.store.worker(sid, wid)

    # ---- worker callbacks ------------------------------------------------------------------------------------------
    def worker_update(self, w, **fields):
        self.store.update("workers", {"session_id": w.sid, "id": w.wid}, **fields)

    def _set_worker_status(self, w, status, **extra):
        cur = self.store.worker(w.sid, w.wid)
        if cur and cur["status"] == status and not extra:
            return
        self.worker_update(w, status=status, **extra)
        self.emit(w.sid, "status", w.wid, {"status": status, **{k: v for k, v in extra.items() if k != "finished_at"}})

    def turn_started(self, w):
        s = self.store.session(w.sid)
        if s and s["status"] in ("done", "stopped", "failed"):
            self.store.update("sessions", {"id": w.sid}, status="running", finished_at=None)
            self.emit(w.sid, "status", None, {"status": "running"})
        self._set_worker_status(w, "working")

    def tool_started(self, w):
        with self.store.lock:
            self.store.db.execute("UPDATE workers SET tool_calls = tool_calls + 1 WHERE session_id = ? AND id = ?",
                                  (w.sid, w.wid))

    def worker_usage(self, w, u):
        f = {}
        if u.get("used") is not None:
            f["context_used"] = int(u["used"])
        if u.get("size") is not None:
            f["context_max"] = int(u["size"])
        for k in ("tokens_in", "tokens_out"):
            if u.get(k) is not None:
                f[k] = int(u[k])
        if f:
            self.worker_update(w, **f)

    def worker_plan(self, w, entries):
        steps = [dict(e, owner=w.wid) for e in entries]
        if w.role == "lead" or not self.store.plan(w.sid):
            self.store.set_plan(w.sid, steps)
        self.emit(w.sid, "plan", w.wid, {"steps": steps})

    async def turn_finished(self, w, stop_reason, text, err=None):
        s = self.store.session(w.sid)
        with self.store.lock:
            self.store.db.execute("UPDATE workers SET turns = turns + 1, last_text = ? WHERE session_id = ? AND id = ?",
                                  ((text or err or "")[:2000], w.sid, w.wid))
        if text:
            self.emit(w.sid, "message", w.wid, {"text": text, "from": w.wid})
        if err:
            self.emit(w.sid, "message", w.wid, {"text": err, "from": "kllm", "error": True})
            self.worker_update(w, error=err)
        elif text:
            self.worker_update(w, error=None)          # a good turn clears the last error
        if s and s.get("worktree_path") and s.get("base_commit"):
            await self._after_turn(s, w)
        wrow = self.store.worker(w.sid, w.wid)
        max_turns = int((s.get("guardrails") or {}).get("max_turns") or 60) if s else 60
        if wrow and wrow["turns"] >= max_turns and not w.stopped:
            self.emit(w.sid, "message", w.wid, {"text": f"{w.wid} reached the {max_turns}-turn limit and stopped.",
                                                "from": "kllm"})
            await w.stop()
            self._set_worker_status(w, "stopped", finished_at=time.time())
        # team: tell the lead when a worker finishes a turn
        if w.role != "lead" and stop_reason != "cancelled" and s and s["mode"] != "one":
            lead = self.workers.get((w.sid, "L"))
            if lead and not lead.stopped:
                lead.send(f"{w.wid} finished a turn ({stop_reason}). Its message:\n{(text or err or '')[:3000]}")

    async def _after_turn(self, s, w):
        """Checkpoint the worktree if this turn changed files, then apply the guardrails and the scope check."""
        sid, wd = s["id"], s["worktree_path"]
        prev = self.store.one("SELECT * FROM checkpoints WHERE session_id = ? ORDER BY n DESC LIMIT 1", (sid,))
        parent = prev["commit_sha"] if prev else s["base_commit"]
        n = (prev["n"] + 1) if prev else 1
        wrow = self.store.worker(sid, w.wid) or {}
        try:
            sha, tree, files = await asyncio.to_thread(gitops.checkpoint, wd, sid, n, parent,
                                                       f"{w.wid} turn {wrow.get('turns', 0)}")
        except gitops.GitError as e:
            self.emit(sid, "message", None, {"text": f"Checkpoint failed: {e}", "from": "kllm", "error": True})
            return
        if not sha:
            return
        g = s.get("guardrails") or {}
        contract = wrow.get("contract") or {}
        never = list(g.get("never_touch") or []) + list(contract.get("must_not_touch") or [])
        blocked = [f for f in files if match_any(f, never)]
        deleted = [f for f in files if not os.path.exists(os.path.join(self._repo_top(wd), f))]
        reverted = list(blocked)
        if g.get("delete_breaker", True) and len(deleted) > int(g.get("delete_limit") or 5):
            reverted += [f for f in deleted if f not in reverted]
        if reverted:
            await asyncio.to_thread(self._revert_paths, wd, parent, reverted)
            sha2, tree, files2 = await asyncio.to_thread(gitops.checkpoint, wd, sid, n, parent,
                                                         f"{w.wid} turn {wrow.get('turns', 0)} (guarded)")
            self.emit(sid, "guardrail", w.wid, {"reverted": reverted,
                                                "why": "never-touch paths" if blocked else "mass delete"})
            if not sha2:
                await asyncio.to_thread(gitops.git, wd, "update-ref", "-d", f"refs/kllm/checkpoints/{sid}/{n}")
                return
            sha, files = sha2, files2
        cid = new_id("c")
        self.store.insert("checkpoints", {"id": cid, "session_id": sid, "worker_id": w.wid, "n": n, "at": time.time(),
                                          "commit_sha": sha, "label": f"{w.wid} turn {wrow.get('turns', 0)}",
                                          "files": files})
        self.emit(sid, "checkpoint", w.wid, {"id": cid, "n": n, "files": files, "sha": sha[:12]})
        may = contract.get("may_edit") or []
        outside = [f for f in files if may and not match_any(f, may)]
        if outside:
            if s["permission_mode"] == "ask":
                diff = await asyncio.to_thread(gitops.diff, wd, parent, sha)
                self._approval(sid, w.wid, "scope", f"{w.wid} edited outside its part: {', '.join(outside[:8])}",
                               diff=diff, options=None, extra={"files": outside, "checkpoint": cid, "parent": parent})
            else:
                undo = (prev or {}).get("id")
                self.emit(sid, "auto_applied", w.wid, {"files": outside, "checkpoint": cid, "undo_to": undo})

    def _repo_top(self, wd):
        return str(gitops.repo_root(wd) or wd)

    def _revert_paths(self, wd, commit, paths):
        top = self._repo_top(wd)
        for p in paths:
            inside = gitops.git(top, "ls-tree", "--name-only", commit, "--", p, check=False).strip()
            if inside:
                gitops.git(top, "checkout", commit, "--", p, check=False)
            else:
                try:
                    os.remove(os.path.join(top, p))
                except OSError:
                    pass

    def worker_idle(self, w):
        row = self.store.worker(w.sid, w.wid)
        if not row or row["status"] in ("stopped", "moved", "failed"):
            return
        if w.paused:
            self._set_worker_status(w, "paused")
            return
        pending = self.store.q("SELECT id FROM approvals WHERE session_id = ? AND worker_id = ? AND status = 'pending'",
                               (w.sid, w.wid))
        self._set_worker_status(w, "needs_you" if pending else "idle")
        s = self.store.session(w.sid)
        if s and s["mode"] == "flow" and not pending:
            asyncio.ensure_future(self._flow_step_done(w.sid, w.wid))
            return
        self._maybe_finish(w.sid)

    def _maybe_finish(self, sid):
        s = self.store.session(sid)
        if not s or s["status"] not in ("running", "needs_you"):
            return
        live = [x for (ssid, _), x in self.workers.items() if ssid == sid and not x.stopped]
        if any(x.turn_running or x.queue for x in live):
            return
        if self.store.q("SELECT id FROM approvals WHERE session_id = ? AND status = 'pending'", (sid,)):
            self._set_session(sid, "needs_you")
            return
        if any(x.paused for x in live):
            return
        # every worker's last turn ended in an error (the harness crashed or never answered): that is a failure
        rows = [r for r in self.store.workers(sid) if r["status"] != "moved"]
        if rows and all(r.get("error") for r in rows):
            self._finish(sid, "failed", summary="Every worker stopped with an error. " + str(rows[0]["error"])[:400])
            return
        if s["mode"] in ("team", "playbook") and any(r["role"] == "lead" for r in rows):
            # a lead that answered and went quiet is waiting for the human, not finished: it ends the session with
            # finish_session, or the human does with Stop
            self._set_session(sid, "waiting")
            return
        if s["mode"] == "flow":
            return
        self._finish(sid, "done")

    def _set_session(self, sid, status, **extra):
        cur = self.store.session(sid)
        if cur and cur["status"] == status:
            return
        self.store.update("sessions", {"id": sid}, status=status, **extra)
        self.emit(sid, "status", None, {"status": status, **{k: v for k, v in extra.items() if k == "summary"}})
        if status == "waiting":
            asyncio.ensure_future(self._close_later(sid, after=1800))

    # ---- flows: a graph of steps across harnesses, run by kllm (no lead needed) ---------------------------------
    def validate_flow(self, steps, perm="skip"):
        """-> {"steps": [...]} with ids, deps and status, or AgentError. Each step: {id, title, harness, model,
        objective, after: [ids], may_edit, must_not_touch}. Steps run as soon as everything in `after` is done;
        independent steps run side by side in the session's worktree."""
        if not isinstance(steps, list) or not steps:
            raise AgentError("a flow needs at least one step")
        if len(steps) > FLOW_MAX:
            raise AgentError(f"a flow has at most {FLOW_MAX} steps")
        out, ids = [], set()
        for i, st in enumerate(steps):
            if not isinstance(st, dict):
                raise AgentError(f"step {i + 1} is not an object")
            sid_ = str(st.get("id") or f"s{i + 1}").strip()[:24]
            if not sid_ or sid_ in ids:
                raise AgentError(f"step ids must be unique (step {i + 1}: {sid_!r})")
            obj = str(st.get("objective") or "").strip()
            if not obj:
                raise AgentError(f"step {sid_} needs an objective")
            h = st.get("harness") or "opencode"
            self._validate_harness(h, perm)
            ids.add(sid_)
            out.append({"id": sid_, "title": str(st.get("title") or obj.splitlines()[0])[:80], "harness": h,
                        "model": st.get("model") or None, "objective": obj,
                        "after": [str(x) for x in (st.get("after") or [])],
                        "may_edit": list(st.get("may_edit") or []), "must_not_touch": list(st.get("must_not_touch") or []),
                        "status": "pending", "worker": None, "summary": None})
        for st in out:
            bad = [x for x in st["after"] if x not in ids or x == st["id"]]
            if bad:
                raise AgentError(f"step {st['id']} waits for unknown steps: {', '.join(bad)}")
        # no cycles: repeatedly take steps whose dependencies are all taken
        left, taken = {st["id"]: set(st["after"]) for st in out}, set()
        while left:
            ready = [k for k, deps in left.items() if deps <= taken]
            if not ready:
                raise AgentError("the flow has a cycle: " + ", ".join(sorted(left)))
            for k in ready:
                taken.add(k)
                del left[k]
        return {"steps": out}

    def _flow_save(self, sid, flow):
        self.store.update("sessions", {"id": sid}, flow=flow)
        self.emit(sid, "flow", None, {"steps": flow["steps"]})

    def _flow_lock(self, sid):
        locks = self.__dict__.setdefault("_flow_locks", {})
        return locks.setdefault(sid, asyncio.Lock())

    async def _flow_advance(self, sid):
        async with self._flow_lock(sid):
            await self._flow_advance_locked(sid)

    async def _flow_advance_locked(self, sid):
        """Start every pending step whose dependencies are done; skip steps behind a failed one; finish at the end."""
        s = self.store.session(sid)
        if not s or not s.get("flow") or s["status"] not in ACTIVE:
            return
        flow = s["flow"]
        by = {st["id"]: st for st in flow["steps"]}
        changed = True
        while changed:
            changed = False
            for st in flow["steps"]:
                if st["status"] == "pending" and any(by[d]["status"] in ("failed", "skipped") for d in st["after"]):
                    st["status"], changed = "skipped", True
        for st in flow["steps"]:
            if st["status"] != "pending" or not all(by[d]["status"] == "done" for d in st["after"]):
                continue
            inputs = [{"title": by[d]["title"], "worker": by[d]["worker"], "harness": by[d]["harness"],
                       "text": by[d]["summary"]} for d in st["after"]]
            try:
                w = await self.spawn_worker(sid, st["harness"], st["model"],
                                            {"objective": st["objective"], "may_edit": st["may_edit"],
                                             "must_not_touch": st["must_not_touch"], "inputs": inputs, "step": st["id"]})
            except AgentError as e:
                if e.code == 409:            # worker limit: try again when a step finishes
                    break
                st["status"], st["summary"] = "failed", e.msg
                continue
            st["status"], st["worker"] = "running", w["id"]
            self.store.update("workers", {"session_id": sid, "id": w["id"]}, step=st["id"])
        self._flow_save(sid, flow)
        sts = [st["status"] for st in flow["steps"]]
        if any(x in ("running", "pending") for x in sts):
            if "running" not in sts and not self._live_workers(sid):
                self._finish(sid, "failed", summary="The flow is stuck: steps are waiting but none can start.")
            return
        done = sum(1 for x in sts if x == "done")
        failed = [st for st in flow["steps"] if st["status"] == "failed"]
        lines = [f"{done} of {len(sts)} steps done."] + [f"- {st['title']} ({st['worker'] or 'not started'}): "
                                                          f"{str(st['summary'] or '')[:300]}" for st in flow["steps"]]
        self._finish(sid, "failed" if failed else "done", summary="\n".join(lines))

    async def _flow_step_done(self, sid, wid):
        # two parallel steps can finish together: read, change and save the flow one at a time
        async with self._flow_lock(sid):
            await self._flow_step_done_locked(sid, wid)

    async def _flow_step_done_locked(self, sid, wid):
        s = self.store.session(sid)
        row = self.store.worker(sid, wid)
        if not s or not s.get("flow") or not row:
            return
        flow = s["flow"]
        st = next((x for x in flow["steps"] if x["worker"] == wid and x["status"] == "running"), None)
        if not st:
            return
        st["status"] = "failed" if row.get("error") else "done"
        st["summary"] = (row.get("error") or row.get("last_text") or "")[:3000]
        self.emit(sid, "message", None, {"text": f"Step \"{st['title']}\" {st['status']} ({wid}).", "from": "kllm"})
        w = self.workers.get((sid, wid))
        if w and not w.stopped:
            await w.stop()
        self.store.update("workers", {"session_id": sid, "id": wid}, status="done" if st["status"] == "done" else "failed",
                          finished_at=time.time())
        self._flow_save(sid, flow)
        await self._flow_advance_locked(sid)

    def pool_models(self):
        """[{key, serving, vision}] chat models on the kllm pool, serving ones first."""
        try:
            from .. import media
            reg = common.load_models()
            st = common.read_state()
        except Exception:
            return []
        serving = {b.get("model_key") for b in st.get("backends", []) if b.get("url")}
        rows = [{"key": k, "serving": k in serving, "ctx": (m or {}).get("ctx")} for k, m in reg.items()
                if isinstance(m, dict) and media.kind(m) == "chat" and not m.get("alias_of")]
        return sorted(rows, key=lambda r: (not r["serving"], r["key"]))

    def _finish(self, sid, status, summary=None):
        now = time.time()
        for (ssid, wid), w in list(self.workers.items()):
            if ssid == sid:
                row = self.store.worker(sid, wid)
                if row and row["status"] in ("idle", "starting", "working", "needs_you", "paused"):
                    self.store.update("workers", {"session_id": sid, "id": wid},
                                      status="done" if status == "done" else "stopped", finished_at=now)
        extra = {"finished_at": now}
        if summary:
            extra["summary"] = summary
            if (self.store.session(sid) or {}).get("status") == status:      # already finished: keep the summary
                self.store.update("sessions", {"id": sid}, summary=summary)
                self.emit(sid, "status", None, {"status": status, "summary": summary})
        self._set_session(sid, status, **extra)
        s = self.store.session(sid)
        if s and s.get("attempt_of"):
            asyncio.ensure_future(self._attempt_done(s))
        # close idle harness processes after a while; a later message reopens them (ACP session/load, --resume)
        asyncio.ensure_future(self._close_later(sid))

    async def _close_later(self, sid, after=600):
        await asyncio.sleep(after)
        s = self.store.session(sid)
        if s and s["status"] in LIVE:
            return
        for (ssid, wid), w in list(self.workers.items()):
            if ssid == sid and not w.turn_running:
                await w.close()

    # ---- permissions and approvals ------------------------------------------------------------------------------
    async def permission(self, w, params):
        """ACP session/request_permission -> outcome. Skip approves (except hard rules); Ask asks the human."""
        from .acp import permission_option
        s = self.store.session(w.sid)
        tool = params.get("toolCall") or {}
        options = params.get("options") or []
        paths = tool_paths(tool)
        cmd = tool_command(tool)
        g = (s or {}).get("guardrails") or {}
        wrow = self.store.worker(w.sid, w.wid) or {}
        never = list(g.get("never_touch") or []) + list((wrow.get("contract") or {}).get("must_not_touch") or [])
        top = self._repo_top(w.cwd)
        rel = [os.path.relpath(p, top) if os.path.isabs(p) else p for p in paths]
        hard = [p for p in rel if match_any(p, never)]
        danger = g.get("delete_breaker", True) and (tool.get("kind") == "delete" or bool(DANGEROUS.search(cmd or "")))
        title = tool.get("title") or tool.get("kind") or "tool call"
        if hard:
            self.emit(w.sid, "guardrail", w.wid, {"refused": title, "paths": hard, "why": "never-touch paths"})
            return {"outcome": "selected", "optionId": permission_option(options, False)} \
                if permission_option(options, False) else {"outcome": "cancelled"}
        if s and s["permission_mode"] == "skip" and not danger:
            oid = permission_option(options, True)
            return {"outcome": "selected", "optionId": oid} if oid else {"outcome": "cancelled"}
        if s and s["permission_mode"] == "skip" and danger:
            self.emit(w.sid, "guardrail", w.wid, {"refused": title, "command": cmd[:300], "why": "delete breaker"})
            oid = permission_option(options, False)
            return {"outcome": "selected", "optionId": oid} if oid else {"outcome": "cancelled"}
        summary = title + (f": {cmd[:200]}" if cmd else (f" ({', '.join(rel[:4])})" if rel else ""))
        diff = "\n".join(f"--- {c.get('path')}\n{c.get('newText', '')[:4000]}" for c in tool.get("content") or []
                         if isinstance(c, dict) and c.get("type") == "diff") or None
        aid = self._approval(w.sid, w.wid, "tool", summary, diff=diff, options=options,
                             extra={"paths": rel, "command": cmd or None, "danger": bool(danger)})
        fut = asyncio.get_running_loop().create_future()
        self.futures[aid] = fut
        self._set_worker_status(w, "needs_you")
        try:
            decision = await fut
        except asyncio.CancelledError:
            decision = "reject"
        finally:
            self.futures.pop(aid, None)
        if not w.stopped:
            self._set_worker_status(w, "working")
        oid = permission_option(options, decision == "accept")
        return {"outcome": "selected", "optionId": oid} if oid else {"outcome": "cancelled"}

    def _approval(self, sid, wid, kind, summary, diff=None, options=None, extra=None):
        aid = new_id("a")
        self.store.insert("approvals", {"id": aid, "session_id": sid, "worker_id": wid, "kind": kind,
                                        "summary": summary[:600], "diff": diff, "options": extra or {},
                                        "status": "pending", "created_at": time.time()})
        self.emit(sid, "approval_requested", wid, {"id": aid, "kind": kind, "summary": summary[:600]})
        self._set_session(sid, "needs_you")
        return aid

    async def decide(self, aid, decision, by="you"):
        a = self.store.one("SELECT * FROM approvals WHERE id = ?", (aid,))
        if not a:
            raise AgentError("no such approval", 404)
        if a["status"] != "pending":
            raise AgentError(f"this approval is already {a['status']}", 409)
        if decision not in ("accept", "reject", "lead"):
            raise AgentError("decision must be accept, reject or lead")
        if decision == "lead":
            lead = self.workers.get((a["session_id"], "L"))
            if not lead or lead.stopped:
                raise AgentError("this session has no running lead to decide", 409)
            lead.send(f"Approval {aid} from {a['worker_id']} needs your decision: {a['summary']}\n"
                      f"Call decide_approval(id=\"{aid}\", decision=\"accept\" or \"reject\").")
            self.emit(a["session_id"], "message", None, {"text": f"Approval {aid} handed to the lead.", "from": "you"})
            return {"status": "pending", "handed_to": "lead"}
        status = "accepted" if decision == "accept" else "rejected"
        self.store.update("approvals", {"id": aid}, status=status, decided_by=by, decided_at=time.time())
        fut = self.futures.get(aid)
        if fut and not fut.done():
            fut.set_result(decision)
        elif a["kind"] == "scope" and decision == "reject":
            info = a.get("options") or {}
            s = self.store.session(a["session_id"])
            if s and s.get("worktree_path") and info.get("parent"):
                await asyncio.to_thread(self._revert_paths, s["worktree_path"], info["parent"], info.get("files") or [])
        self.emit(a["session_id"], "approval_resolved", a["worker_id"], {"id": aid, "decision": decision, "by": by})
        sid = a["session_id"]
        if not self.store.q("SELECT id FROM approvals WHERE session_id = ? AND status = 'pending'", (sid,)):
            s = self.store.session(sid)
            if s and s["status"] == "needs_you":
                self._set_session(sid, "running")
            w = self.workers.get((sid, a["worker_id"]))
            if w and not w.turn_running and not w.queue:
                self.worker_idle(w)
            self._maybe_finish(sid)
        return {"status": status}

    # ---- steering and control -----------------------------------------------------------------------------------
    def _live_workers(self, sid):
        return [(wid, w) for (ssid, wid), w in self.workers.items() if ssid == sid and not w.stopped]

    async def _revive(self, sid, wid):
        """A worker whose process was closed (or a relay restart): rebuild it from the store and resume its
        conversation (ACP session/load or claude --resume)."""
        row = self.store.worker(sid, wid)
        s = self.store.session(sid)
        if not row or not s or row["status"] in ("moved",):
            return None
        d, driver, cmd = self._validate_harness(row["harness"], s["permission_mode"])
        env = harnesses.worker_env(None, self.relay_base(), sid, wid, self.access()["cli"],
                                   kllm_env=d.get("kllm_env", False), model=row["model"])
        env = harnesses.permission_env(env, row["harness"], s["permission_mode"])
        wd = self._workdir(s)
        if driver == "acp":
            w = AcpWorker(self, sid, wid, row["harness"], row["model"], wd, env, cmd, role=row["role"],
                          mcp_servers=self._mcp(sid, wid, "acp") if row["role"] == "lead" else [],
                          acp_session_id=row["acp_session_id"])
        else:
            w = StreamWorker(self, sid, wid, row["harness"], row["model"], wd, env, cmd[0], role=row["role"],
                             mcp_config=self._mcp(sid, wid, "stream") if row["role"] == "lead" else None,
                             resume=row["acp_session_id"], kllm=d.get("kllm_env", False))
        self.workers[(sid, wid)] = w
        return w

    async def message(self, sid, to, text, mode="next_turn", sender=None):
        s = self.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        text = str(text or "").strip()
        if not text:
            raise AgentError("write a message")
        if mode not in ("next_turn", "interrupt"):
            raise AgentError("mode must be next_turn or interrupt")
        rows = self.store.workers(sid)
        if to in (None, "", "lead"):
            to = "L" if any(r["id"] == "L" for r in rows) else (rows[0]["id"] if rows else None)
        targets = [r["id"] for r in rows if r["status"] not in ("moved",) and r["id"] != sender] if to == "all" else [to]
        if sender and targets == [sender]:
            raise AgentError("that is you: message another worker")
        if not targets or not all(self.store.worker(sid, t) for t in targets):
            raise AgentError(f"no worker {to} in this session", 404)
        if s["status"] in ("done", "stopped", "failed") and s.get("worktree_path") and \
                not os.path.isdir(s["worktree_path"]):
            raise AgentError("this session's worktree was removed (applied or chosen elsewhere)", 409)
        if sender is None:
            self.emit(sid, "message", None, {"text": text, "from": "you", "to": to, "mode": mode})
        else:
            text = f"Message from {sender}: {text}"       # the recipient knows a teammate wrote, not the human
        for t in targets:
            w = self.workers.get((sid, t))
            if w is None or w.stopped:
                w = await self._revive(sid, t)
                if w is None:
                    continue
                self.store.update("workers", {"session_id": sid, "id": t}, status="idle", finished_at=None)
            w.send(text, mode)
        return {"sent_to": targets, "mode": mode}

    async def pause(self, sid):
        n = 0
        for wid, w in self._live_workers(sid):
            w.pause(); n += 1
            if not w.turn_running:
                self._set_worker_status(w, "paused")
        self.emit(sid, "status", None, {"status": "paused", "workers": n})
        return {"paused": n}

    async def resume(self, sid):
        n = 0
        for wid, w in self._live_workers(sid):
            if w.paused:
                w.resume(); n += 1
                if not w.queue and not w.turn_running:
                    self._set_worker_status(w, "idle")
        self.emit(sid, "status", None, {"status": "running", "workers": n})
        self._maybe_finish(sid)
        return {"resumed": n}

    async def stop(self, sid, why="stopped by you"):
        s = self.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        for aid in [a["id"] for a in self.store.q("SELECT id FROM approvals WHERE session_id = ? AND status = 'pending'",
                                                 (sid,))]:
            self.store.update("approvals", {"id": aid}, status="expired")
            fut = self.futures.get(aid)
            if fut and not fut.done():
                fut.set_result("reject")
        for wid, w in self._live_workers(sid):
            await w.stop()
        for kid in self.store.q("SELECT id, status FROM sessions WHERE attempt_of = ?", (sid,)):
            if kid["status"] in ACTIVE:
                await self.stop(kid["id"], why)
        self._finish(sid, "stopped")
        self.emit(sid, "message", None, {"text": why, "from": "kllm"})
        return {"status": "stopped"}

    async def set_permission(self, sid, mode):
        s = self.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        if mode not in ("ask", "skip"):
            raise AgentError("mode must be ask or skip")
        if mode == "ask":
            for r in self.store.workers(sid):
                if r["driver"] != "acp" and r["status"] not in ("done", "stopped", "moved", "failed"):
                    raise AgentError(f"{r['id']} runs {r['harness']} without ACP, which cannot ask: stop it first", 409)
        if mode == "skip" and not s.get("worktree_path"):
            raise AgentError("Skip needs a worktree; this session runs in place", 409)
        self.store.update("sessions", {"id": sid}, permission_mode=mode)
        self.emit(sid, "status", None, {"permission_mode": mode})
        if mode == "skip":                      # waiting tool approvals are approved now; scope cards stay
            for a in self.store.q("SELECT id FROM approvals WHERE session_id = ? AND status = 'pending' AND kind = 'tool'",
                                  (sid,)):
                if a["id"] in self.futures:
                    await self.decide(a["id"], "accept", by="skip mode")
        return {"permission_mode": mode}

    # ---- changes, apply, checkpoints, fork --------------------------------------------------------------------------
    def _need_wt(self, sid):
        s = self.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        if not s.get("worktree_path") or not s.get("base_commit"):
            raise AgentError("this session has no worktree, so changes are in place in " + s["cwd"], 409)
        if not os.path.isdir(s["worktree_path"]):
            raise AgentError("this session's worktree was removed", 410)
        return s

    async def changes(self, sid, path=None):
        s = self._need_wt(sid)
        wd = s["worktree_path"]
        tree = await asyncio.to_thread(gitops.snapshot, wd)
        if path:
            return {"file": path, "diff": await asyncio.to_thread(gitops.diff, wd, s["base_commit"], tree, path)}
        files = await asyncio.to_thread(gitops.numstat, wd, s["base_commit"], tree)
        owners = collections.defaultdict(list)
        for c in self.store.q("SELECT worker_id, files FROM checkpoints WHERE session_id = ? AND worker_id IS NOT NULL",
                              (sid,)):
            for f in c["files"] or []:
                if c["worker_id"] not in owners[f]:
                    owners[f].append(c["worker_id"])
        for f in files:
            f["owners"] = owners.get(f["file"], [])
            f["conflict"] = len(f["owners"]) > 1
        diff = await asyncio.to_thread(gitops.diff, wd, s["base_commit"], tree)
        return {"base": s["base_commit"], "files": files, "diff": diff,
                "conflicts": [f["file"] for f in files if f["conflict"]]}

    async def apply(self, sid, target, name=None, message=None):
        s = self._need_wt(sid)
        if any(w.turn_running for _, w in self._live_workers(sid)):
            raise AgentError("a worker is still working: pause or stop the session first", 409)
        pend = self.store.q("SELECT id FROM approvals WHERE session_id = ? AND status = 'pending'", (sid,))
        if pend:
            raise AgentError(f"{len(pend)} approval(s) are waiting: decide them in the Inbox first", 409)
        repo = gitops.repo_root(s["cwd"])
        msg = (message or s["title"]).strip()[:200]
        try:
            if target == "new_branch":
                br = name or f"kllm/{gitops.slug(s['title'])}"
                sha = await asyncio.to_thread(gitops.apply_new_branch, s["worktree_path"], s["base_commit"], br, msg)
                result = {"target": "new_branch", "branch": br, "commit": sha}
            elif target == "commit":
                sha = await asyncio.to_thread(gitops.apply_commit, repo, s["worktree_path"], s["base_commit"], msg)
                result = {"target": "commit", "commit": sha}
            else:
                raise AgentError("target must be new_branch or commit")
        except gitops.GitError as e:
            raise AgentError(str(e), 409)
        for wid, w in self._live_workers(sid):
            await w.stop()
        await asyncio.to_thread(gitops.remove_worktree, repo, self._wt_root(s))
        self.store.update("sessions", {"id": sid}, worktree_path=None)
        if s["status"] in ACTIVE:
            self._finish(sid, "done")
        self.emit(sid, "status", None, {"applied": result})
        return result

    def _wt_root(self, s):
        wd = s.get("worktree_path")
        if not wd:
            return None
        top = gitops.repo_root(wd)
        return str(top) if top else wd

    def checkpoints(self, sid):
        return self.store.q("SELECT * FROM checkpoints WHERE session_id = ? ORDER BY n", (sid,))

    async def restore(self, cid):
        c = self.store.one("SELECT * FROM checkpoints WHERE id = ?", (cid,))
        if not c:
            raise AgentError("no such checkpoint", 404)
        s = self._need_wt(c["session_id"])
        if any(w.turn_running for _, w in self._live_workers(s["id"])):
            raise AgentError("a worker is working: pause the session first", 409)
        top = self._wt_root(s)
        try:
            await asyncio.to_thread(gitops.restore, top, c["commit_sha"])
        except gitops.GitError as e:
            raise AgentError(str(e), 500)
        self.emit(s["id"], "checkpoint", None, {"restored": cid, "n": c["n"], "label": c["label"]})
        return {"restored": cid}

    async def fork(self, cid, goal=None):
        c = self.store.one("SELECT * FROM checkpoints WHERE id = ?", (cid,))
        if not c:
            raise AgentError("no such checkpoint", 404)
        s = self.store.session(c["session_id"])
        first = (self.store.workers(s["id"]) or [{}])[0]
        body = {"goal": goal or s["goal"], "mode": "one", "harness": first.get("harness") or "opencode",
                "model": first.get("model"), "permission_mode": s["permission_mode"], "cwd": s["cwd"],
                "title": f"{s['title']} (fork of {c['label']})", "guardrails": s.get("guardrails") or {}}
        res = await self.create_session(body, base=c["commit_sha"])
        self.store.update("sessions", {"id": res["id"]}, base_commit=s["base_commit"])
        self.emit(s["id"], "message", None, {"text": f"Forked from {c['label']} as {res['id']}.", "from": "kllm"})
        return res

    # ---- best of N --------------------------------------------------------------------------------------------------
    async def _attempt_done(self, s):
        verify = (s.get("guardrails") or {}).get("verify_command")
        if verify and s.get("worktree_path") and os.path.isdir(s["worktree_path"]):
            proc = await asyncio.create_subprocess_shell(verify, cwd=s["worktree_path"], stdout=asyncio.subprocess.PIPE,
                                                         stderr=asyncio.subprocess.STDOUT)
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), 900)
                code = proc.returncode
            except asyncio.TimeoutError:
                proc.kill(); out, code = b"verification timed out after 15 min", -1
            self.emit(s["id"], "verify", None, {"command": verify, "code": code,
                                                "output": out.decode("utf-8", "replace")[-4000:]})
        parent = s["attempt_of"]
        kids = self.store.q("SELECT status FROM sessions WHERE attempt_of = ?", (parent,))
        if kids and all(k["status"] not in ACTIVE for k in kids):
            self._set_session(parent, "needs_you")
            self.emit(parent, "message", None, {"text": "Every attempt finished: compare them and choose one.",
                                                "from": "kllm"})

    async def attempts(self, sid):
        kids = self.store.q("SELECT * FROM sessions WHERE attempt_of = ? ORDER BY attempt_n", (sid,))
        if not kids:
            raise AgentError("this session has no attempts", 404)
        out = []
        for k in kids:
            row = {"id": k["id"], "n": k["attempt_n"], "status": k["status"], "chosen": k["chosen"],
                   "worker": (self.store.workers(k["id"]) or [{}])[0]}
            if k.get("worktree_path") and os.path.isdir(k["worktree_path"]) and k.get("base_commit"):
                tree = await asyncio.to_thread(gitops.snapshot, k["worktree_path"])
                st = await asyncio.to_thread(gitops.numstat, k["worktree_path"], k["base_commit"], tree)
                row["files"] = st
                row["added"] = sum(x["added"] or 0 for x in st)
                row["removed"] = sum(x["removed"] or 0 for x in st)
            vv = self.store.q("SELECT data FROM events WHERE session_id = ? AND type = 'verify' ORDER BY id DESC LIMIT 1",
                              (k["id"],))
            row["verify"] = json.loads(vv[0]["data"]) if vv else None
            out.append(row)
        return out

    async def choose(self, kid):
        k = self.store.session(kid)
        if not k or not k.get("attempt_of"):
            raise AgentError("no such attempt", 404)
        parent = k["attempt_of"]
        for other in self.store.q("SELECT * FROM sessions WHERE attempt_of = ? AND id != ?", (parent, kid)):
            if other["status"] in ACTIVE:
                await self.stop(other["id"], "another attempt was chosen")
            repo = gitops.repo_root(other["cwd"])
            if repo and other.get("worktree_path"):
                await asyncio.to_thread(gitops.remove_worktree, repo, self._wt_root(other))
                self.store.update("sessions", {"id": other["id"]}, worktree_path=None)
        self.store.update("sessions", {"id": kid}, chosen=True)
        self._finish(parent, "done", summary=f"Chose attempt {k['attempt_n']} ({kid}).")
        self.emit(parent, "message", None, {"text": f"Attempt {k['attempt_n']} chosen. Apply it from its session.",
                                            "from": "you", "chosen": kid})
        return {"chosen": kid}

    # ---- workers: move, compact, attach ---------------------------------------------------------------------------
    async def stop_worker(self, ref):
        """Stop one worker; the rest of the session keeps going."""
        sid, wid, row = self._find_worker(ref)
        w = self.workers.get((sid, wid))
        if w and not w.stopped:
            await w.stop()
        if row["status"] not in ("done", "stopped", "moved", "failed"):
            self._set_worker_status(w or SimpleNamespace(sid=sid, wid=wid), "stopped", finished_at=time.time())
        self.emit(sid, "message", None, {"text": f"You stopped {wid}.", "from": "kllm"})
        s = self.store.session(sid)
        if s and s["mode"] == "flow":
            self.store.update("workers", {"session_id": sid, "id": wid}, error="stopped by you")
            await self._flow_step_done(sid, wid)       # its step fails; the steps behind it are skipped
        else:
            self._maybe_finish(sid)
        return {"worker": wid, "status": "stopped"}

    def _find_worker(self, ref):
        sid, _, wid = ref.partition(":")
        if not wid:
            raise AgentError("worker ids look like <session>:<worker>, for example s_1a2b3c4d:W1")
        row = self.store.worker(sid, wid)
        if not row:
            raise AgentError("no such worker", 404)
        return sid, wid, row

    async def move(self, ref, harness, model=None):
        sid, wid, row = self._find_worker(ref)
        old = self.workers.get((sid, wid))
        if old:
            await old.stop()
        s = self.store.session(sid)
        changes = ""
        if s.get("worktree_path") and os.path.isdir(s["worktree_path"]):
            try:
                tree = await asyncio.to_thread(gitops.snapshot, s["worktree_path"])
                st = await asyncio.to_thread(gitops.numstat, s["worktree_path"], s["base_commit"], tree)
                changes = "\n".join(f"- {x['file']} (+{x['added']} -{x['removed']})" for x in st[:40])
            except gitops.GitError:
                pass
        contract = row.get("contract") or {}
        prompt = (self._prompt(s, "?", contract, row["role"]) +
                  f"\n\nYou take over from {wid}, which ran in {row['harness']}. Its last message:\n"
                  f"{(row.get('last_text') or '(none)')[:3000]}\n"
                  + (f"Files changed so far in this worktree:\n{changes}\n" if changes else "")
                  + "Read the current state of those files and continue the objective.")
        new = await self.spawn_worker(sid, harness, model, contract, role="worker", prompt=prompt)
        self.store.update("workers", {"session_id": sid, "id": wid}, status="moved", moved_to=new["id"],
                          finished_at=time.time())
        self.emit(sid, "status", wid, {"status": "moved", "to": new["id"], "harness": harness})
        return {"from": wid, "to": new["id"], "harness": harness}

    async def compact(self, ref):
        sid, wid, row = self._find_worker(ref)
        w = self.workers.get((sid, wid)) or await self._revive(sid, wid)
        if w is None:
            raise AgentError("this worker cannot be resumed", 409)
        if "compact" not in (w.commands or []):
            raise AgentError(f"{row['harness']} did not offer a compact command", 409)
        w.send("/compact")
        self.emit(sid, "message", None, {"text": f"Asked {wid} to compact its context.", "from": "you"})
        return {"sent": "/compact"}

    def attach(self, ref):
        sid, wid, row = self._find_worker(ref)
        s = self.store.session(sid)
        d = harnesses.HARNESSES.get(row["harness"]) or {}
        rid = row.get("acp_session_id")
        if not rid or not d.get("attach"):
            raise AgentError("this worker has no conversation to attach to yet", 409)
        cmd = f"cd {shlex.quote(self._workdir(s))} && " + d["attach"].format(sid=shlex.quote(rid))
        return {"command": cmd, "note": "Pause the session first so two processes never write to one conversation."}

    # ---- lead tools (MCP) ----------------------------------------------------------------------------------------------
    async def tool(self, sid, wid, name, args):
        s = self.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        caller = self.store.worker(sid, wid) or {}
        if caller.get("role") != "lead" and name not in PEER_TOOLS:
            raise AgentError(f"{name} is for the lead; workers can use {', '.join(sorted(PEER_TOOLS))}", 403)
        if name == "post_plan":
            steps = [{"text": str(x.get("text") if isinstance(x, dict) else x), "owner": (x or {}).get("owner")
                      if isinstance(x, dict) else None, "files": (x or {}).get("files") if isinstance(x, dict) else []}
                     for x in args.get("steps") or []]
            self.store.set_plan(sid, steps)
            self.emit(sid, "plan", wid, {"steps": steps})
            return {"ok": True, "steps": len(steps)}
        if name == "spawn_worker":
            if not args.get("objective"):
                raise AgentError("spawn_worker needs an objective")
            w = await self.spawn_worker(sid, args.get("harness") or "opencode", args.get("model"),
                                        {"objective": args["objective"], "may_edit": args.get("may_edit") or [],
                                         "must_not_touch": args.get("must_not_touch") or [],
                                         "report_when": args.get("report_when") or []})
            self.emit(sid, "peer", wid, {"text": f"{wid} started {w['id']} in {w['harness']}: {args['objective'][:200]}"})
            res = {"worker": w["id"], "harness": w["harness"], "model": w["model"]}
            note = self.pool_note(w["harness"], w["model"])
            if note:
                res["note"] = note + " Do not investigate kllm itself; wait with get_session, or start the worker in a harness with its own models."
            return res
        if name == "message_worker":
            res = await self.message(sid, args.get("to"), args.get("text"), args.get("mode") or "next_turn", sender=wid)
            self.emit(sid, "peer", wid, {"text": f"{wid} told {args.get('to')}: {str(args.get('text'))[:200]}"})
            return res
        if name == "list_models":
            return {"models": self.pool_models()[:40],
                    "how": "spawn_worker(harness='opencode', model='kllm/<key>') runs a worker on one of these."}
        if name == "stop_worker":
            to = str(args.get("to") or "")
            w = self.workers.get((sid, to))
            if not w or to == wid:
                raise AgentError(f"no running worker {to}", 404)
            await w.stop()
            self._set_worker_status(w, "stopped", finished_at=time.time())
            self.emit(sid, "peer", wid, {"text": f"{wid} stopped {to}"})
            return {"ok": True}
        if name == "get_session":
            v = self.session_view(sid)
            return {"status": v["status"], "plan": v["plan"],
                    "workers": [{"id": w["id"], "harness": w["harness"], "status": w["status"], "turns": w["turns"],
                                 "last_message": (w.get("last_text") or "")[:1500]} for w in v["workers"]]}
        if name == "request_approval":
            aid = self._approval(sid, wid, "question", str(args.get("summary") or "The lead asks for approval")[:600])
            return {"approval": aid, "note": "Check its status later with get_session; the human decides in the Inbox."}
        if name == "decide_approval":
            return await self.decide(str(args.get("id")), args.get("decision"), by="lead")
        if name == "finish_session":
            summary = str(args.get("summary") or "")[:4000]
            for owid, w in self._live_workers(sid):
                if owid != wid:
                    await w.stop()
            self._finish(sid, "done", summary=summary)
            return {"ok": True}
        raise AgentError(f"unknown tool {name}", 404)

    # ---- kllm actions and access --------------------------------------------------------------------------------------
    def kllm_action(self, sid, worker, what, allowed=True, via="cli"):
        if not self.store.session(sid):
            return
        self.emit(sid, "kllm_action", worker, {"action": str(what)[:300], "allowed": bool(allowed), "via": via})

    # ---- export ------------------------------------------------------------------------------------------------------
    def export(self, sid):
        s = self.session_view(sid)
        lines = [f"# {s['title']}", "", f"Goal: {s['goal']}", "",
                 f"Mode: {s['mode']}. Permissions: {s['permission_mode']}. Status: {s['status']}.",
                 f"Started {time.strftime('%Y-%m-%d %H:%M', time.localtime(s['created_at']))}.", ""]
        if s["workers"]:
            lines += ["## Workers", ""] + [f"- {w['id']}: {w['harness']} {w['model'] or ''} ({w['status']}, "
                                           f"{w['turns']} turns)" for w in s["workers"]] + [""]
        if s["plan"]:
            lines += ["## Plan", ""] + [f"{p['n']}. {p['text']} ({p['status']})" for p in s["plan"]] + [""]
        lines += ["## Timeline", ""]
        for ev in self.store.events(sid, 0, limit=100000):
            t = time.strftime("%H:%M:%S", time.localtime(ev["at"]))
            d = ev["data"]
            who = ev["worker"] or d.get("from") or "kllm"
            if ev["type"] == "message":
                lines += [f"**{t} {d.get('from') or who}**", "", d.get("text", ""), ""]
            elif ev["type"] == "tool_call":
                lines.append(f"- {t} {who} ran {d.get('title') or d.get('kind')}")
            elif ev["type"] in ("approval_requested", "approval_resolved", "auto_applied", "guardrail", "checkpoint",
                                "kllm_action", "peer", "verify"):
                lines.append(f"- {t} {ev['type']}: {json.dumps(d)[:300]}")
        if s.get("summary"):
            lines += ["", "## Summary", "", s["summary"]]
        return "\n".join(lines) + "\n"

    # ---- schedules -----------------------------------------------------------------------------------------------------
    def start(self):
        self.recover()
        if self._sched_task is None or self._sched_task.done():
            self._sched_task = asyncio.ensure_future(self._schedule_loop())

    async def _schedule_loop(self):
        while True:
            try:
                await self.run_due_schedules()
            except Exception as e:
                print("agents schedule error:", repr(e), flush=True)
            await asyncio.sleep(30)

    def _pool_hours_left(self):
        try:
            cfg = self.relay.cfg() if self.relay else common.load_config()
            st = common.read_state()
            return sum(max(0.0, common.quota_left_h(a["name"], st))
                       for a in cfg.get("accounts") or [] if not a.get("disabled"))
        except Exception:
            return None

    def _chat_active(self):
        if not self.relay:
            return False
        from .. import scheduler
        window = scheduler.settings(self.relay.cfg())["session_active_seconds"]
        return bool(self.relay.ROUTER.active_sessions(window))

    async def run_due_schedules(self, now=None):
        now = now or time.time()
        tm = time.localtime(now)
        minute = int(now // 60)
        ran = []
        for sc in self.store.q("SELECT * FROM schedules WHERE enabled = 1"):
            if sc.get("last_run") and int(sc["last_run"] // 60) == minute:
                continue
            if not cron_match(sc["cron"], tm):
                continue
            cond = sc.get("conditions") or {}
            why = None
            if cond.get("min_quota_hours") is not None:
                left = self._pool_hours_left()
                if left is not None and left < float(cond["min_quota_hours"]):
                    why = f"skipped: {left:.1f} h of GPU quota left, below {cond['min_quota_hours']} h"
            if not why and cond.get("no_chat_activity") and self._chat_active():
                why = "skipped: a chat session is active"
            result = why
            if not why:
                pb = self.store.one("SELECT * FROM playbooks WHERE id = ?", (sc["playbook_id"],))
                if not pb:
                    result = "skipped: its playbook was deleted"
                else:
                    try:
                        res = await self.create_session({"goal": pb.get("goal") or pb["description"] or pb["name"],
                                                         "mode": "playbook", "playbook_id": pb["id"],
                                                         "permission_mode": pb.get("permission_mode") or "skip",
                                                         "cwd": pb.get("cwd"), "schedule_id": sc["id"],
                                                         "title": f"{pb['name']} (scheduled)"})
                        result = f"started {res['id']}"
                    except AgentError as e:
                        result = f"failed: {e.msg}"
            self.store.update("schedules", {"id": sc["id"]}, last_run=now, last_result=result)
            ran.append((sc["id"], result))
        return ran


def cron_match(expr, tm):
    """Five-field cron (minute hour day-of-month month day-of-week; 0 = Sunday). Supports *, lists, ranges, steps."""
    fields = (expr or "").split()
    if len(fields) != 5:
        return False
    vals = (tm.tm_min, tm.tm_hour, tm.tm_mday, tm.tm_mon, (tm.tm_wday + 1) % 7)
    bounds = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
    for f, v, (lo, hi) in zip(fields, vals, bounds):
        if not _cron_field(f, v, lo, hi):
            return False
    return True


def _cron_field(f, v, lo, hi):
    weekday = hi == 6
    for part in f.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = max(1, int(s))
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
        else:
            a = b = int(part)
        for x in ((v, 7) if weekday and v == 0 else (v,)):     # Sunday is 0 or 7
            if a <= x <= b and (x - a) % step == 0:
                return True
    return False


def validate_cron(expr):
    fields = (expr or "").split()
    if len(fields) != 5:
        raise AgentError("cron needs five fields: minute hour day month weekday, for example `0 3 * * *`")
    for f, (lo, hi) in zip(fields, ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))):
        for part in f.split(","):
            base = part.split("/", 1)[0]
            if base in ("*", ""):
                continue
            for x in base.split("-", 1):
                if not x.isdigit() or not lo <= int(x) <= hi:
                    raise AgentError(f"cron field `{f}` is out of range {lo} to {hi}")
    return expr


MANAGER = Manager()
