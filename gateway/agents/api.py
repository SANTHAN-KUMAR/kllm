"""HTTP API for kllm Agents: /admin/api/agents/* (BACKEND_SPEC.md, "Which endpoints does the UI call?").

Local only like the rest of /admin: auth.GatewayKeyMiddleware refuses remote callers and auth.LocalGuardMiddleware
refuses cross-site browsers and foreign Host headers.
"""
import asyncio, functools, json, tempfile, time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from . import harnesses
from .runtime import MANAGER, AgentError, validate_cron, GUARDRAIL_DEFAULTS, ACCESS_LEVELS
from .store import new_id

P = "/admin/api/agents"


def make_router(R=None):
    M = MANAGER
    M.relay = R
    r = APIRouter()

    def guarded(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            try:
                return await fn(*a, **kw)
            except AgentError as e:
                return JSONResponse({"ok": False, "msg": e.msg}, status_code=e.code)
        return wrapper

    async def body(request):
        try:
            b = await request.json()
        except ValueError:
            raise AgentError("request body must be JSON")
        if not isinstance(b, dict):
            raise AgentError("request body must be a JSON object")
        return b

    def sse(gen_events, request):
        async def gen():
            async for ev in gen_events:
                if await request.is_disconnected():
                    return
                if ev is None:
                    yield ": keepalive\n\n"
                    continue
                head = f"id: {ev['id']}\n" if ev.get("id") is not None else ""
                yield head + "data: " + json.dumps(ev) + "\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @r.on_event("startup")
    async def _start():
        M.start()

    # ---- sessions -------------------------------------------------------------------------------------------
    @r.get(P + "/sessions")
    @guarded
    async def list_sessions(request: Request, status: str = ""):
        st = [x for x in status.split(",") if x] or None
        return {"sessions": M.list_sessions(st)}

    @r.post(P + "/sessions")
    @guarded
    async def create_session(request: Request):
        b = await body(request)
        if isinstance(b.get("workers"), list):              # an explicit team without a saved playbook
            b.setdefault("mode", "team")
        return dict(await M.create_session(b), ok=True)

    @r.get(P + "/sessions/{sid}")
    @guarded
    async def get_session(request: Request, sid: str):
        return M.session_view(sid)

    @r.get(P + "/sessions/{sid}/events")
    async def session_events(request: Request, sid: str, after: int = 0, follow: int = 1):
        try:
            after = int(request.headers.get("last-event-id") or after)
        except ValueError:
            pass
        if not M.store.session(sid):
            return JSONResponse({"ok": False, "msg": "no such session"}, status_code=404)
        return sse(M.subscribe(sid, after, request.is_disconnected, bool(follow)), request)

    @r.get(P + "/events")
    async def all_events(request: Request):
        """Live events across all sessions (no replay): sidebar, Board and Inbox refresh on these."""
        return sse(M.subscribe("*", 0, request.is_disconnected), request)

    @r.post(P + "/sessions/{sid}/message")
    @guarded
    async def message(request: Request, sid: str):
        b = await body(request)
        return dict(await M.message(sid, b.get("to"), b.get("text"), b.get("mode") or "next_turn"), ok=True)

    @r.post(P + "/sessions/{sid}/pause")
    @guarded
    async def pause(request: Request, sid: str):
        return dict(await M.pause(sid), ok=True)

    @r.post(P + "/sessions/{sid}/resume")
    @guarded
    async def resume(request: Request, sid: str):
        return dict(await M.resume(sid), ok=True)

    @r.post(P + "/sessions/{sid}/stop")
    @guarded
    async def stop(request: Request, sid: str):
        return dict(await M.stop(sid), ok=True)

    @r.post(P + "/sessions/{sid}/permission")
    @guarded
    async def permission(request: Request, sid: str):
        b = await body(request)
        return dict(await M.set_permission(sid, b.get("mode")), ok=True)

    @r.get(P + "/sessions/{sid}/changes")
    @guarded
    async def changes(request: Request, sid: str, file: str = ""):
        return await M.changes(sid, file or None)

    @r.post(P + "/sessions/{sid}/apply")
    @guarded
    async def apply(request: Request, sid: str):
        b = await body(request)
        return dict(await M.apply(sid, b.get("target"), b.get("name"), b.get("message")), ok=True)

    @r.get(P + "/sessions/{sid}/checkpoints")
    @guarded
    async def checkpoints(request: Request, sid: str):
        if not M.store.session(sid):
            raise AgentError("no such session", 404)
        return {"checkpoints": M.checkpoints(sid)}

    @r.post(P + "/checkpoints/{cid}/restore")
    @guarded
    async def restore(request: Request, cid: str):
        return dict(await M.restore(cid), ok=True)

    @r.post(P + "/checkpoints/{cid}/fork")
    @guarded
    async def fork(request: Request, cid: str):
        b = await body(request) if await request.body() else {}
        return dict(await M.fork(cid, b.get("goal")), ok=True)

    @r.get(P + "/sessions/{sid}/attempts")
    @guarded
    async def attempts(request: Request, sid: str):
        return {"attempts": await M.attempts(sid)}

    @r.post(P + "/attempts/{aid}/choose")
    @guarded
    async def choose(request: Request, aid: str):
        return dict(await M.choose(aid), ok=True)

    @r.get(P + "/sessions/{sid}/export")
    @guarded
    async def export(request: Request, sid: str):
        md = M.export(sid)
        return PlainTextResponse(md, media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="kllm-{sid}.md"'})

    @r.post(P + "/sessions/{sid}/delete")
    @guarded
    async def delete_session(request: Request, sid: str):
        s = M.store.session(sid)
        if not s:
            raise AgentError("no such session", 404)
        if s["status"] in ("queued", "running", "needs_you", "waiting"):
            raise AgentError("stop the session before deleting it", 409)
        from . import gitops
        for row in [s] + M.store.q("SELECT * FROM sessions WHERE attempt_of = ?", (sid,)):
            repo = gitops.repo_root(row["cwd"])
            if repo and row.get("worktree_path"):
                await asyncio.to_thread(gitops.remove_worktree, repo, M._wt_root(row))
            if repo:
                await asyncio.to_thread(gitops.drop_refs, repo, row["id"])
            for t in ("events", "workers", "plan_steps", "checkpoints", "approvals"):
                M.store.delete(t, session_id=row["id"])
            M.store.delete("sessions", id=row["id"])
            for k in [k for k in M.workers if k[0] == row["id"]]:
                M.workers.pop(k, None)
        return {"ok": True, "msg": "session deleted with its worktree and checkpoints"}

    @r.post(P + "/sessions/{sid}/kllm_action")
    @guarded
    async def kllm_action(request: Request, sid: str):
        b = await body(request)
        M.kllm_action(sid, b.get("worker"), b.get("action") or "", b.get("allowed", True), b.get("via") or "cli")
        return {"ok": True}

    # ---- workers --------------------------------------------------------------------------------------------
    @r.post(P + "/workers/{ref}/move")
    @guarded
    async def move(request: Request, ref: str):
        b = await body(request)
        if not b.get("harness"):
            raise AgentError("pick the harness to move to")
        return dict(await M.move(ref, b["harness"], b.get("model")), ok=True)

    @r.post(P + "/workers/{ref}/stop")
    @guarded
    async def stop_worker(request: Request, ref: str):
        return dict(await M.stop_worker(ref), ok=True)

    @r.post(P + "/workers/{ref}/compact")
    @guarded
    async def compact(request: Request, ref: str):
        return dict(await M.compact(ref), ok=True)

    @r.post(P + "/workers/{ref}/attach")
    @guarded
    async def attach(request: Request, ref: str):
        return dict(M.attach(ref), ok=True)

    @r.get(P + "/workers")
    @guarded
    async def workers(request: Request, running: int = 0):
        rows = M.store.workers(running=bool(running))
        titles = {s["id"]: s["title"] for s in M.store.q("SELECT id, title FROM sessions")}
        for w in rows:
            w["session_title"] = titles.get(w["session_id"])
            w["ref"] = f"{w['session_id']}:{w['id']}"
        legacy = []
        try:
            from ..spawner import FLEET
            legacy = [x for x in FLEET.list_all() if not running or x.get("status") in ("PENDING", "RUNNING")]
        except Exception:
            pass
        return {"workers": rows, "subagents": legacy[:200], "limit": M.guardrails().get("worker_limit")}

    # ---- approvals ------------------------------------------------------------------------------------------
    @r.get(P + "/approvals")
    @guarded
    async def approvals(request: Request, status: str = "pending"):
        rows = M.store.q("SELECT a.*, s.title AS session_title, s.permission_mode FROM approvals a "
                         "LEFT JOIN sessions s ON s.id = a.session_id"
                         + (" WHERE a.status = ?" if status != "all" else "") + " ORDER BY a.created_at DESC LIMIT 300",
                         (status,) if status != "all" else ())
        for a in rows:
            a["waiting"] = a["id"] in M.futures
        return {"approvals": rows}

    @r.post(P + "/approvals/{aid}")
    @guarded
    async def decide(request: Request, aid: str):
        b = await body(request)
        return dict(await M.decide(aid, b.get("decision")), ok=True)

    # ---- library: playbooks, schedules, notes ------------------------------------------------------------------
    def crud(name, table, prefix, fields, required, clean=None):
        @r.get(P + f"/{name}", name=f"agents_{name}_list")
        @guarded
        async def list_(request: Request):
            return {name: M.store.q(f"SELECT * FROM {table} ORDER BY created_at DESC")}

        @r.post(P + f"/{name}", name=f"agents_{name}_create")
        @guarded
        async def create(request: Request):
            b = await body(request)
            row = {k: b[k] for k in fields if k in b}
            for k in required:
                if not row.get(k):
                    raise AgentError(f"{k} is required")
            if clean:
                row = clean(row)
            row.update(id=new_id(prefix), created_at=time.time())
            M.store.insert(table, row)
            return {"ok": True, "item": M.store.one(f"SELECT * FROM {table} WHERE id = ?", (row["id"],))}

        @r.put(P + f"/{name}/{{item_id}}", name=f"agents_{name}_update")
        @guarded
        async def update(request: Request, item_id: str):
            b = await body(request)
            row = {k: b[k] for k in fields if k in b}
            if clean:
                row = clean(row)
            if not M.store.one(f"SELECT id FROM {table} WHERE id = ?", (item_id,)):
                raise AgentError("not found", 404)
            M.store.update(table, {"id": item_id}, **row)
            return {"ok": True, "item": M.store.one(f"SELECT * FROM {table} WHERE id = ?", (item_id,))}

        @r.delete(P + f"/{name}/{{item_id}}", name=f"agents_{name}_delete")
        @guarded
        async def delete(request: Request, item_id: str):
            if not M.store.delete(table, id=item_id):
                raise AgentError("not found", 404)
            return {"ok": True}

    def clean_playbook(row):
        if "mode" in row and row["mode"] not in ("one", "team", "playbook", "flow"):
            raise AgentError("mode must be one, team, playbook or flow")
        if row.get("flow"):
            steps = row["flow"].get("steps") if isinstance(row["flow"], dict) else row["flow"]
            row["flow"] = {"steps": [{k: v for k, v in st.items() if k not in ("status", "worker", "summary")}
                                     for st in M.validate_flow(steps)["steps"]]}
        for w in row.get("workers") or []:
            if not isinstance(w, dict) or not w.get("harness") or w["harness"] not in harnesses.HARNESSES:
                raise AgentError("each worker needs a known harness")
        if "permission_mode" in row and row["permission_mode"] not in ("ask", "skip"):
            raise AgentError("permission_mode must be ask or skip")
        return row

    def clean_schedule(row):
        if "cron" in row:
            validate_cron(row["cron"])
        if "playbook_id" in row and not M.store.one("SELECT id FROM playbooks WHERE id = ?", (row["playbook_id"],)):
            raise AgentError("no such playbook")
        if "enabled" in row:
            row["enabled"] = bool(row["enabled"])
        return row

    def clean_note(row):
        if "pinned" in row:
            row["pinned"] = bool(row["pinned"])
        if "text" in row:
            row["text"] = str(row["text"])[:4000]
        return row

    crud("playbooks", "playbooks", "pb", ("name", "description", "mode", "lead", "workers", "guardrails", "goal", "cwd",
                                          "permission_mode", "flow"), ("name",), clean_playbook)
    crud("schedules", "schedules", "sc", ("playbook_id", "cron", "conditions", "enabled"), ("playbook_id", "cron"),
         clean_schedule)
    crud("notes", "notes", "n", ("text", "source", "pinned"), ("text",), clean_note)

    # ---- setup: harnesses, access, guardrails ---------------------------------------------------------------------
    @r.get(P + "/harnesses")
    @guarded
    async def list_harnesses(request: Request, refresh: int = 0):
        return {"harnesses": await asyncio.to_thread(harnesses.detect, bool(refresh))}

    @r.get(P + "/harnesses/{name}/models")
    async def harness_models(name: str, refresh: int = 0):
        if name not in harnesses.HARNESSES:
            raise AgentError("unknown harness", 404)
        keys = [m["key"] for m in M.pool_models()]
        return {"models": await asyncio.to_thread(harnesses.models, name, keys, bool(refresh)),
                "default": harnesses.HARNESSES[name].get("default_model")}

    @r.post(P + "/harnesses/{name}/test")
    @guarded
    async def test_harness(request: Request, name: str):
        d, driver, cmd = harnesses.resolve(name)
        if d is None:
            raise AgentError("unknown harness", 404)
        if driver is None:
            raise AgentError(f"{d['name']} is not usable here", 409)
        t0 = time.time()
        if driver == "stream":
            v = await asyncio.to_thread(harnesses._version, cmd[0])
            return {"ok": bool(v), "driver": driver, "version": v, "secs": round(time.time() - t0, 2),
                    "note": "Runs with stream-json; install the ACP adapter for Ask mode and live thinking."}
        from .acp import AcpClient, AcpError
        c = AcpClient(cmd, tempfile.gettempdir())
        try:
            await c.start()
            info = await c.initialize(timeout=45)
            sess = await c.new_session(tempfile.gettempdir(), timeout=60)
        except (AcpError, asyncio.TimeoutError, OSError) as e:
            raise AgentError(f"{d['name']} did not complete the ACP handshake: {e or type(e).__name__}", 502)
        finally:
            await c.close()
        models = [o.get("value") for opt in sess.get("configOptions") or [] if opt.get("category") == "model"
                  for o in opt.get("options") or []]
        models += [m.get("modelId") for m in (sess.get("models") or {}).get("availableModels") or []]
        return {"ok": True, "driver": driver, "agent": info.get("agentInfo"), "protocol": info.get("protocolVersion"),
                "load_session": bool((info.get("agentCapabilities") or {}).get("loadSession")),
                "auth_methods": [a.get("name") for a in info.get("authMethods") or []],
                "models": [m for m in models if m][:80], "secs": round(time.time() - t0, 2)}

    @r.get(P + "/access")
    @guarded
    async def get_access(request: Request):
        return {"access": M.access(), "levels": list(ACCESS_LEVELS)}

    @r.post(P + "/access")
    @guarded
    async def set_access(request: Request):
        b = await body(request)
        cur = M.access()
        for k in ("cli", "api"):
            if k in b:
                if b[k] not in ACCESS_LEVELS:
                    raise AgentError(f"{k} must be off, read or full")
                cur[k] = b[k]
        M.store.set_setting("access", cur)
        return {"ok": True, "access": cur, "msg": "Access saved. It applies to workers started from now on."}

    @r.get(P + "/guardrails")
    @guarded
    async def get_guardrails(request: Request):
        return {"guardrails": M.guardrails(), "defaults": GUARDRAIL_DEFAULTS}

    @r.post(P + "/guardrails")
    @guarded
    async def set_guardrails(request: Request):
        b = await body(request)
        cur = M.store.get_setting("guardrails", {}) or {}
        for k, v in b.items():
            if k not in GUARDRAIL_DEFAULTS:
                raise AgentError(f"unknown guardrail '{k}'")
            d = GUARDRAIL_DEFAULTS[k]
            if isinstance(d, bool):
                v = bool(v)
            elif isinstance(d, int):
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    raise AgentError(f"{k} must be a whole number")
                if v < 0 or v > 1000:
                    raise AgentError(f"{k} must be between 0 and 1000")
            elif isinstance(d, list):
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise AgentError(f"{k} must be a list of paths")
            elif v is not None:
                v = str(v)
            cur[k] = v
        M.store.set_setting("guardrails", cur)
        return {"ok": True, "guardrails": M.guardrails(), "msg": "Guardrails saved for new sessions."}

    # ---- lead tools forwarded by the stdio MCP server -------------------------------------------------------------
    @r.post(P + "/tools/{name}")
    @guarded
    async def tool(request: Request, name: str):
        b = await body(request)
        sid, wid = b.get("session"), b.get("worker")
        if not sid or not M.store.worker(sid, wid or ""):
            raise AgentError("unknown session or worker", 404)
        return {"ok": True, "result": await M.tool(sid, wid, name, b.get("args") or {})}

    return r
