"""Workers: one harness conversation each. AcpWorker speaks ACP; StreamWorker runs `claude -p` per turn.

A worker owns a message queue. Each queued message is one turn. send(mode="interrupt") puts the message first and
cancels the running turn; pause() cancels and holds the queue; resume() continues. The manager (runtime.py) gets
callbacks for events, permission requests and finished turns, and persists everything.
"""
import asyncio, collections, json, os, signal, time
from . import acp

FLUSH_SECONDS = 0.4          # coalesce streamed chunks into one event per worker per kind
TOOL_TEXT_MAX = 4000


def _text_of(content):
    """Text from an ACP content block (or list of them)."""
    if isinstance(content, list):
        return "".join(_text_of(c) for c in content)
    if not isinstance(content, dict):
        return ""
    if content.get("type") == "text":
        return content.get("text") or ""
    if content.get("type") == "content":
        return _text_of(content.get("content"))
    if content.get("type") == "diff":
        return f"diff {content.get('path', '')}"
    return ""


def tool_paths(tool):
    """File paths a tool call touches: ACP locations, diff blocks and common rawInput keys."""
    out = []
    for loc in tool.get("locations") or []:
        if isinstance(loc, dict) and loc.get("path"):
            out.append(loc["path"])
    for c in tool.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "diff" and c.get("path"):
            out.append(c["path"])
    raw = tool.get("rawInput") if isinstance(tool.get("rawInput"), dict) else {}
    for k in ("file_path", "filePath", "path", "notebook_path", "target_file"):
        if isinstance(raw.get(k), str):
            out.append(raw[k])
    return list(dict.fromkeys(out))


def tool_command(tool):
    raw = tool.get("rawInput") if isinstance(tool.get("rawInput"), dict) else {}
    cmd = raw.get("command") or raw.get("cmd")
    return " ".join(cmd) if isinstance(cmd, list) else (cmd if isinstance(cmd, str) else "")


class BaseWorker:
    driver = "?"

    def __init__(self, mgr, sid, wid, harness, model, cwd, env, role="worker"):
        self.mgr, self.sid, self.wid = mgr, sid, wid
        self.harness, self.model, self.cwd, self.env, self.role = harness, model, cwd, env, role
        self.queue = collections.deque()
        self.paused = False
        self.stopped = False
        self.task = None
        self.turn_running = False
        self.interrupted = False
        self.cancelled_mid_turn = False
        self.commands = []                       # slash commands the harness advertised (ACP)
        self.command_info = []                   # the same with their descriptions, for the composer's / menu
        self._buf = {"thought": [], "message": []}
        self._turn_text = []
        self._flusher = None
        self.idle_since = time.time()

    # ---- chunk coalescing -------------------------------------------------------------------------------------
    def _chunk(self, kind, text):
        if not text:
            return
        self._buf[kind].append(text)
        if kind == "message":
            self._turn_text.append(text)
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.ensure_future(self._flush_later())

    async def _flush_later(self):
        await asyncio.sleep(FLUSH_SECONDS)
        self.flush()

    def flush(self):
        for kind, typ in (("thought", "thought_chunk"), ("message", "message_chunk")):
            if self._buf[kind]:
                text = "".join(self._buf[kind])
                self._buf[kind] = []
                self.mgr.emit(self.sid, typ, self.wid, {"text": text})

    # ---- queue and control ------------------------------------------------------------------------------------
    def send(self, text, mode="next_turn"):
        if mode == "interrupt" and self.turn_running:
            self.queue.appendleft(text)
            self.interrupted = True
            asyncio.ensure_future(self.cancel_turn())
        else:
            self.queue.append(text)
        self.kick()

    def kick(self):
        if self.stopped or self.paused:
            return
        if self.task is None or self.task.done():
            self.task = asyncio.ensure_future(self._loop())

    async def _loop(self):
        try:
            while self.queue and not self.paused and not self.stopped:
                text = self.queue.popleft()
                self.turn_running = True
                self._turn_text = []
                self.mgr.turn_started(self)
                stop_reason, err = "end_turn", None
                try:
                    stop_reason = await self.run_turn(text)
                except Exception as e:                     # the harness died or refused: report, keep the session
                    stop_reason, err = "error", str(e)[:600]
                finally:
                    self.turn_running = False
                    self.flush()
                cancelled = stop_reason == "cancelled"
                if cancelled and self.paused:
                    self.cancelled_mid_turn = True
                self.interrupted = False
                await self.mgr.turn_finished(self, stop_reason, "".join(self._turn_text).strip(), err)
        finally:
            self.idle_since = time.time()
            if not self.stopped:
                self.mgr.worker_idle(self)

    def pause(self):
        self.paused = True
        if self.turn_running:
            asyncio.ensure_future(self.cancel_turn())

    def resume(self):
        if not self.paused:
            return
        self.paused = False
        if self.cancelled_mid_turn:
            self.cancelled_mid_turn = False
            self.queue.appendleft("Continue where you stopped.")
        self.kick()

    async def stop(self):
        self.stopped = True
        self.queue.clear()
        if self.turn_running:
            await self.cancel_turn()
        await self.close()

    # ---- driver hooks -------------------------------------------------------------------------------------------
    async def run_turn(self, text):
        raise NotImplementedError

    async def cancel_turn(self):
        raise NotImplementedError

    async def close(self):
        pass

    def resume_id(self):
        return None

    def rel(self, paths):
        """Paths relative to the worker's folder when they are inside it (the timeline shows short paths)."""
        out = []
        base = os.path.realpath(self.cwd)
        for p in paths:
            rp = os.path.realpath(p) if os.path.isabs(p) else p
            out.append(os.path.relpath(rp, base) if os.path.isabs(rp) and (rp + "/").startswith(base + "/") else p)
        return out


class AcpWorker(BaseWorker):
    driver = "acp"

    def __init__(self, mgr, sid, wid, harness, model, cwd, env, cmd, role="worker", mcp_servers=(),
                 acp_session_id=None):
        super().__init__(mgr, sid, wid, harness, model, cwd, env, role)
        self.cmd, self.mcp_servers, self.acp_session_id = cmd, list(mcp_servers), acp_session_id
        self.client = None
        self.tools = {}
        self.updates = 0

    def resume_id(self):
        return self.acp_session_id

    async def ensure_client(self):
        if self.client and not self.client.closed:
            return self.client
        c = acp.AcpClient(self.cmd, self.cwd, self.env, on_update=self.on_update, on_permission=self.on_permission)
        await c.start()
        await c.initialize()
        info = None
        if self.acp_session_id and (c.info.get("agentCapabilities") or {}).get("loadSession"):
            try:
                info = await c.load_session(self.acp_session_id, self.cwd, self.mcp_servers)
            except acp.AcpError:
                info = None
        if info is None:
            info = await c.new_session(self.cwd, self.mcp_servers)
            self.acp_session_id = c.session_id
            self.mgr.worker_update(self, acp_session_id=c.session_id)
        try:
            if self.model:
                await c.set_model(self.model, info)
        except acp.AcpError as e:
            self.mgr.emit(self.sid, "message", self.wid, {"text": f"Could not select model {self.model}: {e}",
                                                           "from": "kllm"})
        self.client = c
        return c

    async def run_turn(self, text):
        for attempt in (1, 2):
            seen = None
            try:
                c = await self.ensure_client()
                seen = self.updates                   # session setup sends updates too: count only this turn's
                r = await c.prompt(text)
                return r.get("stopReason") or "end_turn"
            except acp.AcpError:
                # the harness died before doing anything this turn: restart it once (session/load resumes the chat)
                if attempt == 2 or (seen is not None and self.updates != seen) or self.stopped or self.paused or self.interrupted:
                    raise
                await self.close()
                self.mgr.emit(self.sid, "message", self.wid, {"text": "The harness exited before answering; restarting it once.", "from": "kllm"})

    async def cancel_turn(self):
        # The turn is marked running before the harness process is up (slow to spawn on Windows), and a cancel
        # that arrives before the harness starts the prompt can be dropped (it resets its flag when the prompt
        # begins). So wait for the client and repeat the cancel while the turn is still running.
        for _ in range(100):
            if self.client or not self.turn_running:
                break
            await asyncio.sleep(0.1)
        for _ in range(6):
            if not self.client or not self.turn_running:
                return
            await self.client.cancel()
            for _ in range(10):
                if not self.turn_running:
                    return
                await asyncio.sleep(0.1)

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None

    async def on_permission(self, params):
        return await self.mgr.permission(self, params)

    def on_update(self, u):
        self.updates += 1
        kind = u.get("sessionUpdate")
        if kind == "agent_thought_chunk":
            self._chunk("thought", _text_of(u.get("content")))
        elif kind == "agent_message_chunk":
            self._chunk("message", _text_of(u.get("content")))
        elif kind in ("tool_call", "tool_call_update"):
            self.flush()
            tid = u.get("toolCallId")
            t = self.tools.setdefault(tid, {})
            for k in ("title", "kind", "status", "locations", "rawInput", "content", "rawOutput"):
                if u.get(k) is not None:
                    t[k] = u[k]
            data = {"id": tid, "title": t.get("title"), "kind": t.get("kind"), "status": t.get("status"),
                    "paths": self.rel(tool_paths(t)), "command": tool_command(t)[:400] or None}
            if kind == "tool_call":
                self.mgr.tool_started(self)
                self.mgr.emit(self.sid, "tool_call", self.wid, data)
            else:
                self.mgr.emit(self.sid, "tool_update", self.wid, data)
                if t.get("status") in ("completed", "failed"):
                    out = _text_of(u.get("content")) or (json.dumps(u["rawOutput"])[:TOOL_TEXT_MAX]
                                                         if u.get("rawOutput") is not None else "")
                    if out:
                        self.mgr.emit(self.sid, "tool_output", self.wid, {"id": tid, "text": out[:TOOL_TEXT_MAX]})
        elif kind == "plan":
            entries = [{"text": e.get("content", ""), "status": {"in_progress": "working", "completed": "done"}
                        .get(e.get("status"), "todo")} for e in u.get("entries") or []]
            self.mgr.worker_plan(self, entries)
        elif kind == "usage_update":
            self.mgr.worker_usage(self, u)
        elif kind == "available_commands_update":
            cmds = [c for c in u.get("availableCommands") or [] if isinstance(c, dict) and c.get("name")]
            self.commands = [c["name"] for c in cmds]
            self.command_info = [{"name": c["name"], "description": str(c.get("description") or "")[:160]} for c in cmds]


class StreamWorker(BaseWorker):
    """Claude Code without the ACP adapter: one `claude -p` process per turn, continued with --resume."""
    driver = "stream"

    def __init__(self, mgr, sid, wid, harness, model, cwd, env, claude_bin, role="worker", mcp_config=None,
                 resume=None, kllm=False):
        super().__init__(mgr, sid, wid, harness, model, cwd, env, role)
        self.claude_bin, self.mcp_config, self.kllm = claude_bin, mcp_config, kllm
        self.claude_session = resume
        self.started_once = bool(resume)
        self.proc = None
        self.commands = ["compact"]

    def resume_id(self):
        return self.claude_session

    def command(self, text):
        import uuid
        cmd = [self.claude_bin, "-p", text, "--output-format", "stream-json", "--verbose",
               "--permission-mode", "bypassPermissions"]
        if self.model and not self.kllm:
            cmd += ["--model", self.model]          # on kllm the model travels in ANTHROPIC_MODEL (see worker_env)
        if self.started_once:
            cmd += ["--resume", self.claude_session]
        else:
            self.claude_session = self.claude_session or str(uuid.uuid4())
            cmd += ["--session-id", self.claude_session]
            self.mgr.worker_update(self, acp_session_id=self.claude_session)
        if self.mcp_config:
            cmd += ["--mcp-config", json.dumps(self.mcp_config)]
        return cmd

    async def run_turn(self, text):
        self.proc = await asyncio.create_subprocess_exec(
            *self.command(text), cwd=self.cwd, env=self.env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
            limit=64 * 1024 * 1024)
        self.started_once = True
        result = {}
        async for raw in self.proc.stdout:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if isinstance(ev, dict):
                self._event(ev, result)
        err = (await self.proc.stderr.read()).decode("utf-8", "replace").strip()
        rc = await self.proc.wait()
        self.proc = None
        if self.interrupted or self.paused or self.stopped:
            return "cancelled"
        if rc != 0 and not result:
            raise RuntimeError(err[-500:] or f"claude exited with code {rc}")
        if result.get("is_error"):
            raise RuntimeError(str(result.get("result") or "the model returned an error")[:500])
        return "end_turn"

    def _event(self, ev, result):
        t = ev.get("type")
        if t == "assistant":
            msg = ev.get("message") or {}
            u = msg.get("usage") or {}
            if u:
                self.mgr.worker_usage(self, {"used": int(u.get("input_tokens") or 0)
                                             + int(u.get("cache_read_input_tokens") or 0),
                                             "tokens_in": int(u.get("input_tokens") or 0),
                                             "tokens_out": int(u.get("output_tokens") or 0)})
            for c in msg.get("content") or []:
                ct = c.get("type")
                if ct == "thinking":
                    self._chunk("thought", c.get("thinking") or "")
                elif ct == "text":
                    self._chunk("message", c.get("text") or "")
                elif ct == "tool_use":
                    self.flush()
                    inp = c.get("input") or {}
                    tool = {"rawInput": inp}
                    arg = inp.get("file_path") or inp.get("command") or inp.get("pattern") or inp.get("path") or ""
                    self.mgr.tool_started(self)
                    self.mgr.emit(self.sid, "tool_call", self.wid, {
                        "id": c.get("id"), "title": f"{c.get('name')} {str(arg)[:120]}".strip(),
                        "kind": "edit" if c.get("name") in ("Edit", "Write", "MultiEdit", "NotebookEdit") else
                                "execute" if c.get("name") == "Bash" else "read",
                        "status": "in_progress", "paths": self.rel(tool_paths(tool)),
                        "command": tool_command(tool)[:400] or None})
        elif t == "user":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    content = c.get("content")
                    text = content if isinstance(content, str) else _text_of(content or [])
                    status = "failed" if c.get("is_error") else "completed"
                    self.mgr.emit(self.sid, "tool_update", self.wid, {"id": c.get("tool_use_id"), "status": status})
                    if text:
                        self.mgr.emit(self.sid, "tool_output", self.wid, {"id": c.get("tool_use_id"),
                                                                           "text": text[:TOOL_TEXT_MAX]})
        elif t == "result":
            result.update(ev)

    async def cancel_turn(self):
        p = self.proc
        if p and p.returncode is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
                await asyncio.wait_for(p.wait(), 5)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass

    async def close(self):
        await self.cancel_turn()
