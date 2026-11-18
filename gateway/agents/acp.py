"""Minimal Agent Client Protocol client: kllm is the client, the harness is the agent (JSON-RPC 2.0 over stdio).

Method names follow @agentclientprotocol/sdk 1.2 (protocol version 1): initialize, session/new, session/load,
session/prompt, session/cancel (notification), session/set_config_option, session/set_model; the agent sends
session/update notifications and session/request_permission requests back.

kllm advertises no fs or terminal capability, so the harness does its own file and shell work and asks kllm only
for permission. That is the hook for Ask mode and for the hard rules in Skip mode.
"""
import asyncio, json, os, signal

PROTOCOL_VERSION = 1


class AcpError(Exception):
    pass


class AcpClient:
    """on_update(update: dict) is called for every session/update; on_permission(params) -> outcome dict."""

    def __init__(self, cmd, cwd, env=None, on_update=None, on_permission=None, on_exit=None):
        self.cmd, self.cwd, self.env = cmd, cwd, env
        self.on_update, self.on_permission, self.on_exit = on_update, on_permission, on_exit
        self.proc = None
        self.session_id = None
        self.info = {}
        self._id = 0
        self._pending = {}
        self._reader = None
        self.stderr_tail = []
        self.exit_reason = None
        self.closed = False

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd, cwd=self.cwd, env=self.env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
            limit=64 * 1024 * 1024)
        self._reader = asyncio.ensure_future(self._read())
        asyncio.ensure_future(self._drain_stderr())
        return self

    async def _drain_stderr(self):
        try:
            async for line in self.proc.stderr:
                self.stderr_tail.append(line.decode("utf-8", "replace").rstrip())
                del self.stderr_tail[:-40]
        except Exception:
            pass

    async def _send(self, msg):
        if self.proc is None or self.proc.stdin is None or self.proc.returncode is not None:
            raise AcpError("the harness process is not running")
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self.proc.stdin.drain()

    async def request(self, method, params, timeout=None):
        self._id += 1
        rid = self._id
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        try:
            return await (asyncio.wait_for(fut, timeout) if timeout else fut)
        finally:
            self._pending.pop(rid, None)

    async def notify(self, method, params):
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _respond(self, rid, result=None, error=None):
        msg = {"jsonrpc": "2.0", "id": rid}
        msg["error" if error else "result"] = error or result
        try:
            await self._send(msg)
        except AcpError:
            pass

    async def _handle_request(self, msg):
        method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "session/request_permission":
            try:
                outcome = await self.on_permission(params) if self.on_permission else {"outcome": "cancelled"}
            except Exception:
                outcome = {"outcome": "cancelled"}
            await self._respond(rid, {"outcome": outcome})
        else:
            await self._respond(rid, error={"code": -32601, "message": f"kllm does not implement {method}"})

    async def _read(self):
        try:
            async for raw in self.proc.stdout:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if "method" in msg and "id" in msg:
                    asyncio.ensure_future(self._handle_request(msg))
                elif "method" in msg:
                    if msg["method"] == "session/update" and self.on_update:
                        try:
                            self.on_update((msg.get("params") or {}).get("update") or {})
                        except Exception as e:
                            print("acp update handler error:", repr(e), flush=True)
                elif "id" in msg:
                    fut = self._pending.get(msg["id"])
                    if fut and not fut.done():
                        if msg.get("error"):
                            err = msg["error"]
                            fut.set_exception(AcpError(f"{err.get('message')} ({err.get('code')})"))
                        else:
                            fut.set_result(msg.get("result"))
        finally:
            self.closed = True
            try:                                      # let stderr drain so the error says why
                await asyncio.wait_for(self.proc.wait(), 2)
                await asyncio.sleep(0.2)
            except Exception:
                pass
            rc = self.proc.returncode if self.proc else None
            why = " | ".join(x for x in self.stderr_tail[-3:] if x)
            self.exit_reason = (f"{os.path.basename(self.cmd[0])} exited" + (f" with code {rc}" if rc is not None else "")
                                + (f": {why}" if why else ". Run it in a terminal to check its sign-in and network."))
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(AcpError(self.exit_reason))
            if self.on_exit:
                try:
                    self.on_exit()
                except Exception:
                    pass

    # ---- protocol calls ---------------------------------------------------------------------------------------
    async def initialize(self, timeout=60):
        self.info = await self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": {"name": "kllm", "version": "1"}}, timeout) or {}
        return self.info

    async def new_session(self, cwd, mcp_servers=(), timeout=120):
        r = await self.request("session/new", {"cwd": cwd, "mcpServers": list(mcp_servers)}, timeout) or {}
        self.session_id = r.get("sessionId")
        if not self.session_id:
            raise AcpError("the harness did not return a session id")
        return r

    async def load_session(self, session_id, cwd, mcp_servers=(), timeout=120):
        r = await self.request("session/load", {"sessionId": session_id, "cwd": cwd,
                                                "mcpServers": list(mcp_servers)}, timeout)
        self.session_id = session_id
        return r or {}

    async def set_model(self, model, session_info):
        """Pick a model: a `model` config option (opencode) or session/set_model (Antigravity, Claude adapter)."""
        if not model:
            return False
        for opt in (session_info or {}).get("configOptions") or []:
            if opt.get("category") == "model" or opt.get("id") == "model":
                await self.request("session/set_config_option", {"sessionId": self.session_id,
                                                                 "configId": opt["id"], "value": model}, 30)
                return True
        if (session_info or {}).get("models"):
            await self.request("session/set_model", {"sessionId": self.session_id, "modelId": model}, 30)
            return True
        return False

    async def prompt(self, text):
        r = await self.request("session/prompt", {"sessionId": self.session_id,
                                                  "prompt": [{"type": "text", "text": text}]})
        return r or {}

    async def cancel(self):
        if self.session_id and not self.closed:
            try:
                await self.notify("session/cancel", {"sessionId": self.session_id})
            except AcpError:
                pass

    async def close(self):
        if self.proc and self.proc.returncode is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.closed = True


def permission_option(options, allow):
    """Pick the option id for allow / reject from a request_permission options list (prefer the *_once kinds)."""
    want = ("allow_once", "allow_always") if allow else ("reject_once", "reject_always")
    for kind in want:
        for o in options or []:
            if o.get("kind") == kind:
                return o.get("optionId")
    return None
