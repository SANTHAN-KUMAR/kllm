"""
Universal Model Context Protocol (MCP) Server for KLLM Subagent Spawner.

Exposes tools to Frontier Cloud LLMs (Main Agent in Antigravity IDE, Claude Desktop, Cursor)
so they can spawn, monitor, and coordinate Kaggle GPU worker subagents across harnesses.

Supports:
- Stdio transport (`python -m gateway.mcp_server` or `kllm mcp`)
- HTTP JSON-RPC endpoint (`POST /mcp` and `POST /__bridge/mcp` on KLLM relay)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import common
from .spawner import FLEET, SubagentStatus

logger = logging.getLogger("kllm.mcp")

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "kllm-subagent-spawner"
SERVER_VERSION = "0.2.0"

MCP_TOOLS = [
    {
        "name": "spawn_harness_subagent",
        "description": (
            "Spawns a specialized worker subagent inside a chosen harness (Claude Code, OpenCode, Shell) "
            "powered 100% by free Kaggle GPU LLMs via the local KLLM relay (http://127.0.0.1:8080). "
            "The subagent executes code exploration, line edits, and verification tests on Kaggle GPUs for free, "
            "returning a compact Structured Deliverable Packet (<1,200 tokens) with unified git diff, test outputs, "
            "and execution status back to the Main Agent. "
            "NOTE: Agents and operations must strictly use ONLY 'kg*' accounts (kg1..kg22); "
            "do NOT touch, launch, or use any 'lk*' or 'kx*' accounts. "
            "DO NOT delete or purge any accounts from config.yaml (all accounts remain permanently in config.yaml)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "harness": {
                    "type": "string",
                    "enum": ["claude-code", "claude-kllm", "claude", "opencode", "shell", "bash", "agy"],
                    "default": "claude-code",
                    "description": "Target harness environment to spawn the subagent inside.",
                },
                "task_instructions": {
                    "type": "string",
                    "description": "Clear, detailed task instructions for the subagent. Must adhere to using ONLY 'kg*' accounts (kg1..kg22); never use 'lk*' or 'kx*' accounts; NEVER delete or purge accounts from config.yaml.",
                },
                "permitted_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of permitted files/directories the subagent is allowed to edit.",
                },
                "verification_command": {
                    "type": "string",
                    "description": "Optional command the subagent or spawner runs to verify success (e.g. 'pytest tests/test_gateway_key.py').",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 600,
                    "description": "Maximum execution timeout in seconds.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory for the subagent (defaults to workspace root).",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override (e.g. 'qwen3.6-35b').",
                },
                "wait": {
                    "type": "boolean",
                    "default": True,
                    "description": "If true, blocks until the subagent finishes and returns deliverable. If false, returns subagent ID immediately.",
                },
                "resume_session_id": {
                    "type": "string",
                    "description": "Claude Code session ID to resume (the `claude_session_id` returned by get_subagent_status / the result of an interrupted worker). The worker continues the same conversation.",
                },
                "open_terminal": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, opens the subagent session in a visible desktop terminal window (wt.exe/powershell).",
                },
            },
            "required": ["task_instructions"],
        },
    },
    {
        "name": "get_subagent_status",
        "description": (
            "Queries the status of a spawned subagent. While RUNNING it reports live `progress` (turns, tool_calls, last_tool, "
            "last_text, idle_seconds: a large idle_seconds means the worker is stalled or waiting on a slow model/API) and "
            "`elapsed_seconds`. Every Claude Code worker has a `claude_session_id` from the moment it starts; if the worker "
            "dies, times out or hits API errors, call spawn_harness_subagent again with resume_session_id=<that id> and a short "
            "'continue' instruction to resume the SAME conversation instead of starting over. Terminal states also return the "
            "git diff, scope_violations (files changed outside permitted_files) and verification output."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "The unique ID of the spawned subagent.",
                },
            },
            "required": ["subagent_id"],
        },
    },
    {
        "name": "list_subagents",
        "description": "Lists all active, completed, or failed worker subagents in the KLLM fleet.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["PENDING", "RUNNING", "COMPLETED", "NEEDS_REVIEW", "FAILED", "TIMEOUT", "CANCELLED"],
                    "description": "Optional filter by status.",
                },
            },
        },
    },
    {
        "name": "cancel_subagent",
        "description": "Terminates and cancels a currently running worker subagent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "The unique ID of the subagent to cancel.",
                },
            },
            "required": ["subagent_id"],
        },
    },
]


# Session tools for the lead of a kllm Agents session. Listed only when this server runs inside a session
# (KLLM_AGENT_SESSION and KLLM_AGENT_WORKER are set in its environment by the agents runtime); each call is
# forwarded to the relay, which owns the session (POST /admin/api/agents/tools/<name>).
AGENT_TOOLS = [
    {"name": "post_plan", "description": "Publish or replace the session plan shown to the human.",
     "inputSchema": {"type": "object", "required": ["steps"], "properties": {"steps": {"type": "array", "items": {
         "type": "object", "required": ["text"], "properties": {"text": {"type": "string"},
                                                              "owner": {"type": "string", "description": "worker id"},
                                                              "files": {"type": "array", "items": {"type": "string"}}}}}}}},
    {"name": "spawn_worker", "description": "Start a worker on one part of the goal, in the session's worktree. "
                                            "Workers may use different harnesses. For a model on the kllm pool use "
                                            "harness opencode and model kllm/<key> (see list_models).",
     "inputSchema": {"type": "object", "required": ["objective"], "properties": {
         "objective": {"type": "string"}, "harness": {"type": "string", "description": "opencode, agy, claude-code, claude-kllm or codex"},
         "model": {"type": "string", "description": "harness model id, or kllm/<key> for a kllm pool model"},
         "may_edit": {"type": "array", "items": {"type": "string"}},
         "must_not_touch": {"type": "array", "items": {"type": "string"}},
         "report_when": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "message_worker", "description": "Send a worker a message: next_turn queues it, interrupt stops its current turn.",
     "inputSchema": {"type": "object", "required": ["to", "text"], "properties": {
         "to": {"type": "string", "description": "worker id such as W1, or all"}, "text": {"type": "string"},
         "mode": {"type": "string", "enum": ["next_turn", "interrupt"]}}}},
    {"name": "get_session", "description": "Session status, plan, and each worker's status and last message.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_models", "description": "Chat models on the human's kllm pool, serving ones first.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "stop_worker", "description": "Stop a worker that has finished or gone off track.",
     "inputSchema": {"type": "object", "required": ["to"], "properties": {"to": {"type": "string", "description": "worker id such as W2"}}}},
    {"name": "request_approval", "description": "Ask the human to approve a step. Returns an approval id at once.",
     "inputSchema": {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}}},
    {"name": "decide_approval", "description": "Accept or reject an approval the human handed to you.",
     "inputSchema": {"type": "object", "required": ["id", "decision"], "properties": {
         "id": {"type": "string"}, "decision": {"type": "string", "enum": ["accept", "reject"]}}}},
    {"name": "finish_session", "description": "End the session with a summary once the goal is met.",
     "inputSchema": {"type": "object", "properties": {"summary": {"type": "string"}}}},
]
AGENT_TOOL_NAMES = {t["name"] for t in AGENT_TOOLS}


def _agent_context():
    import os
    sid, wid = os.environ.get("KLLM_AGENT_SESSION"), os.environ.get("KLLM_AGENT_WORKER")
    return (sid, wid, os.environ.get("KLLM_RELAY") or "http://127.0.0.1:8080") if sid and wid else None


PEER_TOOL_NAMES = {"get_session", "message_worker", "list_models"}


def _session_tools():
    """The lead sees every session tool; a worker sees the peer tools only."""
    import os
    if os.environ.get("KLLM_AGENT_ROLE", "lead") == "lead":
        return AGENT_TOOLS
    return [t for t in AGENT_TOOLS if t["name"] in PEER_TOOL_NAMES]


def _call_agent_tool(name, args):
    import urllib.request, urllib.error
    sid, wid, relay = _agent_context()
    req = urllib.request.Request(f"{relay.rstrip('/')}/admin/api/agents/tools/{name}",
                                 data=json.dumps({"session": sid, "worker": wid, "args": args}).encode(),
                                 headers={"Content-Type": "application/json", "X-Kllm-Panel": "1"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return False, json.loads(r.read().decode()).get("result")
    except urllib.error.HTTPError as e:
        try:
            return True, json.loads(e.read().decode()).get("msg") or str(e)
        except ValueError:
            return True, str(e)
    except Exception as e:
        return True, f"kllm relay unreachable: {e}"


class MCPServer:
    """Core Model Context Protocol message dispatcher."""

    def __init__(self, fleet=FLEET):
        self.fleet = fleet

    async def handle_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch an incoming JSON-RPC 2.0 message."""
        if not isinstance(msg, dict):
            return self._error_response(None, -32600, "Invalid Request: expected JSON object")

        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        # Handle notifications (no response required)
        if msg_id is None and method:
            if method == "notifications/initialized":
                logger.info("Client MCP initialized notification received.")
            return None

        if not method:
            return self._error_response(msg_id, -32600, "Invalid Request: missing method")

        try:
            if method == "initialize":
                return self._handle_initialize(msg_id, params)
            elif method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            elif method == "tools/list":
                return self._handle_tools_list(msg_id)
            elif method == "tools/call":
                return await self._handle_tool_call(msg_id, params)
            elif method == "resources/list":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}}
            elif method == "prompts/list":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}}
            else:
                return self._error_response(msg_id, -32601, f"Method not found: '{method}'")
        except Exception as e:
            logger.exception(f"Error handling MCP method '{method}': {e}")
            return self._error_response(msg_id, -32603, f"Internal error: {str(e)}")

    def _handle_initialize(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    def _handle_tools_list(self, msg_id: Any) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": MCP_TOOLS + (_session_tools() if _agent_context() else []),
            },
        }

    async def _handle_tool_call(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = params.get("name")
        args = params.get("arguments") or {}

        if tool_name in AGENT_TOOL_NAMES and _agent_context():
            is_err, res = await asyncio.to_thread(_call_agent_tool, tool_name, args)
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": res if isinstance(res, str) else json.dumps(res, indent=2)}],
                "isError": is_err}}

        if tool_name == "spawn_harness_subagent":
            try:
                res = await self.fleet.spawn(
                    harness=args.get("harness", "claude-code"),
                    task_instructions=args.get("task_instructions", ""),
                    permitted_files=args.get("permitted_files"),
                    verification_command=args.get("verification_command"),
                    timeout_seconds=args.get("timeout_seconds", 600),
                    cwd=args.get("cwd"),
                    model=args.get("model"),
                    wait=args.get("wait", True),
                    resume_session_id=args.get("resume_session_id"),
                    open_terminal=args.get("open_terminal", False),
                )
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(res, indent=2),
                            }
                        ],
                        "isError": res.get("status") in (SubagentStatus.FAILED.value, SubagentStatus.TIMEOUT.value),
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error spawning subagent: {e}"}],
                        "isError": True,
                    },
                }

        elif tool_name == "get_subagent_status":
            subagent_id = args.get("subagent_id")
            if not subagent_id:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": "Missing 'subagent_id' parameter."}],
                        "isError": True,
                    },
                }
            res = self.fleet.get(subagent_id)
            if not res:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Subagent '{subagent_id}' not found."}],
                        "isError": True,
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                    "isError": False,
                },
            }

        elif tool_name == "list_subagents":
            status_filter = args.get("status")
            items = self.fleet.list_all(status=status_filter)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"subagents": items}, indent=2)}],
                    "isError": False,
                },
            }

        elif tool_name == "cancel_subagent":
            subagent_id = args.get("subagent_id")
            if not subagent_id:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": "Missing 'subagent_id' parameter."}],
                        "isError": True,
                    },
                }
            ok = self.fleet.cancel(subagent_id)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Subagent '{subagent_id}' cancellation: {'success' if ok else 'failed / not found'}.",
                        }
                    ],
                    "isError": not ok,
                },
            }

        else:
            return self._error_response(msg_id, -32601, f"Unknown tool: '{tool_name}'")

    def _error_response(self, msg_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message,
            },
        }


# Global Server Instance
mcp_server = MCPServer()


# ---- Stdio Runner ------------------------------------------------------------

async def run_stdio():
    """Run MCP server over standard input / standard output."""
    if sys.platform == "win32":
        # Windows ProactorEventLoop does not support connect_read_pipe on stdin
        while True:
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
            except (ValueError, EOFError):
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception:
                resp = mcp_server._error_response(None, -32700, "Parse error: invalid JSON")
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
                continue

            if isinstance(req, list):
                responses = []
                for item in req:
                    r = await mcp_server.handle_message(item)
                    if r is not None:
                        responses.append(r)
                if responses:
                    sys.stdout.write(json.dumps(responses) + "\n")
                    sys.stdout.flush()
            else:
                resp = await mcp_server.handle_message(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
    else:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception:
                resp = mcp_server._error_response(None, -32700, "Parse error: invalid JSON")
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
                continue

            if isinstance(req, list):
                responses = []
                for item in req:
                    r = await mcp_server.handle_message(item)
                    if r is not None:
                        responses.append(r)
                if responses:
                    sys.stdout.write(json.dumps(responses) + "\n")
                    sys.stdout.flush()
            else:
                resp = await mcp_server.handle_message(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()


def main():
    """CLI entrypoint for running MCP server over stdio."""
    # Ensure all logging writes to stderr so stdio JSON-RPC stream is not corrupted
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
    try:
        asyncio.run(run_stdio())
    except (KeyboardInterrupt, BrokenPipeError):
        pass


# ---- FastAPI HTTP Router -----------------------------------------------------

router = APIRouter(prefix="", tags=["mcp"])


@router.post("/mcp")
@router.post("/__bridge/mcp")
async def http_mcp_endpoint(req: Request):
    """Universal HTTP endpoint accepting JSON-RPC MCP requests (single or batch)."""
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=mcp_server._error_response(None, -32700, "Parse error: invalid JSON"),
        )

    if isinstance(body, list):
        # JSON-RPC 2.0 Batch request
        responses = []
        for item in body:
            r = await mcp_server.handle_message(item)
            if r is not None:
                responses.append(r)
        if not responses:
            # 204 must carry no body: JSONResponse would render b"{}" while Starlette omits Content-Length for
            # 204, so h11 frames the body as length 0 and uvicorn raises "Too much data for declared Content-Length".
            return Response(status_code=204)
        return JSONResponse(status_code=200, content=responses)

    resp = await mcp_server.handle_message(body)
    if resp is None:
        return Response(status_code=204)
    return JSONResponse(status_code=200, content=resp)


def get_mcp_config_snippet() -> Dict[str, Any]:
    """Helper returning standard configuration snippet for IDEs."""
    if sys.platform == "win32":
        venv_py = common.ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = common.ROOT / ".venv" / "bin" / "python"
    py_exe = str(venv_py) if venv_py.exists() else sys.executable
    return {
        "mcpServers": {
            "kllm-spawner": {
                "command": py_exe,
                "args": ["-m", "gateway.mcp_server"],
                "cwd": str(common.ROOT),
                "env": {
                    "PYTHONPATH": str(common.ROOT),
                },
            }
        }
    }


if __name__ == "__main__":
    main()

