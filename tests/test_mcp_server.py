"""Tests for Model Context Protocol (MCP) server implementation and endpoints."""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from gateway import mcp_server, relay
from gateway.mcp_server import MCPServer, get_mcp_config_snippet
from gateway.spawner import FleetManager, SubagentStatus


def test_mcp_initialize():
    server = MCPServer(fleet=FleetManager())
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    resp = asyncio.run(server.handle_message(req))
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    res = resp["result"]
    assert res["protocolVersion"] == "2024-11-05"
    assert res["serverInfo"]["name"] == "kllm-subagent-spawner"
    assert "tools" in res["capabilities"]


def test_mcp_ping():
    server = MCPServer(fleet=FleetManager())
    req = {"jsonrpc": "2.0", "id": 10, "method": "ping"}
    resp = asyncio.run(server.handle_message(req))
    assert resp == {"jsonrpc": "2.0", "id": 10, "result": {}}


def test_mcp_notification_initialized():
    server = MCPServer(fleet=FleetManager())
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = asyncio.run(server.handle_message(req))
    assert resp is None


def test_mcp_tools_list():
    server = MCPServer(fleet=FleetManager())
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = asyncio.run(server.handle_message(req))
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "spawn_harness_subagent" in tool_names
    assert "get_subagent_status" in tool_names
    assert "list_subagents" in tool_names
    assert "cancel_subagent" in tool_names


def test_mcp_tool_call_spawn_success(tmp_path):
    fleet = FleetManager()
    server = MCPServer(fleet=fleet)

    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "spawn_harness_subagent",
            "arguments": {
                "harness": "shell",
                "task_instructions": "echo 'mcp test task'",
                "cwd": str(tmp_path),
                "wait": True,
            },
        },
    }
    resp = asyncio.run(server.handle_message(req))
    assert resp["id"] == 3
    result = resp["result"]
    assert result["isError"] is False
    assert len(result["content"]) == 1
    text = result["content"][0]["text"]
    data = json.loads(text)
    assert data["status"] == SubagentStatus.COMPLETED.value
    assert data["subagent_summary"].strip("'\"") == "mcp test task"


def test_mcp_tool_call_get_status(tmp_path):
    fleet = FleetManager()
    server = MCPServer(fleet=fleet)

    # Spawn first
    spawn_res = asyncio.run(fleet.spawn(
        harness="shell",
        task_instructions="echo 'get status task'",
        cwd=str(tmp_path),
        wait=True,
    ))
    sid = spawn_res["subagent_id"]

    # Query via MCP tool
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "get_subagent_status",
            "arguments": {"subagent_id": sid},
        },
    }
    resp = asyncio.run(server.handle_message(req))
    assert resp["id"] == 4
    result = resp["result"]
    assert result["isError"] is False
    data = json.loads(result["content"][0]["text"])
    assert data["subagent_id"] == sid
    assert data["status"] == SubagentStatus.COMPLETED.value


def test_mcp_tool_call_list_subagents(tmp_path):
    fleet = FleetManager()
    server = MCPServer(fleet=fleet)

    asyncio.run(fleet.spawn(harness="shell", task_instructions="echo 1", cwd=str(tmp_path), wait=True))
    asyncio.run(fleet.spawn(harness="shell", task_instructions="echo 2", cwd=str(tmp_path), wait=True))

    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "list_subagents",
            "arguments": {},
        },
    }
    resp = asyncio.run(server.handle_message(req))
    assert resp["id"] == 5
    data = json.loads(resp["result"]["content"][0]["text"])
    assert len(data["subagents"]) >= 2


def test_mcp_tool_call_cancel_subagent(tmp_path):
    fleet = FleetManager()
    server = MCPServer(fleet=fleet)

    spawn_res = asyncio.run(fleet.spawn(
        harness="shell",
        task_instructions="sleep 10",
        cwd=str(tmp_path),
        wait=False,
    ))
    sid = spawn_res["subagent_id"]

    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "cancel_subagent",
            "arguments": {"subagent_id": sid},
        },
    }
    resp = asyncio.run(server.handle_message(req))
    assert resp["id"] == 6
    assert resp["result"]["isError"] is False


def test_mcp_error_unknown_tool():
    server = MCPServer(fleet=FleetManager())
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    }
    resp = asyncio.run(server.handle_message(req))
    assert resp["id"] == 7
    assert resp["error"]["code"] == -32601


def test_mcp_error_unknown_method():
    server = MCPServer(fleet=FleetManager())
    req = {"jsonrpc": "2.0", "id": 8, "method": "unknown/method"}
    resp = asyncio.run(server.handle_message(req))
    assert resp["id"] == 8
    assert resp["error"]["code"] == -32601


def test_mcp_http_endpoints():
    client = TestClient(relay.app, client=("127.0.0.1", 50000))

    # 1. POST /mcp with tools/list
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 100, "method": "tools/list", "params": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 100
    assert len(data["result"]["tools"]) == 4

    # 2. POST /__bridge/mcp with ping
    resp2 = client.post(
        "/__bridge/mcp",
        json={"jsonrpc": "2.0", "id": 101, "method": "ping"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["result"] == {}


def test_mcp_http_notification_returns_empty_204():
    """A 204 must carry no body.

    Regression: JSONResponse(status_code=204, content={}) rendered b"{}" while Starlette omits
    Content-Length for 204, so h11 framed the body as length 0 and uvicorn raised
    "Too much data for declared Content-Length" on every MCP notification.
    """
    client = TestClient(relay.app, client=("127.0.0.1", 50000))
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert resp.status_code == 204
    assert resp.content == b""
    assert "content-length" not in {k.lower() for k in resp.headers}


def test_mcp_http_batch_of_notifications_returns_empty_204():
    client = TestClient(relay.app, client=("127.0.0.1", 50000))
    batch = [
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}},
    ]
    resp = client.post("/mcp", json=batch)
    assert resp.status_code == 204
    assert resp.content == b""
    assert "content-length" not in {k.lower() for k in resp.headers}


def test_mcp_config_snippet():
    cfg = get_mcp_config_snippet()
    assert "mcpServers" in cfg
    assert "kllm-spawner" in cfg["mcpServers"]
    server_info = cfg["mcpServers"]["kllm-spawner"]
    assert server_info["args"] == ["-m", "gateway.mcp_server"]
    assert "cwd" in server_info
    assert "PYTHONPATH" in server_info.get("env", {})


def test_mcp_http_batch_requests():
    client = TestClient(relay.app, client=("127.0.0.1", 50000))
    batch = [
        {"jsonrpc": "2.0", "id": 201, "method": "ping"},
        {"jsonrpc": "2.0", "id": 202, "method": "tools/list", "params": {}},
    ]
    resp = client.post("/mcp", json=batch)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == 201
    assert data[1]["id"] == 202


def test_mcp_resources_and_prompts_list():
    server = MCPServer(fleet=FleetManager())
    req_res = {"jsonrpc": "2.0", "id": 301, "method": "resources/list"}
    resp_res = asyncio.run(server.handle_message(req_res))
    assert resp_res["id"] == 301
    assert resp_res["result"] == {"resources": []}

    req_prompt = {"jsonrpc": "2.0", "id": 302, "method": "prompts/list"}
    resp_prompt = asyncio.run(server.handle_message(req_prompt))
    assert resp_prompt["id"] == 302
    assert resp_prompt["result"] == {"prompts": []}


def test_claude_kllm_harness_supported():
    fleet = FleetManager()
    driver = fleet.get_driver("claude-kllm")
    assert driver is not None


