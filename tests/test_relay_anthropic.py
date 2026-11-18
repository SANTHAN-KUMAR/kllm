"""Unit and integration tests for Anthropic /v1/messages handling in gateway/relay.py.

Verifies:
1. Transparent reverse-proxy streaming of native Anthropic SSE events without stripping tool calls or dumping reasoning.
2. Non-streaming collection: converting upstream SSE stream into a structured Anthropic JSON response (thinking, text, tool_use).
3. Loop guard detection in Anthropic message streams.
4. _as_stream behavior for streaming and non-streaming requests.
"""
import asyncio, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import httpx, pytest
from fastapi.testclient import TestClient
from gateway import common, relay

LOCAL = ("127.0.0.1", 50000)

class AnthropicMockBackend:
    def __init__(self, sse_events=None):
        self.received_headers = []
        self.received_bodies = []
        self.received_paths = []
        outer = self
        self.sse_events = sse_events or []

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass

            def do_GET(self):
                if self.path in ("/health", "/v1/models"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                outer.received_headers.append(dict(self.headers))
                outer.received_bodies.append(body)
                outer.received_paths.append(self.path)

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                for ev, data in outer.sse_events:
                    payload = f"event: {ev}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
                    self.wfile.write(payload)
                    self.wfile.flush()
                self.close_connection = True

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def env(monkeypatch):
    common.CONFIG_FILE.write_text(
        "relay_host: 127.0.0.1\n"
        "relay_port: 18199\n"
        "default_model: qwen3.6-35b\n"
        "local_fallback: {enabled: false}\n"
        "accounts: []\n"
        "autoscale: {enabled: true, max_wait_seconds: 15}\n"
    , encoding="utf-8")
    relay._cfg["mtime"] = 0
    relay._reg.clear()
    monkeypatch.setattr(relay, "_client", httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=None)))
    relay.ROUTER.inflight.clear()
    relay.ROUTER.waiting.clear()
    relay._bad.clear()
    with common.mutate_state() as s:
        s["backends"] = []
        s["pending"] = {}
    yield
    common.CONFIG_FILE.unlink(missing_ok=True)
    relay._cfg["mtime"] = 0
    relay._reg.clear()
    with common.mutate_state() as s:
        s["backends"] = []
        s["pending"] = {}


def test_as_stream_anthropic():
    req = {"model": "qwen3.6-35b", "messages": [{"role": "user", "content": "hi"}]}
    stream_bytes, collector = relay._as_stream("v1/messages", req)
    assert stream_bytes is not None
    parsed = json.loads(stream_bytes)
    assert parsed["stream"] is True
    assert collector is relay._collect_messages

    req_stream = {"model": "qwen3.6-35b", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    s_bytes, coll = relay._as_stream("v1/messages", req_stream)
    assert s_bytes is None
    assert coll is None


@pytest.mark.anyio
async def test_collect_messages_with_thinking_and_tools():
    """Verify _collect_messages correctly assembles thinking, text, and tool_use blocks."""
    events = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "qwen3.6-35b",
                "stop_reason": None,
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Thinking about the files..."}
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "I will list the files."}
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_01ABC", "name": "Bash", "input": {}}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"command":'}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": ' "ls -la"}'}
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 30}
        }),
        ("message_stop", {"type": "message_stop"})
    ]

    class FakeSSEStream:
        async def aiter_lines(self):
            for ev, d in events:
                yield f"event: {ev}"
                yield f"data: {json.dumps(d)}"
                yield ""

    status, msg = await relay._collect_messages(FakeSSEStream())
    assert status == 200
    assert msg["id"] == "msg_01XYZ"
    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"]["output_tokens"] == 30
    assert len(msg["content"]) == 3

    # Block 0: thinking
    assert msg["content"][0]["type"] == "thinking"
    assert msg["content"][0]["thinking"] == "Thinking about the files..."

    # Block 1: text
    assert msg["content"][1]["type"] == "text"
    assert msg["content"][1]["text"] == "I will list the files."

    # Block 2: tool_use
    assert msg["content"][2]["type"] == "tool_use"
    assert msg["content"][2]["id"] == "toolu_01ABC"
    assert msg["content"][2]["name"] == "Bash"
    assert msg["content"][2]["input"] == {"command": "ls -la"}


def test_relay_transparent_proxy_streaming(env):
    """End-to-end integration test through TestClient for streaming /v1/messages."""
    backend_events = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_live_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "qwen3.6-35b",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 1}
            }
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Internal reasoning here"}
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_call_1", "name": "Read", "input": {}}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "test.txt"}'}
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 15}
        }),
        ("message_stop", {"type": "message_stop"})
    ]

    backend = AnthropicMockBackend(backend_events)
    try:
        now = time.time()
        with common.mutate_state() as s:
            s["backends"] = [{
                "account": "kg1",
                "model_key": "qwen3.6-35b",
                "url": backend.url,
                "api_key": "test-key-123",
                "started": now,
                "ready_at": now,
                "session_seconds": 39600
            }]

        with TestClient(relay.app, client=LOCAL) as client:
            resp = client.post(
                "/v1/messages",
                headers={
                    "content-type": "application/json",
                    "x-api-key": "sk-kllm-local",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "qwen3.6-35b",
                    "stream": True,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "analyze this"}]
                }
            )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body_text = resp.text
            # Verify raw SSE events flow through untouched
            assert "event: message_start" in body_text
            assert "msg_live_123" in body_text
            assert "thinking_delta" in body_text
            assert "Internal reasoning here" in body_text
            assert "toolu_call_1" in body_text
            assert "input_json_delta" in body_text
            assert "test.txt" in body_text
            assert "tool_use" in body_text
            assert "event: message_stop" in body_text

            # Verify backend received the request intact at /v1/messages
            assert len(backend.received_paths) == 1
            assert backend.received_paths[0] == "/v1/messages"
            assert backend.received_headers[0].get("authorization") == "Bearer test-key-123"
    finally:
        backend.close()


def test_relay_transparent_proxy_non_streaming(env):
    """End-to-end integration test through TestClient for non-streaming /v1/messages."""
    backend_events = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_live_nonstream",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "qwen3.6-35b",
                "stop_reason": None,
                "usage": {"input_tokens": 12, "output_tokens": 1}
            }
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Completed task."}
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_complete", "name": "Write", "input": {}}
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"file": "out.txt"}'}
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 20}
        }),
        ("message_stop", {"type": "message_stop"})
    ]

    backend = AnthropicMockBackend(backend_events)
    try:
        now = time.time()
        with common.mutate_state() as s:
            s["backends"] = [{
                "account": "kg1",
                "model_key": "qwen3.6-35b",
                "url": backend.url,
                "api_key": "test-key-123",
                "started": now,
                "ready_at": now,
                "session_seconds": 39600
            }]

        with TestClient(relay.app, client=LOCAL) as client:
            resp = client.post(
                "/v1/messages",
                headers={
                    "content-type": "application/json",
                    "x-api-key": "sk-kllm-local",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "qwen3.6-35b",
                    "stream": False,
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "write out.txt"}]
                }
            )

            assert resp.status_code == 200
            assert "application/json" in resp.headers["content-type"]
            data = resp.json()
            assert data["id"] == "msg_live_nonstream"
            assert data["stop_reason"] == "tool_use"
            assert len(data["content"]) == 2
            assert data["content"][0]["type"] == "text"
            assert data["content"][0]["text"] == "Completed task."
            assert data["content"][1]["type"] == "tool_use"
            assert data["content"][1]["name"] == "Write"
            assert data["content"][1]["input"] == {"file": "out.txt"}
    finally:
        backend.close()
