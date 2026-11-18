"""Kernel keep-alive proxy: SSE behaviour unchanged, long JSON (image/video) requests survive the
tunnel's 120 s limit via whitespace keep-alives. HOLD/BEAT are shrunk to fractions of a second."""
import http.client, json, socket, sys, threading, time, pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kernel_loader import load_kernel

_next = [18240]


def free_port():
    while True:
        _next[0] += 1
        assert _next[0] < 18300
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", _next[0]))
                return _next[0]
            except OSError:
                continue


class Upstream:
    """A model server that answers after `delay` seconds with (status, content-type, body)."""
    def __init__(self, delay, status=200, ctype="application/json", body=b'{"ok": true}', drop=False):
        outer = self
        self.port = free_port()
        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass
            def _h(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                time.sleep(delay)
                if drop:
                    self.connection.shutdown(socket.SHUT_RDWR); self.close_connection = True; return
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            do_GET = do_POST = _h
        self.srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


@pytest.fixture
def proxy():
    mod = load_kernel("validate-3b")
    mod.HOLD, mod.BEAT = 0.3, 0.1
    made = []
    def make(up):
        port = free_port()
        made.append(mod.start_proxy(port, up.port))
        return port
    yield make
    for s in made:
        s.shutdown()


def post(port, path, body=b"{}"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, r.getheader("Content-Type"), raw


def test_slow_json_media_request_gets_whitespace_then_body(proxy):
    up = Upstream(delay=1.0, body=b'{"created": 1, "data": [{"b64_json": "QQ=="}]}')
    st, ctype, raw = post(proxy(up), "/v1/images/generations")
    assert st == 200 and ctype == "application/json"
    assert raw.startswith(b" ") and len(raw) - len(raw.lstrip()) >= 3          # keep-alive spaces first
    assert json.loads(raw)["data"][0]["b64_json"] == "QQ=="                   # still valid JSON


def test_slow_video_request_same(proxy):
    up = Upstream(delay=0.8, body=b'{"data": [{"b64_mp4": "AA=="}], "fps": 8, "frames": 9}')
    st, ctype, raw = post(proxy(up), "/v1/videos/generations")
    assert st == 200 and raw.startswith(b" ") and json.loads(raw)["frames"] == 9


def test_late_error_is_a_json_error_object(proxy):
    up = Upstream(delay=0.8, status=500, body=b'{"error": {"message": "GPU out of memory", "type": "out_of_memory"}}')
    st, ctype, raw = post(proxy(up), "/v1/images/generations")
    assert st == 200 and ctype == "application/json"      # already committed
    assert json.loads(raw)["error"]["message"] == "GPU out of memory"


def test_late_non_json_error_is_wrapped(proxy):
    up = Upstream(delay=0.8, status=502, ctype="text/plain", body=b"bad gateway")
    st, _, raw = post(proxy(up), "/v1/videos/generations")
    assert st == 200 and json.loads(raw)["error"]["message"] == "bad gateway"


def test_upstream_dies_mid_request_is_json_error(proxy):
    up = Upstream(delay=0.8, drop=True)
    st, _, raw = post(proxy(up), "/v1/images/generations")
    assert st == 200 and "error" in json.loads(raw)


def test_fast_media_request_passes_through_with_real_status(proxy):
    up = Upstream(delay=0.0, status=400, body=b'{"error": {"message": "prompt is required"}}')
    st, ctype, raw = post(proxy(up), "/v1/images/generations")
    assert st == 400 and json.loads(raw)["error"]["message"] == "prompt is required"
    assert not raw.startswith(b" ")


def test_sse_behaviour_unchanged(proxy):
    up = Upstream(delay=0.8, ctype="text/event-stream", body=b'data: {"x": 1}\n\ndata: [DONE]\n\n')
    st, ctype, raw = post(proxy(up), "/v1/chat/completions")
    assert st == 200 and ctype == "text/event-stream"
    assert raw.startswith(b": kllm-kernel: working") and raw.endswith(b'data: [DONE]\n\n')


def test_sse_late_error_still_an_sse_event(proxy):
    up = Upstream(delay=0.8, status=500, body=b'{"error": {"message": "boom"}}')
    st, ctype, raw = post(proxy(up), "/v1/chat/completions")
    assert st == 200 and ctype == "text/event-stream"
    assert b'data: {"error": {"message": "boom", "code": 500}}' in raw and raw.endswith(b"data: [DONE]\n\n")


def test_other_paths_never_commit(proxy):
    up = Upstream(delay=0.8, status=404, body=b'{"error": "nope"}')
    c = http.client.HTTPConnection("127.0.0.1", proxy(up), timeout=20)
    c.request("GET", "/props")
    r = c.getresponse()
    assert r.status == 404 and r.read() == b'{"error": "nope"}'


def test_upstream_dead_before_hold_is_502_json(proxy):
    up = Upstream(delay=0.0, drop=True)
    st, ctype, raw = post(proxy(up), "/v1/images/generations")
    assert st == 502 and "error" in json.loads(raw)
