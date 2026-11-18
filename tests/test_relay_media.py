"""Relay routing for /v1/images/generations and /v1/videos/generations: routed by model, one request at
a time per backend, counted as demand, none of the chat logic applied."""
import asyncio, json, threading, time, pathlib, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import httpx, pytest
from fastapi.testclient import TestClient
from gateway import common, relay, scheduler, media

LOCAL = ("127.0.0.1", 50000)
MODELS_EXTRA = """
  img-test:
    runtime: diffusers
    kind: image
    hf: acme/img
    vram_gb: 8
  img-test-2:
    runtime: diffusers
    kind: image
    hf: acme/img2
  vid-test:
    runtime: diffusers
    kind: video
    hf: acme/vid
"""


class Backend:
    """Stands in for a diffusers kernel: records bodies, tracks concurrency."""
    def __init__(self, delay=0.0, status=200, body=None):
        self.bodies, self.auth, self.paths = [], [], []
        self.active = self.peak = 0
        outer = self
        reply = body or {"created": 1, "data": [{"b64_json": "QQ=="}]}
        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.bodies.append(self.rfile.read(n)); outer.auth.append(self.headers.get("Authorization"))
                outer.paths.append(self.path)
                outer.active += 1; outer.peak = max(outer.peak, outer.active)
                time.sleep(delay)
                outer.active -= 1
                raw = (b"  " + json.dumps(reply).encode()) if status == 200 else json.dumps(
                    {"error": {"message": "backend says no"}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers(); self.wfile.write(raw)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def register(*pairs):
    """pairs: (account, model_key, Backend|url)"""
    now = time.time()
    with common.mutate_state() as s:
        s["backends"] = [{"account": a, "model_key": m, "url": getattr(b, "url", b), "api_key": f"key-{a}",
                          "started": now, "ready_at": now, "session_seconds": 39600} for a, m, b in pairs]
        s["pending"] = {}


@pytest.fixture
def env(monkeypatch):
    saved = common.MODELS_FILE.read_text(encoding="utf-8")
    common.MODELS_FILE.write_text(saved.rstrip("\n") + "\n" + MODELS_EXTRA, encoding="utf-8")
    common.CONFIG_FILE.write_text("relay_host: 127.0.0.1\nrelay_port: 18199\ndefault_model: qwen3.6-35b-abliterated\n"
                                  "local_fallback: {enabled: false}\naccounts: []\n"
                                  "autoscale: {enabled: true, max_wait_seconds: 15}\n", encoding="utf-8")
    relay._cfg["mtime"] = 0; relay._reg.clear()
    monkeypatch.setattr(relay, "_client", httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=None)))
    relay.ROUTER.inflight.clear(); relay.ROUTER.waiting.clear(); relay._bad.clear()
    register()
    made = []
    def backend(**kw):
        b = Backend(**kw); made.append(b); return b
    yield backend
    for b in made:
        b.srv.shutdown()
    common.MODELS_FILE.write_text(saved, encoding="utf-8")
    common.CONFIG_FILE.unlink(missing_ok=True)
    relay._cfg["mtime"] = 0; relay._reg.clear()
    register()


def client():
    return TestClient(relay.app, client=LOCAL)


def test_routed_by_model_to_the_right_backend(env):
    img, vid = env(), env()
    register(("kg1", "img-test", img), ("kg2", "vid-test", vid))
    with client() as c:
        r = c.post("/v1/images/generations", json={"model": "img-test", "prompt": "a cat", "n": 1})
        assert r.status_code == 200 and r.json()["data"][0]["b64_json"] == "QQ=="
        assert r.headers["x-kllm-backend"] == "kaggle:kg1"
        r = c.post("/v1/videos/generations", json={"model": "vid-test", "prompt": "waves"})
        assert r.status_code == 200 and r.headers["x-kllm-backend"] == "kaggle:kg2"
    assert len(img.bodies) == 1 and len(vid.bodies) == 1
    assert img.paths == ["/v1/images/generations"] and vid.paths == ["/v1/videos/generations"]
    assert img.auth == ["Bearer key-kg1"]                    # backend key injected


def test_chat_logic_is_not_applied(env):
    """No stream conversion, no stream_options, body forwarded untouched."""
    img = env()
    register(("kg1", "img-test", img))
    body = {"model": "img-test", "prompt": "x", "size": "512x512", "negative_prompt": "y"}
    with client() as c:
        assert c.post("/v1/images/generations", json=body).status_code == 200
    assert json.loads(img.bodies[0]) == body
    assert b"stream" not in img.bodies[0]


def test_one_request_at_a_time_per_backend(env):
    img = env(delay=0.5)
    register(("kg1", "img-test", img))
    out = []
    with client() as c:
        ts = [threading.Thread(target=lambda: out.append(c.post("/v1/images/generations", json={"model": "img-test", "prompt": "x"}).status_code))
              for _ in range(3)]
        t0 = time.time()
        [t.start() for t in ts]; [t.join() for t in ts]
    assert out == [200, 200, 200] and img.peak == 1 and time.time() - t0 >= 1.4
    assert relay.ROUTER.inflight.get("kaggle:kg1", 0) == 0     # slot released


def test_second_backend_takes_overflow(env):
    a, b = env(delay=0.6), env(delay=0.6)
    register(("kg1", "img-test", a), ("kg2", "img-test", b))
    with client() as c:
        ts = [threading.Thread(target=lambda: c.post("/v1/images/generations", json={"model": "img-test", "prompt": "x"}))
              for _ in range(2)]
        t0 = time.time()
        [t.start() for t in ts]; [t.join() for t in ts]
    assert a.peak == 1 and b.peak == 1 and len(a.bodies) == 1 and len(b.bodies) == 1 and time.time() - t0 < 1.15


def test_cold_start_counts_as_demand_then_completes(env):
    img = env()
    out = []
    with client() as c:
        t = threading.Thread(target=lambda: out.append(c.post("/v1/images/generations", json={"model": "img-test", "prompt": "x"})))
        t.start()
        d = {}
        for _ in range(30):
            d = c.get("/__relay/demand").json()
            if d["models"].get("img-test", {}).get("waiting"):
                break
            time.sleep(0.1)
        assert d["models"]["img-test"]["waiting"] == 1
        # the planner launches the model for that demand (and one backend per waiting media request)
        reg = common.load_models()
        acts = scheduler.plan(time.time(), {"autoscale": {"enabled": True, "max_backends": 3}}, reg,
                              "qwen3.6-35b-abliterated", [], {}, ["kg1", "kg2"], {}, {"kg1": 28, "kg2": 28}, d)
        assert [a[:3] for a in acts] == [("launch", "img-test", "kg1")]
        register(("kg1", "img-test", img))                  # the supervisor's launch became ready
        t.join(timeout=10)
    assert out and out[0].status_code == 200 and len(img.bodies) == 1


def test_planner_one_backend_per_waiting_media_request():
    reg = {"qwen": {"hf": "x", "quant": "Q4"}, "img": {"hf": "y", "runtime": "diffusers", "kind": "image"}}
    cfg = {"autoscale": {"enabled": True, "max_backends": 3}}
    q = {"kg1": 28, "kg2": 28, "kg3": 28}
    def launches(model, waiting):
        acts = scheduler.plan(1e6, cfg, reg, "qwen", [], {}, ["kg1", "kg2", "kg3"], {}, q,
                              {"models": {model: {"waiting": waiting, "sessions": 0}}, "backends": {}})
        return [a for a in acts if a[0] == "launch"]
    assert len(launches("img", 2)) == 2 and len(launches("qwen", 2)) == 1


def test_no_backend_and_autoscale_off_is_immediate_503(env):
    common.CONFIG_FILE.write_text(common.CONFIG_FILE.read_text(encoding="utf-8").replace("enabled: true", "enabled: false"), encoding="utf-8")
    relay._cfg["mtime"] = 0
    with client() as c:
        t0 = time.time()
        r = c.post("/v1/images/generations", json={"model": "img-test", "prompt": "x"})
    assert r.status_code == 503 and r.json()["error"]["type"] == "no_backend" and "kllm up --model img-test" in r.json()["error"]["message"]
    assert time.time() - t0 < 3


def test_model_resolution_errors(env):
    img = env()
    register(("kg1", "img-test", img))
    with client() as c:
        r = c.post("/v1/images/generations", json={"model": "vid-test", "prompt": "x"})
        assert r.status_code == 400 and "video model" in r.json()["error"]["message"]
        r = c.post("/v1/images/generations", json={"model": "qwen3.6-35b-abliterated", "prompt": "x"})
        assert r.status_code == 400 and r.json()["error"]["type"] == "model_not_found"
        # "kllm" / OpenAI aliases -> the running image model
        r = c.post("/v1/images/generations", json={"model": "dall-e-3", "prompt": "x"})
        assert r.status_code == 200 and r.headers["x-kllm-backend"] == "kaggle:kg1"
        r = c.post("/v1/images/generations", content=b"nope")
        assert r.status_code == 400
        register()                                            # nothing running, two image models: ask which
        r = c.post("/v1/images/generations", json={"prompt": "x"})
        assert r.status_code == 400 and "img-test" in r.json()["error"]["message"] and "img-test-2" in r.json()["error"]["message"]


def test_backend_error_passes_through_and_frees_slot(env):
    img = env(status=500)
    register(("kg1", "img-test", img))
    with client() as c:
        r = c.post("/v1/images/generations", json={"model": "img-test", "prompt": "x"})
        assert r.status_code == 500 and r.json()["error"]["message"] == "backend says no"
        assert relay.ROUTER.inflight.get("kaggle:kg1", 0) == 0


def test_unreachable_backend_is_502_json_and_frees_slot(env):
    register(("kg1", "img-test", "http://127.0.0.1:18299"))        # nothing listens there
    with client() as c:
        r = c.post("/v1/images/generations", json={"model": "img-test", "prompt": "x"})
        assert r.status_code == 502 and r.json()["error"]["type"] == "all_down"
        assert relay.ROUTER.inflight.get("kaggle:kg1", 0) == 0


def test_request_log_and_leading_whitespace_passthrough(env):
    img = env()
    register(("kg1", "img-test", img))
    before = relay.REQ_LOG.stat().st_size if relay.REQ_LOG.exists() else 0
    with client() as c:
        r = c.post("/v1/images/generations", json={"model": "img-test", "prompt": "x"})
    assert r.content.startswith(b"  ")                      # kernel keep-alive whitespace is passed through
    assert r.json()["data"]
    rec = json.loads(relay.REQ_LOG.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["kind"] == "image" and rec["model"] == "img-test" and rec["status"] == 200 and rec["backend"] == "kaggle:kg1"


def test_chat_never_resolves_to_a_media_backend(env):
    img = env()
    register(("kg1", "img-test", img))
    reg = common.load_models()
    assert scheduler.resolve_model("kllm", reg, "qwen3.6-35b-abliterated", ["img-test"]) == "qwen3.6-35b-abliterated"
    assert scheduler.resolve_model("kllm", reg, "qwen3.6-35b-abliterated", ["img-test", "validate-3b"]) == "validate-3b"
    with client() as c:      # model-less /v1/models must not be answered by a diffusers kernel
        r = c.get("/v1/models")
        assert r.status_code == 200 and "qwen3.6-35b-abliterated" in {m["id"] for m in r.json()["data"]}
    assert not img.paths


def test_backends_not_idle_since_boot_after_relay_restart(env, monkeypatch):
    """Usage history is in memory: after a relay restart a backend that booted hours ago must not look idle for
    hours (the idle rule would retire it before its sessions reconnect)."""
    img = env()
    register(("kg1", "img-test", img))
    with common.mutate_state() as s:
        s["backends"][0]["ready_at"] = s["backends"][0]["started"] = time.time() - 4 * 3600
    relay.ROUTER.last_used.pop("kaggle:kg1", None)
    monkeypatch.setattr(relay, "_RELAY_STARTED", time.time())
    with client() as c:
        d = c.get("/__relay/demand").json()
    assert d["backends"]["kaggle:kg1"]["idle_s"] < 60
