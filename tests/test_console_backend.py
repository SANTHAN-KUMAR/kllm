"""Console backend gaps: local guard, live kernel telemetry, media history, connect, rpm, download bytes."""
import asyncio, base64, json, os, pathlib, time

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import auth, common, connect, media_store, relay, scheduler, telemetry

LOCAL = ("127.0.0.1", 5)


@pytest.fixture
def cfgfile():
    common.CONFIG_FILE.write_text("relay_host: 127.0.0.1\nrelay_port: 18199\ndefault_model: x\naccounts:\n"
                                  "  - {name: kg1, user: u1}\npanel_telemetry: false\n", encoding="utf-8")
    relay._cfg["mtime"] = 0
    yield
    common.CONFIG_FILE.unlink(missing_ok=True)
    relay._cfg["mtime"] = 0


# ---- local guard ---------------------------------------------------------------------------------------------------
def H(**kw):
    return [(k.replace("_", "-").encode(), v.encode()) for k, v in kw.items()]


def test_guard_rules():
    g = auth.guard_local
    assert g("POST", "/v1/chat/completions", H(host="evil.com")) is None             # model API is not guarded here
    assert g("GET", "/admin/api/state", H(host="127.0.0.1:8080")) is None
    assert g("GET", "/admin/api/state", H(host="evil.com"))[0] == 403                 # DNS rebinding
    assert g("POST", "/mcp", H(host="127.0.0.1:8080", content_type="text/plain", content_length="10"))[0] == 403
    assert g("POST", "/mcp", H(host="127.0.0.1:8080", content_type="application/json", content_length="10")) is None
    assert g("POST", "/mcp", H(host="localhost", origin="https://evil.com", content_type="application/json",
                               content_length="2"))[0] == 403                          # cross-site browser
    assert g("POST", "/mcp", H(host="localhost", origin="null", content_type="application/json"))[0] == 403
    assert g("POST", "/admin/api/x", H(host="localhost:8080", origin="http://localhost:8080",
                                       x_kllm_panel="1", content_length="5")) is None  # the panel itself
    assert g("POST", "/admin/api/subagents/x/cancel", H(host="127.0.0.1")) is None     # bodyless, no Origin
    assert g("GET", "/admin", H(host="kllm.lan"), ["kllm.lan"]) is None


def test_cross_site_post_to_mcp_is_refused(cfgfile):
    c = TestClient(relay.app, client=LOCAL)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    r = c.post("/mcp", content=body, headers={"Content-Type": "text/plain"})
    assert r.status_code == 403
    r = c.post("/mcp", content=body, headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
    assert r.status_code == 403
    r = c.post("/mcp", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 200 and r.json()["result"]["tools"]


def test_removed_and_unknown_admin_paths_never_reach_a_backend(cfgfile):
    c = TestClient(relay.app, client=LOCAL)
    for p in ("/admin/api/launch", "/admin/api/stop", "/admin/api/rotate", "/admin/api/set-default",
              "/admin/api/set-key", "/admin/api/turn-on", "/admin/api/nope"):
        r = c.post(p, json={})
        assert r.status_code in (404, 405) and "no such endpoint" in r.text, p


# ---- telemetry -----------------------------------------------------------------------------------------------------
def sample(ts, **kw):
    s = {"ts": ts, "cg_used": 10, "cg_limit": 100, "cg_ev_oom_kill": 0,
         "gpu": [{"i": 0, "used_mib": 1000, "total_mib": 15360, "util": 50, "temp": 60}]}
    s.update(kw)
    return s


def test_hub_buffer_summary_and_alerts():
    h = telemetry.TelemetryHub(keep_s=100)
    now = time.time()
    assert h.add("kg1", [sample(1), sample(2), sample(2), sample(1)], now) == 2      # duplicates and old ones dropped
    assert [s["ts"] for s in h.since("kg1", 1)] == [2]
    h.add("kg1", [sample(500)], now)
    assert [s["ts"] for s in h.since("kg1")] == [500]                                  # older than an hour (keep_s)
    sm = h.summary("kg1", now)
    assert sm["gpu_used_mib"] == 1000 and sm["gpu_util"] == 50 and sm["stale"] is False
    backends = [{"account": "kg1", "ready_at": now - 600}]
    assert h.alerts(backends, now) == []
    h.add("kg1", [sample(501, cg_used=95, cg_ev_oom_kill=2, disk={"/kaggle/working": {"used": 95, "total": 100}},
                         gpu=[{"i": 1, "temp": 88}])], now)
    ids = {a["id"] for a in h.alerts(backends, now)}
    assert {"mem-kg1", "oom-kg1", "gpu-temp-kg1-1", "disk-kg1-/kaggle/working"} <= ids
    assert "telemetry-silent-kg1" in {a["id"] for a in h.alerts(backends, now + 31)}
    h.status["kg2"] = "unsupported"
    assert not [a for a in h.alerts([{"account": "kg2", "ready_at": 0}], now) if a["account"] == "kg2"]
    h.forget_missing({"kg9"})
    assert h.latest("kg1") is None


def test_hub_polls_backends_with_since():
    seen = []

    def handler(req):
        seen.append((req.url.path, dict(req.url.params), req.headers.get("authorization")))
        if "dead" in req.url.host:
            return httpx.Response(404)
        since = float(req.url.params["since"])
        return httpx.Response(200, json={"samples": [s for s in (sample(10), sample(20)) if s["ts"] > since]})

    async def go():
        h = telemetry.TelemetryHub()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            bs = [{"account": "kg1", "url": "https://a.example/", "api_key": "k1"},
                  {"account": "kg2", "url": "https://dead.example", "api_key": "k2"}]
            assert await h.poll_once(c, bs) == 2
            assert await h.poll_once(c, bs) == 0
            assert h.status == {"kg1": "ok", "kg2": "unsupported"}
            assert {seen[2][1]["since"], seen[3][1]["since"]} == {"20", "0"}     # kg1 resumes after its newest sample
            assert seen[0][2] == "Bearer k1"
            seq = h.seq
            assert await h.wait_new(seq, 0.05) == seq
    asyncio.run(go())


def test_telemetry_endpoints_and_state_fields(cfgfile, monkeypatch):
    with common.mutate_state() as s:
        s["backends"] = [{"account": "kg1", "url": "https://x.example", "api_key": "k", "model_key": "m",
                          "started": time.time() - 100, "session_seconds": 3600}]
    hub = telemetry.TelemetryHub()
    hub.add("kg1", [sample(time.time())])
    monkeypatch.setattr(telemetry, "HUB", hub)
    relay.ROUTER.begin("kaggle:kg1"); relay.ROUTER.end("kaggle:kg1")
    try:
        c = TestClient(relay.app, client=LOCAL)
        r = c.get("/admin/api/backends/kg1/telemetry").json()
        assert r["status"] == "ok" and len(r["samples"]) == 1
        assert c.get("/admin/api/backends/kg7/telemetry").status_code == 404
        b = c.get("/admin/api/backends").json()["backends"][0]
        assert b["res"]["gpu_used_mib"] == 1000 and b["rpm"] >= 1
        st = c.get("/admin/api/state").json()["pool"][0]
        assert st["res"]["gpu"][0]["util"] == 50 and st["rpm"] >= 1
    finally:
        with common.mutate_state() as s:
            s["backends"] = []


def test_router_rpm_window():
    r = scheduler.Router()
    for _ in range(3):
        r.begin("kaggle:kg1")
    assert r.rpm("kaggle:kg1") == 3 and r.rpm("kaggle:kg2") == 0
    r.recent["kaggle:kg1"][0] = time.time() - 120
    assert r.rpm("kaggle:kg1") == 2


def test_timeline_reports_download_bytes(cfgfile):
    with common.mutate_state() as s:
        s["pending"] = {"kg1": {"model_key": "m", "started": time.time() - 300, "boot_at": time.time() - 200,
                                "phase": "downloading", "bytes_done": 5 * 2**30, "bytes_total": 16 * 2**30,
                                "bytes_at": int(time.time())}}
    try:
        p = TestClient(relay.app, client=LOCAL).get("/admin/api/backends").json()["pending"][0]
        assert p["bytes_done"] == 5 * 2**30 and p["bytes_total"] == 16 * 2**30
        dl = p["steps"][4]
        assert dl["key"] == "download" and dl["state"] == "active" and dl["bytes_done"] == 5 * 2**30
    finally:
        with common.mutate_state() as s:
            s["pending"] = {}


# ---- media history -------------------------------------------------------------------------------------------------
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


def test_media_store_save_list_prune_delete():
    body = json.dumps({"data": [{"b64_json": base64.b64encode(PNG).decode(), "seed": 7},
                                {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}]})
    rows = media_store.save("image", {"prompt": "a cat", "size": "512x512", "model": "flux"}, body, backend="kaggle:kg1")
    assert len(rows) == 2 and rows[0]["file"].endswith(".png") and rows[0]["seed"] == 7
    vid = json.dumps({"data": [{"b64_mp4": base64.b64encode(b"mp4data").decode()}]})
    v = media_store.save("video", {"prompt": "waves", "seconds": 4}, vid)[0]
    assert v["file"].endswith(".mp4") and v["seconds"] == 4
    assert [r["kind"] for r in media_store.list_media()][:1] == ["video"]
    assert len(media_store.list_media("image")) >= 2
    p, row = media_store.path_of(rows[0]["id"])
    assert p.read_bytes() == PNG and row["prompt"] == "a cat"
    assert media_store.delete(rows[0]["id"]) and not p.exists() and not media_store.delete(rows[0]["id"])
    assert media_store.save("image", {}, "not json") == [] and media_store.save("image", {}, '{"data": 1}') == []
    big = json.dumps({"data": [{"b64_json": base64.b64encode(PNG + b"x" * (1024 * 1024)).decode()}]})
    for _ in range(3):
        media_store.save("image", {"prompt": "big"}, big, max_mb=2)
    total = sum(r["bytes"] for r in media_store.list_media(limit=1000))
    assert total <= 2 * 1024 * 1024


def test_media_endpoints(cfgfile):
    row = media_store.save("image", {"prompt": "endpoint"},
                           json.dumps({"data": [{"b64_json": base64.b64encode(PNG).decode()}]}))[0]
    c = TestClient(relay.app, client=LOCAL)
    items = c.get("/admin/api/media?kind=image").json()["media"]
    hit = next(x for x in items if x["id"] == row["id"])
    assert hit["url"] == f"/admin/api/media/{row['id']}/file"
    assert c.get(hit["url"]).content == PNG
    assert c.get("/admin/api/media?kind=audio").status_code == 400
    assert c.post(f"/admin/api/media/{row['id']}/delete", json={}, headers={"X-Kllm-Panel": "1"}).json()["ok"]
    assert c.get(hit["url"]).status_code == 404


def test_relay_pump_stores_successful_media(cfgfile):
    class Resp:
        status_code = 200
        headers = {}
        async def aiter_raw(self):
            data = json.dumps({"data": [{"b64_json": base64.b64encode(PNG).decode()}]}).encode()
            yield data[:10]; yield data[10:]
        async def aclose(self):
            pass

    async def go():
        relay.ROUTER.begin("kaggle:kg1")
        rec = {"kind": "image", "model": "flux"}
        out = b"".join([c async for c in relay._media_pump(Resp(), "kaggle:kg1", rec, time.time(), {"prompt": "pumped"})])
        assert json.loads(out)["data"]
        for _ in range(50):
            if any(r["prompt"] == "pumped" for r in media_store.list_media("image")):
                return
            await asyncio.sleep(0.05)
        raise AssertionError("media not stored")
    asyncio.run(go())


# ---- connect -------------------------------------------------------------------------------------------------------
def test_connect_detect_and_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(connect, "_version", lambda b: "1.0")
    which = {"claude": "/x/claude", "opencode": "/x/opencode"}
    monkeypatch.setattr(connect.common, "which", lambda b: which.get(b))
    connect._CACHE["rows"] = None
    rows = {r["id"]: {o["what"]: o for o in r["options"]} for r in connect.detect()}
    assert rows["claude-code"]["models"]["can_setup"] and rows["claude-code"]["models"]["state"] == "off"
    assert rows["opencode"]["models"]["can_setup"] and rows["codex"]["models"]["state"] == "missing"
    msg = connect.setup("claude-code")
    assert connect._ours(connect._launcher(tmp_path / ".local" / "bin")) and "Installed" in msg
    msg = connect.setup("opencode")
    cfg = tmp_path / ".config" / "opencode" / "opencode.jsonc"
    assert cfg.exists() and '"kllm"' in cfg.read_text(encoding="utf-8")
    assert "already" in connect.setup("opencode")
    cfg.write_text('{"provider": {"other": {}', encoding="utf-8")                         # a file it cannot parse
    with pytest.raises(connect.SetupError, match="not valid JSON"):
        connect.setup("opencode")
    assert cfg.read_text(encoding="utf-8") == '{"provider": {"other": {}'              # never rewrites a user's file
    with pytest.raises(connect.SetupError):
        connect.setup("cursor")
    connect._CACHE["rows"] = None


def test_launch_refuses_accounts_outside_allow_prefixes(monkeypatch):
    common.CONFIG_FILE.write_text("relay_host: 127.0.0.1\nrelay_port: 18199\ndefault_model: qwen3.8-27b\n"
                                  "account_allow_prefixes: [kg]\naccounts:\n  - {name: kg1, user: u1}\n"
                                  "  - {name: lk4, user: u4}\npanel_telemetry: false\n", encoding="utf-8")
    relay._cfg["mtime"] = 0
    spawned = []
    monkeypatch.setattr(relay, "_spawn", lambda args: spawned.append(args))
    try:
        c = TestClient(relay.app, client=LOCAL)
        model = next(iter(relay._registry()))
        r = c.post("/admin/api/backends/launch", json={"model": model, "account": "lk4"}, headers={"X-Kllm-Panel": "1"})
        assert r.status_code == 403 and "account_allow_prefixes" in r.json()["msg"] and not spawned
        accts = {a["name"]: a["allowed"] for a in c.get("/admin/api/backends").json()["accounts"]}
        assert accts == {"kg1": True, "lk4": False}
    finally:
        common.CONFIG_FILE.unlink(missing_ok=True)
        relay._cfg["mtime"] = 0
