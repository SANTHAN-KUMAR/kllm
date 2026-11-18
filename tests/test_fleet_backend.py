"""Fleet map plumbing: client classification, in-memory traffic trackers and the /backends
contract the Fleet page reads. Everything here exercises the real router and the real log
backfill, against the throwaway KLLM_ROOT used by the rest of the suite."""
import collections, json, os, time

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from gateway import common, relay, scheduler

LOCAL = ("127.0.0.1", 5)


@pytest.fixture
def cfgfile():
    common.CONFIG_FILE.write_text("relay_host: 127.0.0.1\nrelay_port: 18199\ndefault_model: x\naccounts:\n"
                                  "  - {name: kg1, user: u1}\npanel_telemetry: false\n", encoding="utf-8")
    relay._cfg["mtime"] = 0
    yield
    common.CONFIG_FILE.unlink(missing_ok=True)
    relay._cfg["mtime"] = 0


def req_with(headers):
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                    "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})


def test_client_kind_from_user_agent():
    assert relay._client_kind(req_with({"user-agent": "claude-cli/2.1.322 (external, cli)"})) == "Claude Code"
    assert relay._client_kind(req_with({"user-agent": "opencode/2.2.0"})) == "opencode"
    assert relay._client_kind(req_with({"x-kllm-agent-session": "s_1"})) == "kllm Agents"
    assert relay._client_kind(req_with({"user-agent": "antigravity/1.5.4"})) == "Antigravity"
    assert relay._client_kind(req_with({"user-agent": "gemini-cli/0.12.0"})) == "Gemini CLI"
    assert relay._client_kind(req_with({"user-agent": "codex/0.48.0"})) == "Codex"
    assert relay._client_kind(req_with({"user-agent": "python-requests/2.32.3"})) == "API"
    assert relay._client_kind(None) is None


def test_touch_session_tracks_client_and_recent():
    r = scheduler.Router()
    r.touch_session("u:abc", "qwen3.8-27b", 100, client="Claude Code")
    r.touch_session("u:abc", "qwen3.8-27b", 100, client="Claude Code")
    assert r.sessions["u:abc"]["client"] == "Claude Code"
    assert r.client_rpm(60.0)["Claude Code"] == 2.0
    # clients with no recent traffic are absent, not zero
    assert "opencode" not in r.client_rpm(60.0)


def test_traffic_series_buckets_by_minute():
    r = scheduler.Router()
    now = time.time()
    for i in range(7):
        r.recent.setdefault("kaggle:kg1", collections.deque(maxlen=4000)).append(now - 30)      # this minute
    for i in range(4):
        r.recent.setdefault("kaggle:kg2", collections.deque(maxlen=4000)).append(now - 170)     # two minutes ago
    series = r.traffic_series(60, 60)
    assert len(series) == 60
    assert series[59] == 7                             # most recent bucket last
    assert series[57] == 4
    assert sum(series) == 11
    assert r.ttft_p50(3600) is None                    # no samples yet


def test_backfill_router_from_log_rebuilds_traffic(monkeypatch, tmp_path):
    log = tmp_path / "requests.jsonl"
    now = time.time()
    rows = [
        {"ts": int(now - 10), "model": "qwen3.8-27b", "backend": "kaggle:kg1", "status": 200, "ttft_s": 1.0, "client": "Claude Code"},
        {"ts": int(now - 20), "model": "qwen3.8-27b", "backend": "kaggle:kg1", "status": 200, "ttft_s": 3.0, "client": "opencode"},
        {"ts": int(now - 500), "model": "qwen3.8-27b", "backend": "kaggle:kg2", "status": 200, "ttft_s": 2.0, "client": "Claude Code"},
        {"ts": int(now - 99999), "model": "old", "backend": "kaggle:kg9", "status": 200, "ttft_s": 9.9, "client": "API"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    old_recent, old_client, old_ttft = dict(relay.ROUTER.recent), list(relay.ROUTER.client_recent), list(relay.ROUTER.ttft_recent)
    relay.ROUTER.recent.clear()
    relay.ROUTER.client_recent.clear()
    relay.ROUTER.ttft_recent.clear()
    monkeypatch.setattr(relay, "REQ_LOG", log)
    try:
        relay._backfill_router_from_log(3600)
        assert sorted(t for q in relay.ROUTER.recent.values() for t in q) == sorted([rows[0]["ts"], rows[1]["ts"], rows[2]["ts"]])
        assert [c for _, c in relay.ROUTER.client_recent].count("Claude Code") == 2
        assert "opencode" in [c for _, c in relay.ROUTER.client_recent]
        assert sorted(relay.ROUTER.ttft_recent, key=lambda x: x[1])[-1][1] == 3.0
        assert relay.ROUTER.ttft_p50(3600) == 2.0                     # median of [1.0, 2.0, 3.0]
    finally:
        relay.ROUTER.recent.clear(); relay.ROUTER.recent.update(old_recent)
        relay.ROUTER.client_recent.clear(); relay.ROUTER.client_recent.extend(old_client)
        relay.ROUTER.ttft_recent.clear(); relay.ROUTER.ttft_recent.extend(old_ttft)


def test_backends_endpoint_has_client_and_traffic_contract(cfgfile, monkeypatch):
    traffic = {"series": list(range(60)), "rpm": 3.5, "ttft_p50": 1.8}
    monkeypatch.setattr(relay.ROUTER, "active_sessions", lambda window, **kw: [
        {"client": "Claude Code", "model": "m", "backend": "kaggle:kg1", "last": time.time(), "first": time.time() - 10, "n": 5},
        {"client": "Claude Code", "model": "m", "backend": "kaggle:kg1", "last": time.time(), "first": time.time() - 10, "n": 2},
        {"client": "opencode", "model": "m", "backend": "kaggle:kg1", "last": time.time(), "first": time.time() - 10, "n": 4},
    ])
    monkeypatch.setattr(relay.ROUTER, "client_rpm", lambda window=60.0: {"Claude Code": 3.5, "opencode": 1.0})
    monkeypatch.setattr(relay.ROUTER, "traffic_series", lambda b, s: traffic["series"])
    monkeypatch.setattr(relay.ROUTER, "ttft_p50", lambda window=3600: 1.8)
    c = TestClient(relay.app, client=LOCAL)
    r = c.get("/admin/api/backends")
    assert r.status_code == 200
    d = r.json()
    assert d["traffic"]["ttft_p50"] == 1.8 and d["traffic"]["rpm"] >= 0 and len(d["traffic"]["series"]) == 60
    kinds = {x["kind"]: x for x in d["clients"]}
    assert kinds["Claude Code"]["sessions"] == 2 and kinds["Claude Code"]["rpm"] == 3.5
    assert kinds["opencode"]["sessions"] == 1


def test_matches_kind_filter():
    from gateway import metrics
    img = {"model": "flux", "backend": "kaggle:kg1", "kind": "image", "status": 200}
    vid = {"model": "wan", "backend": "kaggle:kg1", "kind": "video", "status": 200}
    chat = {"model": "qwen", "backend": "kaggle:kg1", "status": 200}
    assert metrics.matches(img, {"kind": "image"}) and not metrics.matches(vid, {"kind": "image"})
    assert metrics.matches(chat, {"kind": "chat"})        # missing kind reads as chat
    assert not metrics.matches(img, {"kind": "chat"})
    assert metrics.matches(vid, {"kind": "video"})


def test_backends_rows_carry_model_spec(cfgfile):
    common.STATE_FILE.write_text(json.dumps({
        "backends": [{"account": "kg1", "user": "u1", "model_key": "qwen3.8-27b",
                      "url": "https://example.invalid", "started": time.time() - 100, "ready_at": time.time() - 90,
                      "session_seconds": 39600, "ctx": 131072}],
        "pending": {}, "accounts": {}, "pinned": {}, "health": {"kg1": {"ok": True}}}), encoding="utf-8")
    c = TestClient(relay.app, client=LOCAL)
    r = c.get("/admin/api/backends")
    assert r.status_code == 200
    row = r.json()["backends"][0]
    assert row["spec"]["quant"] == "Q4_K_M"
    assert row["spec"]["vram_gb"] == 27
    assert row["spec"]["kind"] == "chat"
    assert row["spec"]["desc"]
