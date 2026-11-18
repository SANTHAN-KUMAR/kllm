"""gateway_api_key on the relay: local callers need no key, everyone else does."""
import pytest
from fastapi.testclient import TestClient
from gateway import auth, common, relay

KEY = "sk-kllm-s3cret"
LOCAL = ("127.0.0.1", 50000)
LAN = ("192.168.1.20", 50000)


def hdrs(**kw):
    return [(k.replace("_", "-").encode(), v.encode()) for k, v in kw.items()]


def test_check_local_needs_no_key():
    cfg = {"gateway_api_key": KEY}
    for host in ("127.0.0.1", "::1"):
        assert auth.check(cfg, host, [], "/v1/chat/completions") is None
        assert auth.check(cfg, host, [], "/admin/api/state") is None
        assert auth.check({}, host, [], "/v1/models") is None      # even with no key configured


def test_check_remote_requires_key_on_model_api_only():
    cfg = {"gateway_api_key": KEY}
    assert auth.check(cfg, "10.0.0.2", hdrs(authorization=f"Bearer {KEY}"), "/v1/models") is None
    assert auth.check(cfg, "10.0.0.2", hdrs(x_api_key=KEY), "/v1/messages") is None   # Anthropic SDK style
    assert auth.check(cfg, "10.0.0.2", [], "/v1/models")[0] == 401
    assert auth.check(cfg, "10.0.0.2", hdrs(authorization="Bearer nope"), "/v1/models")[0] == 401
    # panel and relay internals are never exposed remotely, even with the right key
    for p in ("/admin", "/admin/api/state", "/__relay/status", "/chat", "/"):
        assert auth.check(cfg, "10.0.0.2", hdrs(authorization=f"Bearer {KEY}"), p)[0] == 403


def test_check_refuses_remote_when_no_real_key():
    for k in (None, "", "sk-change-me"):
        d = auth.check({"gateway_api_key": k}, "10.0.0.2", hdrs(authorization=f"Bearer {k}"), "/v1/models")
        assert d and d[0] == 403


def test_tunnel_connections_from_localhost_are_not_local():
    """cloudflared connects from 127.0.0.1 but adds proxy headers: the key is still enforced."""
    cfg = {"gateway_api_key": KEY}
    for h in ("x-forwarded-for", "cf-connecting-ip", "x-real-ip", "forwarded"):
        d = auth.check(cfg, "127.0.0.1", [(h.encode(), b"203.0.113.9")], "/v1/models")
        assert d and d[0] == 401, h
        assert auth.check(cfg, "127.0.0.1", [(h.encode(), b"1.2.3.4"), (b"authorization", f"Bearer {KEY}".encode())],
                          "/v1/models") is None


@pytest.fixture
def relay_cfg():
    def write(**extra):
        lines = ["relay_host: 127.0.0.1", "relay_port: 18199", "default_model: qwen3.6-35b-abliterated",
                 "local_fallback: {enabled: false}", "accounts: []"] + [f"{k}: {v}" for k, v in extra.items()]
        common.CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        relay._cfg["mtime"] = 0
    yield write
    common.CONFIG_FILE.unlink(missing_ok=True); relay._cfg["mtime"] = 0


def test_relay_enforces_key_for_non_local_clients(relay_cfg):
    relay_cfg(gateway_api_key=KEY)
    local, lan = TestClient(relay.app, client=LOCAL), TestClient(relay.app, client=LAN)
    r = local.get("/v1/models")
    assert r.status_code == 200 and {m["id"] for m in r.json()["data"]} >= {"kllm", "qwen3.6-35b-abliterated"}
    assert lan.get("/v1/models").status_code == 401
    ok = lan.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    assert ok.status_code == 200 and ok.json()["object"] == "list"
    assert lan.get("/v1/models", headers={"x-api-key": KEY}).status_code == 200
    assert lan.get("/admin/api/state", headers={"Authorization": f"Bearer {KEY}"}).status_code == 403
    assert local.get("/__relay/health").json()["ok"] is True
    assert lan.get("/__relay/health").status_code == 403
    assert local.get("/v1/models", headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 401


def test_relay_without_key_configured_keeps_local_open_and_refuses_remote(relay_cfg):
    relay_cfg()
    assert TestClient(relay.app, client=LOCAL).get("/v1/models").status_code == 200
    r = TestClient(relay.app, client=LAN).get("/v1/models", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 403 and "gateway_api_key" in r.json()["error"]["message"]


def test_key_change_is_hot_reloaded(relay_cfg):
    relay_cfg(gateway_api_key=KEY)
    lan = TestClient(relay.app, client=LAN)
    assert lan.get("/v1/models", headers={"Authorization": "Bearer sk-other-key"}).status_code == 401
    import os, time
    relay_cfg(gateway_api_key="sk-other-key")
    os.utime(common.CONFIG_FILE, (time.time() + 5, time.time() + 5))
    assert lan.get("/v1/models", headers={"Authorization": "Bearer sk-other-key"}).status_code == 200
