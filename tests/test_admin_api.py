"""Control-center API: every endpoint under /admin/api added for the panel."""
import json, os, sys, time, threading, types
import httpx, pytest, yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gateway import common, relay, admin_api, accounts as acct_mod, metrics, media, tunnel
from gateway import inspect_model as im
from tests.fakelog import make_records, write
from tests.test_relay_media import Backend

LOCAL = ("127.0.0.1", 50000)
H = {"x-kllm-panel": "1"}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = """# my config
default_model: qwen3.6-35b-abliterated
gateway_api_key: sk-kllm-supersecretvalue1234
hf_token: hf_abcdef
gist_token: ghp_zzzz
relay_host: 127.0.0.1
relay_port: 18199
desired_pool: 0
health_interval_seconds: 30
local_fallback:
  enabled: true
  url: http://127.0.0.1:8081
  api_key: sk-local
accounts:
  - {name: kg1, user: alice}   # first
  - {name: kg2, user: bob}
autoscale:
  enabled: true
  min_backends: 0
  max_backends: 3      # most kernels
  sessions_per_backend: 2
"""


def register(*pairs):
    now = time.time()
    with common.mutate_state() as s:
        s["backends"] = [{"account": a, "model_key": m, "url": getattr(b, "url", b), "api_key": f"key-{a}", "user": a + "u",
                          "started": now - 600, "ready_at": now - 300, "session_seconds": 39600, "ctx": 65536}
                         for a, m, b in pairs]
        s["pending"] = {}; s["pinned"] = {}; s["models_failing"] = {}; s["health"] = {}


@pytest.fixture
def api(monkeypatch, tmp_path):
    common.CONFIG_FILE.write_text(CONFIG, encoding="utf-8")
    common.MODELS_FILE.write_text(open(os.path.join(HERE, "models.yaml"), encoding="utf-8").read(), encoding="utf-8")
    for f in (metrics.ROLLUP_FILE, metrics.DECISIONS_FILE, metrics.REQ_LOG):
        if f.exists(): f.unlink()
    for p in metrics.log_files(): os.unlink(p)
    relay._cfg["mtime"] = 0; relay._reg.clear()
    monkeypatch.setattr(relay, "_client", httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=None)))
    relay.ROUTER.sessions.clear(); relay.ROUTER.inflight.clear(); relay.ROUTER.waiting.clear(); relay._bad.clear()
    register()
    spawned = []
    monkeypatch.setattr(relay, "_spawn", lambda args: spawned.append(args))
    monkeypatch.setattr(relay, "_refresh_litellm", lambda: None)
    monkeypatch.setattr(acct_mod.pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "nobin"))            # no real kgN wrappers are ever found
    app = FastAPI()
    router = admin_api.make_router(relay)
    app.include_router(router)
    c = TestClient(app, client=LOCAL)
    c.spawned = spawned; c.router = router; c.tmp = tmp_path
    yield c
    # leave the shared test root as other test modules expect it
    with common.mutate_state() as s:
        for k in ("backends", "pending", "pinned", "models_failing", "health", "quota", "accounts", "scheduler"):
            s.pop(k, None)
    common.MODELS_FILE.write_text(open(os.path.join(HERE, "models.yaml"), encoding="utf-8").read(), encoding="utf-8")
    common.CONFIG_FILE.unlink(missing_ok=True); relay._cfg["mtime"] = 0; relay._reg.clear()
    relay.ROUTER.sessions.clear()
    for f in (metrics.ROLLUP_FILE, metrics.DECISIONS_FILE, metrics.REQ_LOG):
        f.unlink(missing_ok=True)


def post(c, path, body=None, **kw):
    return c.post(path, json=body or {}, headers=H, **kw)


# ---- guard -----------------------------------------------------------------------------------------------
def test_state_changing_calls_need_the_panel_header_and_a_local_host(api):
    assert api.post("/admin/api/sessions/forget", json={"id": "x"}).status_code == 403
    assert api.post("/admin/api/sessions/forget", json={"id": "x"}, headers={"x-kllm-panel": "1"}).status_code == 404
    assert api.get("/admin/api/config", headers={"host": "evil.example.com"}).status_code == 403
    assert api.get("/admin/api/config", headers={"host": "localhost:8080"}).status_code == 200
    assert api.get("/admin/api/config", headers={"host": "127.0.0.1:8080"}).status_code == 200
    assert api.get("/admin/api/config", headers={"host": "[::1]:8080"}).status_code == 200


def test_whole_router_is_local_only_through_the_relay_app(api):
    remote = TestClient(relay.app, client=("203.0.113.9", 1234))
    for path in ("/admin/api/metrics", "/admin/api/config/key", "/admin/api/config/raw"):
        assert remote.get(path, headers={"authorization": "Bearer sk-kllm-supersecretvalue1234"}).status_code in (401, 403)
    local = TestClient(relay.app, client=LOCAL)
    assert local.get("/admin/api/metrics").status_code == 200


# ---- metrics / compute / alerts -------------------------------------------------------------------------------
def test_metrics_endpoint_on_fixture_log(api):
    now = time.time()
    recs = make_records(now)
    write(metrics.REQ_LOG, recs)
    r = api.get("/admin/api/metrics?range=24h&tz=330")
    assert r.status_code == 200
    v = r.json()
    assert v["range"] == "24h" and v["totals"]["n"] == sum(h["n"] for h in v["hourly"]) > 0
    assert v["tz"] == 330 and len(v["daily"]) == 7 and v["latency"]["ttft"]["p50"] is not None
    assert api.get("/admin/api/metrics?range=7d").json()["totals"]["n"] > v["totals"]["n"]
    assert api.get("/admin/api/metrics?range=nope").status_code == 400
    # log grows -> next call picks it up without re-reading the old bytes
    n7 = api.get("/admin/api/metrics?range=7d").json()["totals"]["n"]
    with open(metrics.REQ_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": int(time.time()), "backend": "kaggle:kg1", "model": "m", "status": 200}) + "\n")
    assert api.get("/admin/api/metrics?range=7d").json()["totals"]["n"] == n7 + 1


def test_metrics_with_no_log_is_empty_not_an_error(api):
    v = api.get("/admin/api/metrics?range=7d").json()
    assert v["totals"]["n"] == 0 and v["by_model"] == []


def test_compute_endpoint(api):
    ws = common._week_start(time.time())
    with common.mutate_state() as s:
        s["quota"] = {"kg1": {"week": ws, "used": 12 * 3600}}
        s["accounts"] = {"kg2": {"last_error": "queue", "cooldown_until": time.time() + 120}}
    write(metrics.REQ_LOG, make_records(time.time(), per_hour=2))
    v = api.get("/admin/api/compute").json()
    a1, a2 = v["accounts"]
    assert a1["used_h"] == 12 and a1["left_h"] == 18 and a2["last_error"] == "queue" and a2["cooldown_left_s"] > 60
    assert v["pool"]["left_h"] == 48 and v["log_bytes"] > 0


def alerts_for(cfg, state, demand=None, sup=True):
    return {a["id"]: a for a in admin_api.compute_alerts(cfg, state, demand, sup)}


def test_alert_rules():
    now = time.time(); ws = common._week_start(now)
    cfg = {"accounts": [{"name": "kg1"}, {"name": "kg2"}], "autoscale": {"enabled": False}}
    st = {"backends": [{"account": "kg1", "model_key": "m", "started": now - 600, "session_seconds": 39600}],
          "health": {"kg1": {"ok": False}}, "quota": {"kg1": {"week": ws, "used": 29 * 3600}},
          "models_failing": {"bad-model": {"until": now + 600, "reason": "OOM on load"}},
          "pending": {"kg2": {"model_key": "m", "started": now - 400, "phase": None}}}
    a = alerts_for(cfg, st, {"models": {"m": {"waiting": 2}}}, sup=False)
    assert a["quota-kg1"]["level"] == "warn" and "1.0 h" in a["quota-kg1"]["text"]
    assert a["model-bad-model"]["level"] == "crit" and "OOM" in a["model-bad-model"]["text"]
    assert a["unhealthy-kg1"]["level"] == "crit" and a["unhealthy-kg1"]["tab"] == "backends"
    assert a["queued-kg2"]["level"] == "warn"
    assert a["autoscale-off"]["level"] == "crit" and "2 request" in a["autoscale-off"]["text"]
    assert "supervisor" not in a                      # autoscale off and no always-on minimum: nothing to supervise
    assert alerts_for({**cfg, "autoscale": {"enabled": True}}, {"backends": [], "pending": {}}, None, sup=False)["supervisor"]["level"] == "crit"
    assert "pool-quota" in alerts_for({"accounts": [{"name": "kg1"}]}, {"quota": {"kg1": {"week": ws, "used": 29 * 3600}},
                                                                         "backends": [], "pending": {}})
    st2 = {"backends": [], "pending": {"kg1": {"model_key": "m", "started": now - 3000, "boot_at": now - 2900}}}
    assert "stuck-kg1" in alerts_for({"accounts": [{"name": "kg1"}]}, st2)
    assert "launch-failed-kg1" in alerts_for({"accounts": [{"name": "kg1"}]},
                                              {"pending": {"kg1": {"started": now, "phase": "crash"}}, "backends": []})
    assert "all-disabled" in alerts_for({"accounts": [{"name": "kg1", "disabled": True}]}, {"backends": [], "pending": {}})
    assert "no-accounts" in alerts_for({"accounts": []}, {"backends": [], "pending": {}})
    healthy = alerts_for({"accounts": [{"name": "kg1"}], "autoscale": {"enabled": True}},
                         {"backends": [], "pending": {}}, {"models": {}})
    assert healthy == {}


def test_alerts_endpoint(api):
    v = api.get("/admin/api/alerts").json()
    assert set(v) >= {"alerts", "supervisor", "backends", "pending", "waiting", "autoscale"}


# ---- sessions ---------------------------------------------------------------------------------------------
def mk_session(key, model="qwen3.6-35b-abliterated", backend="kaggle:kg1", last_req=True, **kw):
    now = time.time()
    s = {"backend": backend, "model": model, "first": now - 1200, "last": now - 20, "n": 5, "big": True}
    if last_req:
        s["last_req"] = ("v1/chat/completions", json.dumps({"model": model, "messages": [{"role": "user", "content": "hi" * 500}]}).encode(),
                         "application/json")
    s.update(kw)
    relay.ROUTER.sessions[key] = s
    return s


def test_sessions_list_and_forget(api):
    b1, b2 = Backend(), Backend()
    register(("kg1", "qwen3.6-35b-abliterated", b1), ("kg2", "qwen3.6-35b-abliterated", b2))
    mk_session("u:aaaaaaaaaaaaaaaaaaaa"); mk_session("h:bbbbbbbbbbbbbbbbbbbb", last_req=False, n=1, big=False)
    v = api.get("/admin/api/sessions").json()
    assert v["total"] == 2 and {s["id"] for s in v["sessions"]} == {"u:aaaaaaaaaaaa", "h:bbbbbbbbbbbb"}
    s = next(s for s in v["sessions"] if s["id"].startswith("u:"))
    assert s["model"] == "qwen3.6-35b-abliterated" and s["busy"] and s["counts"] and s["ctx_est"] > 200 and s["replayable"]
    assert set(v["targets"]["qwen3.6-35b-abliterated"]) == {"kaggle:kg1", "kaggle:kg2"}
    r = post(api, "/admin/api/sessions/forget", {"id": s["id"]})
    assert r.json()["ok"] and "u:aaaaaaaaaaaaaaaaaaaa" not in relay.ROUTER.sessions
    assert post(api, "/admin/api/sessions/forget", {"id": "nope"}).status_code == 404
    b1.srv.shutdown(); b2.srv.shutdown()


def test_session_move_warms_the_target_then_moves(api):
    b1, b2, b3 = Backend(), Backend(), Backend()
    register(("kg1", "qwen3.6-35b-abliterated", b1), ("kg2", "qwen3.6-35b-abliterated", b2), ("kg3", "qwen3-coder-30b-abliterated", b3))
    sess = mk_session("u:cccccccccccccccccccc")
    r = post(api, "/admin/api/sessions/move", {"id": "u:cccccccccccc", "backend": "kg2"})
    assert r.json()["ok"], r.text
    assert sess["backend"] == "kaggle:kg2" and sess["moved"] == 1 and not sess["warming"]
    body = json.loads(b2.bodies[0])
    assert body["max_tokens"] == 1 and body["stream"] is True and b2.auth[0] == "Bearer key-kg2"   # 1-token cache warm-up
    assert not b1.bodies and not b3.bodies
    for payload, code in (({"backend": "kg2"}, 400), ({"backend": "kg3"}, 400), ({"backend": "kg9"}, 400)):
        r = post(api, "/admin/api/sessions/move", {"id": "u:cccccccccccc", **payload})
        assert r.status_code == code and not r.json()["ok"]
    for b in (b1, b2, b3): b.srv.shutdown()


def test_session_move_cold_when_nothing_to_replay_and_failure_modes(api):
    b1, b2 = Backend(), Backend(status=500)
    register(("kg1", "qwen3.6-35b-abliterated", b1), ("kg2", "qwen3.6-35b-abliterated", b2))
    cold = mk_session("u:dddddddddddddddddddd", last_req=False)
    assert post(api, "/admin/api/sessions/move", {"id": "u:dddddddddddd", "backend": "kaggle:kg2"}).json()["ok"]
    assert cold["backend"] == "kaggle:kg2" and not b2.bodies
    warm = mk_session("u:eeeeeeeeeeeeeeeeeeee")
    r = post(api, "/admin/api/sessions/move", {"id": "u:eeeeeeeeeeee", "backend": "kg2"})     # b2 answers 500
    assert r.status_code == 502 and r.json()["can_cold"] and warm["backend"] == "kaggle:kg1" and not warm["warming"]
    assert post(api, "/admin/api/sessions/move", {"id": "u:eeeeeeeeeeee", "backend": "kg2", "cold": True}).json()["ok"]
    assert warm["backend"] == "kaggle:kg2"
    for b in (b1, b2): b.srv.shutdown()


# ---- backends ---------------------------------------------------------------------------------------------
def test_backend_rows_and_pending_timeline(api):
    b1 = Backend()
    register(("kg1", "qwen3.6-35b-abliterated", b1))
    now = time.time()
    with common.mutate_state() as s:
        s["pending"] = {"kg2": {"model_key": "qwen3-coder-30b-abliterated", "started": now - 500, "boot_at": now - 300,
                                "phase": "downloading", "origin": "auto"},
                        "kg3": {"model_key": "validate-3b", "started": now - 60, "origin": "manual"}}
        s["health"] = {"kg1": {"ok": True, "ts": int(now)}}
    v = api.get("/admin/api/backends").json()
    row = v["backends"][0]
    assert row["account"] == "kg1" and row["healthy"] and row["left_s"] > 30000 and not row["pinned"] and row["host"].startswith("127.0.0.1")
    by = {p["account"]: p for p in v["pending"]}
    st = {x["key"]: x["state"] for x in by["kg2"]["steps"]}
    assert st == {"push": "done", "queue": "done", "boot": "done", "build": "done", "download": "active", "serve": "todo", "ready": "todo"}
    assert by["kg2"]["queue_s"] == 200
    assert {x["key"]: x["state"] for x in by["kg3"]["steps"]}["queue"] == "active"
    with common.mutate_state() as s:
        s["pending"]["kg3"]["phase"] = "crash"
    assert next(p for p in api.get("/admin/api/backends").json()["pending"] if p["account"] == "kg3")["failed"]
    b1.srv.shutdown()


def test_launch_validation_and_spawn(api):
    assert post(api, "/admin/api/backends/launch", {"model": "nope"}).status_code == 400
    assert post(api, "/admin/api/backends/launch", {"model": "sdxl-turbo", "account": "kg9"}).status_code == 400
    r = post(api, "/admin/api/backends/launch", {"model": "qwen3-coder-30b-abliterated", "account": "kg2"})
    assert r.json()["ok"] and api.spawned[-1] == ["up", "--no-wait", "--model", "qwen3-coder-30b-abliterated", "--account", "kg2"]
    assert post(api, "/admin/api/backends/launch", {}).json()["ok"] and api.spawned[-1] == ["up", "--no-wait", "--model", "qwen3.6-35b-abliterated"]
    # busy account / disabled account (force overrides) / low quota / circuit breaker
    register(("kg1", "qwen3.6-35b-abliterated", "http://127.0.0.1:1"))
    assert post(api, "/admin/api/backends/launch", {"account": "kg1"}).status_code == 409
    acct_mod.set_disabled("kg2", True); relay._cfg["mtime"] = 0
    r = post(api, "/admin/api/backends/launch", {"account": "kg2"})
    assert r.status_code == 409 and r.json()["can_force"]
    n = len(api.spawned)
    assert post(api, "/admin/api/backends/launch", {"account": "kg2", "force": True}).json()["ok"] and len(api.spawned) == n + 1
    ws = common._week_start(time.time())
    with common.mutate_state() as s:
        s["models_failing"] = {"validate-3b": {"until": time.time() + 500, "reason": "boom"}}
        s["backends"] = []
        s["quota"] = {"kg1": {"week": ws, "used": 30 * 3600}}
    assert post(api, "/admin/api/backends/launch", {"model": "validate-3b"}).status_code == 409
    r = post(api, "/admin/api/backends/launch", {"account": "kg1"}); assert r.status_code == 409 and "quota" in r.json()["msg"]
    acct_mod.set_disabled("kg2", False)
    with common.mutate_state() as s:
        s["pending"] = {"kg1": {"model_key": "x", "started": 1}, "kg2": {"model_key": "x", "started": 1}}
    assert post(api, "/admin/api/backends/launch", {}).status_code == 409           # no free account


def test_stop_rotate_pin(api):
    b1 = Backend()
    register(("kg1", "qwen3.6-35b-abliterated", b1))
    assert post(api, "/admin/api/backends/stop", {"account": "kg2"}).status_code == 404
    assert post(api, "/admin/api/backends/stop", {"account": "kg1"}).json()["ok"] and api.spawned[-1] == ["down", "--account", "kg1"]
    assert post(api, "/admin/api/backends/rotate", {"account": "kg2"}).status_code == 404
    assert post(api, "/admin/api/backends/rotate", {"account": "kg1"}).json()["ok"]
    assert api.spawned[-1] == ["rotate", "--account", "kg1", "--model", "qwen3.6-35b-abliterated"]
    assert post(api, "/admin/api/backends/pin", {"account": "kg2"}).status_code == 404
    assert post(api, "/admin/api/backends/pin", {"account": "kg1"}).json()["ok"]
    assert "kg1" in common.read_state()["pinned"]
    assert api.get("/admin/api/backends").json()["backends"][0]["pinned"]
    post(api, "/admin/api/backends/pin", {"account": "kg1", "pinned": False})
    assert common.read_state()["pinned"] == {}
    b1.srv.shutdown()


def test_pinned_backend_never_looks_idle_to_the_planner():
    from gateway import supervisor, scheduler
    now = 1_000_000.0
    demand = {"models": {}, "backends": {"kaggle:kg1": {"idle_s": 99999, "inflight": 0, "sessions": 0},
                                          "kaggle:kg2": {"idle_s": 99999, "inflight": 0, "sessions": 0}}}
    be = [{"account": a, "model_key": "m", "started": now - 600, "session_seconds": 39600} for a in ("kg1", "kg2")]
    cfg = {"autoscale": {"enabled": True, "spare_backends": 0}}
    run = lambda d: scheduler.plan(now, cfg, {"m": {}}, "m", be, {}, ["kg1", "kg2"], {}, {"kg1": 20, "kg2": 20}, d)
    assert {a[1] for a in run(demand) if a[0] == "retire"} == {"kg1", "kg2"}
    supervisor.apply_pins(demand, {"pinned": {"kg1": 1}})
    assert {a[1] for a in run(demand) if a[0] == "retire"} == {"kg2"}


# ---- autoscale --------------------------------------------------------------------------------------------
def test_autoscale_config_roundtrip_block_style(api):
    v = api.get("/admin/api/autoscale/config").json()
    assert v["settings"]["max_backends"] == 3 and v["spec"]["max_backends"]["max"] == 10 and v["defaults"]["spare_backends"] == 1
    r = post(api, "/admin/api/autoscale/config", {"settings": {"max_backends": 5, "idle_stop_minutes": 20, "spare_backends": 0,
                                                              "min_quota_hours": 2.5, "enabled": False, "queue_timeout_seconds": 300,
                                                              "max_wait_seconds": 900, "sessions_per_backend": 4}})
    assert r.json()["ok"], r.text
    cfg = yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))
    assert cfg["autoscale"] == {"enabled": False, "min_backends": 0, "max_backends": 5, "sessions_per_backend": 4,
                                "idle_stop_minutes": 20, "spare_backends": 0, "min_quota_hours": 2.5,
                                "queue_timeout_seconds": 300, "max_wait_seconds": 900}
    assert "# most kernels" in common.CONFIG_FILE.read_text(encoding="utf-8") and cfg["accounts"][0]["name"] == "kg1"
    s = api.get("/admin/api/autoscale/config").json()["settings"]
    assert s["max_backends"] == 5 and s["enabled"] is False and s["min_quota_hours"] == 2.5


def test_autoscale_min_backends_syncs_always_on_and_flow_style(api):
    assert post(api, "/admin/api/autoscale/config", {"min_backends": 2}).json()["ok"]
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["desired_pool"] == 2
    common.CONFIG_FILE.write_text("default_model: m\naccounts: []\nautoscale: {enabled: true, max_wait_seconds: 15}\nz: 1\n", encoding="utf-8")
    relay._cfg["mtime"] = 0
    assert post(api, "/admin/api/autoscale/config", {"max_backends": 4}).json()["ok"]
    cfg = yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))
    assert cfg["autoscale"] == {"enabled": True, "max_wait_seconds": 15, "max_backends": 4} and cfg["z"] == 1


@pytest.mark.parametrize("body", [{"max_backends": 0}, {"max_backends": 99}, {"idle_stop_minutes": "abc"}, {"bogus": 1},
                                  {"spare_backends": 1.5}, {"min_backends": 5, "max_backends": 2}, {}, {"enabled": "maybe"}])
def test_autoscale_rejects_bad_values_and_leaves_config(api, body):
    before = common.CONFIG_FILE.read_text(encoding="utf-8")
    r = post(api, "/admin/api/autoscale/config", body)
    assert r.status_code == 400 and not r.json()["ok"]
    assert common.CONFIG_FILE.read_text(encoding="utf-8") == before


def test_autoscale_decision_history_endpoint(api):
    with common.mutate_state() as s:
        s["scheduler"] = {"tick": time.time(), "log": [{"ts": 100, "action": "launch", "target": "kg1", "model": "m", "why": "demand"}]}
    api.router.housekeeping_once()
    d = api.get("/admin/api/autoscale/config").json()
    assert d["decisions"][0]["why"] == "demand" and d["tick"]


# ---- accounts ---------------------------------------------------------------------------------------------
def fake_kaggle(user="carol"):
    def run(cmd, **kw):
        if cmd[1:3] == ["config", "view"]:
            return types.SimpleNamespace(stdout=f"username: {user}\n", stderr="", returncode=0)
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)
    return run


def test_accounts_list_add_disable_remove(api, monkeypatch):
    v = api.get("/admin/api/accounts").json()
    assert [a["name"] for a in v["accounts"]] == ["kg1", "kg2"] and v["next_name"] == "kg3"
    assert v["accounts"][0]["credentials"]["ok"] is False and "no credentials" in v["accounts"][0]["credentials"]["error"]
    token = "KGAT_" + "0f" * 16
    monkeypatch.setattr(acct_mod.subprocess, "run", fake_kaggle())
    r = post(api, "/admin/api/accounts/add", {"token": token})
    assert r.json()["ok"] and r.json()["name"] == "kg3" and token not in r.text
    assert [a["name"] for a in yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["accounts"]] == ["kg1", "kg2", "kg3"]
    assert post(api, "/admin/api/accounts/add", {"token": "nope"}).status_code == 400
    # disable -> config flag -> planner ignores it
    assert post(api, "/admin/api/accounts/disable", {"name": "kg3", "disabled": True}).json()["ok"]
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["accounts"][2]["disabled"] is True
    assert next(a for a in api.get("/admin/api/accounts").json()["accounts"] if a["name"] == "kg3")["disabled"]
    assert post(api, "/admin/api/accounts/disable", {"name": "zz"}).status_code == 400
    assert post(api, "/admin/api/accounts/disable", {"name": "kg3", "disabled": False}).json()["ok"]
    # remove: config only by default; deleting credential files needs an explicit second confirmation
    home, tok, wrapper = acct_mod.credential_paths("kg3")
    assert tok.exists()
    r = post(api, "/admin/api/accounts/remove", {"name": "kg3", "delete_credentials": True})
    assert r.status_code == 400 and tok.exists() and len(yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["accounts"]) == 3
    r = post(api, "/admin/api/accounts/remove", {"name": "kg3"})
    assert r.json()["ok"] and tok.exists() and wrapper.exists()
    monkeypatch.setattr(acct_mod.subprocess, "run", fake_kaggle("dave"))
    post(api, "/admin/api/accounts/add", {"token": token, "name": "kg3"})
    r = post(api, "/admin/api/accounts/remove", {"name": "kg3", "delete_credentials": True, "confirm_credentials": "kg3"})
    assert r.json()["ok"] and not home.exists() and not wrapper.exists()
    assert post(api, "/admin/api/accounts/remove", {"name": "kg3"}).status_code == 400


def test_accounts_clear_error_and_sync_quota(api, monkeypatch):
    with common.mutate_state() as s:
        s["accounts"] = {"kg1": {"last_error": "push failed", "fails": 2, "cooldown_until": time.time() + 300},
                         "kg2": {"last_error": "sat in queue", "fails": 1, "cooldown_until": 0}}
    v = api.get("/admin/api/accounts").json()
    kg1_row = next(a for a in v["accounts"] if a["name"] == "kg1")
    kg2_row = next(a for a in v["accounts"] if a["name"] == "kg2")
    assert kg1_row["error_active"] is True
    assert kg1_row["last_error"] == "push failed"

    # clear single error
    r = post(api, "/admin/api/accounts/clear-error", {"account": "kg1"})
    assert r.json()["ok"]
    st = common.read_state().get("accounts", {}).get("kg1", {})
    assert st.get("last_error") is None and st.get("fails") == 0 and st.get("cooldown_until") == 0

    # clear all errors
    r = post(api, "/admin/api/accounts/clear-error", {"all": True})
    assert r.json()["ok"]
    st2 = common.read_state().get("accounts", {}).get("kg2", {})
    assert st2.get("last_error") is None and st2.get("fails") == 0

    # sync quota mocked
    def fake_quota(acct, cfg=None):
        return {"ok": True, "account": acct.get("name"), "used_s": 36000.0, "used_h": 10.0, "refresh_time": "2026-10-17T00:00:00.000Z"}
    monkeypatch.setattr(admin_api.kaggle, "fetch_account_quota", fake_quota)
    r = post(api, "/admin/api/accounts/sync-quota")
    assert r.json()["ok"] and "kg1" in r.json()["synced"]
    s = common.read_state()
    assert s["quota"]["kg1"]["used"] == 36000.0
    assert s["quota"]["kg1"]["refresh_time"] == "2026-10-17T00:00:00.000Z"



# ---- models -----------------------------------------------------------------------------------------------
def test_models_library_flags(api):
    register(("kg1", "qwen3.6-35b-abliterated", "http://127.0.0.1:1"))
    v = api.get("/admin/api/models").json()
    by = {m["key"]: m for m in v["models"]}
    q = by["qwen3.6-35b-abliterated"]
    assert q["default"] and q["running"] == ["kg1"] and q["verified"] and q["abliterated"] and q["kind"] == "chat"
    assert not by["gpt-oss-20b-abliterated"]["verified"] and by["sdxl-turbo"]["kind"] == "image"
    assert by["x-qwen36-512k"]["experiment"] and by["sdxl-turbo"]["problems"] == []
    assert not by["sdxl-turbo"]["abliterated"]                       # not a chat model name: just a flag
    assert q["args"][0] == "--temp" and q["ctx"] == 262144


def test_model_update_validation_and_roundtrip(api):
    r = post(api, "/admin/api/models/update", {"key": "validate-3b", "changes": {"ctx": 16384, "desc": "new text",
                                                                               "args": "--temp 0.2 --top-p 0.9", "vram_gb": 4}})
    assert r.json()["ok"], r.text
    m = yaml.safe_load(common.MODELS_FILE.read_text(encoding="utf-8"))["models"]["validate-3b"]
    assert (m["ctx"], m["desc"], m["args"], m["vram_gb"]) == (16384, "new text", ["--temp", "0.2", "--top-p", "0.9"], 4)
    assert os.path.exists(str(common.MODELS_FILE) + ".bak")
    assert len(yaml.safe_load(common.MODELS_FILE.read_text(encoding="utf-8"))["models"]) == len(yaml.safe_load(open(os.path.join(HERE, "models.yaml"), encoding="utf-8"))["models"])
    assert post(api, "/admin/api/models/update", {"key": "validate-3b", "changes": {"ctx": 16384}}).json()["msg"] == "nothing changed"
    before = common.MODELS_FILE.read_text(encoding="utf-8")
    for changes in ({"ctx": 5}, {"ctx": "x"}, {"hf": "nope"}, {"runtime": "diffusers"}, {"args": ["a\nb"]}, {"vram_gb": 999},
                    {"template": "../../etc/passwd"}, {"desc": "a\nb"}, {}, {"tags": ["bad tag!"]}, {"quant": None, "file": None}):
        r = post(api, "/admin/api/models/update", {"key": "validate-3b", "changes": changes})
        assert r.status_code == 400, changes
    assert common.MODELS_FILE.read_text(encoding="utf-8") == before
    assert post(api, "/admin/api/models/update", {"key": "nope", "changes": {"ctx": 4096}}).status_code == 404


def test_model_update_notes_running_backends_and_edits_diffusers_opts(api):
    register(("kg1", "validate-3b", "http://127.0.0.1:1"))
    r = post(api, "/admin/api/models/update", {"key": "validate-3b", "changes": {"ctx": 4096}})
    assert "relaunched" in r.json()["msg"]
    r = post(api, "/admin/api/models/update", {"key": "sdxl-turbo", "changes": {"opts": {"size": "768x768", "steps": 2}}})
    assert r.json()["ok"]
    assert yaml.safe_load(common.MODELS_FILE.read_text(encoding="utf-8"))["models"]["sdxl-turbo"]["opts"] == {"size": "768x768", "steps": 2}


def test_model_delete_protections(api):
    before = len(yaml.safe_load(common.MODELS_FILE.read_text(encoding="utf-8"))["models"])
    section_comments = [l for l in common.MODELS_FILE.read_text(encoding="utf-8").splitlines() if l.strip().startswith("# ----")]
    assert post(api, "/admin/api/models/delete", {"key": "qwen3.6-35b-abliterated"}).status_code == 400      # default
    assert post(api, "/admin/api/models/delete", {"key": "nope"}).status_code == 400
    register(("kg1", "validate-3b", "http://127.0.0.1:1"))
    assert "running" in post(api, "/admin/api/models/delete", {"key": "validate-3b"}).json()["msg"]
    register()
    common.set_config_value("default_image_model", "sdxl-turbo"); relay._cfg["mtime"] = 0
    assert "default image" in post(api, "/admin/api/models/delete", {"key": "sdxl-turbo"}).json()["msg"]
    assert post(api, "/admin/api/models/delete", {"key": "validate-3b"}).json()["ok"]
    m = yaml.safe_load(common.MODELS_FILE.read_text(encoding="utf-8"))["models"]
    assert "validate-3b" not in m and len(m) == before - 1
    after = common.MODELS_FILE.read_text(encoding="utf-8")
    assert section_comments and all(c in after for c in section_comments)    # comments kept


def test_model_default(api):
    assert post(api, "/admin/api/models/default", {"key": "sdxl-turbo"}).status_code == 400
    assert post(api, "/admin/api/models/default", {"key": "nope"}).status_code == 400
    assert post(api, "/admin/api/models/default", {"key": "qwen3-coder-30b-abliterated"}).json()["ok"]
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["default_model"] == "qwen3-coder-30b-abliterated"


def test_abliterated_suggestions_come_from_hf_on_demand(api, monkeypatch):
    seen = []
    def fake_get(url, token=None, headers=None, timeout=20):
        seen.append(url)
        return json.dumps([{"id": "mradermacher/Qwen3-32B-abliterated-GGUF", "downloads": 900, "likes": 4},
                           {"id": "someone/Qwen3-32B-abliterated-GGUF", "downloads": 5},
                           {"id": "org/Qwen3-32B-Instruct-GGUF", "downloads": 1}]).encode(), {}
    monkeypatch.setattr(im, "_get", fake_get)
    assert seen == []
    r = post(api, "/admin/api/models/suggest", {"repo": "bartowski/Qwen3-32B-Instruct-GGUF"})
    j = r.json()
    assert j["ok"] and j["base"] == "Qwen3-32B" and len(seen) == 1
    assert "search=Qwen3-32B+abliterated" in seen[0] and "filter=gguf" in seen[0] and "limit=8" in seen[0]
    assert [x["repo"] for x in j["results"]][0] == "mradermacher/Qwen3-32B-abliterated-GGUF"
    assert j["results"][0]["abliterated"] and not j["results"][-1]["abliterated"]
    assert "huggingface.co/models?search=" in j["search_url"]
    j = post(api, "/admin/api/models/search", {"q": "qwen3 coder"}).json()
    assert j["ok"] and "search=qwen3+coder+abliterated" in seen[-1]
    assert post(api, "/admin/api/models/search", {"q": "x"}).status_code == 400
    def boom(*a, **k): raise im.InspectError("cannot reach huggingface.co")
    monkeypatch.setattr(im, "_get", boom)
    r = post(api, "/admin/api/models/suggest", {"repo": "a/b"})
    assert r.status_code == 502 and "huggingface" in r.json()["msg"]


def test_base_name_extraction():
    f = admin_api.base_name
    assert f("mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-i1-GGUF") == "Qwen3.6-35B-A3B"
    assert f("bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF") == "Qwen2.5-Coder-32B"
    assert f("bartowski/huihui-ai_QwQ-32B-abliterated-GGUF") == "QwQ-32B"
    assert f("org/Plain-Model") == "Plain-Model"
    assert admin_api.guess_abliterated("x/Foo-abliterated-GGUF") and not admin_api.guess_abliterated("x/Foo-GGUF")


def test_inspect_response_flags_non_abliterated_chat_models(api, monkeypatch):
    from gateway import admin_models
    app = FastAPI(); app.include_router(admin_models.make_router(lambda: None))
    c = TestClient(app, client=LOCAL)
    res = {"runtime": "llama.cpp", "kind": "chat", "repo": "org/Plain-GGUF", "key": "plain", "entry": {}, "files": [],
           "chosen": "Plain-Q4_K_M.gguf", "weights_gb": 5, "vram_gb": 8, "ctx": 8192, "warnings": [], "fits": True}
    monkeypatch.setattr(im, "inspect_source", lambda **k: dict(res))
    assert c.post("/admin/api/inspect", json={"source": "org/Plain-GGUF"}).json()["abliterated"] is False
    res.update(repo="org/Plain-abliterated-GGUF")
    assert c.post("/admin/api/inspect", json={"source": "x"}).json()["abliterated"] is True
    res.update(repo="org/img", kind="image")
    assert c.post("/admin/api/inspect", json={"source": "x"}).json()["abliterated"] is True


# ---- config -----------------------------------------------------------------------------------------------
def test_config_view_key_rotate_and_raw_masking(api):
    v = api.get("/admin/api/config").json()
    assert v["gateway_key_set"] and v["gateway_key_preview"] == "sk-kllm-...234" and "supersecret" not in json.dumps(v)
    assert api.get("/admin/api/config/key").json()["key"] == "sk-kllm-supersecretvalue1234"
    raw = api.get("/admin/api/config/raw").json()
    assert raw["masked"] and "supersecret" not in raw["text"] and "hf_abcdef" not in raw["text"] and "ghp_zzzz" not in raw["text"]
    assert "gateway_api_key: ******** (set)" in raw["text"] and "sk-local" not in raw["text"]
    assert "default_model: qwen3.6-35b-abliterated" in raw["text"] and "# first" in raw["text"] and "relay_port: 18199" in raw["text"]
    assert "# most kernels" in raw["text"]
    models = api.get("/admin/api/config/raw?file=models").json()
    assert "validate-3b" in models["text"] and not models["masked"]
    r = post(api, "/admin/api/config/rotate-key").json()
    assert r["ok"] and r["key"].startswith("sk-kllm-") and len(r["key"]) > 30
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["gateway_api_key"] == r["key"]
    assert api.get("/admin/api/config/key").json()["key"] == r["key"]


def test_mask_secrets_edge_cases():
    m = admin_api.mask_secrets
    assert m("hf_token: \ngist_token: null\n") == "hf_token: \ngist_token: null\n"        # empty stays empty
    assert m("  api_key: sk-local   # note\n") == "  api_key: ******** (set)   # note\n"
    assert m("kaggle_json: ~/.kaggle/kaggle.json\n") == "kaggle_json: ~/.kaggle/kaggle.json\n"
    assert "KGAT" not in m("my_secret: KGAT_abc\n")


def test_config_set_whitelist_and_validation(api, monkeypatch):
    started = []
    monkeypatch.setattr(tunnel.CTL, "start", lambda log=print, cfg_fn=None: started.append("start") or True)
    monkeypatch.setattr(tunnel.CTL, "stop", lambda: started.append("stop") or True)
    assert post(api, "/admin/api/config/set", {"key": "remote_enabled", "value": True}).json()["ok"]
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["remote_enabled"] is True and started == ["start"]
    assert post(api, "/admin/api/config/set", {"key": "remote_enabled", "value": False}).json()["ok"] and started[-1] == "stop"
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["remote_enabled"] is False
    assert post(api, "/admin/api/config/set", {"key": "health_interval_seconds", "value": 45}).json()["config"]["health_interval_seconds"] == 45
    assert post(api, "/admin/api/config/set", {"key": "default_image_model", "value": "sdxl-turbo"}).json()["ok"]
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["default_image_model"] == "sdxl-turbo"
    assert post(api, "/admin/api/config/set", {"key": "default_video_model", "value": "sdxl-turbo"}).status_code == 400     # not a video model
    assert post(api, "/admin/api/config/set", {"key": "default_video_model", "value": "ltx-video"}).json()["ok"]
    assert post(api, "/admin/api/config/set", {"key": "default_image_model", "value": ""}).json()["ok"]
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["default_image_model"] is None
    assert post(api, "/admin/api/config/set", {"key": "default_model", "value": "qwen3-coder-30b-abliterated"}).json()["ok"]
    assert post(api, "/admin/api/config/set", {"key": "default_model", "value": "ltx-video"}).status_code == 400
    assert post(api, "/admin/api/config/set", {"key": "local_fallback_enabled", "value": False}).json()["ok"]
    assert yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))["local_fallback"]["enabled"] is False
    for key, val in (("health_interval_seconds", 5), ("health_interval_seconds", "x"), ("gateway_api_key", "x"),
                     ("accounts", []), ("relay_port", 1), ("desired_pool", 99), ("remote_enabled", "maybe")):
        assert post(api, "/admin/api/config/set", {"key": key, "value": val}).status_code == 400, key
    cfg = yaml.safe_load(common.CONFIG_FILE.read_text(encoding="utf-8"))
    assert cfg["relay_port"] == 18199 and len(cfg["accounts"]) == 2 and cfg["gateway_api_key"].startswith("sk-kllm-super")


# ---- diagnostics ------------------------------------------------------------------------------------------
DOCTOR_OUT = """=== kllm doctor ===
  OK   account kg1                  json: /home/u/kaggle.json
  FAIL rendezvous gist              gist_id not set
       -> kllm init --github-token <token>
  OK   relay                        http://127.0.0.1:8080
  FAIL generation speed             12 t/s
1 problem(s)
"""


def test_parse_doctor_output():
    checks, summary = admin_api.parse_doctor(DOCTOR_OUT.splitlines(True))
    assert [(c["ok"], c["name"]) for c in checks] == [(True, "account kg1"), (False, "rendezvous gist"), (True, "relay"), (False, "generation speed")]
    assert checks[1]["fix"].startswith("kllm init") and checks[0]["detail"] == "json: /home/u/kaggle.json"
    assert checks[3]["detail"] == "12 t/s" and summary == "1 problem(s)"


def test_doctor_job_runs_once_and_reports(api):
    script = "import time; print('=== kllm doctor ==='); print('  OK   relay                        up'); time.sleep(0.4); print('  FAIL pool   none'); print('       -> kllm up'); print('1 problem(s)')"
    assert admin_api.start_doctor([sys.executable, "-c", script]) is True
    assert admin_api.start_doctor([sys.executable, "-c", script]) is False                # already running
    assert api.get("/admin/api/doctor").json()["running"] is True
    for _ in range(50):
        v = api.get("/admin/api/doctor").json()
        if not v["running"]: break
        time.sleep(0.1)
    assert not v["running"] and v["rc"] == 0 and v["failed"] == 1 and v["summary"] == "1 problem(s)"
    assert [c["name"] for c in v["checks"]] == ["relay", "pool"] and v["checks"][1]["fix"] == "kllm up"
    r = post(api, "/admin/api/doctor/run")           # real command: launches `kllm doctor`, which fails fast without a relay
    assert r.json()["ok"]
    for _ in range(100):
        if not api.get("/admin/api/doctor").json()["running"]: break
        time.sleep(0.1)


def test_service_log_journal_and_fallback(api, monkeypatch):
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="2026-10-06T10:00:00+0530 host kllm[1]: hello\n2026-10-06T10:00:01+0530 host kllm[1]: world\n", stderr="")
    monkeypatch.setattr(admin_api.subprocess, "run", fake_run)
    v = api.get("/admin/api/logs/service?n=50").json()
    assert v["lines"][-1].endswith("world") and "journalctl" in v["source"]
    assert calls[0][:4] == ["journalctl", "--user", "-u", "kllm"] and "50" in calls[0] and "--no-pager" in calls[0]
    api.get("/admin/api/logs/service?unit=sshd;rm")
    assert calls[-1][3] == "kllm"                                    # only whitelisted units
    api.get("/admin/api/logs/service?n=999999")
    assert "1000" in calls[-1]
    monkeypatch.setattr(admin_api.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("journalctl")))
    (common.LOG_DIR / "serve.log").write_text("line1\nline2\n", encoding="utf-8")
    v = api.get("/admin/api/logs/service").json()
    assert v["lines"] == ["line1", "line2"] and v["source"].endswith("serve.log")
    (common.LOG_DIR / "serve.log").unlink()
    v = api.get("/admin/api/logs/service").json()
    assert v["lines"] == [] and v["note"]


def test_request_explorer_and_csv(api):
    now = time.time()
    recs = make_records(now, days=2)
    write(metrics.REQ_LOG, recs)
    v = api.get("/admin/api/logs/requests?status=error&limit=500").json()
    assert len(v["records"]) == sum(1 for r in recs if r["status"] != 200) > 0 and v["scanned"] >= len(v["records"])
    v = api.get("/admin/api/logs/requests?model=coder&backend=kg2&since=3600&limit=50").json()
    assert all("coder" in r["model"] and r["backend"] == "kaggle:kg2" and r["ts"] >= now - 3700 for r in v["records"])
    assert len(api.get("/admin/api/logs/requests?limit=7").json()["records"]) == 7
    assert api.get("/admin/api/logs/requests?loop=1").json()["records"] and api.get("/admin/api/logs/requests?min_total=abc").status_code == 400
    r = api.get("/admin/api/logs/requests.csv?status=error")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("time,ts,backend,model") and len(lines) - 1 == sum(1 for x in recs if x["status"] != 200)


def test_housekeeping_rotates_syncs_decisions_and_purges_pins(api, monkeypatch):
    monkeypatch.setattr(metrics, "rotate_if_needed", lambda *a, **k: calls.append("rotate"))
    calls = []
    with common.mutate_state() as s:
        s["pinned"] = {"kg1": 1, "kg2": 2}
        s["backends"] = [{"account": "kg1", "model_key": "m", "url": "http://x"}]
        s["scheduler"] = {"log": [{"ts": 5, "action": "retire", "target": "kg2", "model": None, "why": "idle"}]}
    api.router.housekeeping_once()
    assert calls == ["rotate"] and common.read_state()["pinned"] == {"kg1": 1}
    assert [e["target"] for e in metrics.read_decisions()] == ["kg2"]
