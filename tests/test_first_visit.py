"""A fresh install opened in a browser: http://127.0.0.1:8080 shows the panel, and "nothing running" errors say
where to go next instead of pointing at a local server the user never set up."""
import pathlib

import yaml
from fastapi.testclient import TestClient

from gateway import relay

LOCAL = ("127.0.0.1", 50000)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_browser_at_root_lands_on_the_panel():
    c = TestClient(relay.app, client=LOCAL)
    r = c.get("/", headers={"accept": "text/html,application/xhtml+xml"}, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/admin"


def test_api_client_at_root_gets_a_pointer_not_an_error():
    r = TestClient(relay.app, client=LOCAL).get("/", headers={"accept": "application/json"})
    assert r.status_code == 200 and r.json()["panel"] == "/admin" and r.json()["openai_api"] == "/v1"


def test_local_fallback_is_off_in_a_fresh_config():
    cfg = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    assert cfg["local_fallback"]["enabled"] is False


def test_old_template_config_gets_local_fallback_turned_off(tmp_path):
    from gateway import setup
    old = ("default_model: x\n"
           "# Last-resort fallback: your existing local llama-server (abliterated models/serve.sh).\n"
           "# If every Kaggle backend is down, the relay falls back here so you are never fully offline.\n"
           "local_fallback:\n  enabled: true\n  url: http://127.0.0.1:8081\n  api_key: sk-local\n")
    p = tmp_path / "config.yaml"
    p.write_text(old, encoding="utf-8")
    assert setup.migrate_local_fallback(p) is True
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert cfg["local_fallback"] == {"enabled": False, "url": "http://127.0.0.1:8081", "api_key": "sk-local"}
    assert setup.migrate_local_fallback(p) is False                    # once only


def test_a_deliberately_enabled_fallback_is_left_alone(tmp_path):
    from gateway import setup
    mine = "local_fallback:\n  enabled: true   # my own llama-server\n  url: http://127.0.0.1:8081\n"
    p = tmp_path / "config.yaml"
    p.write_text(mine, encoding="utf-8")
    assert setup.migrate_local_fallback(p) is False and p.read_text(encoding="utf-8") == mine


def test_no_account_fails_fast_instead_of_waiting(monkeypatch):
    monkeypatch.setattr(relay, "cfg", lambda: {"accounts": [], "local_fallback": {"enabled": False}, "relay_port": 8080,
                                                "autoscale": {"enabled": True}})
    monkeypatch.setattr(relay, "_admit", lambda *a, **k: ([], None))
    import time
    t0 = time.time()
    r = TestClient(relay.app, client=LOCAL).post("/v1/chat/completions", json={"model": "kllm", "messages": [{"role": "user", "content": "hi"}]})
    assert time.time() - t0 < 10
    assert r.status_code == 503 and r.json()["error"]["type"] == "no_account" and "/admin" in r.json()["error"]["message"]
