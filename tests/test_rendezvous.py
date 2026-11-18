"""Gist client: talks to the GitHub REST API with urllib; `gh` only as a fallback."""
import json, subprocess
from gateway import rendezvous

CFG = {"gist_id": "g1", "gist_token": "ghp_realtoken"}


def test_token_precedence_and_placeholders(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False); monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert rendezvous.token({"github_token": "a", "gist_token": "b"}) == "a"
    assert rendezvous.token({"gist_token": "b"}) == "b"
    assert rendezvous.token({"gist_token": "ghp_YOUR_GIST_SCOPED_TOKEN"}) is None
    monkeypatch.setenv("GH_TOKEN", "envtok")
    assert rendezvous.token({"gist_token": "ghp_YOUR_GIST_SCOPED_TOKEN"}) == "envtok"
    assert rendezvous.token({}) == "envtok"


def test_latest_events_reads_status_files_newest_first(fake_github):
    fake_github.set_status("kg1", event="alive", ts=100, url="https://a.trycloudflare.com")
    fake_github.set_status("kg2", event="ready", ts=200, url="https://b.trycloudflare.com")
    fake_github.gists["g1"]["files"]["c-kg1.json"] = {"content": "{}"}      # control files are ignored
    fake_github.gists["g1"]["files"]["s-bad.json"] = {"content": "not json"}
    evs = rendezvous.latest_events(CFG)
    assert [e["account"] for e in evs] == ["kg2", "kg1"]
    assert set(rendezvous.latest_by_account(CFG)) == {"kg1", "kg2"}
    method, path, auth, _ = fake_github.requests[0]
    assert (method, path, auth) == ("GET", "/gists/g1", "Bearer ghp_realtoken")


def test_conditional_get_uses_etag(fake_github):
    fake_github.set_status("kg1", event="alive", ts=1)
    assert rendezvous.latest_events(CFG)[0]["account"] == "kg1"
    again = rendezvous.latest_events(CFG)          # server answers 304; cached copy is returned
    assert again and again[0]["account"] == "kg1"
    fake_github.set_status("kg1", event="alive", ts=2)      # changed -> fresh body
    assert rendezvous.latest_events(CFG)[0]["ts"] == 2


def test_publish_control_patches_the_gist(fake_github):
    assert rendezvous.publish_control(CFG, "kg3", "stop") is True
    method, path, auth, body = fake_github.requests[-1]
    assert (method, path) == ("PATCH", "/gists/g1")
    content = json.loads(body["files"]["c-kg3.json"]["content"])
    assert content["cmd"] == "stop" and content["account"] == "kg3"


def test_errors_never_raise(fake_github):
    fake_github.fail = 500
    assert rendezvous.latest_events(CFG) == []
    assert rendezvous.publish_control(CFG, "kg1") is False
    fake_github.fail = None
    assert rendezvous.latest_events({"gist_id": "missing", "gist_token": "t"}) == []
    assert rendezvous.latest_events({"gist_id": "g1", "gist_token": "t", "github_api": "http://127.0.0.1:9"}) == []


def test_create_gist_is_secret_and_returns_owner(fake_github):
    gid, login = rendezvous.create_gist({}, "tok")
    assert login == "octo" and fake_github.gists[gid]["public"] is False
    assert rendezvous.whoami({}, "tok") == "octo"
    fake_github.fail = 403
    try:
        rendezvous.create_gist({}, "tok"); assert False
    except RuntimeError as e:
        assert "gist" in str(e)


def test_falls_back_to_gh_cli_without_token(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False); monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = json.dumps({"files": {"s-kg1.json": {"content": json.dumps({"account": "kg1", "ts": 5})}}})
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(rendezvous.subprocess, "run", fake_run)
    cfg = {"gist_id": "g1", "gist_token": "ghp_YOUR_GIST_SCOPED_TOKEN"}
    assert rendezvous.latest_events(cfg)[0]["account"] == "kg1"
    assert rendezvous.publish_control(cfg, "kg1") is True
    assert all(c[0] == "gh" for c in calls) and len(calls) == 2
