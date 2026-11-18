"""kllm Agents: sessions driven end to end through a fake ACP harness (a real subprocess), real git worktrees,
checkpoints, approvals, guardrails, steering, apply, fork, best of N, schedules and the HTTP API. No Kaggle."""
import asyncio, json, os, pathlib, subprocess, sys, time

import pytest

from gateway.agents import gitops, harnesses, runtime
from gateway.agents.runtime import Manager, AgentError, cron_match, match_any
from gateway.agents.store import Store

FAKE = str(pathlib.Path(__file__).parent / "fixtures" / "fake_acp_agent.py")


@pytest.fixture(autouse=True)
def fake_harness(monkeypatch):
    monkeypatch.setitem(harnesses.HARNESSES, "fake", {"name": "Fake", "bin": sys.executable,
                                                      "acp": [sys.executable, FAKE], "default_model": "m2",
                                                      "attach": "fake --resume {sid}"})
    harnesses._CACHE["rows"] = None
    yield
    harnesses._CACHE["rows"] = None


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=r, check=True)
    (r / "a.txt").write_text("one\n", encoding="utf-8")
    (r / "config.yaml").write_text("secret: 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=r, check=True)
    return r


@pytest.fixture
def mgr(tmp_path):
    return Manager(store=Store(tmp_path / "agents.db"))


async def until(pred, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


def types(m, sid):
    return [e["type"] for e in m.store.events(sid)]


def run(coro):
    return asyncio.run(coro)


# ---- pure helpers -------------------------------------------------------------------------------------------------
def test_cron_match():
    t = time.strptime("2026-11-09 03:00", "%Y-%m-%d %H:%M")         # a Monday
    assert cron_match("0 3 * * *", t)
    assert cron_match("*/15 1-4 * * 1-5", t)
    assert not cron_match("5 3 * * *", t)
    assert not cron_match("0 3 * * 0,6", t)
    sunday = time.strptime("2026-11-08 03:00", "%Y-%m-%d %H:%M")
    assert cron_match("0 3 * * 7", sunday) and cron_match("0 3 * * 0", sunday)
    assert not cron_match("bad", t)
    with pytest.raises(AgentError):
        runtime.validate_cron("61 * * * *")


def test_match_any():
    assert match_any("config.yaml", ["config.yaml"])
    assert match_any("kaggle_accounts/kg1/kaggle.json", ["kaggle_accounts/**"])
    assert match_any("deep/dir/key.pem", ["**/*.pem"])
    assert match_any("x/.env", [".env"])
    assert not match_any("gateway/relay.py", ["config.yaml", "kaggle_accounts/**"])


def test_gitops_checkpoint_restore_apply(repo):
    base = gitops.head(repo)
    wt = repo.parent / "wt"
    gitops.add_worktree(repo, wt, base)
    assert gitops.checkpoint(wt, "s1", 1, base, "nothing")[0] is None
    (wt / "a.txt").write_text("two\n", encoding="utf-8")
    (wt / "new.txt").write_text("new\n", encoding="utf-8")
    sha, tree, files = gitops.checkpoint(wt, "s1", 1, base, "turn 1")
    assert sorted(files) == ["a.txt", "new.txt"]
    assert "refs/kllm/checkpoints/s1/1" in gitops.git(repo, "for-each-ref")
    (wt / "a.txt").write_text("three\n", encoding="utf-8")
    gitops.restore(wt, base)
    assert (wt / "a.txt").read_text(encoding="utf-8") == "one\n" and not (wt / "new.txt").exists()
    gitops.restore(wt, sha)
    assert (wt / "new.txt").exists()
    gitops.apply_new_branch(wt, base, "kllm/test", "msg")
    assert gitops.git(repo, "show", "kllm/test:new.txt") == "new\n"
    with pytest.raises(gitops.GitError):
        gitops.apply_new_branch(wt, base, "kllm/test", "msg")
    gitops.apply_commit(repo, wt, base, "apply it")
    assert (repo / "new.txt").read_text(encoding="utf-8") == "new\n" and gitops.git(repo, "status", "--porcelain").strip() == ""
    gitops.drop_refs(repo, "s1")
    assert "refs/kllm" not in gitops.git(repo, "for-each-ref")


# ---- sessions -----------------------------------------------------------------------------------------------------
def test_skip_session_runs_checkpoints_and_applies(mgr, repo):
    async def go():
        res = await mgr.create_session({"goal": "WRITE:b.txt=hello", "mode": "one", "harness": "fake",
                                        "permission_mode": "skip", "cwd": str(repo)})
        sid = res["id"]
        assert res["worktree"] and os.path.isdir(res["worktree"])
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        t = types(mgr, sid)
        for want in ("thought_chunk", "plan", "tool_call", "tool_update", "checkpoint", "message"):
            assert want in t, (want, t)
        w = mgr.store.worker(sid, "W1")
        assert w["model"] == "m2" and w["turns"] == 1 and w["tool_calls"] == 1 and w["context_max"] == 32768
        assert mgr.store.plan(sid)[0]["text"] == "Do the task"
        ch = await mgr.changes(sid)
        assert [f["file"] for f in ch["files"]] == ["b.txt"] and ch["files"][0]["owners"] == ["W1"]
        assert "+hello" in (await mgr.changes(sid, "b.txt"))["diff"]
        assert not (repo / "b.txt").exists()                     # the main checkout is untouched until apply
        out = await mgr.apply(sid, "new_branch", "kllm/b")
        assert out["branch"] == "kllm/b"
        assert gitops.git(repo, "show", "kllm/b:b.txt") == "hello\n"
        assert mgr.store.session(sid)["worktree_path"] is None
        md = mgr.export(sid)
        assert "WRITE:b.txt=hello" in md and "Done: b.txt" in md
    run(go())


def test_ask_session_waits_for_approval(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "WRITE:c.txt=x", "harness": "fake", "permission_mode": "ask",
                                         "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.q("SELECT * FROM approvals WHERE session_id = ?", (sid,)))
        a = mgr.store.q("SELECT * FROM approvals WHERE session_id = ?", (sid,))[0]
        assert a["kind"] == "tool" and "Write c.txt" in a["summary"]
        assert mgr.store.session(sid)["status"] == "needs_you"
        assert mgr.store.worker(sid, "W1")["status"] == "needs_you"
        await mgr.decide(a["id"], "reject")
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        assert not os.path.exists(os.path.join(mgr.store.session(sid)["worktree_path"], "c.txt"))
        assert "approval_resolved" in types(mgr, sid)
        with pytest.raises(AgentError):
            await mgr.decide(a["id"], "accept")                   # already decided
        # a follow-up message reopens the worker; accept this time
        await mgr.message(sid, "W1", "WRITE:c.txt=y")
        await until(lambda: mgr.store.q("SELECT * FROM approvals WHERE session_id = ? AND status = 'pending'", (sid,)))
        a2 = mgr.store.q("SELECT * FROM approvals WHERE session_id = ? AND status = 'pending'", (sid,))[0]
        await mgr.decide(a2["id"], "accept")
        await until(lambda: mgr.store.session(sid)["status"] == "done"
                    and os.path.exists(os.path.join(mgr.store.session(sid)["worktree_path"], "c.txt")))
    run(go())


def test_never_touch_is_refused_and_reverted(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "WRITE:config.yaml=pwned WRITE:ok.txt=fine DELETE:a.txt",
                                         "harness": "fake", "permission_mode": "skip", "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        wt = mgr.store.session(sid)["worktree_path"]
        assert open(os.path.join(wt, "config.yaml"), encoding="utf-8").read() == "secret: 1\n"      # refused at permission time
        assert os.path.exists(os.path.join(wt, "a.txt"))                          # delete breaker in Skip
        assert open(os.path.join(wt, "ok.txt"), encoding="utf-8").read() == "fine\n"
        g = [e["data"] for e in mgr.store.events(sid) if e["type"] == "guardrail"]
        assert any(x.get("why") == "never-touch paths" for x in g) and any(x.get("why") == "delete breaker" for x in g)
    run(go())


def test_scope_outside_contract_is_logged_in_skip(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "x", "harness": "fake", "permission_mode": "skip",
                                         "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        w = await mgr.spawn_worker(sid, "fake", None, {"objective": "WRITE:docs/x.md=1", "may_edit": ["src/**"]})
        await until(lambda: "auto_applied" in types(mgr, sid))
        ev = [e for e in mgr.store.events(sid) if e["type"] == "auto_applied"][0]
        assert ev["worker"] == w["id"] and ev["data"]["files"] == ["docs/x.md"] and ev["data"]["undo_to"]
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        cps = mgr.checkpoints(sid)
        assert [c["n"] for c in cps] == [0, 1]
        await mgr.restore(cps[0]["id"])
        assert not os.path.exists(os.path.join(mgr.store.session(sid)["worktree_path"], "docs", "x.md"))
        fork = await mgr.fork(cps[1]["id"], goal="nothing")
        await until(lambda: mgr.store.session(fork["id"])["status"] == "done")
        assert os.path.exists(os.path.join(mgr.store.session(fork["id"])["worktree_path"], "docs", "x.md"))
        assert mgr.store.session(fork["id"])["base_commit"] == mgr.store.session(sid)["base_commit"]
    run(go())


def test_interrupt_pause_resume_stop(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "SLOW", "harness": "fake", "permission_mode": "skip",
                                         "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.worker(sid, "W1")["status"] == "working")
        await mgr.message(sid, "W1", "WRITE:i.txt=1", mode="interrupt")
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        assert os.path.exists(os.path.join(mgr.store.session(sid)["worktree_path"], "i.txt"))
        assert mgr.store.worker(sid, "W1")["turns"] == 2
        await mgr.message(sid, "lead", "SLOW")                    # no lead: goes to the first worker
        await until(lambda: mgr.store.worker(sid, "W1")["status"] == "working")
        await mgr.pause(sid)
        await until(lambda: mgr.store.worker(sid, "W1")["status"] == "paused")
        await mgr.resume(sid)                                     # replays "Continue where you stopped."
        await until(lambda: mgr.store.worker(sid, "W1")["turns"] == 4 and mgr.store.session(sid)["status"] == "done")
        msgs = [e["data"].get("text") for e in mgr.store.events(sid) if e["type"] == "message"]
        assert msgs[-1].startswith("Done")
        await mgr.message(sid, "W1", "SLOW")
        await until(lambda: mgr.store.worker(sid, "W1")["status"] == "working")
        await mgr.stop(sid)
        assert mgr.store.session(sid)["status"] == "stopped"
        with pytest.raises(AgentError):
            await mgr.message(sid, "W9", "hi")
    run(go())


def test_team_lead_tools(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "lead it", "mode": "team", "lead": {"harness": "fake"},
                                         "permission_mode": "skip", "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.worker(sid, "L"))
        assert mgr.store.worker(sid, "L")["role"] == "lead"
        await mgr.tool(sid, "L", "post_plan", {"steps": [{"text": "write w", "owner": "W2"}]})
        assert mgr.store.plan(sid)[0]["owner_worker_id"] == "W2"
        r = await mgr.tool(sid, "L", "spawn_worker", {"objective": "WRITE:w.txt=1", "harness": "fake"})
        assert r["worker"] == "W2"
        await until(lambda: (mgr.store.worker(sid, "L") or {}).get("turns", 0) >= 2)   # the lead heard back
        info = await mgr.tool(sid, "L", "get_session", {})
        assert {w["id"] for w in info["workers"]} == {"L", "W2"}
        ap = await mgr.tool(sid, "L", "request_approval", {"summary": "ok to ship?"})
        assert mgr.store.session(sid)["status"] == "needs_you"
        await mgr.decide(ap["approval"], "accept")
        await mgr.tool(sid, "L", "finish_session", {"summary": "all done"})
        s = mgr.store.session(sid)
        assert s["status"] == "done" and s["summary"] == "all done"
        assert "peer" in types(mgr, sid)
    run(go())


def test_best_of_n_attempts(mgr, repo):
    async def go():
        res = await mgr.create_session({"goal": "WRITE:n.txt=1", "harness": "fake", "permission_mode": "skip",
                                        "cwd": str(repo), "attempts": 2,
                                        "guardrails": {"verify_command": "test -f n.txt"}})
        pid = res["id"]
        assert len(res["attempts"]) == 2
        await until(lambda: mgr.store.session(pid)["status"] == "needs_you", 30)
        rows = await mgr.attempts(pid)
        assert [r["n"] for r in rows] == [1, 2] and all(r["verify"]["code"] == 0 for r in rows)
        assert rows[0]["added"] == 1
        await mgr.choose(rows[1]["id"])
        assert mgr.store.session(pid)["status"] == "done"
        assert mgr.store.session(rows[0]["id"])["worktree_path"] is None
        assert mgr.store.session(rows[1]["id"])["chosen"]
        assert [s["id"] for s in mgr.list_sessions()] == [pid]        # attempts stay out of the sidebar
    run(go())


def test_validation_and_limits(mgr, repo, tmp_path):
    async def go():
        with pytest.raises(AgentError, match="describe the goal"):
            await mgr.create_session({"goal": " ", "harness": "fake"})
        with pytest.raises(AgentError, match="unknown harness"):
            await mgr.create_session({"goal": "x", "harness": "nope", "cwd": str(repo)})
        import tempfile
        plain = tempfile.mkdtemp(prefix="kllm-plain-")            # outside any git checkout
        with pytest.raises(AgentError, match="git repository"):
            await mgr.create_session({"goal": "x", "harness": "fake", "permission_mode": "skip", "cwd": str(plain)})
        mgr.store.set_setting("guardrails", {"worker_limit": 1})
        sid = (await mgr.create_session({"goal": "SLOW", "harness": "fake", "cwd": str(repo)}))["id"]
        with pytest.raises(AgentError, match="limit"):
            await mgr.spawn_worker(sid, "fake", None, {"objective": "x"})
        await mgr.stop(sid)
    run(go())


def test_stream_harness_rejects_ask(mgr, repo, monkeypatch):
    monkeypatch.setitem(harnesses.HARNESSES, "streamy", {"name": "Streamy", "bin": sys.executable, "stream": True})
    with pytest.raises(AgentError, match="Ask mode needs an ACP harness"):
        run(mgr.create_session({"goal": "x", "harness": "streamy", "permission_mode": "ask", "cwd": str(repo)}))


def test_recover_keeps_sessions_resumable(tmp_path):
    """A relay restart kills the harness processes but not the conversations: the session waits for a message."""
    st = Store(tmp_path / "a.db")
    st.insert("sessions", {"id": "s_1", "title": "t", "goal": "g", "mode": "one", "status": "running",
                           "created_at": time.time()})
    st.insert("workers", {"id": "W1", "session_id": "s_1", "harness": "fake", "status": "working",
                          "created_at": time.time()})
    m = Manager(store=st)
    m.recover()
    assert st.session("s_1")["status"] == "waiting" and st.worker("s_1", "W1")["status"] == "stopped"
    assert st.events("s_1")[-1]["data"]["why"].startswith("the relay restarted")


def test_schedules_run_playbooks(mgr, repo):
    async def go():
        mgr.store.insert("playbooks", {"id": "pb_1", "name": "Nightly", "description": "d", "mode": "playbook",
                                       "workers": [{"harness": "fake", "objective": "WRITE:p.txt=1"}],
                                       "goal": "nightly run", "cwd": str(repo), "permission_mode": "skip",
                                       "created_at": time.time()})
        mgr.store.insert("schedules", {"id": "sc_1", "playbook_id": "pb_1", "cron": "* * * * *",
                                       "conditions": {"min_quota_hours": 999999}, "enabled": True,
                                       "created_at": time.time()})
        mgr.relay = None
        monkey = mgr._pool_hours_left
        mgr._pool_hours_left = lambda: 5.0
        ran = await mgr.run_due_schedules()
        assert ran and ran[0][1].startswith("skipped: 5.0 h")
        assert await mgr.run_due_schedules() == []                 # once per minute
        mgr.store.update("schedules", {"id": "sc_1"}, conditions={}, last_run=0)
        ran = await mgr.run_due_schedules()
        sid = ran[0][1].split()[-1]
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        assert mgr.store.session(sid)["schedule_id"] == "sc_1"
        assert mgr.store.one("SELECT uses FROM playbooks WHERE id = 'pb_1'")["uses"] == 1
        mgr._pool_hours_left = monkey
    run(go())


def test_thought_cap(mgr, repo):
    async def go():
        mgr.store.set_setting("guardrails", {"thought_cap_mb": 0})
        sid = (await mgr.create_session({"goal": "x", "harness": "fake", "permission_mode": "skip",
                                         "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        assert "thought_chunk" not in types(mgr, sid)              # streamed live, not stored past the cap
        assert any(ev["type"] == "thought_chunk" for _, ev in mgr.live[sid])
    run(go())


# ---- HTTP API and the SSE stream ------------------------------------------------------------------------------------
def test_http_api_end_to_end(repo, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from gateway import relay
    from gateway.agents import api
    m = Manager(store=Store(tmp_path / "api.db"))
    monkeypatch.setattr(runtime, "MANAGER", m)
    monkeypatch.setattr(api, "MANAGER", m)
    relay.app.router.routes[:] = [r for r in relay.app.router.routes if not getattr(r, "path", "").startswith(api.P)]
    for route in api.make_router(relay).routes:
        relay.app.router.routes.insert(0, route)
    c = TestClient(relay.app, client=("127.0.0.1", 5))
    with c:
        hs = c.get(api.P + "/harnesses").json()["harnesses"]
        assert any(h["id"] == "fake" and h["acp"] for h in hs)
        t = c.post(api.P + "/harnesses/fake/test", json={}).json()
        assert t["ok"] and t["agent"]["name"] == "Fake" and t["models"] == ["m1", "m2"]
        r = c.post(api.P + "/sessions", json={"goal": "WRITE:h.txt=1", "harness": "fake", "permission_mode": "skip",
                                              "cwd": str(repo)}).json()
        sid = r["id"]
        for _ in range(200):
            if c.get(api.P + f"/sessions/{sid}").json()["status"] == "done":
                break
            time.sleep(0.05)
        v = c.get(api.P + f"/sessions/{sid}").json()
        assert v["status"] == "done" and v["workers"][0]["id"] == "W1" and v["counts"]["checkpoints"] == 2
        assert c.get(api.P + "/sessions").json()["sessions"][0]["id"] == sid
        assert c.get(api.P + f"/sessions/{sid}/changes").json()["files"][0]["file"] == "h.txt"
        assert c.post(api.P + f"/workers/{sid}:W1/attach", json={}).json()["command"].endswith("fake --resume fake-1")
        assert "Done: h.txt" in c.get(api.P + f"/sessions/{sid}/export").text
        assert c.post(api.P + "/sessions", json={"goal": ""}).status_code == 400
        assert c.post(api.P + f"/sessions/{sid}/message", json={"to": "W7", "text": "x"}).status_code == 404
        # library CRUD
        pb = c.post(api.P + "/playbooks", json={"name": "P", "workers": [{"harness": "fake"}]}).json()["item"]
        assert c.post(api.P + "/playbooks", json={"name": "Bad", "workers": [{"harness": "zzz"}]}).status_code == 400
        sc = c.post(api.P + "/schedules", json={"playbook_id": pb["id"], "cron": "0 3 * * *"}).json()["item"]
        assert sc["enabled"] is True
        assert c.post(api.P + "/schedules", json={"playbook_id": pb["id"], "cron": "nope"}).status_code == 400
        assert c.put(api.P + f"/schedules/{sc['id']}", json={"enabled": False}).json()["item"]["enabled"] is False
        n = c.post(api.P + "/notes", json={"text": "Use kg accounts only", "pinned": True}).json()["item"]
        assert n["pinned"] is True and len(c.get(api.P + "/notes").json()["notes"]) == 1
        assert c.delete(api.P + f"/notes/{n['id']}").json()["ok"]
        assert c.post(api.P + "/access", json={"cli": "full"}).json()["access"]["cli"] == "full"
        assert c.post(api.P + "/access", json={"cli": "root"}).status_code == 400
        assert c.post(api.P + "/guardrails", json={"max_workers": 2}).json()["guardrails"]["max_workers"] == 2
        assert c.post(api.P + "/guardrails", json={"bogus": 1}).status_code == 400
        assert c.get(api.P + "/workers").json()["workers"][0]["ref"] == f"{sid}:W1"
        # lead tool forwarding endpoint refuses unknown workers
        assert c.post(api.P + "/tools/get_session", json={"session": sid, "worker": "W9"}).status_code == 404
        assert c.post(api.P + "/tools/get_session", json={"session": sid, "worker": "W1"}).json()["result"]["status"] == "done"
        # the event stream replays stored events with ids, resumable with Last-Event-ID
        with c.stream("GET", api.P + f"/sessions/{sid}/events?follow=0", headers={"Last-Event-ID": "0"}) as s:
            seen = []
            for line in s.iter_lines():
                if line.startswith("data: "):
                    seen.append(json.loads(line[6:]))
                if len(seen) >= 3:
                    break
        assert seen[0]["type"] == "status" and seen[0]["id"] < seen[1]["id"] < seen[2]["id"]
        # kllm actions are logged; the admin API refuses writes from a read-only agent session
        assert c.post(api.P + f"/sessions/{sid}/kllm_action", json={"worker": "W1", "action": "kllm status"}).json()["ok"]
        c.post(api.P + "/access", json={"api": "read"})
        r = c.post("/admin/api/backends/pin", json={"account": "kg1"},
                   headers={"X-Kllm-Panel": "1", "X-Kllm-Agent-Session": sid})
        assert r.status_code == 403 and "Access page" in r.json()["detail"]
        acts = [e["data"] for e in m.store.events(sid) if e["type"] == "kllm_action"]
        assert acts[0]["action"] == "kllm status" and acts[-1]["allowed"] is False and acts[-1]["via"] == "api"
        assert c.post(api.P + f"/sessions/{sid}/delete", json={}).json()["ok"]
        assert c.get(api.P + f"/sessions/{sid}").status_code == 404


def test_cli_agent_access(monkeypatch):
    from gateway import cli
    import argparse
    calls = []
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(OSError()))
    env = {"KLLM_AGENT_SESSION": "s_1", "KLLM_ACCESS": "read"}
    assert cli.agent_access_ok(argparse.Namespace(cmd="status"), env)
    assert not cli.agent_access_ok(argparse.Namespace(cmd="down", account=None), env)
    assert cli.agent_access_ok(argparse.Namespace(cmd="account", action="list"), env)
    assert not cli.agent_access_ok(argparse.Namespace(cmd="status"), dict(env, KLLM_ACCESS="off"))
    assert cli.agent_access_ok(argparse.Namespace(cmd="down"), dict(env, KLLM_ACCESS="full"))
    assert cli.agent_access_ok(argparse.Namespace(cmd="down"), {})          # outside a session: no restriction
    assert len(calls) == 5


def test_mcp_lists_agent_tools_only_inside_a_session(monkeypatch):
    from gateway import mcp_server
    srv = mcp_server.MCPServer()
    names = lambda: {t["name"] for t in run(srv.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
                     ["result"]["tools"]}
    monkeypatch.delenv("KLLM_AGENT_SESSION", raising=False)
    assert "post_plan" not in names()
    monkeypatch.setenv("KLLM_AGENT_SESSION", "s_1")
    monkeypatch.setenv("KLLM_AGENT_WORKER", "L")
    monkeypatch.setenv("KLLM_RELAY", "http://127.0.0.1:9")
    assert {"post_plan", "spawn_worker", "finish_session"} <= names()
    r = run(srv.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                "params": {"name": "get_session", "arguments": {}}}))
    assert r["result"]["isError"] and "unreachable" in r["result"]["content"][0]["text"]


def test_harness_crash_is_retried_once_then_reported(mgr, repo, monkeypatch, tmp_path):
    async def go():
        monkeypatch.setenv("FAKE_EXIT_ONCE", str(tmp_path / "crashed-once"))
        sid = (await mgr.create_session({"goal": "WRITE:r.txt=1", "harness": "fake", "permission_mode": "skip",
                                         "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.session(sid)["status"] == "done")
        msgs = [e["data"]["text"] for e in mgr.store.events(sid) if e["type"] == "message"]
        assert any("restarting it once" in m for m in msgs) and msgs[-1].startswith("Done: r.txt")
        monkeypatch.delenv("FAKE_EXIT_ONCE")
        monkeypatch.setenv("FAKE_EXIT_ALWAYS", "1")
        sid2 = (await mgr.create_session({"goal": "x", "harness": "fake", "permission_mode": "skip", "cwd": str(repo)}))["id"]
        # a session whose only worker crashed is a failure, not "done"
        await until(lambda: mgr.store.session(sid2)["status"] == "failed")
        err = mgr.store.worker(sid2, "W1")["error"]
        assert "exited with code 3" in err and "crashed on purpose" in err
        assert "Every worker stopped with an error" in (mgr.store.session(sid2).get("summary") or "")
    run(go())


def test_pool_note_when_a_kllm_worker_has_no_model(mgr, monkeypatch):
    """A worker on kllm's own models with nothing serving must say so, not sit at 'working' in silence."""
    from gateway.agents import runtime as rt
    reg = {"q": {"key": "q", "kind": "chat"}, "img": {"key": "img", "kind": "image"}}
    monkeypatch.setattr(rt.common, "load_models", lambda: reg)
    monkeypatch.setattr(rt.common, "read_state", lambda: {"backends": [{"model_key": "img"}], "pending": {"kg2": {"model_key": "q"}}})
    note = mgr.pool_note("opencode", "kllm/kllm")
    assert note and "q is starting" in note
    monkeypatch.setattr(rt.common, "read_state", lambda: {"backends": [{"model_key": "q"}], "pending": {}})
    assert mgr.pool_note("opencode", "kllm/kllm") is None          # a chat model serves: no note
    assert mgr.pool_note("agy", None) is None                       # own models: never a note


def test_opencode_worker_permissions_follow_the_session():
    import json
    from gateway.agents import harnesses
    ask = json.loads(harnesses.permission_env({}, "opencode", "ask")["OPENCODE_CONFIG_CONTENT"])
    assert ask["permission"] == {"edit": "ask", "bash": "ask", "webfetch": "ask"}
    skip = json.loads(harnesses.permission_env({"OPENCODE_CONFIG_CONTENT": '{"theme": "x"}'}, "opencode", "skip")["OPENCODE_CONFIG_CONTENT"])
    assert skip["theme"] == "x" and skip["permission"]["edit"] == "allow"
    assert "OPENCODE_CONFIG_CONTENT" not in harnesses.permission_env({}, "agy", "ask")


# ---- flows: a dependency graph of steps across harnesses, run by kllm ---------------------------------------------
def test_flow_validation(mgr):
    ok = mgr.validate_flow([{"id": "a", "objective": "x", "harness": "fake"},
                            {"id": "b", "objective": "y", "harness": "fake", "after": ["a"]}])
    assert [s["status"] for s in ok["steps"]] == ["pending", "pending"] and ok["steps"][1]["after"] == ["a"]
    for bad, frag in [([], "at least one"), ([{"id": "a", "objective": ""}], "objective"),
                      ([{"id": "a", "objective": "x", "harness": "fake"}, {"id": "a", "objective": "y", "harness": "fake"}], "unique"),
                      ([{"id": "a", "objective": "x", "harness": "fake", "after": ["zz"]}], "unknown"),
                      ([{"id": "a", "objective": "x", "harness": "fake", "after": ["b"]},
                        {"id": "b", "objective": "y", "harness": "fake", "after": ["a"]}], "cycle")]:
        with pytest.raises(AgentError, match=frag):
            mgr.validate_flow(bad)


def test_flow_runs_steps_in_order_and_passes_results_on(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "two steps", "mode": "flow", "permission_mode": "skip", "cwd": str(repo),
                                         "steps": [{"id": "write", "objective": "WRITE:f1.txt=one", "harness": "fake"},
                                                   {"id": "more", "objective": "WRITE:f2.txt=two", "harness": "fake", "after": ["write"]},
                                                   {"id": "side", "objective": "WRITE:f3.txt=three", "harness": "fake"}]}))["id"]
        # independent steps start together; the dependent one waits
        flow = mgr.store.session(sid)["flow"]
        assert {s["id"]: s["status"] for s in flow["steps"]} == {"write": "running", "more": "pending", "side": "running"}
        await until(lambda: mgr.store.session(sid)["status"] == "done", timeout=40)
        s = mgr.store.session(sid)
        assert all(st["status"] == "done" for st in s["flow"]["steps"]) and "3 of 3 steps done" in s["summary"]
        wt = pathlib.Path(s["worktree_path"])
        assert (wt / "f1.txt").exists() and (wt / "f2.txt").exists() and (wt / "f3.txt").exists()
        second = next(w for w in mgr.store.workers(sid) if w.get("step") == "more")
        assert second["contract"]["inputs"][0]["title"].startswith("WRITE:f1")       # got the first step's report
    run(go())


def test_lead_that_goes_quiet_waits_instead_of_finishing(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "hi", "mode": "team", "lead": {"harness": "fake"},
                                         "permission_mode": "skip", "cwd": str(repo)}))["id"]
        await until(lambda: mgr.store.session(sid)["status"] == "waiting")
        assert mgr.store.session(sid).get("finished_at") is None
        models = await mgr.tool(sid, "L", "list_models", {})
        assert "kllm/<key>" in models["how"]
        await mgr.tool(sid, "L", "finish_session", {"summary": "bye"})
        assert mgr.store.session(sid)["status"] == "done"
    run(go())


def test_workers_in_a_flow_get_peer_tools_only(mgr, repo):
    async def go():
        sid = (await mgr.create_session({"goal": "two", "mode": "flow", "permission_mode": "skip", "cwd": str(repo),
                                         "steps": [{"id": "a", "objective": "SLOW", "harness": "fake"},
                                                   {"id": "b", "objective": "SLOW", "harness": "fake"}]}))["id"]
        await until(lambda: len(mgr.store.workers(sid)) == 2)
        info = await mgr.tool(sid, "W1", "get_session", {})                 # a worker can see its teammates
        assert {w["id"] for w in info["workers"]} == {"W1", "W2"}
        r = await mgr.tool(sid, "W1", "message_worker", {"to": "W2", "text": "hello"})
        assert r["sent_to"] == ["W2"]
        with pytest.raises(AgentError, match="for the lead"):
            await mgr.tool(sid, "W1", "spawn_worker", {"objective": "x", "harness": "fake"})
        assert mgr.workers[(sid, "W1")].mcp_servers                         # wired to the kllm MCP server
        await mgr.stop(sid)
    run(go())


def test_claude_on_kllm_gets_a_model_name_claude_code_accepts(monkeypatch):
    from gateway.agents import harnesses as H
    assert H.kllm_model("kllm/qwen3.8-27b") == "qwen3.8-27b"           # opencode-style name from a moved worker
    monkeypatch.setattr(H.common, "read_state", lambda: {"backends": [{"model_key": "qwen3.8-27b"}]})
    monkeypatch.setattr(H.common, "load_models", lambda: {"qwen3.8-27b": {}})
    monkeypatch.setattr(H.common, "load_config", lambda: {"default_model": "other"})
    assert H.kllm_model("kllm") == "qwen3.8-27b" and H.kllm_model(None) == "qwen3.8-27b"
    from gateway.agents.workers import StreamWorker
    w = StreamWorker.__new__(StreamWorker)
    w.claude_bin, w.model, w.kllm, w.started_once, w.claude_session, w.mcp_config = "claude", "kllm", True, False, None, None
    w.mgr = type("M", (), {"worker_update": lambda *a, **k: None})()
    assert "--model" not in w.command("hi")
