"""Planner + router policy tests (pure, no network). Run: .venv/bin/python -m pytest -q tests"""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gateway import scheduler
from gateway.scheduler import plan, Router, resolve_model

NOW = 1_000_000.0
REG = {"big": {}, "fast": {}}
ACCTS = ["kg1", "kg2", "kg3", "kg4"]
CFG = {"prelaunch_lead_seconds": 1500, "autoscale": {"enabled": True, "max_backends": 3, "spare_backends": 0}}
QUOTA = {a: 28.0 for a in ACCTS}

def be(acct, model="big", started=NOW - 3600, session=39600):
    return {"account": acct, "model_key": model, "started": started, "session_seconds": session}

def demand(models=None, backends=None):
    return {"models": models or {}, "backends": backends or {}}

def run(backends=(), pending=None, stats=None, quota=None, dem=None, cfg=CFG, healthy=None):
    return plan(NOW, cfg, REG, "big", list(backends), pending or {}, ACCTS, stats or {},
                quota or QUOTA, dem if dem is not None else demand(), healthy)

def kinds(acts):
    return sorted((a[0], a[1] if a[0] != "launch" else a[2]) for a in acts)

def test_nothing_running_no_demand_does_nothing():
    assert run() == []

def test_waiting_request_launches_default_on_best_account():
    stats = {"kg1": {"ready_ewma_s": 900}, "kg2": {"ready_ewma_s": 300}}
    acts = run(stats=stats, dem=demand({"big": {"waiting": 1, "sessions": 0}}))
    assert acts == [("launch", "big", "kg2", "demand")]

def test_accounts_in_cooldown_or_low_quota_are_skipped():
    stats = {"kg1": {"cooldown_until": NOW + 60}}
    quota = dict(QUOTA, kg2=0.5)
    acts = run(stats=stats, quota=quota, dem=demand({"big": {"waiting": 1}}))
    assert acts[0][2] in ("kg3", "kg4")

def test_queued_launch_gets_hedged_but_kept():
    pending = {"kg1": {"model_key": "big", "started": NOW - 300}}       # no boot after 5 min
    acts = run(pending=pending, dem=demand({"big": {"waiting": 1}}))
    assert ("launch", "big") == acts[0][:2] and acts[0][2] != "kg1" and "hedge" in acts[0][3]
    assert not any(a[0] == "cancel" for a in acts)                      # keeps its queue place

def test_queued_launch_cancelled_once_hedge_boots():
    pending = {"kg1": {"model_key": "big", "started": NOW - 600},
               "kg2": {"model_key": "big", "started": NOW - 200, "boot_at": NOW - 150}}
    acts = run(pending=pending, dem=demand({"big": {"waiting": 1}}))
    assert kinds(acts) == [("cancel", "kg1")]

def test_no_endless_hedging_when_everything_queues():
    pending = {"kg1": {"model_key": "big", "started": NOW - 900},
               "kg2": {"model_key": "big", "started": NOW - 500}}
    acts = run(pending=pending, dem=demand({"big": {"waiting": 1}}))
    assert not any(a[0] == "launch" for a in acts)

def test_parallel_sessions_scale_out_and_cap():
    acts = run(backends=[be("kg1")], dem=demand({"big": {"sessions": 3}}, {"kaggle:kg1": {"idle_s": 5}}))
    assert kinds(acts) == [("launch", "kg2")] or kinds(acts)[0][0] == "launch"
    acts = run(backends=[be("kg1")], dem=demand({"big": {"sessions": 20}}, {"kaggle:kg1": {"idle_s": 5}}))
    assert len([a for a in acts if a[0] == "launch"]) == 2                # max_backends 3

def test_surplus_pending_cancelled_when_served():
    acts = run(backends=[be("kg1")], pending={"kg2": {"model_key": "big", "started": NOW - 60}},
               dem=demand({"big": {"sessions": 1}}, {"kaggle:kg1": {"idle_s": 5}}))
    assert kinds(acts) == [("cancel", "kg2")]

def test_idle_backend_retired_unless_minimum():
    d = demand({}, {"kaggle:kg1": {"idle_s": 3600}})
    assert kinds(run(backends=[be("kg1")], dem=d)) == [("retire", "kg1")]
    cfg = {"prelaunch_lead_seconds": 1500, "desired_pool": 1, "autoscale": {"enabled": True}}
    assert run(backends=[be("kg1")], dem=d, cfg=cfg) == []

def test_unknown_demand_never_retires_for_idleness():
    assert run(backends=[be("kg1")], dem=None) == []

def test_expiring_backend_in_use_gets_successor_then_retires():
    old = be("kg1", started=NOW - 39600 + 600)                           # 10 min left
    d = demand({"big": {"sessions": 1}}, {"kaggle:kg1": {"idle_s": 10, "sessions": 1}})
    acts = run(backends=[old], dem=d)
    assert [a[0] for a in acts] == ["launch"]
    # successor serving but the session has not been handed off yet: keep the old one
    assert run(backends=[old, be("kg2", started=NOW - 60)], dem=d) == []
    d2 = demand({"big": {"sessions": 1}}, {"kaggle:kg1": {"idle_s": 10, "sessions": 0},
                                           "kaggle:kg2": {"idle_s": 5, "sessions": 1}})
    acts = run(backends=[old, be("kg2", started=NOW - 60)], dem=d2)
    assert kinds(acts) == [("retire", "kg1")]
    # at the hard cap it goes regardless
    near = be("kg1", started=NOW - 39600 + 60)
    assert ("retire", "kg1") in kinds(run(backends=[near, be("kg2", started=NOW - 60)], dem=d))

def test_hard_expiry_retires():
    acts = run(backends=[be("kg1", started=NOW - 40000)])
    assert ("retire", "kg1") in kinds(acts)

def test_quota_expiring_backend_gets_successor_then_retires():
    # kg1 has plenty of session RUNTIME left but is about to run out of weekly QUOTA
    # (0.2 h < the 25-min prelaunch lead), so it must be treated as expiring and get a successor.
    old = be("kg1")                                              # ~10 h of runtime left
    q = dict(QUOTA); q["kg1"] = 0.2                              # ~12 min of weekly quota left
    d = demand({"big": {"sessions": 1}}, {"kaggle:kg1": {"idle_s": 10, "sessions": 1}})
    acts = run(backends=[old], quota=q, dem=d)
    assert [a[0] for a in acts] == ["launch"]                    # a replacement is launched...
    assert acts[0][2] != "kg1"                                  # ...on a different, higher-quota account
    # once the successor serves and kg1's session has been handed off (0 sessions), kg1 is retired
    d2 = demand({"big": {"sessions": 1}}, {"kaggle:kg1": {"idle_s": 10, "sessions": 0},
                                           "kaggle:kg2": {"idle_s": 5, "sessions": 1}})
    acts2 = run(backends=[old, be("kg2", started=NOW - 60)], quota=q, dem=d2)
    assert ("retire", "kg1") in kinds(acts2)

def test_unhealthy_backend_is_not_capacity():
    acts = run(backends=[be("kg1")], healthy={"kg1": False},
               dem=demand({"big": {"sessions": 1}}, {"kaggle:kg1": {"idle_s": 5}}))
    assert [a[0] for a in acts] == ["launch"]

def test_manual_mode_does_nothing():
    cfg = {"prelaunch_lead_seconds": 1500, "autoscale": {"enabled": False}}
    assert run(dem=demand({"big": {"waiting": 3}}), cfg=cfg) == []

# ---- router ---------------------------------------------------------------------------
def B(label, age_left=1e6):
    return {"label": label, "url": "u-" + label, "key": "k", "age_left": age_left, "lead": 1500}

def test_sessions_are_sticky_and_new_sessions_balance():
    r = Router()
    s1 = r.touch_session("a", "big", 50000)
    first = r.order(s1, [B("x"), B("y")], 900)[0]["label"]
    s2 = r.touch_session("b", "big", 50000)
    second = r.order(s2, [B("x"), B("y")], 900)[0]["label"]
    assert first != second                                       # least-loaded placement
    # adding a backend never moves an existing conversation
    assert r.order(s1, [B("x"), B("y"), B("z")], 900)[0]["label"] == first

def test_dead_backend_session_replaced():
    r = Router()
    s = r.touch_session("a", "big", 50000)
    r.order(s, [B("x")], 900)
    assert r.order(s, [B("y")], 900)[0]["label"] == "y" and s["backend"] == "y"

def test_new_sessions_avoid_expiring_backends():
    r = Router()
    s = r.touch_session("a", "big", 50000)
    assert r.order(s, [B("old", age_left=600), B("new")], 900)[0]["label"] == "new"

def test_side_requests_are_not_demand():
    r = Router()
    r.touch_session("title", "big", 800)                         # one small request
    assert r.demand(900, {})["models"] == {}

def test_resolve_model():
    reg = {"big": {}, "fast": {}, "alias-to-fast": {"alias_of": "fast"}}
    assert resolve_model("fast", reg, "big", []) == "fast"
    assert resolve_model("alias-to-fast", reg, "big", []) == "fast"
    assert resolve_model("kllm", reg, "big", ["fast"]) == "fast"          # whatever runs
    assert resolve_model("kllm", reg, "big", ["fast", "big"]) == "big"    # default preferred
    assert resolve_model("claude-haiku", reg, "big", []) == "big"         # nothing runs -> default

def test_manual_prewarm_is_not_cancelled_without_demand():
    pending = {"kg1": {"model_key": "big", "started": NOW - 120, "origin": "manual"}}
    assert run(pending=pending) == []

def test_overloaded_backend_sheds_its_newest_session_only():
    r = Router()
    ss = []
    for k in ("old1", "old2", "new3"):
        x = r.touch_session(k, "big", 50000); r.order(x, [B("x")], 900, per_backend=2); ss.append(x)
        time.sleep(0.01)
    # a second backend appears: only the newest session on the overloaded backend moves
    assert r.order(ss[0], [B("x"), B("y")], 900, per_backend=2)[0]["label"] == "x"
    assert r.order(ss[1], [B("x"), B("y")], 900, per_backend=2)[0]["label"] == "x"
    assert r.order(ss[2], [B("x"), B("y")], 900, per_backend=2)[0]["label"] == "y"
    # and it then stays there
    assert r.order(ss[2], [B("x"), B("y")], 900, per_backend=2)[0]["label"] == "y"

def test_claude_launcher_sizes_for_model_that_autoscale_will_start():
    from gateway.claude_launcher import pick_model
    st = {"default_model": "big", "pool": [], "models": [{"key": "big", "ctx": 262144}, {"key": "fast", "ctx": 98304}]}
    assert pick_model(st) == ("big", 262144)
    assert pick_model(st, "fast") == ("fast", 98304)
    st["pool"] = [{"model_key": "fast", "ctx": 98304}]
    assert pick_model(st) == ("kllm", 98304)

def test_idle_sessions_keep_but_do_not_grow_the_pool():
    d = demand({"big": {"sessions": 5, "busy": 1}}, {"kaggle:kg1": {"idle_s": 5}})
    assert run(backends=[be("kg1")], dem=d) == []
    d = demand({"big": {"sessions": 5, "busy": 3}}, {"kaggle:kg1": {"idle_s": 5}})
    assert [a[0] for a in run(backends=[be("kg1")], dem=d)] == ["launch"]

def test_probe_sessions_are_not_demand():
    r = Router()
    s = r.touch_session("doctor", "big", 50000); s["probe"] = True
    assert r.demand(900, {})["models"] == {}

SPARE = {"prelaunch_lead_seconds": 1500, "autoscale": {"enabled": True, "max_backends": 3, "spare_backends": 1}}

def test_hot_standby_only_while_someone_is_working():
    busy = demand({"big": {"sessions": 1, "busy": 1, "sustained": 1}}, {"kaggle:kg1": {"idle_s": 5}})
    assert [a[0] for a in run(backends=[be("kg1")], dem=busy, cfg=SPARE)] == ["launch"]
    quick = demand({"big": {"sessions": 1, "busy": 1}}, {"kaggle:kg1": {"idle_s": 5}})
    assert run(backends=[be("kg1")], dem=quick, cfg=SPARE) == []          # short work: no standby
    idle = demand({"big": {"sessions": 1, "busy": 0}}, {"kaggle:kg1": {"idle_s": 600}})
    assert run(backends=[be("kg1")], dem=idle, cfg=SPARE) == []
    # standby stops once nobody works for idle_stop_minutes
    gone = demand({}, {"kaggle:kg1": {"idle_s": 3600}, "kaggle:kg2": {"idle_s": 3600}})
    assert kinds(run(backends=[be("kg1"), be("kg2")], dem=gone, cfg=SPARE)) == [("retire", "kg1"), ("retire", "kg2")]

def test_failing_model_is_not_relaunched():
    acts = plan(NOW, CFG, REG, "big", [], {}, ACCTS, {}, QUOTA,
                demand({"big": {"waiting": 2}}), None, {"big": NOW + 600})
    assert acts == []
    acts = plan(NOW, CFG, REG, "big", [], {}, ACCTS, {}, QUOTA,
                demand({"big": {"waiting": 2}}), None, {"big": NOW - 1})      # breaker expired
    assert [a[0] for a in acts] == ["launch"]


def test_disabled_account_is_never_chosen_for_a_launch():
    cfg = dict(CFG, accounts=[{"name": "kg1", "disabled": True}, {"name": "kg2"}, {"name": "kg3"}, {"name": "kg4"}])
    stats = {"kg1": {"ready_ewma_s": 60}}                 # kg1 would be the best-ranked account
    acts = run(cfg=cfg, stats=stats, dem=demand({"big": {"waiting": 1}}))
    assert acts and acts[0][0] == "launch" and acts[0][2] != "kg1"
    # all but one disabled -> the remaining one; all disabled -> nothing to launch on
    cfg2 = dict(CFG, accounts=[{"name": a, "disabled": a != "kg4"} for a in ACCTS])
    assert run(cfg=cfg2, dem=demand({"big": {"waiting": 1}}))[0][2] == "kg4"
    cfg3 = dict(CFG, accounts=[{"name": a, "disabled": True} for a in ACCTS])
    assert run(cfg=cfg3, dem=demand({"big": {"waiting": 1}})) == []


def test_disabled_account_still_serves_and_retires_normally():
    cfg = dict(CFG, accounts=[{"name": "kg1", "disabled": True}])
    acts = run(cfg=cfg, backends=[be("kg1")], dem=demand({"big": {"sessions": 1}}, {"kaggle:kg1": {"idle_s": 5}}))
    assert acts == []                                     # a disabled account's running kernel is left alone


def test_pick_next_account_skips_disabled():
    from gateway import orchestrator
    cfg = {"accounts": [{"name": "kg1", "disabled": True}, {"name": "kg2"}]}
    assert orchestrator.pick_next_account(cfg, {"last_account_index": -1}) == (1, "kg2")
    assert orchestrator.pick_next_account(cfg, {"last_account_index": 0}) == (1, "kg2")
    only = {"accounts": [{"name": "kg1", "disabled": True}]}
    assert orchestrator.pick_next_account(only, {})[1] == "kg1"      # last resort


def test_subagent_swarm_affinity_separation():
    from gateway.relay import affinity_key
    # Parent session: initial turn
    parent_t1 = {"metadata": {"user_id": "session-123"},
                 "system": "You are a helpful coding assistant.",
                 "messages": [{"role": "user", "content": "Refactor the authentication module."}]}
    parent_k1 = affinity_key(parent_t1)

    # Parent session: second turn (adds assistant response and next user instruction)
    parent_t2 = {"metadata": {"user_id": "session-123"},
                 "system": "You are a helpful coding assistant.",
                 "messages": [{"role": "user", "content": "Refactor the authentication module."},
                              {"role": "assistant", "content": "I will examine auth.py first."},
                              {"role": "user", "content": "Also check test_auth.py."}]}
    parent_k2 = affinity_key(parent_t2)
    assert parent_k1 == parent_k2, "Parent follow-up turns must retain the exact same affinity key"

    # Sub-agent 1: spawned with subagent_id metadata
    sub1_t1 = {"metadata": {"user_id": "session-123", "subagent_id": "search-agent"},
               "system": "You are a subagent specialized in codebase search.",
               "messages": [{"role": "user", "content": "Find all usages of generate_token() in tests/"}]}
    sub1_k1 = affinity_key(sub1_t1)
    assert sub1_k1 != parent_k1, "Sub-agent must receive a distinct affinity key from parent"

    # Sub-agent 1: follow-up turn
    sub1_t2 = {"metadata": {"user_id": "session-123", "subagent_id": "search-agent"},
               "system": "You are a subagent specialized in codebase search.",
               "messages": [{"role": "user", "content": "Find all usages of generate_token() in tests/"},
                            {"role": "assistant", "content": "Found 3 occurrences."},
                            {"role": "user", "content": "Now grep for verify_token()."}]}
    assert affinity_key(sub1_t2) == sub1_k1, "Sub-agent follow-up turns must retain the sub-agent affinity key"

    # Sub-agent 2: another concurrent sub-task
    sub2_t1 = {"metadata": {"user_id": "session-123", "subagent_id": "test-agent"},
               "system": "You are a test runner subagent.",
               "messages": [{"role": "user", "content": "Execute pytest on tests/test_auth.py"}]}
    sub2_k1 = affinity_key(sub2_t1)
    assert sub2_k1 != parent_k1 and sub2_k1 != sub1_k1, "Concurrent subagents must receive distinct affinity keys"


def test_subagent_swarm_balances_across_backends():
    from gateway.relay import affinity_key
    r = Router()
    parent_req = {"metadata": {"user_id": "swarm-session"}, "messages": [{"role": "user", "content": "Main task"}]}
    sub1_req = {"metadata": {"user_id": "swarm-session", "subagent_id": "search"}, "messages": [{"role": "user", "content": "Sub-agent search"}]}
    sub2_req = {"metadata": {"user_id": "swarm-session", "subagent_id": "tests"}, "messages": [{"role": "user", "content": "Sub-agent tests"}]}

    sp = r.touch_session(affinity_key(parent_req), "big", 50000)
    be_parent = r.order(sp, [B("kg1"), B("kg2"), B("kg3")], 900, per_backend=1)[0]["label"]

    s1 = r.touch_session(affinity_key(sub1_req), "big", 50000)
    be_sub1 = r.order(s1, [B("kg1"), B("kg2"), B("kg3")], 900, per_backend=1)[0]["label"]

    s2 = r.touch_session(affinity_key(sub2_req), "big", 50000)
    be_sub2 = r.order(s2, [B("kg1"), B("kg2"), B("kg3")], 900, per_backend=1)[0]["label"]

    # All three sessions land on separate backends
    placed = {be_parent, be_sub1, be_sub2}
    assert len(placed) == 3, f"Parent and subagents should distribute across distinct backends, got {placed}"



def test_active_session_without_waiting_request_does_not_relaunch_a_stopped_backend():
    # session still "active" (last request < 15 min ago) but its backend was stopped and nothing is waiting
    assert run(dem=demand({"big": {"sessions": 1, "busy": 0, "waiting": 0}})) == []
    # the agent's next request (waiting) does launch
    assert kinds(run(dem=demand({"big": {"sessions": 1, "waiting": 1}}))) == [("launch", "kg1")]
    # and a session on a RUNNING backend still keeps it (no retire)
    assert all(a[0] != "retire" for a in run(backends=[be("kg1")], dem=demand({"big": {"sessions": 1}},
                                                                               {"kaggle:kg1": {"sessions": 1, "idle_s": 30}})))


def test_manual_launch_queued_in_kaggle_keeps_its_place():
    """Busy Kaggle days queue kernels 25+ min. A manual launch that has not booted must not be cancelled at the
    25-min load timeout (that re-queues the user's request at the back); a booted-but-hung one still is."""
    import time as _t
    from gateway import scheduler as sc
    now = _t.time()
    reg = {"m": {"key": "m", "ctx": 8192}}
    pend = {"kg1": {"model": "m", "model_key": "m", "origin": "manual", "started": now - 40 * 60},
            "kg2": {"model": "m", "model_key": "m", "origin": "manual", "started": now - 40 * 60, "boot_at": now - 35 * 60}}
    acts = sc.plan(now, {"autoscale": {"enabled": True, "max_backends": 4}}, reg, "m", [], pend, ["kg3"], {},
                   {"kg1": 20, "kg2": 20, "kg3": 20}, {"models": {}, "backends": {}})
    cancels = {a[1] for a in acts if a[0] == "cancel"}
    assert "kg1" not in cancels          # still queued at Kaggle: keeps its place
    assert "kg2" in cancels              # booted 35 min ago and still not ready: broken


def test_manual_launch_never_picks_a_busy_account(monkeypatch, tmp_path):
    """`kllm up` without --account must skip accounts that run a backend or have a launch pending (pushing to them
    replaces that kernel), and reserve its pick so a concurrent `kllm up` cannot take the same account."""
    from gateway import orchestrator as orch, common
    cfg = {"accounts": [{"name": f"kg{i}", "user": f"u{i}"} for i in range(1, 5)], "default_model": "m",
           "session_seconds": 3600}
    state = {"backends": [{"account": "kg1"}], "pending": {"kg2": {"account": "kg2"}}, "last_account_index": -1}
    monkeypatch.setattr(common, "load_config", lambda: cfg)
    monkeypatch.setattr(common, "read_state", lambda: state)
    import contextlib
    @contextlib.contextmanager
    def ms():
        yield state
    monkeypatch.setattr(common, "mutate_state", ms)
    monkeypatch.setattr(common, "resolve_model", lambda k: {"key": "m", "hf": "x/y", "quant": "Q4"})
    monkeypatch.setattr(orch.media, "problems", lambda m: [])
    monkeypatch.setattr(orch.media, "runtime", lambda m: "llama.cpp")
    picked = []
    def fake_render(cfg, account, *a, **k):
        picked.append(account); raise RuntimeError("stop after pick")
    monkeypatch.setattr(orch, "render_kernel", fake_render)
    for _ in range(2):
        try:
            orch.launch("m", wait=False, log=lambda *a: None)
        except RuntimeError:
            pass
    assert picked == ["kg3", "kg4"]            # never kg1 (backend) or kg2 (pending); round-robin moves on
    assert set(state["pending"]) == {"kg2"}   # a failed render releases its reservation (no account left blocked)


def test_pick_next_account_respects_account_allow_prefixes():
    from gateway import orchestrator
    cfg = {
        "accounts": [
            {"name": "kg1"},
            {"name": "lk4"},
            {"name": "kx2"},
            {"name": "kg2"},
        ],
        "account_allow_prefixes": ["kg"],
    }
    idx1, acct1 = orchestrator.pick_next_account(cfg, {"last_account_index": -1})
    assert acct1 == "kg1"
    idx2, acct2 = orchestrator.pick_next_account(cfg, {"last_account_index": 0})
    assert acct2 == "kg2"
    idx3, acct3 = orchestrator.pick_next_account(cfg, {"last_account_index": 1})
    assert acct3 == "kg1"


def test_launch_rejects_disallowed_account_prefix(monkeypatch):
    from gateway import orchestrator, common
    cfg = {
        "accounts": [{"name": "kg1"}, {"name": "lk4"}],
        "default_model": "m",
        "session_seconds": 3600,
        "account_allow_prefixes": ["kg"],
    }
    monkeypatch.setattr(common, "load_config", lambda: cfg)
    monkeypatch.setattr(common, "read_state", lambda: {"backends": [], "pending": {}})
    monkeypatch.setattr(common, "resolve_model", lambda k: {"key": "m", "hf": "x/y"})
    monkeypatch.setattr(orchestrator.media, "problems", lambda m: [])
    monkeypatch.setattr(orchestrator.media, "runtime", lambda m: "llama.cpp")

    logs = []
    res = orchestrator.launch(model_key="m", account="lk4", log=logs.append)
    assert res is None
    assert any("not permitted" in l for l in logs)


def test_scheduler_step_respects_account_allow_prefixes():
    cfg = dict(CFG, accounts=[{"name": "lk4"}, {"name": "kx2"}, {"name": "kg1"}],
               account_allow_prefixes=["kg"])
    acts = run(cfg=cfg, dem=demand({"big": {"waiting": 1}}))
    assert acts and acts[0][0] == "launch"
    assert acts[0][2] == "kg1"


def test_pick_next_account_raises_when_no_accounts_match_prefix():
    import pytest
    from gateway import orchestrator
    cfg = {
        "accounts": [{"name": "lk4"}, {"name": "kx2"}],
        "account_allow_prefixes": ["kg"],
    }
    with pytest.raises(RuntimeError, match="No accounts configured matching"):
        orchestrator.pick_next_account(cfg, {})


def test_launch_swarm_cluster_rejects_disallowed_accounts(monkeypatch):
    from gateway import orchestrator, common
    cfg = {
        "accounts": [{"name": "kg1"}, {"name": "kg2"}, {"name": "lk4"}],
        "default_model": "m",
        "account_allow_prefixes": ["kg"],
    }
    monkeypatch.setattr(common, "load_config", lambda: cfg)
    monkeypatch.setattr(common, "read_state", lambda: {"backends": [], "pending": {}})
    monkeypatch.setattr(common, "resolve_model", lambda k: {"key": "m", "stages": 2})

    logs = []
    res = orchestrator.launch_swarm_cluster(model_key="m", accounts=["kg1", "lk4"], log=logs.append)
    assert res is None
    assert any("not permitted" in l for l in logs)


def test_build_cache_rejects_disallowed_account(monkeypatch):
    from gateway import orchestrator, common
    cfg = {
        "accounts": [{"name": "kg1"}, {"name": "lk4"}],
        "default_model": "m",
        "account_allow_prefixes": ["kg"],
    }
    monkeypatch.setattr(common, "load_config", lambda: cfg)
    monkeypatch.setattr(common, "read_state", lambda: {"backends": [], "pending": {}})

    logs = []
    res = orchestrator.build_cache(account="lk4", log=logs.append)
    assert res is None
    assert any("not permitted" in l for l in logs)


def test_kaggle_run_and_quota_reject_disallowed_prefix(monkeypatch):
    import pytest
    from gateway import kaggle, common
    cfg = {
        "accounts": [{"name": "kg1"}],
        "account_allow_prefixes": ["kg"],
    }
    monkeypatch.setattr(common, "load_config", lambda: cfg)

    rc, out = kaggle.run("lk4", "kernels", "list")
    assert rc == 127
    assert "not permitted by account_allow_prefixes" in out

    with pytest.raises(RuntimeError, match="not permitted by account_allow_prefixes"):
        kaggle.run("lk4", "kernels", "list", check=True)

    q = kaggle.fetch_account_quota("lk4")
    assert q["ok"] is False
    assert "not permitted by account_allow_prefixes" in q["error"]




def test_load_timeout_counts_from_boot_not_from_queue():
    """2026-11-10: three panel launches queued ~10 min at Kaggle, then needed ~26 min to build and load on fresh
    accounts. Timing the load from launch killed two of them a minute before they registered. Load time starts at
    boot; the queue has its own limit."""
    import time as _t
    from gateway import scheduler as sc
    now = _t.time()
    reg = {"m": {"key": "m", "ctx": 8192}}
    pend = {"kg1": {"model": "m", "model_key": "m", "origin": "manual", "started": now - 37 * 60, "boot_at": now - 20 * 60},
            "kg2": {"model": "m", "model_key": "m", "origin": "manual", "started": now - 37 * 60, "boot_at": now - 26 * 60}}
    acts = sc.plan(now, {"autoscale": {"enabled": True, "max_backends": 4}}, reg, "m", [], pend, ["kg3"], {},
                   {"kg1": 20, "kg2": 20, "kg3": 20}, {"models": {}, "backends": {}})
    cancels = {a[1] for a in acts if a[0] == "cancel"}
    assert "kg1" not in cancels          # 37 min since launch but only 20 min since boot: still loading
    assert "kg2" in cancels              # 26 min since boot: past the 25 min load timeout


def test_cold_start_launches_one_backend_however_busy():
    """Nothing serves yet: a retrying agent looks like many busy sessions, but one launch is all that can help
    (2026-11-10: six launches of one model in four minutes)."""
    acts = run(dem=demand({"big": {"sessions": 6, "busy": 6, "waiting": 4}}))
    assert len([a for a in acts if a[0] == "launch"]) == 1
    # surplus launches already queued for it are cancelled down to the most advanced one
    pend = {a: {"model_key": "big", "started": NOW - 120 + i} for i, a in enumerate(["kg1", "kg2", "kg3"])}
    acts = run(pending=pend, dem=demand({"big": {"sessions": 6, "busy": 6, "waiting": 4}}))
    assert sorted(a[1] for a in acts if a[0] == "cancel") == ["kg2", "kg3"] and not [a for a in acts if a[0] == "launch"]


def test_repeat_failure_demotes_below_unfailed_accounts():
    stats = {"kg-a": {"fails": 2, "queue_ewma_s": 3.9, "ready_ewma_s": 429},
             "kg-b": {"fails": 0, "queue_ewma_s": 3.9, "ready_ewma_s": 429},
             "kg-c": {"fails": 1, "queue_ewma_s": 3.9, "ready_ewma_s": 429}}
    scores = {a: scheduler.account_score(a, stats, 30) for a in stats}
    # kg-b (clean) must outrank kg-c (one fail) must outrank kg-a (repeat fail)
    assert scores["kg-b"] > scores["kg-c"] > scores["kg-a"]
    # 30h quota for kg-a must not keep it competitive once it has failed twice in a row
    assert scores["kg-a"] < 0


def test_one_failure_does_not_demote_into_last_resort():
    stats_failed_once = {"kg-a": {"fails": 1, "queue_ewma_s": 3.9, "ready_ewma_s": 429}}
    stats_clean = {"kg-b": {"fails": 0, "queue_ewma_s": 200.0, "ready_ewma_s": 1200}}
    assert scheduler.account_score("kg-a", stats_failed_once, 30) > scheduler.account_score("kg-b", stats_clean, 30)
