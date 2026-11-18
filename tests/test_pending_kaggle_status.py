"""Pending-launch dead-kernel detection: a kernel that dies before ever talking to the gist
(no GPU allocated, instant crash) must show as failed in the panel instead of 'queue active'
forever. Uses the real supervisor bookkeeping; Kaggle's CLI is the only thing stubbed."""
import time

from gateway import common, supervisor


def _sup(monkeypatch, cfg_text="relay_host: 127.0.0.1\nrelay_port: 18199\naccounts:\n  - {name: kg3, user: u1}\n  - {name: kg4, user: u2}\n  - {name: kg5, user: u3}\n"):
    common.CONFIG_FILE.write_text(cfg_text, encoding="utf-8")
    sup = supervisor.Supervisor()
    return sup


def test_kernel_error_marks_pending_failed(monkeypatch):
    sup = _sup(monkeypatch)
    calls = []
    monkeypatch.setattr(sup, "_fail", lambda a, why, minutes=None: calls.append(("fail", a)))
    monkeypatch.setattr(sup, "_stop", lambda a, reason, queued=False: calls.append(("stop", a)))
    monkeypatch.setattr(supervisor.kaggle, "kernel_status",
                        lambda a, r: {"kg3": "ERROR", "kg4": "ERROR", "kg5": "RUNNING"}[a])
    now = time.time()
    state = {"pending": {
        "kg3": {"account": "kg3", "started": now - 400, "since": now - 400, "model_key": "m", "kernel_ref": "u/r1"},
        "kg4": {"account": "kg4", "started": now - 10, "since": now - 10, "model_key": "m", "kernel_ref": "u/r2"},
        "kg5": {"account": "kg5", "started": now - 400, "since": now - 400, "model_key": "m", "kernel_ref": "u/r3"},
    }}
    # kg5 has a fresh gist event, kg4 is too young to judge, kg3 is dead on Kaggle
    by_acct = {"kg5": {"event": "boot", "ts": now}}
    sup._check_dead_pending_launches(state, by_acct, now)
    assert calls == [("fail", "kg3"), ("stop", "kg3")]


def test_kernel_status_is_rate_limited(monkeypatch):
    sup = _sup(monkeypatch)
    n = {"calls": 0}

    def fake_status(a, r):
        n["calls"] += 1
        return "QUEUED"

    monkeypatch.setattr(supervisor.kaggle, "kernel_status", fake_status)
    p = {"account": "kg3", "kernel_ref": "u/r"}
    assert sup._pending_kernel_status(p) == "QUEUED"
    assert sup._pending_kernel_status(p) == "QUEUED"        # cached: no second Kaggle call
    assert n["calls"] == 1


def test_kernel_status_api_error_never_fails_a_launch(monkeypatch):
    sup = _sup(monkeypatch)
    calls = []
    monkeypatch.setattr(sup, "_fail", lambda a, why, minutes=None: calls.append(("fail", a)))
    monkeypatch.setattr(sup, "_stop", lambda a, reason, queued=False: calls.append(("stop", a)))

    def boom(a, r):
        raise RuntimeError("Kaggle API down")

    monkeypatch.setattr(supervisor.kaggle, "kernel_status", boom)
    now = time.time()
    state = {"pending": {
        "kg3": {"account": "kg3", "started": now - 400, "since": now - 400, "model_key": "m", "kernel_ref": "u/r"},
    }}
    sup._check_dead_pending_launches(state, {}, now)
    assert calls == []                                        # UNKNOWN must not punish a live launch
