import time
from gateway import common


def test_quota_left_is_zero_while_kaggle_reports_weekly_cap():
    s = {"quota": {}, "accounts": {"kg4": {"quota_exhausted_until": time.time() + 3600}}}
    assert common.quota_left_h("kg4", s) == 0.0
    # The local estimate still applies to other accounts and after the reset passes.
    assert common.quota_left_h("kg5", s) == common.QUOTA_LIMIT / 3600
    s["accounts"]["kg4"]["quota_exhausted_until"] = time.time() - 1
    assert common.quota_left_h("kg4", s) == common.QUOTA_LIMIT / 3600


def test_compute_report_shows_zero_left_while_kaggle_reports_weekly_cap():
    import time as _time
    from gateway import common, metrics
    now = _time.time()
    ws = common._week_start(now)
    state = {"backends": [], "pending": {},
             "quota": {"kg1": {"week": ws, "used": 7 * 3600}},
             "accounts": {"kg1": {"quota_exhausted_until": now + 3600}}}
    cfg = {"accounts": [{"name": "kg1", "user": "a"}]}
    r = metrics.compute_report(cfg, state, {}, [], now=now)
    row = r["accounts"][0]
    assert row["left_h"] == 0.0
    assert r["pool"]["left_h"] == 0.0
    # after the exhaustion window passes, the local estimate applies again
    state["accounts"]["kg1"]["quota_exhausted_until"] = now - 1
    r = metrics.compute_report(cfg, state, {}, [], now=now)
    assert r["accounts"][0]["left_h"] == common.QUOTA_LIMIT / 3600 - 7




def test_mark_quota_exhausted_sets_flag_and_zeroes_left_h():
    import time as _time
    from gateway import common
    name = "kg_unit_test_flag"
    until = common.mark_quota_exhausted(name)
    try:
        s = common.read_state()
        st = (s.get("accounts") or {}).get(name) or {}
        assert st.get("quota_exhausted_until", 0) > _time.time()
        assert st["quota_exhausted_until"] == int(until)
        assert common.quota_left_h(name, s) == 0.0
    finally:
        with common.mutate_state() as s:
            (s.get("accounts") or {}).pop(name, None)
