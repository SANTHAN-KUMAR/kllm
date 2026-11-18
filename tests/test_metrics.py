"""Metrics aggregation on a fixture log: totals, splits, percentiles, incremental rollup, rotation, CSV."""
import json, os, time
import pytest
from gateway import metrics
from tests import fakelog
from tests.fakelog import make_records, write


@pytest.fixture
def log(tmp_path):
    now = time.time()
    path = tmp_path / "requests.jsonl"
    recs = make_records(now)
    write(path, recs)
    return path, recs, now, tmp_path


def rollup(log):
    path, _, _, tmp = log
    return metrics.Rollup(path, tmp / "rollup.json")


def expected(recs, since):
    rs = [r for r in recs if r["ts"] >= since]
    return rs


def test_totals_match_brute_force_for_24h_and_7d(log):
    path, recs, now, _ = log
    ro = rollup(log)
    for rng, span in (("24h", 86400), ("7d", 7 * 86400)):
        v = ro.query(rng, now=now)
        # the view counts whole hour buckets: compare against the same bucket boundary
        since = int(now // 3600) * 3600 - (span // 3600 - 1) * 3600
        rs = [r for r in recs if r["ts"] >= since]
        t = v["totals"]
        assert t["n"] == len(rs)
        assert t["err"] == sum(1 for r in rs if r["status"] != 200)
        assert t["loop"] == sum(1 for r in rs if r.get("loop_cut"))
        assert t["new_in"] == sum(r.get("new_in", 0) for r in rs)
        assert t["cached_in"] == sum(r.get("cached_in", 0) for r in rs)
        assert t["out"] == sum(r.get("out", 0) for r in rs)
        assert t["think_c"] == sum(r.get("think_chars", 0) for r in rs)
        assert t["text_c"] == sum(r.get("text_chars", 0) for r in rs)
        assert t["cache_hit"] == pytest.approx(t["cached_in"] / (t["new_in"] + t["cached_in"]), abs=1e-4)
        assert sum(m["n"] for m in v["by_model"]) == len(rs)
        assert sum(m["n"] for m in v["by_backend"]) == len(rs)


def test_split_by_model_backend_kind(log):
    _, recs, now, _ = log
    v = rollup(log).query("7d", now=now)
    q36 = next(m for m in v["by_model"] if m["key"] == "qwen3.6-35b-abliterated")
    rs = [r for r in recs if r["model"] == "qwen3.6-35b-abliterated" and r["ts"] >= int(now // 3600) * 3600 - (7 * 24 - 1) * 3600]
    assert q36["n"] == len(rs) and q36["out"] == sum(r["out"] for r in rs)
    assert q36["think_tok"] > 0 and next(m for m in v["by_model"] if "coder" in m["key"])["think_tok"] == 0
    assert {b["key"] for b in v["by_backend"]} == set(fakelog.BACKENDS)
    kinds = {k["key"]: k["n"] for k in v["by_kind"]}
    assert kinds["image"] == 1 and kinds["chat"] == v["totals"]["n"] - 1


def test_hourly_and_daily_series_cover_the_window_and_sum_up(log):
    _, recs, now, _ = log
    v = rollup(log).query("24h", now=now)
    assert len(v["hourly"]) == 24 and len(v["daily"]) == 7
    assert [h["t"] for h in v["hourly"]] == sorted(h["t"] for h in v["hourly"])
    assert v["hourly"][-1]["t"] == int(now // 3600 * 3600)
    assert sum(h["n"] for h in v["hourly"]) == v["totals"]["n"]          # 24h range == hourly window
    day_total = sum(d["n"] for d in rollup(log).query("7d", now=now)["daily"])
    assert day_total == len([r for r in recs if r["ts"] >= v["daily"][0]["t"]])


def test_daily_buckets_follow_timezone_offset(log):
    _, _, now, _ = log
    ro = rollup(log)
    utc = ro.query("7d", now=now, tz_minutes=0)["daily"]
    ist = ro.query("7d", now=now, tz_minutes=330)["daily"]
    assert all(d["t"] % 86400 == 0 for d in utc)
    assert all((d["t"] + 330 * 60) % 86400 == 0 for d in ist)


def test_percentiles_and_ctx_buckets(log):
    _, recs, now, _ = log
    v = rollup(log).query("7d", now=now)
    since = int(now // 3600) * 3600 - (7 * 24 - 1) * 3600
    ttfts = sorted(r["ttft_s"] for r in recs if r["ts"] >= since and r.get("ttft_s") is not None and r["status"] == 200)
    tps = sorted(r["gen_tps"] for r in recs if r["ts"] >= since and r["status"] == 200 and r.get("gen_tps"))
    lt = v["latency"]
    assert lt["ttft"]["n"] == len(ttfts) and lt["gen_tps"]["n"] == len(tps)
    # at most SAMPLE_CAP samples per hour are kept, so the percentiles are approximate
    assert lt["ttft"]["p50"] == pytest.approx(metrics.pct(ttfts, 50), rel=0.12)
    assert lt["ttft"]["p90"] > lt["ttft"]["p50"] and lt["gen_tps"]["p10"] < lt["gen_tps"]["p50"] < lt["gen_tps"]["p90"]
    labels = [b["label"] for b in v["ctx_buckets"]]
    assert labels == sorted(labels, key=[b[2] for b in metrics.CTX_BUCKETS].index)
    small = next(b for b in v["ctx_buckets"] if b["label"] == "<4K")
    big = next(b for b in v["ctx_buckets"] if b["label"] == "128K+")
    assert small["tps_mean"] > big["tps_mean"] and big["ttft_mean"] > small["ttft_mean"]


def test_pct_helper():
    assert metrics.pct([], 50) is None
    assert metrics.pct([1, 2, 3, 4], 50) == 2.5 and metrics.pct([5], 99) == 5


def test_incremental_update_parses_only_new_bytes_and_persists(log):
    path, recs, now, tmp = log
    ro = rollup(log)
    n0 = ro.query("7d", now=now)["totals"]["n"]
    first = ro.parsed_bytes
    assert first == os.path.getsize(path)
    ro.query("24h", now=now)
    assert ro.parsed_bytes == first                      # nothing new -> nothing re-read
    extra = [{"ts": int(now), "backend": "kaggle:kg1", "model": "x", "status": 200, "new_in": 5, "cached_in": 0, "out": 1}]
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(extra[0]) + "\n")
        f.write('{"ts": 1, "half-written')                # an unfinished record is left for next time
    assert ro.query("7d", now=now)["totals"]["n"] == n0 + 1
    assert ro.parsed_bytes - first < 300
    ro._save(force=True)
    again = metrics.Rollup(path, tmp / "rollup.json")    # restart: loads the cache, parses nothing
    assert again.query("7d", now=now)["totals"]["n"] == n0 + 1 and again.parsed_bytes < 10
    with open(path, "a", encoding="utf-8") as f:
        f.write('"}\n')                                  # the record completes later (invalid JSON line: skipped)
    again.update()


def test_rotation_chain_and_keep(tmp_path):
    p = tmp_path / "requests.jsonl"
    assert metrics.rotate_if_needed(p, max_bytes=10) is False        # missing file
    for gen in range(1, 6):
        p.write_text(f"gen{gen}" * 10, encoding="utf-8")
        assert metrics.rotate_if_needed(p, max_bytes=10, keep=3) is True
        assert not p.exists()
    assert (tmp_path / "requests.jsonl.1").read_text(encoding="utf-8").startswith("gen5")
    assert (tmp_path / "requests.jsonl.2").read_text(encoding="utf-8").startswith("gen4")
    assert (tmp_path / "requests.jsonl.3").read_text(encoding="utf-8").startswith("gen3")
    assert not (tmp_path / "requests.jsonl.4").exists()
    p.write_text("tiny", encoding="utf-8")
    assert metrics.rotate_if_needed(p, max_bytes=10) is False


def test_rollup_follows_rotation_without_losing_or_double_counting(log):
    path, recs, now, tmp = log
    ro = rollup(log)
    n0 = ro.query("7d", now=now)["totals"]["n"]
    # records written after the rollup last looked, then the file rotates, then new ones land
    with open(path, "a", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"ts": int(now) - i, "backend": "kaggle:kg2", "model": "m", "status": 200}) + "\n")
    assert metrics.rotate_if_needed(path, max_bytes=1, keep=3)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(2):
            f.write(json.dumps({"ts": int(now), "backend": "kaggle:kg3", "model": "m", "status": 200}) + "\n")
    assert ro.query("7d", now=now)["totals"]["n"] == n0 + 5
    assert ro.query("7d", now=now)["totals"]["n"] == n0 + 5
    fresh = metrics.Rollup(path, tmp / "other.json")                   # a rebuild reads .1 + current
    assert fresh.query("7d", now=now)["totals"]["n"] == n0 + 5


def test_old_hours_are_pruned_and_old_lines_skipped(tmp_path):
    now = time.time()
    p = tmp_path / "requests.jsonl"
    write(p, [{"ts": int(now) - 40 * 86400, "backend": "b", "model": "m", "status": 200},
              {"ts": int(now) - 60, "backend": "b", "model": "m", "status": 200}])
    ro = metrics.Rollup(p, tmp_path / "r.json")
    assert ro.query("14d", now=now)["totals"]["n"] == 1


def test_search_filters_and_order(log):
    path, recs, now, _ = log
    got, scanned = metrics.search({"status": "error"}, limit=1000, path=path)
    assert len(got) == sum(1 for r in recs if r["status"] != 200)
    assert [r["ts"] for r in got] == sorted((r["ts"] for r in got), reverse=True)
    got, _ = metrics.search({"model": "coder", "backend": "kg2", "since": now - 86400}, limit=1000, path=path)
    assert got and all("coder" in r["model"] and r["backend"] == "kaggle:kg2" and r["ts"] >= now - 86400 for r in got)
    got, _ = metrics.search({"loop": True}, limit=1000, path=path)
    assert got and all(r.get("loop_cut") for r in got)
    got, _ = metrics.search({}, limit=5, path=path)
    assert len(got) == 5 and got[0]["ts"] == max(r["ts"] for r in recs)
    got, _ = metrics.search({"q": "sdxl-turbo"}, limit=10, path=path)
    assert len(got) == 1 and got[0]["kind"] == "image"


def test_search_reads_across_rotated_files_and_survives_garbage(tmp_path):
    p = tmp_path / "requests.jsonl"
    (tmp_path / "requests.jsonl.1").write_text(json.dumps({"ts": 1, "model": "old"}) + "\n", encoding="utf-8")
    p.write_text("not json\n" + json.dumps({"ts": 2, "model": "new"}) + "\n\n", encoding="utf-8")
    got, _ = metrics.search({}, limit=10, path=p)
    assert [r["model"] for r in got] == ["new", "old"]


def test_csv_export(log):
    path, recs, now, _ = log
    text = "".join(metrics.csv_rows({"status": "error"}, path=path))
    lines = text.strip().splitlines()
    assert lines[0].split(",")[:4] == ["time", "ts", "backend", "model"]
    assert len(lines) - 1 == sum(1 for r in recs if r["status"] != 200)
    tss = [int(l.split(",")[1]) for l in lines[1:]]
    assert tss == sorted(tss)                                          # oldest first


def test_decision_history_dedupes_and_counts(tmp_path):
    f = tmp_path / "d.jsonl"
    st = {"scheduler": {"log": [{"ts": 100, "action": "launch", "target": "kg1", "model": "m", "why": "demand"},
                                {"ts": 101, "action": "retire", "target": "kg1", "model": None, "why": "idle 50 min"}]}}
    assert metrics.sync_decisions(st, f) == 2
    assert metrics.sync_decisions(st, f) == 0
    st["scheduler"]["log"].append({"ts": 102, "action": "cancel", "target": "kg2", "model": None, "why": "x"})
    assert metrics.sync_decisions(st, f) == 1
    assert [e["action"] for e in metrics.read_decisions(path=f)] == ["launch", "retire", "cancel"]
    assert [e["ts"] for e in metrics.read_decisions(since=101, path=f)] == [101, 102]


def test_compute_report_quota_burn_and_standby():
    from gateway import common
    now = time.time()
    ws = common._week_start(now)
    state = {"backends": [{"account": "kg1"}], "pending": {"kg3": {"boot_at": 1}},
             "quota": {"kg1": {"week": ws, "used": 10 * 3600}, "kg2": {"week": ws - 1, "used": 29 * 3600}},
             "accounts": {"kg2": {"cooldown_until": now + 600, "last_error": "boom", "fails": 2,
                                  "queue_ewma_s": 120, "ready_ewma_s": 480}}}
    cfg = {"accounts": [{"name": "kg1", "user": "a"}, {"name": "kg2", "user": "b", "disabled": True}, {"name": "kg3", "user": "c"}]}
    dec = [{"ts": int(now), "action": "launch", "target": "kg1"}, {"ts": int(now), "action": "retire", "target": "kg1"},
           {"ts": ws - 5, "action": "launch", "target": "kg1"}]
    r = metrics.compute_report(cfg, state, {"kaggle:kg1": 4}, dec, now=now)
    a1, a2, a3 = r["accounts"]
    assert a1["used_h"] == 10 and a1["left_h"] == 20 and a1["running"] and a1["exhaust_in_h"] == 20
    assert a1["busy_h"] == 4 and a1["standby_h"] == 6 and a1["launches"] == 1 and a1["retires"] == 1
    assert a2["used_h"] == 0 and a2["disabled"] and a2["cooldown_left_s"] > 500 and a2["last_error"] == "boom"
    assert a2["queue_s"] == 120 and a2["ready_s"] == 480 and a2["exhaust_in_h"] is None
    assert a3["launching"] and a3["exhaust_in_h"] == 30
    p = r["pool"]
    assert p["burn_now"] == 2 and p["left_h"] == 50 and p["exhaust_in_h"] == 25.0       # disabled account excluded
