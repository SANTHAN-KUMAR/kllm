"""Kernel resource sampler: the fields the panel's Resources tab reads, and download progress reporting."""
import pathlib, sys, types

import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kernel_loader import load_kernel


def kernel():
    return load_kernel("validate-3b")


@pytest.mark.skipif(not pathlib.Path("/proc/stat").exists(), reason="the kernel runs on Kaggle's Linux; it samples /proc")
def test_sample_has_cpu_net_disk_after_two_reads():
    mod = kernel()
    mod._gpu_stats = lambda: []
    first = mod._telemetry_sample()
    assert "cpu_pct" not in first and "disk" in first           # rates need a previous read
    second = mod._telemetry_sample()
    assert 0.0 <= second["cpu_pct"] <= 100.0
    assert second["net_rx_bps"] >= 0 and second["net_tx_bps"] >= 0
    assert second["disk"]["/tmp"]["total"] > 0
    assert "slots" not in second                                 # no llama-server running here


def test_slots_parse_and_download_watch():
    mod = kernel()

    class R:
        def __init__(self, body): self.body = body
        def read(self): return self.body

    import json
    mod.urllib = types.SimpleNamespace(request=types.SimpleNamespace(
        Request=lambda *a, **k: a[0],
        urlopen=lambda req, timeout=0: R(json.dumps([{"id": 0, "n_ctx": 131072, "is_processing": True,
                                                       "next_token": {"n_past": 4000}},
                                                      {"id": 1, "n_ctx": 131072, "n_prompt_tokens": 10,
                                                       "next_token": [{"n_decoded": 5}]}]).encode())))
    assert mod._slots() == [{"id": 0, "n_ctx": 131072, "used": 4000, "busy": True},
                            {"id": 1, "n_ctx": 131072, "used": 15, "busy": False}]

    sent = []
    sizes = iter([0, 0, 2 * 2**30, 8 * 2**30, 8 * 2**30])
    mod._dir_bytes = lambda paths: next(sizes)
    mod._gguf_total_bytes = lambda: 8 * 2**30
    mod.ntfy = lambda ev, **kw: sent.append((ev, kw))
    mod.time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0)
    mod.urllib = types.SimpleNamespace(request=types.SimpleNamespace(
        urlopen=lambda *a, **k: (_ for _ in ()).throw(OSError("not healthy"))))
    srv = types.SimpleNamespace(poll=lambda: None)
    mod._download_watch(srv)
    assert [kw["bytes_done"] for _, kw in sent] == [2 * 2**30, 8 * 2**30]      # 0 B (cached) is never reported
    assert all(ev == "downloading" and kw["bytes_total"] == 8 * 2**30 for ev, kw in sent)


def test_tps_rates_from_metrics():
    import json, types
    mod = kernel()

    class R:
        def __init__(self, b): self.b = b
        def read(self): return self.b.encode()

    body1 = ("llamacpp_prompt_tokens_total 1000\n"
             "llamacpp_prompt_seconds_total 5.0\n"
             "llamacpp_tokens_predicted_total 200\n"
             "llamacpp_tokens_predicted_seconds_total 10.0\n")
    mod.urllib = types.SimpleNamespace(request=types.SimpleNamespace(
        Request=lambda *a, **k: a[0], urlopen=lambda *a, **k: R(body1)))
    m1 = mod._tps()
    assert m1["prefill_tokens_total"] == 1000 and "prefill_tps" not in m1     # first read: no delta yet

    body2 = ("llamacpp_prompt_tokens_total 1250\n"
             "llamacpp_prompt_seconds_total 6.25\n"     # +250 tok / +1.25 s -> 200 t/s
             "llamacpp_tokens_predicted_total 440\n"
             "llamacpp_tokens_predicted_seconds_total 17.0\n")  # +240 tok / +7.0 s -> 34.3 t/s
    mod.urllib = types.SimpleNamespace(request=types.SimpleNamespace(
        Request=lambda *a, **k: a[0], urlopen=lambda *a, **k: R(body2)))
    m2 = mod._tps()
    assert abs(m2["prefill_tps"] - 200.0) < 0.2
    assert abs(m2["decode_tps"] - 240 / 7.0) < 0.2
    assert m2["decode_tokens_total"] == 440
