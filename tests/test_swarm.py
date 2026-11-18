"""Unit and integration tests for kllm-swarm distributed MoE pipeline protocol."""

import math
import uuid
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import pytest

from gateway import common, hub, relay, swarm
from gateway.swarm import (
    HEADER_SIZE,
    SwarmFrameHeader,
    calculate_moe_expert_amplification,
    dequantize_activations_q8_0,
    is_valid_gcp_zone,
    optimal_speculative_depth,
    pack_activation_frame,
    quantize_activations_q8_0,
    unpack_activation_frame,
    verify_node_latency,
)


def test_swarm_frame_header_pack_unpack():
    req_id = uuid.uuid4()
    header = SwarmFrameHeader(
        req_id=req_id,
        seq_id=42,
        step_id=108,
        stage_idx=2,
        total_stages=6,
        num_tokens=5,
        hidden_dim=5120,
        pos_offset=1024,
        flags=1,
    )
    raw = header.pack()
    assert len(raw) == swarm.HEADER_SIZE
    assert swarm.HEADER_SIZE == 64  # Strictly verify 64-byte binary specification

    unpacked = SwarmFrameHeader.unpack(raw)
    assert unpacked.req_id == req_id
    assert unpacked.seq_id == 42
    assert unpacked.step_id == 108
    assert unpacked.stage_idx == 2
    assert unpacked.total_stages == 6
    assert unpacked.num_tokens == 5
    assert unpacked.hidden_dim == 5120
    assert unpacked.pos_offset == 1024
    assert unpacked.flags == 1


def test_swarm_frame_header_errors():
    with pytest.raises(ValueError, match="less than header size"):
        SwarmFrameHeader.unpack(b"short")

    corrupted = b"XXXX" + b"\x00" * (swarm.HEADER_SIZE - 4)
    with pytest.raises(ValueError, match="Invalid magic header"):
        SwarmFrameHeader.unpack(corrupted)


def test_q8_0_quantize_dequantize_roundtrip():
    # Synthetic activation vector with positive, negative, and zero values
    dim = 5120
    test_values = [math.sin(i * 0.05) * 3.5 for i in range(dim)]

    quantized = quantize_activations_q8_0(test_values, block_size=32)
    # 5120 / 32 = 160 blocks; each block = 2 bytes fp16 + 32 bytes int8 = 34 bytes
    assert len(quantized) == 160 * 34

    dequantized = dequantize_activations_q8_0(quantized, num_elements=dim, block_size=32)
    assert len(dequantized) == dim

    # Max quantization error should be bounded by scale / 2
    max_err = max(abs(orig - deq) for orig, deq in zip(test_values, dequantized))
    assert max_err < 0.05


def test_q8_0_quantize_nan_inf_robustness():
    # Activation vectors with NaN and Infs must not crash the quantizer
    raw_vals = [float("nan"), float("inf"), float("-inf"), 1.5, -2.0, 0.0]
    quantized = quantize_activations_q8_0(raw_vals, block_size=32)
    assert len(quantized) == 34  # 1 block padded to 32

    deq = dequantize_activations_q8_0(quantized, num_elements=len(raw_vals), block_size=32)
    assert len(deq) == len(raw_vals)
    # All dequantized values must be finite
    for val in deq:
        assert math.isfinite(val)


def test_q8_0_quantize_empty_and_block_boundaries():
    assert quantize_activations_q8_0([]) == b""
    assert dequantize_activations_q8_0(b"", num_elements=0) == []

    # Non-multiple of 32 (33 elements -> 2 blocks = 68 bytes)
    vals = [float(i) for i in range(33)]
    quantized = quantize_activations_q8_0(vals, block_size=32)
    assert len(quantized) == 68

    deq = dequantize_activations_q8_0(quantized, num_elements=33, block_size=32)
    assert len(deq) == 33

    with pytest.raises(ValueError, match="block_size must be positive"):
        quantize_activations_q8_0([1.0], block_size=0)


def test_q8_0_dequantize_truncated_raises_value_error():
    # Truncated payload should raise clean ValueError instead of struct.error
    with pytest.raises(ValueError, match="Truncated Q8_0 payload"):
        dequantize_activations_q8_0(b"\x00" * 10, num_elements=32, block_size=32)


def test_frame_packing_roundtrip():
    req_id = uuid.uuid4()
    header = SwarmFrameHeader(
        req_id=req_id,
        seq_id=1,
        step_id=1,
        stage_idx=0,
        total_stages=6,
        num_tokens=6,
        hidden_dim=256,
        pos_offset=0,
    )
    payload = b"test_payload_activation_tensor"
    frame = pack_activation_frame(header, payload)
    assert len(frame) == swarm.HEADER_SIZE + len(payload)

    unpacked_hdr, unpacked_payload = unpack_activation_frame(frame)
    assert unpacked_hdr.req_id == req_id
    assert unpacked_payload == payload


def test_moe_expert_amplification():
    # 1 token -> exactly 1.0 amplification
    assert calculate_moe_expert_amplification(128, 8, 1) == 1.0

    # Qwen3-235B style (128 experts, top 8)
    amp_6 = calculate_moe_expert_amplification(128, 8, 6, routing_correlation=0.45)
    amp_12 = calculate_moe_expert_amplification(128, 8, 12, routing_correlation=0.45)

    # 6 tokens should yield ~2.9x amplification; 12 tokens should yield ~4.8x amplification
    assert 2.5 <= amp_6 <= 3.3
    assert 4.0 <= amp_12 <= 5.5
    assert amp_6 < amp_12


def test_moe_expert_amplification_edge_cases():
    # When top_k >= num_experts, should return 1.0 (real float, never complex)
    amp = calculate_moe_expert_amplification(8, 16, 5)
    assert isinstance(amp, float)
    assert amp == 1.0

    # Boundary conditions
    assert calculate_moe_expert_amplification(0, 8, 5) == 1.0
    assert calculate_moe_expert_amplification(128, 0, 5) == 1.0
    assert calculate_moe_expert_amplification(128, 8, 0) == 1.0


def test_optimal_speculative_depth():
    # Ultra-low latency GCP hub (<= 4ms) should choose K=5 to minimize expert blowout
    assert optimal_speculative_depth(1.5, num_experts=128, top_k=8) == 5
    assert optimal_speculative_depth(1.5, num_experts=256, top_k=8) == 5
    # Moderate latency WAN should choose K=6 or K=7
    assert optimal_speculative_depth(12.0) == 6
    assert optimal_speculative_depth(25.0) == 7
    # High latency Cloudflare WAN should choose K=8 to amortize roundtrips
    assert optimal_speculative_depth(55.0) == 8


def test_gcp_zone_validation():
    assert is_valid_gcp_zone("projects/12345/zones/us-central1-a") is True
    assert is_valid_gcp_zone("us-east1-c") is True
    assert is_valid_gcp_zone("us-central1") is True
    # us-central11 must NOT match us-central1
    assert is_valid_gcp_zone("us-central11-a") is False
    assert is_valid_gcp_zone("projects/12345/zones/europe-west4-b") is False
    assert is_valid_gcp_zone("asia-east1-a") is False
    assert is_valid_gcp_zone("") is False


def test_verify_node_latency():
    assert verify_node_latency(1.8) is True
    assert verify_node_latency(34.9) is True
    assert verify_node_latency(35.0) is True
    assert verify_node_latency(35.1) is False
    assert verify_node_latency(-5.0) is False
    assert verify_node_latency(0.0) is False
    assert verify_node_latency(float("nan")) is False
    assert verify_node_latency(float("inf")) is False


@pytest.mark.anyio
async def test_hub_error_isolation_on_broken_target():
    packet_switch = hub.SwarmPacketSwitch()
    ws0 = AsyncMock()
    ws1 = AsyncMock()
    # Simulate broken socket on stage 1
    ws1.send_bytes.side_effect = RuntimeError("Broken pipe")

    await packet_switch.register_stage(0, 2, ws0)
    await packet_switch.register_stage(1, 2, ws1)

    hdr = SwarmFrameHeader(uuid.uuid4(), 1, 1, 0, 2, 1, 128, 0)
    frame = pack_activation_frame(hdr, b"\x00" * 34)

    # Forwarding to stage 1 should catch error, unregister stage 1, and return None
    res = await packet_switch.forward_frame(frame)
    assert res is None
    # Stage 0 remains registered; broken stage 1 was cleaned up
    assert 0 in packet_switch.stages
    assert 1 not in packet_switch.stages


@pytest.mark.anyio
async def test_hub_malformed_frame_dropped_cleanly():
    packet_switch = hub.SwarmPacketSwitch()
    # Malformed frame should be dropped without raising
    res = await packet_switch.forward_frame(b"XXXX" + b"\x00" * 60)
    assert res is None


def test_hub_status_and_ping_api():
    common.CONFIG_FILE.write_text("relay_host: 127.0.0.1\nrelay_port: 18199\ndefault_model: x\naccounts: []\n", encoding="utf-8")
    relay._cfg["mtime"] = 0
    client = TestClient(relay.app, client=("127.0.0.1", 5))
    try:
        # Ping endpoint
        resp = client.get("/swarm/ping")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

        # Status endpoint
        resp = client.get("/swarm/status")
        assert resp.status_code == 200
        status_data = resp.json()
        assert "total_stages" in status_data
        assert "connected_stages" in status_data
        assert "all_stages_ready" in status_data
    finally:
        common.CONFIG_FILE.unlink(missing_ok=True)
        relay._cfg["mtime"] = 0


def test_models_yaml_contains_swarm_entries():
    models = common.load_models()
    assert "qwen3-235b-swarm" in models
    assert "deepseek-v3-671b-swarm" in models
    assert "minimax-m2-230b-swarm" in models

    qwen_swarm = models["qwen3-235b-swarm"]
    assert qwen_swarm.get("stages") == 6
    assert qwen_swarm.get("runtime") == "swarm"
    assert "-sm" in qwen_swarm.get("args", [])
    assert "tensor" in qwen_swarm.get("args", [])

    dseek_swarm = models["deepseek-v3-671b-swarm"]
    assert dseek_swarm.get("stages") == 8
    assert dseek_swarm.get("runtime") == "swarm"


@pytest.mark.anyio
async def test_hub_dispatch_api_behavior():
    packet_switch = hub.SwarmPacketSwitch()
    # 1. Dispatch without stage 0 should return False or raise
    assert packet_switch.stages.get(0) is None

    # 2. Register mock stage 0
    ws0 = AsyncMock()
    await packet_switch.register_stage(0, 6, ws0)
    assert packet_switch.stages.get(0) is ws0

    # 3. Simulate message forwarding from tail stage to stage 0 and coordinator
    ws_coord = AsyncMock()
    packet_switch.coordinator_ws = ws_coord

    hdr = SwarmFrameHeader(uuid.uuid4(), 1, 1, stage_idx=5, total_stages=6, num_tokens=1, hidden_dim=128, pos_offset=0)
    frame = pack_activation_frame(hdr, b"\x00" * 34)
    next_st = await packet_switch.forward_frame(frame)
    assert next_st == 0
    ws0.send_bytes.assert_awaited_once_with(frame)
    ws_coord.send_bytes.assert_awaited_once_with(frame)


def test_kernel_template_contains_swarm_support():
    from gateway.orchestrator import KROOT
    tmpl = (KROOT / "server_kernel.tmpl.py").read_text(encoding="utf-8")
    assert "start_swarm" in tmpl
    assert "probe_gcp_zone_and_latency" in tmpl
    assert "_SwarmHandle" in tmpl
    assert 'elif RUNTIME == "swarm":' in tmpl
    assert '"-sm", "tensor"' in tmpl


def test_slice_gguf_partition_plan():
    from scripts.kllm_slice_gguf import plan_slice, calculate_layer_partition
    partition = calculate_layer_partition(94, 8)
    assert len(partition) == 8
    assert partition[0] == (0, 12)
    assert partition[-1] == (83, 94)

    plan = plan_slice("qwen3-235b-swarm", total_stages=8)
    assert plan["total_layers"] == 94
    assert plan["total_stages"] == 8
    assert len(plan["stages"]) == 8
    assert plan["stages"][0]["role"] == "head"
    assert plan["stages"][-1]["role"] == "tail"


def test_kernel_template_pack_activation_frame_11_args_and_reserved():
    """Verify server_kernel.tmpl.py pack_activation_frame handles >= 10 args with reserved field."""
    from gateway.orchestrator import KROOT
    code = (KROOT / "server_kernel.tmpl.py").read_text(encoding="utf-8")
    scope = {"struct": __import__("struct"), "math": __import__("math"), "uuid": __import__("uuid")}
    start = code.find("SWARM_MAGIC =")
    end = code.find("class SwarmTokenizer:")
    exec(code[start:end], scope)

    req_id = uuid.uuid4()
    reserved_20 = b"RESERVED_METADATA_20"
    # Call with 11 arguments as intermediate workers and stage 0 do
    frame = scope["pack_activation_frame"](
        req_id, 1, 10, 0, 6, 4, 256, 12, 0, b"TEST_PAYLOAD", reserved_20
    )
    assert len(frame) == 64 + len(b"TEST_PAYLOAD")

    hdr_dict, payload = scope["unpack_activation_frame"](frame)
    assert hdr_dict["req_id"] == req_id
    assert hdr_dict["step_id"] == 10
    assert hdr_dict["reserved"] == reserved_20
    assert payload == b"TEST_PAYLOAD"


def test_swarm_drafter_empty_and_no_duplicate_edge_cases():
    """Verify SwarmDrafter does not spam repetitive single tokens when no matches exist."""
    from gateway.orchestrator import KROOT
    code = (KROOT / "server_kernel.tmpl.py").read_text(encoding="utf-8")
    scope = {"struct": __import__("struct"), "math": __import__("math"), "time": __import__("time")}
    start = code.find("class SwarmDrafter:")
    end = code.find("class SwarmWorker:")
    exec(code[start:end], scope)

    drafter = scope["SwarmDrafter"](k=6)
    # Empty context
    assert drafter.draft([], k=4) == []
    assert drafter.draft_tree([], k=4) == [[]]

    # Single token context
    assert drafter.draft([42], k=4) == []
    assert drafter.draft_tree([42], k=4) == [[]]

    # No duplicate n-grams in context
    no_dups = [100, 200, 300, 400, 500, 600, 700]
    assert drafter.draft(no_dups, k=4) == []
    assert drafter.draft_tree(no_dups, k=4) == [[]]


def test_optimal_speculative_depth_low_acceptance():
    """Verify optimal_speculative_depth drops to 1 on near-zero acceptance rate."""
    # When acceptance is near zero (< 0.20), depth drops to 1 to eliminate MoE expert read amplification
    assert optimal_speculative_depth(hop_rtt_ms=2.0, acceptance_rate=0.10) == 1
    assert optimal_speculative_depth(hop_rtt_ms=20.0, acceptance_rate=0.05) == 1
    # Moderate acceptance (< 0.50) scales down to 2-4
    assert optimal_speculative_depth(hop_rtt_ms=10.0, acceptance_rate=0.40) <= 4
    # High acceptance scales up to 10
    assert optimal_speculative_depth(hop_rtt_ms=35.0, acceptance_rate=0.90) == 10


def test_gateway_tunnel_http2_heartbeat_args(monkeypatch):
    """Verify gateway/tunnel.py passes HTTP/2 multiplexing and heartbeat keepalive flags."""
    from gateway import tunnel
    captured_cmd = []

    stop = tunnel.threading.Event()
    def mock_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        stop.set()  # Signal termination once command has been captured
        class DummyProc:
            stdout = []
            def poll(self): return 0
            def terminate(self): pass
            def wait(self, timeout=None): pass
        return DummyProc()

    monkeypatch.setattr(tunnel.subprocess, "Popen", mock_popen)
    monkeypatch.setattr(tunnel, "ensure_cloudflared", lambda *args: "/fake/cloudflared")
    tunnel.run_forever(cfg_fn=lambda: {"relay_port": 8080}, stop=stop)

    assert "--protocol" in captured_cmd
    assert "http2" in captured_cmd
    assert "--heartbeat-interval" in captured_cmd
    assert "10s" in captured_cmd
    assert "--heartbeat-count" in captured_cmd
    assert "5" in captured_cmd
    assert "--edge-bind-address" in captured_cmd



