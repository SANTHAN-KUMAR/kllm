"""kllm-swarm: Distributed MoE Pipeline & Userspace Packet Switch.

Provides multi-node pipeline parallelism across independent Kaggle GPU kernels,
dynamic speculative drafting, Q8_0 activation quantization, and low-latency WebSocket routing.
"""

from .protocol import (
    CandidateBranch,
    CandidateTree,
    FLAG_ACTIVATION,
    FLAG_EOS,
    FLAG_TOKEN,
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC_HEADER,
    SwarmFrameHeader,
    calculate_moe_expert_amplification,
    dequantize_activations_q8_0,
    is_valid_gcp_zone,
    optimal_speculative_depth,
    pack_activation_frame,
    pack_token_payload,
    quantize_activations_q8_0,
    unpack_activation_frame,
    unpack_token_payload,
    verify_node_latency,
)
from .hub import SwarmPacketSwitch, router

__all__ = [
    "CandidateBranch",
    "CandidateTree",
    "FLAG_ACTIVATION",
    "FLAG_EOS",
    "FLAG_TOKEN",
    "HEADER_SIZE",
    "HEADER_STRUCT",
    "MAGIC_HEADER",
    "SwarmFrameHeader",
    "SwarmPacketSwitch",
    "calculate_moe_expert_amplification",
    "dequantize_activations_q8_0",
    "is_valid_gcp_zone",
    "optimal_speculative_depth",
    "pack_activation_frame",
    "pack_token_payload",
    "quantize_activations_q8_0",
    "router",
    "unpack_activation_frame",
    "unpack_token_payload",
    "verify_node_latency",
]
