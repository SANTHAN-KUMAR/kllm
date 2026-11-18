"""Unit tests for double-quantized NF4 scale dequantization and nibble unpacking.

Tests the mathematical correctness of:
1. Double-quantized NF4 absmax dequantization:
   absmax = (qmap[abs_idx].view(n_blocks, b_sz) * n_abs.view(n_blocks, 1)).flatten()
2. High-nibble-first unpacking:
   high = (b >> 4) & 0x0F
   low  = b & 0x0F
   unpacked = [high, low]
"""

import numpy as np
import pytest

NF4_TABLE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


def test_nf4_nibble_unpacking_order():
    """Verify that bitsandbytes packs high nibble first, low nibble second."""
    # Byte 0x12 -> high nibble 1, low nibble 2
    # Byte 0x34 -> high nibble 3, low nibble 4
    # Byte 0xFA -> high nibble 15 (0xF), low nibble 10 (0xA)
    b = np.array([0x12, 0x34, 0xFA], dtype=np.uint8)

    high = ((b >> 4) & 0x0F).astype(np.int64)
    low = (b & 0x0F).astype(np.int64)
    idx = np.stack([high, low], axis=-1).flatten()

    expected = np.array([1, 2, 3, 4, 15, 10], dtype=np.int64)
    np.testing.assert_array_equal(idx, expected)

    # Verify that the old buggy unpacking (low first) produced incorrect ordering
    buggy_idx = np.stack([low, high], axis=-1).flatten()
    assert not np.array_equal(buggy_idx, expected)
    np.testing.assert_array_equal(buggy_idx, np.array([2, 1, 4, 3, 10, 15]))


def test_nf4_dequant_value_lookup():
    """Verify NF4 codebook lookup for unpacked nibbles."""
    nf4_t = np.array(NF4_TABLE, dtype=np.float32)

    # Byte with high=0 (val=-1.0), low=15 (val=1.0) -> 0x0F
    # Byte with high=7 (val=0.0), low=8 (val=0.079580...) -> 0x78
    packed = np.array([0x0F, 0x78], dtype=np.uint8)
    high = ((packed >> 4) & 0x0F).astype(np.int64)
    low = (packed & 0x0F).astype(np.int64)
    idx = np.stack([high, low], axis=-1).flatten()

    vals = nf4_t[idx]
    assert len(vals) == 4
    assert vals[0] == -1.0
    assert vals[1] == 1.0
    assert vals[2] == 0.0
    assert abs(vals[3] - 0.0795803) < 1e-6


def test_double_quantized_nf4_scale_dequantization_shape():
    """Verify nested absmax dequantization broadcasting across blocks."""
    # 256 codebook entries for 8-bit scale quantization
    qmap = np.linspace(-1.0, 1.0, 256, dtype=np.float32)

    # 4 outer blocks, each with nested_absmax scale
    n_blocks = 4
    b_sz = 64  # 64 scales per block -> 256 total scales
    n_abs = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)  # shape (4,)

    # 256 scale indices (pointing into qmap)
    abs_idx = np.random.randint(0, 256, size=n_blocks * b_sz, dtype=np.int64)

    # The fixed implementation:
    # absmax = (qmap[abs_idx].view(n_blocks, b_sz) * n_abs.view(n_blocks, 1)).flatten()
    dequant_fixed = (qmap[abs_idx].reshape(n_blocks, b_sz) * n_abs.reshape(n_blocks, 1)).flatten()
    assert dequant_fixed.shape == (n_blocks * b_sz,)

    # Verify each block i was scaled specifically by n_abs[i]
    for i in range(n_blocks):
        block_slice = dequant_fixed[i * b_sz : (i + 1) * b_sz]
        expected_block = qmap[abs_idx[i * b_sz : (i + 1) * b_sz]] * n_abs[i]
        np.testing.assert_allclose(block_slice, expected_block, rtol=1e-5)

    # Verify that the buggy unshaped multiplication fails (shape mismatch)
    with pytest.raises(ValueError):
        _ = qmap[abs_idx] * n_abs  # shape (256,) * (4,) cannot broadcast in numpy


def test_double_quantized_nf4_single_block_boundary():
    """Boundary test: 1 outer block with block size 16."""
    qmap = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    n_blocks = 1
    b_sz = 16
    n_abs = np.array([2.5], dtype=np.float32)
    abs_idx = np.array([1, 2, 3, 0] * 4, dtype=np.int64)

    dequant = (qmap[abs_idx].reshape(n_blocks, b_sz) * n_abs.reshape(n_blocks, 1)).flatten()
    assert len(dequant) == 16
    expected = qmap[abs_idx] * 2.5
    np.testing.assert_allclose(dequant, expected, rtol=1e-5)


def test_full_dequant_weight_reconstruction():
    """End-to-end simulation of _get_dequant with double quantization."""
    # 64 weights = 32 packed bytes
    num_weights = 64
    block_size = 64
    packed_bytes = np.array([0x0F] * 32, dtype=np.uint8)  # Each byte: high=0, low=15 -> [-1.0, 1.0]

    # Double-quantized scales: 1 scale for the 64 weights
    # absmax = qmap[abs_idx] * n_abs
    qmap = np.array([0.5], dtype=np.float32)
    n_abs = np.array([0.8], dtype=np.float32)
    abs_idx = np.array([0], dtype=np.int64)

    n_blocks = n_abs.shape[0]
    b_sz = abs_idx.size // n_blocks
    absmax = (qmap[abs_idx].reshape(n_blocks, b_sz) * n_abs.reshape(n_blocks, 1)).flatten()
    assert absmax[0] == pytest.approx(0.4)

    # Dequantize NF4
    b = packed_bytes.flatten()
    high = ((b >> 4) & 0x0F).astype(np.int64)
    low = (b & 0x0F).astype(np.int64)
    idx = np.stack([high, low], axis=-1).flatten()

    nf4_t = np.array(NF4_TABLE, dtype=np.float32)
    vals = nf4_t[idx].reshape(-1, block_size)
    scale = absmax.reshape(-1, 1)
    reconstructed = (vals * scale).flatten()

    assert reconstructed.shape == (64,)
    # high was 0 (-1.0), low was 15 (1.0), scale was 0.4
    # Even indices should be -0.4, odd indices should be +0.4
    np.testing.assert_allclose(reconstructed[0::2], -0.4, rtol=1e-5)
    np.testing.assert_allclose(reconstructed[1::2], 0.4, rtol=1e-5)
