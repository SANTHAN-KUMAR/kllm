#!/usr/bin/env python3
"""kllm_slice_gguf: Offline & Remote GGUF Layer Slicer for Distributed MoE Pipelines.

Partitions massive GGUF models (e.g. Qwen3-235B, DeepSeek-V3 671B) into N self-contained
layer shards along transformer layer boundaries. Supports streaming remote GGUF weights
via HTTP range requests directly into Kaggle Dataset targets without requiring local storage.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


class ByteSource:
    """Abstract byte source supporting ranged reads over local file or remote HTTP URL."""

    def read_range(self, start: int, length: int) -> bytes:
        raise NotImplementedError

    @property
    def total_size(self) -> int:
        raise NotImplementedError


class FileByteSource(ByteSource):
    def __init__(self, path: str):
        self.path = path
        self._size = os.path.getsize(path)
        self._file = open(path, "rb")

    def read_range(self, start: int, length: int) -> bytes:
        self._file.seek(start)
        return self._file.read(length)

    @property
    def total_size(self) -> int:
        return self._size

    def close(self):
        self._file.close()


class HttpByteSource(ByteSource):
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {"User-Agent": "kllm-slicer/1.0"}
        req = urllib.request.Request(self.url, headers=self.headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            self._size = int(resp.headers.get("Content-Length", 0))

    def read_range(self, start: int, length: int) -> bytes:
        end = start + length - 1
        hdrs = dict(self.headers)
        hdrs["Range"] = f"bytes={start}-{end}"
        req = urllib.request.Request(self.url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) != length:
                raise IOError(f"Short read from {self.url}: expected {length}, got {len(data)}")
            return data

    @property
    def total_size(self) -> int:
        return self._size


def read_gguf_string(stream: io.BytesIO) -> str:
    raw_len = stream.read(8)
    if len(raw_len) < 8:
        raise EOFError("Unexpected EOF reading string length")
    length = struct.unpack("<Q", raw_len)[0]
    data = stream.read(length)
    return data.decode("utf-8", errors="replace")


def write_gguf_string(val: str) -> bytes:
    data = val.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def parse_gguf_header(src: ByteSource) -> Tuple[Dict[str, any], List[Dict[str, any]], int]:
    """Parse GGUF metadata and tensor index from the start of a byte source."""
    # Read the first 32 MB which is almost always sufficient to cover all metadata
    initial_read_size = min(32 * 1024 * 1024, src.total_size)
    chunk = src.read_range(0, initial_read_size)
    stream = io.BytesIO(chunk)

    magic = stream.read(4)
    if magic != GGUF_MAGIC:
        raise ValueError(f"Invalid GGUF magic: {magic!r}")

    version = struct.unpack("<I", stream.read(4))[0]
    tensor_count = struct.unpack("<Q", stream.read(8))[0]
    metadata_kv_count = struct.unpack("<Q", stream.read(8))[0]

    metadata: Dict[str, any] = {}
    for _ in range(metadata_kv_count):
        k = read_gguf_string(stream)
        v_type = struct.unpack("<I", stream.read(4))[0]
        v = _read_gguf_val(stream, v_type)
        metadata[k] = v

    tensors: List[Dict[str, any]] = []
    for _ in range(tensor_count):
        t_name = read_gguf_string(stream)
        n_dims = struct.unpack("<I", stream.read(4))[0]
        dims = struct.unpack(f"<{n_dims}Q", stream.read(8 * n_dims))
        t_type = struct.unpack("<I", stream.read(4))[0]
        offset = struct.unpack("<Q", stream.read(8))[0]
        tensors.append({
            "name": t_name,
            "n_dims": n_dims,
            "dims": dims,
            "type": t_type,
            "offset": offset,
        })

    alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    cur_pos = stream.tell()
    # Header padded to alignment
    header_padding = (alignment - (cur_pos % alignment)) % alignment
    tensor_data_start = cur_pos + header_padding
    return metadata, tensors, tensor_data_start


def _read_gguf_val(stream: io.BytesIO, v_type: int) -> any:
    if v_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", stream.read(1))[0]
    elif v_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", stream.read(1))[0]
    elif v_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", stream.read(2))[0]
    elif v_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", stream.read(2))[0]
    elif v_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", stream.read(4))[0]
    elif v_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", stream.read(4))[0]
    elif v_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", stream.read(4))[0]
    elif v_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", stream.read(1))[0])
    elif v_type == GGUF_TYPE_STRING:
        return read_gguf_string(stream)
    elif v_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", stream.read(8))[0]
    elif v_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", stream.read(8))[0]
    elif v_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", stream.read(8))[0]
    elif v_type == GGUF_TYPE_ARRAY:
        item_type = struct.unpack("<I", stream.read(4))[0]
        count = struct.unpack("<Q", stream.read(8))[0]
        return [_read_gguf_val(stream, item_type) for _ in range(count)]
    else:
        raise ValueError(f"Unknown GGUF value type {v_type}")


def calculate_layer_partition(total_layers: int, num_stages: int) -> List[Tuple[int, int]]:
    """Partition `total_layers` contiguously into `num_stages` intervals [start, end)."""
    base = total_layers // num_stages
    rem = total_layers % num_stages
    ranges = []
    idx = 0
    for i in range(num_stages):
        size = base + (1 if i < rem else 0)
        ranges.append((idx, idx + size))
        idx += size
    return ranges


def filter_stage_tensors(
    tensors: List[Dict[str, any]],
    stage_idx: int,
    total_stages: int,
    layer_start: int,
    layer_end: int,
) -> List[Dict[str, any]]:
    """Select tensors belonging to this pipeline stage."""
    pat_layer = re.compile(r"^blk\.(\d+)\.")
    selected = []

    for t in tensors:
        name = t["name"]
        m = pat_layer.match(name)
        if m:
            layer_id = int(m.group(1))
            if layer_start <= layer_id < layer_end:
                selected.append(dict(t))
        else:
            # Non-layer tensors: embeddings go to Stage 0; output norm/head go to Tail Stage
            if stage_idx == 0 and any(k in name for k in ("token_embd", "position_embd")):
                selected.append(dict(t))
            elif stage_idx == total_stages - 1 and any(k in name for k in ("output_norm", "output.")):
                selected.append(dict(t))

    return selected


def plan_slice(model_key: str, total_stages: Optional[int] = None) -> Dict[str, any]:
    """Generate partition plan for a model."""
    try:
        from gateway import common
        model_info = common.resolve_model(model_key)
    except Exception:
        model_info = {}

    total_layers = model_info.get("total_layers")
    if total_layers is None:
        layer_counts = {
            "qwen3-235b-swarm": 94,
            "deepseek-v3-671b-swarm": 61,
            "minimax-m2-230b-swarm": 80,
        }
        total_layers = layer_counts.get(model_key, 94)

    l_per_stage = model_info.get("layers_per_stage")
    if total_stages is None:
        total_stages = len(l_per_stage) if l_per_stage else int(model_info.get("stages", 6))

    if l_per_stage and len(l_per_stage) == total_stages:
        partition = []
        cur = 0
        for sz in l_per_stage:
            partition.append((cur, cur + sz))
            cur += sz
    else:
        partition = calculate_layer_partition(total_layers, total_stages)

    stages = []
    for idx, (s, e) in enumerate(partition):
        stages.append({
            "stage_idx": idx,
            "layer_start": s,
            "layer_end": e,
            "num_layers": e - s,
            "role": "head" if idx == 0 else ("tail" if idx == total_stages - 1 else "worker"),
            "dataset_slug": f"{model_key}-shard-{idx:02d}",
        })
    return {
        "model_key": model_key,
        "total_layers": total_layers,
        "total_stages": total_stages,
        "stages": stages,
    }


def export_stage_shard(
    src: ByteSource,
    out_path: str,
    stage_idx: int,
    total_stages: int,
    model_key: str = "qwen3-235b-swarm",
) -> int:
    """Slice and write a self-contained GGUF shard containing only the assigned layers."""
    metadata, all_tensors, tensor_data_start = parse_gguf_header(src)
    plan = plan_slice(model_key, total_stages)
    stage_info = plan["stages"][stage_idx]
    layer_start = stage_info["layer_start"]
    layer_end = stage_info["layer_end"]

    selected = filter_stage_tensors(all_tensors, stage_idx, total_stages, layer_start, layer_end)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)

    # Prepare header in memory
    hdr_stream = io.BytesIO()
    hdr_stream.write(GGUF_MAGIC)
    hdr_stream.write(struct.pack("<I", GGUF_VERSION))
    hdr_stream.write(struct.pack("<Q", len(selected)))

    # Basic metadata
    meta_kvs = {
        "general.architecture": metadata.get("general.architecture", "qwen2moe"),
        "kllm.stage_idx": stage_idx,
        "kllm.total_stages": total_stages,
        "kllm.layer_start": layer_start,
        "kllm.layer_end": layer_end,
    }
    hdr_stream.write(struct.pack("<Q", len(meta_kvs)))
    for k, v in meta_kvs.items():
        hdr_stream.write(write_gguf_string(k))
        if isinstance(v, int):
            hdr_stream.write(struct.pack("<I", GGUF_TYPE_INT32))
            hdr_stream.write(struct.pack("<i", v))
        else:
            hdr_stream.write(struct.pack("<I", GGUF_TYPE_STRING))
            hdr_stream.write(write_gguf_string(str(v)))

    # Compute tensor offsets
    cur_offset = 0
    tensor_headers = []
    tensor_payload_slices = []

    for t in selected:
        tensor_headers.append({
            "name": t["name"],
            "n_dims": t["n_dims"],
            "dims": t["dims"],
            "type": t["type"],
            "offset": cur_offset,
        })
        # Estimate size or read raw data
        # Alignment padding for tensor
        pad = (alignment - (cur_offset % alignment)) % alignment
        cur_offset += pad

    for th in tensor_headers:
        hdr_stream.write(write_gguf_string(th["name"]))
        hdr_stream.write(struct.pack("<I", th["n_dims"]))
        hdr_stream.write(struct.pack(f"<{th['n_dims']}Q", *th["dims"]))
        hdr_stream.write(struct.pack("<I", th["type"]))
        hdr_stream.write(struct.pack("<Q", th["offset"]))

    cur_pos = hdr_stream.tell()
    pad = (alignment - (cur_pos % alignment)) % alignment
    hdr_stream.write(b"\x00" * pad)

    with open(out_path, "wb") as f_out:
        f_out.write(hdr_stream.getvalue())
        # Copy tensor data slices
        for t in selected:
            # Type element size heuristic
            elem_count = 1
            for d in t["dims"]:
                elem_count *= d
            t_size = max(1, elem_count // 2)  # 4-bit quantization default
            raw_data = src.read_range(tensor_data_start + t["offset"], min(t_size, src.total_size - (tensor_data_start + t["offset"])))
            f_out.write(raw_data)
            pad = (alignment - (len(raw_data) % alignment)) % alignment
            if pad:
                f_out.write(b"\x00" * pad)

    return os.path.getsize(out_path)


def main():
    parser = argparse.ArgumentParser(description="kllm GGUF Layer Slicer")
    parser.add_argument("--model", default="qwen3-235b-swarm", help="Target model key")
    parser.add_argument("--stages", type=int, default=8, help="Number of stages (default 8)")
    parser.add_argument("--plan", action="store_true", help="Print shard partition plan")
    parser.add_argument("--src", help="Source GGUF file path or HTTP URL")
    parser.add_argument("--export-stage", type=int, help="Export specific stage index")
    parser.add_argument("--out", help="Output path for exported shard")
    args = parser.parse_args()

    plan = plan_slice(args.model, args.stages)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return

    if args.src and args.export_stage is not None and args.out:
        src = HttpByteSource(args.src) if args.src.startswith("http") else FileByteSource(args.src)
        bytes_written = export_stage_shard(src, args.out, args.export_stage, args.stages, args.model)
        print(f"Exported Stage {args.export_stage} to {args.out} ({bytes_written} bytes)")
        return

    print(f"Partition Plan for {args.model} ({args.stages} stages):")
    for s in plan["stages"]:
        print(f"  Stage {s['stage_idx']:2d} [{s['role']:6s}]: Layers {s['layer_start']:2d}..{s['layer_end']-1:2d} ({s['num_layers']} layers) -> {s['dataset_slug']}")


if __name__ == "__main__":
    main()
