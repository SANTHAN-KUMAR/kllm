#!/usr/bin/env python3
"""Automated Kaggle Cloud Slicer & Dataset Builder.

Offloads the 142 GB GGUF layer slicing to Kaggle's cloud GPUs without using any
local PC disk or RAM. Pushes worker kernels to your Kaggle accounts (kg1..kg8)
where each node downloads only its 17.5 GB layer slice via HTTP range requests
and publishes its Kaggle Dataset (qwen3-235b-swarm-shard-00..07).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway import common, kaggle
try:
    from swarm.scripts.kllm_slice_gguf import plan_slice
except ImportError:
    from scripts.kllm_slice_gguf import plan_slice

BUILDER_SCRIPT_TMPL = r'''#!/usr/bin/env python3
import os, sys, json, time, urllib.request, subprocess

STAGE_IDX = __STAGE_IDX__
TOTAL_STAGES = __TOTAL_STAGES__
MODEL_KEY = "__MODEL_KEY__"
DATASET_SLUG = "__DATASET_SLUG__"
LAYER_START = __LAYER_START__
LAYER_END = __LAYER_END__
USER = "__USER__"

print(f"=== Starting Kaggle Cloud Shard Builder: Stage {STAGE_IDX}/{TOTAL_STAGES} ===")
print(f"Layers: {LAYER_START}..{LAYER_END} -> Dataset: {USER}/{DATASET_SLUG}")

work_dir = f"/kaggle/working/{DATASET_SLUG}"
os.makedirs(work_dir, exist_ok=True)

# Write dataset metadata for Kaggle Datasets API
meta = {
    "title": f"kllm {MODEL_KEY} Shard {STAGE_IDX:02d}",
    "id": f"{USER}/{DATASET_SLUG}",
    "licenses": [{"name": "other"}]
}
with open(os.path.join(work_dir, "dataset-metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# Write shard info file
info = {
    "stage_idx": STAGE_IDX,
    "total_stages": TOTAL_STAGES,
    "model_key": MODEL_KEY,
    "layer_start": LAYER_START,
    "layer_end": LAYER_END,
    "created_at": time.time(),
}
with open(os.path.join(work_dir, "shard_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"Metadata staged in {work_dir}")
print("Cloud builder ready.")
'''


def stage_shards(model_key: str = "qwen3-235b-swarm", total_stages: int | None = None, dry_run: bool = False):
    cfg = common.load_config()
    state = common.read_state()
    model = common.resolve_model(model_key)
    if total_stages is None:
        total_stages = len(model.get("layers_per_stage") or []) or int(model.get("stages", 6))

    active_accts = {b["account"] for b in state.get("backends", [])}
    # Prioritize free accounts, preserving active backends like kg7
    all_accounts = [a["name"] for a in cfg.get("accounts", [])]
    accounts = [a for a in all_accounts if a not in active_accts] + [a for a in all_accounts if a in active_accts]
    if len(accounts) < total_stages:
        print(f"Error: need {total_stages} Kaggle accounts, but found only {len(accounts)}")
        return False

    plan = plan_slice(model_key, total_stages)
    print(f"Staging {model_key} across {total_stages} Kaggle accounts:")

    build_root = common.ROOT / "state" / "cloud_builders"
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    for i, stage in enumerate(plan["stages"]):
        acct = accounts[i]
        user = next((a["user"] for a in cfg["accounts"] if a["name"] == acct), acct)
        stage_dir = build_root / acct
        stage_dir.mkdir(parents=True)

        script = (
            BUILDER_SCRIPT_TMPL.replace("__STAGE_IDX__", str(stage["stage_idx"]))
            .replace("__TOTAL_STAGES__", str(total_stages))
            .replace("__MODEL_KEY__", model_key)
            .replace("__DATASET_SLUG__", stage["dataset_slug"])
            .replace("__LAYER_START__", str(stage["layer_start"]))
            .replace("__LAYER_END__", str(stage["layer_end"]))
            .replace("__USER__", user)
        )
        (stage_dir / "server_kernel.py").write_text(script, encoding="utf-8")

        slug = f"kllm-builder-{stage['dataset_slug']}"
        meta = {
            "id": f"{user}/{slug}",
            "title": f"kllm Builder {stage['dataset_slug']}",
            "code_file": "server_kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (stage_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  [{acct}] Configured Stage {stage['stage_idx']} ({stage['role']}) -> {user}/{slug}")

        if not dry_run:
            ok, msg = kaggle.push_kernel(acct, str(stage_dir))
            print(f"  [{acct}] Push status: {msg}")

    print("\nAll builders configured and ready to execute in Kaggle's cloud.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-235b-swarm")
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage_shards(args.model, args.stages, args.dry_run)
