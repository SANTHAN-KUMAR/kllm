"""Static checks for the ComfyUI video runtime (kernel/comfy_video.py + kernel/comfy_kllm_nodes.py).

These run without ComfyUI/torch: they catch the bugs that otherwise only surface as an HTTP 400 from ComfyUI
on a Kaggle GPU 20 minutes into a run (an unregistered custom node, a dangling node link in a graph).
"""
import ast
import itertools
import re
import types
from pathlib import Path

import pytest

KROOT = Path(__file__).resolve().parents[1] / "kernel"


def _module():
    mod = types.ModuleType("kllm_comfy_video_test")
    exec(compile((KROOT / "comfy_video.py").read_text(encoding="utf-8"), "comfy_video.py", "exec"), mod.__dict__)
    return mod


def _runtime(mod, ray):
    rt = mod.Runtime({"opts": {"raylight": ray}}, lambda *a: None, lambda *a, **k: None)
    rt.comfy = "/tmp/comfy"
    return rt


def _dangling(g):
    out = []
    for nid, node in g.items():
        for key, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], int) and val[0] not in g:
                out.append((nid, key, val[0]))
    return out


def test_every_kllm_node_class_is_registered_and_every_used_node_exists():
    src = (KROOT / "comfy_kllm_nodes.py").read_text(encoding="utf-8")
    registered = set(re.findall(r'"(Kllm\w+)":\s*Kllm\w+', src))
    classes = {n.name for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name.startswith("Kllm")}
    used = set(re.findall(r'"class_type":\s*"(Kllm\w+)"', (KROOT / "comfy_video.py").read_text(encoding="utf-8")))
    assert classes <= registered, f"defined but not registered: {classes - registered}"
    assert used <= registered, f"graphs use unregistered nodes: {used - registered}"


@pytest.mark.parametrize("ray", [True, False])
def test_graphs_have_no_dangling_links(ray):
    mod = _module()
    rt = _runtime(mod, ray)
    for windows, quality in itertools.product((1, 2, 3), ("best", "fast")):
        g = rt.g_i2v("t", "/tmp/comfy/input/r.png", "p", 480, 832, windows, quality, 1, {})
        assert not _dangling(g), ("i2v", windows, quality, _dangling(g))
    for windows in (1, 2, 3):
        g = rt.g_replace("t", "/tmp/r.png", "v.mp4", "p", "man in blue shirt", 480, 832, windows, 1, {})
        assert not _dangling(g), ("replace", windows, _dangling(g))
    for quality in ("best", "fast"):
        g = rt.g_recreate("t", "/tmp/r.png", "v.mp4", "p", 480, 832, quality, 1, {})
        assert not _dangling(g), ("recreate", quality, _dangling(g))


def test_replace_protects_other_people_and_uses_no_limb_capsules():
    rt = _runtime(_module(), True)
    g = rt.g_replace("t", "/tmp/r.png", "v.mp4", "p", "man", 480, 832, 1, 1, {})
    assert g["grow"]["inputs"]["mask"] == ["prot", 0]
    assert g["cm"]["inputs"]["limb_scale"] == 0.0


def test_size_buckets_follow_orientation():
    mod = _module()
    assert mod.Runtime._grid("480p", 0.56) == (480, 832)
    assert mod.Runtime._grid("480p", 1.78) == (832, 480)
    assert mod.Runtime._grid("720p", 0.56) == (720, 1280)
    assert mod.Runtime._grid("720x1280", 2.0) == (720, 1280)


def test_long_i2v_uses_svi_loras_and_anchor():
    rt = _runtime(_module(), True)
    g = rt.g_i2v("t", "/tmp/comfy/input/r.png", "p", 480, 832, 3, "best", 1, {})
    loras = {n["inputs"]["lora_name"] for n in g.values() if n["class_type"] == "RayLoraLoader"}
    assert any("SVI_v2_PRO" in name for name in loras)
    assert sum(1 for n in g.values() if n["class_type"] == "WanImageToVideoSVIPro") == 3


def test_svi_lora_is_on_in_every_phase_including_the_undistilled_one():
    """Regression (e19/e20): 3-phase + SVI dropped the SVI LoRA in the CFG phase -> output collapsed to noise."""
    rt = _runtime(_module(), True)
    g = rt.g_i2v("t", "/tmp/comfy/input/r.png", "p", 480, 832, 2, "best", 1, {})
    for nid, n in g.items():
        if n["class_type"] != "XFuserKSamplerAdvanced":
            continue
        loader = g[n["inputs"]["ray_actors"][0]]["inputs"]["ray_actors"][0]          # m_<tag> -> u_<tag>
        lora_ref = g[loader]["inputs"].get("lora")
        names = []
        while lora_ref:
            node = g[lora_ref[0]]
            names.append(node["inputs"]["lora_name"])
            lora_ref = node["inputs"].get("prev_ray_lora")
        assert any("SVI_v2_PRO" in x for x in names), (nid, names)


def test_mode_switch_resets_ray_workers_and_uses_a_fresh_namespace():
    """Regression (P1): Ray workers keep every loaded model; switching mode without a reset OOMs."""
    mod = _module()
    rt = _runtime(mod, True)
    ran = []
    rt.run_graph = lambda g, timeout=7200: ran.append(g)
    rt.free = lambda: ran.append("free")
    rt.reset_ray()
    assert ran[0]["kill"]["class_type"] == "RayKillInit" and ran[0]["ri"]["inputs"]["ray_cluster_namespace"] == "kllm0"
    assert ran[1] == "free" and rt.ray_gen == 1
    g = rt.g_i2v("t", "/tmp/comfy/input/r.png", "p", 480, 832, 1, "best", 1, {})
    assert g["ri"]["inputs"]["ray_cluster_namespace"] == "kllm1"


def test_optional_skirt_and_hybrid_control_are_wired():
    rt = _runtime(_module(), True)
    g = rt.g_replace("t", "/tmp/r.png", "v.mp4", "p", "man", 480, 832, 1, 1, {"outfit": "long"})
    assert g["grow"]["inputs"]["mask"] == ["skirt", 0] and g["skirt"]["class_type"] == "KllmSkirtMask"
    assert not _dangling(g)
    g = rt.g_recreate("t", "/tmp/r.png", "v.mp4", "p", 480, 832, "best", 1, {"target": "man in blue shirt"})
    assert g["vc"]["inputs"]["control_video"] == ["hy", 0] and not _dangling(g)
    g = rt.g_recreate("t", "/tmp/r.png", "v.mp4", "p", 480, 832, "best", 1, {})
    assert g["vc"]["inputs"]["control_video"] == ["dr", 0]


def test_comfy_client_accepts_empty_json_bodies(monkeypatch):
    """Regression (P5): POST /free returns an empty 200 body; parsing it crashed every request after a reset."""
    mod = _module()
    rt = _runtime(mod, True)

    class R:
        def read(self):
            return b""
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: R())
    assert rt._req("/free", {"unload_models": True}) == {}


def test_edit_modes_run_native_even_when_raylight_is_on():
    """Regression (P4/P6): replace/recreate on Raylight exhausted the 30 GB host RAM on every request."""
    rt = _runtime(_module(), True)
    for g in (rt.g_replace("t", "/tmp/r.png", "v.mp4", "p", "man", 480, 832, 1, 1, {}),
              rt.g_recreate("t", "/tmp/r.png", "v.mp4", "p", 480, 832, "best", 1, {})):
        kinds = {n["class_type"] for n in g.values()}
        assert "RayInitializer" not in kinds and "XFuserKSamplerAdvanced" not in kinds and "KSamplerAdvanced" in kinds
        assert not _dangling(g)
    g = rt.g_i2v("t", "/tmp/comfy/input/r.png", "p", 480, 832, 1, "best", 1, {})
    assert "XFuserKSamplerAdvanced" in {n["class_type"] for n in g.values()}


def test_model_set_change_restarts_comfy_when_a_process_is_running():
    """Regression (P7/P8): after a native edit, RayKillInit + /free left ~3-4 GB and long Raylight i2v OOM'd."""
    mod = _module()
    rt = _runtime(mod, True)
    calls = []
    rt.proc = object()
    rt.restart_comfy = lambda: calls.append("restart")
    rt.reset_ray()
    assert calls == ["restart"]


def test_staged_replace_splits_preprocessing_from_two_gpu_sampling():
    rt = _runtime(_module(), True)
    for windows in (1, 2):
        prep = rt.g_replace_prep("t", "v.mp4", "man", 480, 832, windows, {})
        kinds = {n["class_type"] for n in prep.values()}
        assert "SAM3_VideoTrack" in kinds and "KllmPoseFromMask" in kinds and not _dangling(prep)
        assert not kinds & {"RayInitializer", "XFuserKSamplerAdvanced", "KSamplerAdvanced", "CLIPLoader"}
        saved = {n["inputs"]["filename_prefix"] for n in prep.values() if n["class_type"] == "SaveImage"}
        assert saved == {"t_prep/pose/f", "t_prep/face/f", "t_prep/mask/f"}
        g = rt.g_replace("t", "/tmp/r.png", "v.mp4", "p", "man", 480, 832, windows, 1, {}, staged="/o/t_prep")
        kinds = {n["class_type"] for n in g.values()}
        assert "XFuserKSamplerAdvanced" in kinds and not kinds & {"SAM3_VideoTrack", "OnnxDetectionModelLoader"}
        assert not _dangling(g)
        assert g["wa0"]["inputs"]["pose_video"] == ["ldp", 0] and g["wa0"]["inputs"]["face_video"] == ["ldf", 0]


def test_main_comfy_is_lean_when_raylight_holds_the_model():
    """Regression (P12): pinned memory + RAM cache in the main process pushed staged replace over 30 GB."""
    mod = _module()
    assert _runtime(mod, True).comfy_mem_flags() == ["--disable-pinned-memory", "--cache-none"]
    assert _runtime(mod, False).comfy_mem_flags() == ["--cache-none"]



RAY_WORKER_STUB = """import os
import gc
import torch


class RayWorker:
    def _free_current_model(self):
        self.model = None

    def clear_sampling_vram(self):
        \"\"\"Release worker-side CUDA memory after a Ray sampling node finishes.\"\"\"
        import comfy.model_management as comfy_model_management

        comfy_model_management.unload_all_models()
        self.model.unpatch_model(device_to=getattr(self.model, "offload_device", None))
"""


def test_raylight_patch_drops_weights_instead_of_offloading(tmp_path):
    """Regression (P14 memtrace): clear_sampling_vram copied ~11 GB/worker of weights to host RAM -> OOM kill."""
    d = tmp_path / "raylight" / "src" / "raylight" / "distributed_worker"
    d.mkdir(parents=True)
    (d / "ray_worker.py").write_text(RAY_WORKER_STUB, encoding="utf-8")
    rt = _runtime(_module(), True)
    assert rt.patch_raylight(str(tmp_path / "raylight"))
    out = (d / "ray_worker.py").read_text(encoding="utf-8")
    compile(out, "ray_worker.py", "exec")
    body = out.split("def clear_sampling_vram(self):", 1)[1]
    assert out.count(rt.RAY_DROP_MARK) == 1 and body.index("KLLM_RAY_DROP") < body.index("unload_all_models")
    assert body.index('"""', body.index('"""') + 3) < body.index("KLLM_RAY_DROP")      # inserted after the docstring
    assert rt.patch_raylight(str(tmp_path / "raylight"))                               # idempotent
    assert (d / "ray_worker.py").read_text(encoding="utf-8").count(rt.RAY_DROP_MARK) == 1


def test_anchored_recreate_keyframe_and_wiring():
    """Anchored recreate (e31c-e31i): Qwen keyframe edits only a crop around the SAM3 target and pastes back inside
    the grown mask; the VACE stage keeps that keyframe as frame 0 and drops the (person-unbound) reference image."""
    rt = _runtime(_module(), False)
    kf = rt.g_keyframe("t", "/tmp/r.png", "v.mp4", "man in yellow shirt", 480, 832, 1, {})
    assert not _dangling(kf)
    assert kf["save"]["inputs"]["images"] == ["paste", 0] and kf["paste"]["inputs"]["mask"] == ["grow", 0]
    assert kf["qpos"]["inputs"]["image1"] == ["crop", 0] and "man in yellow shirt" in kf["qpos"]["inputs"]["prompt"]
    g = rt.g_recreate("t", "/tmp/r.png", "v.mp4", "p", 480, 832, "best", 1, {"target": "man in yellow shirt"}, anchor_image="t_kf.png")
    assert not _dangling(g)
    vc = g["vc"]["inputs"]
    assert vc["control_video"] == ["cvid", 0] and vc["control_masks"] == ["cmask", 0] and "reference_image" not in vc
    assert g["crest"]["inputs"]["image"] == ["hy", 0]                     # hybrid depth+pose control for frames 1..
    plain = rt.g_recreate("t", "/tmp/r.png", "v.mp4", "p", 480, 832, "best", 1, {})
    assert "reference_image" in plain["vc"]["inputs"] and "cvid" not in plain


def test_quality_real_uses_undistilled_q6_with_svi_even_for_one_window():
    """quality=real (L1/c2): 20 undistilled steps, CFG 3.5 both experts, Q6_K, SVI conditioning + SVI LoRA, no distill LoRA."""
    mod = _module()
    rt = _runtime(mod, True)
    g = rt.g_i2v("t", "/tmp/comfy/input/r.png", "p", 480, 832, 1, "real", 1, {})
    assert not _dangling(g)
    assert "WanImageToVideoSVIPro" in {n["class_type"] for n in g.values()}
    import json
    text = json.dumps(g)
    assert "Q6_K" in text and "SVI" in text and "lightx2v" not in text
    cfgs = {n["inputs"]["cfg"] for n in g.values() if n["class_type"] == "XFuserKSamplerAdvanced"}
    assert cfgs == {3.5}
    assert mod.PHASES["real"][0] == 20
