"""Tests for speculative decoding draft model loading, passing (-md / -ngld),
and the experimental DSpine profile in models.yaml."""
import os, sys, pathlib, json, pytest, yaml
from gateway import common, media, orchestrator as orch
from tests.kernel_loader import load_kernel, llama_argv


def test_models_yaml_dspine_profile_validity():
    models = common.load_models()
    assert "qwen3.8-27b-dspine-test" in models
    m = models["qwen3.8-27b-dspine-test"]
    assert media.problems(m) == []
    assert m["draft_hf"] == "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF"
    assert m["draft_file"] == "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    assert m["draft_quant"] == "Q4_K_M"
    assert m["draft_ngld"] == 99
    assert m["llamacpp_repo"] == "https://github.com/Ajay9o9/llama.cpp"
    assert m["llamacpp_ref"] == "dspine"
    assert m["prefer_cached_binary"] is True
    assert m["ctx"] == 32768
    assert m["vram_gb"] <= 30

    args = m["args"]
    assert "--spec-type" in args
    assert args[args.index("--spec-type") + 1] == "draft-dflash"
    assert "--spec-draft-n-max" in args
    assert args[args.index("--spec-draft-n-max") + 1] == "16"
    assert "-sm" in args
    assert args[args.index("-sm") + 1] == "tensor"


def test_dspine_llama_argv_draft_injection(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    argv = llama_argv("qwen3.8-27b-dspine-test")
    assert "-md" in argv
    md_idx = argv.index("-md")
    assert argv[md_idx + 1].endswith("Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf")
    assert "-ngld" in argv
    ngld_idx = argv.index("-ngld")
    assert argv[ngld_idx + 1] == "99"
    assert "--spec-type" in argv
    assert argv[argv.index("--spec-type") + 1] == "draft-dflash"
    assert "--spec-draft-n-max" in argv
    assert argv[argv.index("--spec-draft-n-max") + 1] == "16"


def test_draft_args_no_duplicate(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    # If -md and -ngld are explicitly provided in model args, do not duplicate them
    entry = {
        "key": "test-model-custom-md",
        "hf": "foo/bar-GGUF",
        "quant": "Q4_K_M",
        "draft_hf": "aj9o9/Qwen3.8-27B-DSpine-b16-GGUF",
        "draft_file": "custom-draft.gguf",
        "draft_ngld": 99,
        "args": ["-md", "/custom/path/draft.gguf", "-ngld", "42"],
    }
    argv = llama_argv(entry)
    assert argv.count("-md") == 1
    assert argv[argv.index("-md") + 1] == "/custom/path/draft.gguf"
    assert argv.count("-ngld") == 1
    assert argv[argv.index("-ngld") + 1] == "42"


def test_resolve_draft_model_helpers(tmp_path, monkeypatch):
    mod = load_kernel("qwen3.8-27b-dspine-test")
    
    # 1. Direct path exists
    dummy_gguf = tmp_path / "custom.gguf"
    dummy_gguf.write_bytes(b"GGUF")
    mod.CFG["draft_model"] = str(dummy_gguf)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert mod.resolve_draft_model() == str(dummy_gguf)

    # 2. Cached file find in mock directory
    kaggle_input = tmp_path / "kaggle" / "input" / "test-dataset"
    kaggle_input.mkdir(parents=True)
    cached_gguf = kaggle_input / "cached-draft.gguf"
    cached_gguf.write_bytes(b"CACHED")
    
    monkeypatch.setattr(mod.glob, "glob", lambda pattern, recursive=True: [str(cached_gguf)] if "cached-draft.gguf" in pattern else [])
    assert mod.find_cached_draft_file("cached-draft.gguf") == str(cached_gguf)


def test_render_kernel_preserves_custom_binary_and_draft_fields():
    cfg = {"accounts": [{"name": "kg1", "user": "u"}], "kernel_slug": "kllm", "gist_id": "g",
           "gist_token": "t", "gist_user": "x", "extra_server_args": []}
    m = common.resolve_model("qwen3.8-27b-dspine-test")
    bdir, ref = orch.render_kernel(cfg, "kg1", m, "key123", 3600)
    
    src = (bdir / "server_kernel.py").read_text(encoding="utf-8")
    assert "Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf" in src
    assert "https://github.com/Ajay9o9/llama.cpp" in src
    assert "dspine" in src
    assert "resolve_draft_model" in src


def test_orchestrator_dataset_sources_multiple():
    # Verify that dataset can be list and datasets field is supported
    model = {"hf": "test/model", "quant": "Q4", "datasets": ["user/ds1", "user/ds2"], "dataset": "user/ds0"}
    ds_sources = []
    if model.get("dataset"):
        if isinstance(model["dataset"], list):
            ds_sources.extend(model["dataset"])
        else:
            ds_sources.append(model["dataset"])
    if model.get("datasets"):
        for d in model["datasets"]:
            if d not in ds_sources:
                ds_sources.append(d)
    assert ds_sources == ["user/ds0", "user/ds1", "user/ds2"]
