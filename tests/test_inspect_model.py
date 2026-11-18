"""kllm add / inspect_model: type detection, quant choice, ctx and VRAM estimates, models.yaml writing,
CLI and panel endpoints. HuggingFace is faked at the single network seam (inspect_model._get)."""
import json, struct, sys, pathlib
import pytest, yaml
from fastapi.testclient import TestClient
from gateway import common, inspect_model as im, media, relay, cli

GB = 10 ** 9


def gguf_header(arch="qwen2", ctx=32768, layers=64, emb=5120, heads=40, kv=8, extra=None, pad_tokens=True):
    """A GGUF header like llama.cpp writes it: general.* and {arch}.* first, tokenizer tables after."""
    def s(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    kvs = [("general.architecture", 8, s(arch)), ("general.name", 8, s("test")),
           (f"{arch}.context_length", 4, struct.pack("<I", ctx)), (f"{arch}.block_count", 4, struct.pack("<I", layers)),
           (f"{arch}.embedding_length", 4, struct.pack("<I", emb)),
           (f"{arch}.attention.head_count", 4, struct.pack("<I", heads)),
           (f"{arch}.attention.head_count_kv", 4, struct.pack("<I", kv))]
    kvs += extra or []
    if pad_tokens:   # a huge tokenizer array: parsing must stop before it
        kvs.append(("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 150000) + b"\x05\x00\x00\x00\x00\x00\x00\x00abc"))
    out = b"GGUF" + struct.pack("<IQQ", 3, 100, len(kvs))
    for k, t, v in kvs:
        out += s(k) + struct.pack("<I", t) + v
    return out


class FakeHF:
    def __init__(self):
        self.routes, self.calls = {}, []

    def add(self, suffix, obj):
        self.routes[suffix] = obj

    def get(self, url, token=None, headers=None, timeout=20):
        self.calls.append((url, token, headers))
        for suf, obj in self.routes.items():
            if url.endswith(suf) or suf in url:
                if isinstance(obj, im.InspectError):
                    raise obj
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                if headers and "Range" in headers and isinstance(obj, bytes):
                    body = body[:int(headers["Range"].split("-")[1]) + 1]
                return body, {}
        raise im.InspectError(f"HuggingFace returned HTTP 404 ({url})")


def tree(*files):
    return [{"type": "file", "path": p, "size": int(sz * GB), "lfs": {"size": int(sz * GB)}} for p, sz in files]


@pytest.fixture
def hf(monkeypatch):
    f = FakeHF()
    monkeypatch.setattr(im, "_get", f.get)
    return f


def repo(hf, name, files, info=None, gguf_file=None, **hdr):
    hf.add(f"/api/models/{name}/tree/main?recursive=true&expand=false", tree(*files))
    hf.add(f"/api/models/{name}", info or {"id": name, "tags": []})
    if gguf_file:
        hf.add(f"/{name}/resolve/main/{gguf_file}", gguf_header(**hdr))


CODER = "bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF"


# ---- source parsing -----------------------------------------------------------------------------
@pytest.mark.parametrize("src,want", [
    ("org/name", ("org/name", None)),
    ("https://huggingface.co/org/name", ("org/name", None)),
    ("https://huggingface.co/org/name/tree/main", ("org/name", None)),
    ("https://huggingface.co/org/name/blob/main/sub/m-Q5_K_M.gguf", ("org/name", "sub/m-Q5_K_M.gguf")),
    ("https://huggingface.co/org/name/resolve/main/m.gguf?download=true", ("org/name", "m.gguf")),
    ("hf.co/org/name:Q4_K_M", ("org/name", "Q4_K_M")),
    ("org/name:i1-Q4_K_S", ("org/name", "i1-Q4_K_S")),
])
def test_parse_source(src, want):
    assert im.parse_source(src) == want


@pytest.mark.parametrize("bad", ["", "justaname", "https://example.com/a/b", "https://huggingface.co/onlyorg"])
def test_parse_source_errors(bad):
    with pytest.raises(im.InspectError):
        im.parse_source(bad)


# ---- GGUF: grouping and the quant choice ---------------------------------------------------------
def cands(*pairs):
    c, _ = im.group_ggufs(tree(*pairs))
    return c


def test_quant_names():
    q = im.quant_of
    assert q("Huihui-Qwen3.6-35B-A3B-abliterated.i1-Q4_K_M.gguf") == "i1-Q4_K_M"
    assert q("Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert q("sub/dir/model-IQ4_XS-00001-of-00003.gguf") == "IQ4_XS"
    assert q("huihui-ai_Huihui-gpt-oss-20b-BF16-abliterated-MXFP4_MOE.gguf") == "MXFP4_MOE"


def test_split_files_are_grouped_and_junk_skipped():
    files = tree(("m-Q4_K_M-00001-of-00002.gguf", 10), ("m-Q4_K_M-00002-of-00002.gguf", 9),
                 ("m-Q8_0-00001-of-00002.gguf", 18),                      # incomplete: part 2 missing
                 ("mmproj-F16.gguf", 1), ("m.imatrix.gguf", 1), ("m-Q2_K.gguf.part1of2", 5), ("README.md", 0))
    c, notes = im.group_ggufs(files)
    assert [(x["file"], x["size_gb"], x["parts"], x["quant"]) for x in c] == [("m-Q4_K_M-00001-of-00002.gguf", 19.0, 2, "Q4_K_M")]
    assert any("incomplete" in n for n in notes) and any("raw-split" in n for n in notes)


def test_prefers_imatrix_q4_k_m():
    c = cands(("m.i1-Q4_K_M.gguf", 19.9), ("m.Q4_K_M.gguf", 19.9), ("m.i1-Q4_K_S.gguf", 18.8), ("m.Q6_K.gguf", 26))
    assert im.pick_gguf(c)[0]["file"] == "m.i1-Q4_K_M.gguf"
    assert im.pick_gguf(cands(("m.Q4_K_M.gguf", 19.9), ("m.i1-Q4_K_S.gguf", 18.8)))[0]["file"] == "m.Q4_K_M.gguf"


def test_falls_back_when_q4_k_m_is_too_big():
    c = cands(("m-Q4_K_M.gguf", 24.0), ("m-Q4_K_S.gguf", 22.5), ("m-IQ4_XS.gguf", 21.0), ("m-Q3_K_M.gguf", 19.0))
    assert im.pick_gguf(c)[0]["file"] == "m-IQ4_XS.gguf"
    c = cands(("m-Q4_K_M.gguf", 40.0), ("m-IQ2_XS.gguf", 25.5))
    ch, warn = im.pick_gguf(c)
    assert ch["file"] == "m-IQ2_XS.gguf" and warn and "will not leave room" in warn[0]


def test_small_model_takes_q5_k_m():
    c = cands(("m-Q4_K_M.gguf", 4.7), ("m-Q5_K_M.gguf", 5.5), ("m-Q8_0.gguf", 8.5))
    assert im.pick_gguf(c)[0]["file"] == "m-Q5_K_M.gguf"
    c = cands(("m-Q4_K_M.gguf", 19.9), ("m-Q5_K_M.gguf", 21.0))         # big: Q5 is not worth the context
    assert im.pick_gguf(c)[0]["file"] == "m-Q4_K_M.gguf"


def test_explicit_file_or_quant():
    c = cands(("m.i1-Q4_K_M.gguf", 19.9), ("m.i1-Q5_K_M.gguf", 23.0), ("m.Q5_K_M.gguf", 23.0), ("sub/m-Q6_K.gguf", 26))
    assert im.pick_gguf(c, "Q5_K_M")[0]["file"] == "m.i1-Q5_K_M.gguf"
    assert im.pick_gguf(c, "m.Q5_K_M.gguf")[0]["file"] == "m.Q5_K_M.gguf"
    assert im.pick_gguf(c, "sub/m-Q6_K.gguf")[1]                       # fits warning (26 GB of weights)
    with pytest.raises(im.InspectError, match="Available"):
        im.pick_gguf(c, "Q9_Z")


def test_gguf_header_parsing_stops_before_tokenizer():
    meta = im.parse_gguf_header(gguf_header(), want=("context_length",))
    assert meta["qwen2.context_length"] == 32768
    assert im.parse_gguf_header(gguf_header()[:90]).get("general.architecture") == "qwen2"   # truncated: partial, no crash
    assert im.parse_gguf_header(b"nope") == {}
    arch = im._arch_info(im.parse_gguf_header(gguf_header(layers=64, heads=40, kv=8, emb=5120)))
    assert arch["native_ctx"] == 32768 and arch["kv_bytes_per_token"] == 64 * 8 * (128 + 128) * 2


def test_hybrid_attention_uses_only_full_attention_layers():
    extra = [("qwen35moe.full_attention_interval", 4, struct.pack("<I", 4))]
    h = gguf_header(arch="qwen35moe", ctx=262144, layers=40, emb=2048, heads=16, kv=2, extra=extra)
    a = im._arch_info(im.parse_gguf_header(h))
    assert a["kv_bytes_per_token"] == 10 * 2 * (128 + 128) * 2          # 10 of 40 layers hold KV


def test_ctx_is_bounded_by_vram_and_native():
    kv = 64 * 8 * 256 * 2
    assert im._pick_ctx(32768, kv, 19.85) == 24576                       # dense 32B: little room
    assert im._pick_ctx(32768, kv, 5.0) == 32768                         # native cap
    assert im._pick_ctx(262144, 20 * 1024, 21.2) == 262144               # hybrid MoE: whole native window
    assert im._pick_ctx(None, None, 10) == 32768                         # no metadata: default
    assert im._pick_ctx(131072, 10 ** 6, 20) == 4096


def test_inspect_gguf_repo_end_to_end(hf):
    f = "Qwen2.5-Coder-32B-Instruct-abliterated-Q4_K_M.gguf"
    repo(hf, CODER, [(f, 19.85), ("Qwen2.5-Coder-32B-Instruct-abliterated-Q8_0.gguf", 34.8),
                     ("Qwen2.5-Coder-32B-Instruct-abliterated-IQ4_XS.gguf", 17.7), ("README.md", 0)], gguf_file=f)
    res = im.inspect_source(f"https://huggingface.co/{CODER}")
    e = res["entry"]
    assert res["runtime"] == "llama.cpp" and res["kind"] == "chat" and res["key"] == "qwen2.5-coder-32b-instruct-abliterated"
    assert (e["hf"], e["quant"], e["file"]) == (CODER, "Q4_K_M", f)
    assert e["ctx"] == 24576 and 26 <= e["vram_gb"] <= 29 and res["fits"] and res["native_ctx"] == 32768
    assert "coder" in e["tags"] and "abliterated" in e["tags"] and "unverified" in e["tags"]
    a = e["args"]
    assert a[:4] == ["--temp", "0.7", "--top-p", "0.8"]                  # coder sampling
    assert "-sm" in a and a[a.index("-sm") + 1] == "tensor" and a.count("f16") == 2
    assert "--reasoning-format" not in a
    assert any(c["file"].endswith("Q8_0.gguf") for c in res["files"])
    assert any("Range" in (h or {}) for _, _, h in hf.calls)             # metadata came from a Range read, not a download


def test_sampling_by_model_family():
    s = lambda n: im._sampling(n)[0]
    assert s("Qwen3-Coder-30B-A3B-Instruct-GGUF")[:4] == ["--temp", "0.7", "--top-p", "0.8"]
    q3 = s("Qwen3-32B-abliterated-GGUF")
    assert q3[:6] == ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20"] and "--reasoning-format" in q3
    assert "--reasoning-format" in s("DeepSeek-R1-Distill-Qwen-32B-GGUF") and "--reasoning-format" in s("QwQ-32B-GGUF")
    assert "--reasoning-format" not in s("Qwen3-30B-A3B-Instruct-2507-GGUF")      # non-thinking Instruct
    assert s("gpt-oss-20b-GGUF")[:4] == ["--temp", "1.0", "--top-p", "1.0"]
    assert s("Llama-3.1-8B-Instruct-GGUF") == ["--temp", "0.7", "--top-p", "0.9"]


def test_no_metadata_still_works_with_default_ctx(hf):
    f = "m-Q4_K_M.gguf"
    repo(hf, "a/m-GGUF", [(f, 4.7)])                      # no resolve route -> metadata fetch fails
    res = im.inspect_source("a/m-GGUF")
    assert res["ctx"] == 32768 and res["entry"]["vram_gb"] >= 8 and any("rough" in w for w in res["warnings"])


# ---- diffusers ----------------------------------------------------------------------------------
def diffusers_repo(hf, name, cls, comps, extra=()):
    idx = {"_class_name": cls, **{c: ["diffusers", "X"] for c in comps}, "scheduler": ["diffusers", "S"]}
    files = [("model_index.json", 0.00001)]
    for c, fs in comps.items():
        files += [(f"{c}/{p}", sz) for p, sz in fs]
    files += list(extra)
    hf.add(f"/api/models/{name}/tree/main?recursive=true&expand=false", tree(*files))
    hf.add(f"/api/models/{name}", {"id": name, "tags": ["diffusers"]})
    hf.add(f"/{name}/resolve/main/model_index.json", idx)


def test_sdxl_image_repo_with_fp16_variants(hf):
    diffusers_repo(hf, "stabilityai/sdxl-turbo", "StableDiffusionXLPipeline", {
        "unet": [("diffusion_pytorch_model.safetensors", 10.3), ("diffusion_pytorch_model.fp16.safetensors", 5.1)],
        "vae": [("diffusion_pytorch_model.safetensors", 0.33), ("diffusion_pytorch_model.fp16.safetensors", 0.17)],
        "text_encoder": [("model.safetensors", 0.49), ("model.fp16.safetensors", 0.25)],
        "text_encoder_2": [("model.fp16.safetensors", 1.39), ("model.safetensors", 2.78)]},
        extra=[("sd_xl_turbo_1.0.safetensors", 13.9)])           # root single-file checkpoint: not counted
    res = im.inspect_source("stabilityai/sdxl-turbo")
    e = res["entry"]
    assert res["runtime"] == "diffusers" and res["kind"] == "image" and e["runtime"] == "diffusers" and e["kind"] == "image"
    assert res["weights_gb"] == 6.9 and "quant" not in e and e["vram_gb"] <= 10
    assert e["opts"]["steps"] == 4 and e["opts"]["guidance_scale"] == 0.0 and e["opts"]["size"] == "512x512"
    assert "variant" not in e["opts"] and res["fits"] and res["ctx"] is None


def test_video_repos_are_video(hf):
    for cls in ("WanPipeline", "LTXPipeline", "CogVideoXPipeline", "HunyuanVideoPipeline", "MochiPipeline"):
        hf.routes.clear()
        diffusers_repo(hf, "x/vid", cls, {"transformer": [("diffusion_pytorch_model.safetensors", 5.0)],
                                           "vae": [("diffusion_pytorch_model.safetensors", 0.5)]})
        hf.add("range", b"")
        res = im.inspect_source("x/vid")
        assert res["kind"] == "video" and res["entry"]["opts"]["num_frames"] and res["entry"]["opts"]["fps"], cls
        assert res["entry"]["opts"]["variant"] is None                   # no fp16 variant files


def test_fp32_stored_weights_are_halved(hf, monkeypatch):
    diffusers_repo(hf, "x/vid", "WanPipeline", {"text_encoder": [("model.safetensors", 22.7)],
                                                 "transformer": [("diffusion_pytorch_model.safetensors", 5.7)],
                                                 "vae": [("diffusion_pytorch_model.safetensors", 0.5)]})
    monkeypatch.setattr(im, "safetensors_dtype", lambda repo, path, token=None: "F32" if "text_encoder" in path or "transformer" in path else "BF16")
    res = im.inspect_source("x/vid")
    assert res["weights_gb"] == 14.7 and "model CPU offload" in res["placement"]


def test_safetensors_dtype_reads_header_via_range(hf):
    hdr = json.dumps({"__metadata__": {}, "w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    hf.add("/r/resolve/main/a/model.safetensors", struct.pack("<Q", len(hdr)) + hdr + b"\0\0")
    assert im.safetensors_dtype("r", "a/model.safetensors") == "BF16"
    assert im.safetensors_dtype("r", "missing.safetensors") is None


def test_huge_diffusers_model_does_not_fit(hf):
    diffusers_repo(hf, "x/huge", "FluxPipeline", {"transformer": [("diffusion_pytorch_model.safetensors", 24.0)],
                                                   "text_encoder_2": [("model.safetensors", 9.5)]})
    res = im.inspect_source("x/huge")
    assert not res["fits"] and any("RAM" in w for w in res["warnings"]) and res["entry"]["opts"]["guidance_scale"] == 3.5


def test_image_to_video_is_refused(hf):
    diffusers_repo(hf, "x/i2v", "WanImageToVideoPipeline", {"transformer": [("diffusion_pytorch_model.safetensors", 5.0)]})
    with pytest.raises(im.InspectError, match="input image"):
        im.inspect_source("x/i2v")


# ---- unsupported / errors ------------------------------------------------------------------------
def test_unsupported_repo_explains_what_kllm_runs(hf):
    repo(hf, "Qwen/Qwen3-0.6B", [("config.json", 0), ("model.safetensors", 1.2)])
    hf.add("/api/models?search=", [{"id": "Qwen/Qwen3-0.6B-GGUF"}, {"id": "other/Thing"}])
    with pytest.raises(im.InspectError) as e:
        im.inspect_source("Qwen/Qwen3-0.6B")
    m = str(e.value)
    assert "neither a GGUF repo nor a diffusers pipeline" in m and "GGUF" in m and "model_index.json" in m
    assert "Qwen/Qwen3-0.6B-GGUF" in m and "other/Thing" not in m


def test_missing_repo_and_gated(hf):
    with pytest.raises(im.InspectError, match="404"):
        im.inspect_source("nobody/nothing")
    hf.add("/api/models/g/model", im.InspectError("HuggingFace refused access (gated or private repo)"))
    with pytest.raises(im.InspectError, match="gated"):
        im.inspect_source("g/model")


def test_gated_warning_and_token_is_sent(hf):
    f = "m-Q4_K_M.gguf"
    repo(hf, "a/gated-GGUF", [(f, 4.7)], info={"id": "a/gated-GGUF", "tags": [], "gated": "auto"}, gguf_file=f)
    res = im.inspect_source("a/gated-GGUF")
    assert any("gated" in w for w in res["warnings"])
    hf.calls.clear()
    res = im.inspect_source("a/gated-GGUF", token="hf_secret")
    assert not any("gated" in w for w in res["warnings"]) and all(t == "hf_secret" for _, t, _ in hf.calls)


def test_name_collisions_and_validation(hf):
    f = "m-Q4_K_M.gguf"
    repo(hf, "a/Thing-i1-GGUF", [(f, 4.7)], gguf_file=f)
    assert im.inspect_source("a/Thing-i1-GGUF")["key"] == "thing"
    assert im.inspect_source("a/Thing-i1-GGUF", taken={"thing"})["key"] == "thing-2"
    assert im.inspect_source("a/Thing-i1-GGUF", name="mine")["key"] == "mine"
    with pytest.raises(im.InspectError, match="already exists"):
        im.inspect_source("a/Thing-i1-GGUF", name="thing", taken={"thing"})
    with pytest.raises(im.InspectError, match="must be"):
        im.inspect_source("a/Thing-i1-GGUF", name="bad name!")


# ---- writing models.yaml ----------------------------------------------------------------------------
BASE = """# my registry
# keep this comment
models:
  old:                # trailing comment
    hf: a/old
    quant: Q4_K_M
    args: ["--temp", "0.7"]
    desc: "old"
"""


def test_write_entry_appends_valid_yaml_and_keeps_comments(tmp_path):
    p = tmp_path / "models.yaml"; p.write_text(BASE, encoding="utf-8")
    llama = {"hf": "a/new", "quant": "Q4_K_M", "file": "n-Q4_K_M.gguf", "vram_gb": 20, "ctx": 32768,
             "args": ["--temp", "0.6", "-sm", "tensor"], "desc": 'has "quotes": and colons', "tags": ["unverified"]}
    diff = {"runtime": "diffusers", "kind": "image", "hf": "a/img", "vram_gb": 9,
            "opts": {"size": "512x512", "steps": 4, "variant": None}, "desc": "img", "tags": ["image"]}
    im.write_entry("new", llama, p)
    im.write_entry("img", diff, p)
    txt = p.read_text(encoding="utf-8")
    assert txt.startswith(BASE) and "# keep this comment" in txt and "# trailing comment" in txt
    models = yaml.safe_load(txt)["models"]
    assert list(models) == ["old", "new", "img"]
    assert models["new"] == llama and models["img"]["opts"] == {"size": "512x512", "steps": 4, "variant": None}
    with pytest.raises(im.InspectError, match="already exists"):
        im.write_entry("old", llama, p)
    assert p.read_text(encoding="utf-8") == txt


def test_write_entry_refuses_to_corrupt(tmp_path):
    p = tmp_path / "models.yaml"; p.write_text("models: [broken\n", encoding="utf-8")
    with pytest.raises(im.InspectError, match="not valid YAML"):
        im.write_entry("x", {"hf": "a/b"}, p)
    assert p.read_text(encoding="utf-8") == "models: [broken\n"


def test_shipped_registry_is_valid_and_existing_entries_unchanged():
    reg = yaml.safe_load((pathlib.Path(__file__).resolve().parent.parent / "models.yaml").read_text(encoding="utf-8"))["models"]
    assert reg["qwen3.6-35b-abliterated"]["quant"] == "i1-Q4_K_M" and "runtime" not in reg["qwen3.6-35b-abliterated"]
    for k, m in reg.items():
        assert media.problems(m) == [], k


# ---- CLI and panel ------------------------------------------------------------------------------------
@pytest.fixture
def registry_backup():
    saved = common.MODELS_FILE.read_text(encoding="utf-8")
    yield
    common.MODELS_FILE.write_text(saved, encoding="utf-8")


def _sdxl(hf):
    diffusers_repo(hf, "stabilityai/sdxl-turbo", "StableDiffusionXLPipeline", {
        "unet": [("diffusion_pytorch_model.fp16.safetensors", 5.1)], "vae": [("diffusion_pytorch_model.fp16.safetensors", 0.17)]})


def test_cli_add_dry_run_then_write(hf, registry_backup, capsys):
    _sdxl(hf)
    before = common.MODELS_FILE.read_text(encoding="utf-8")
    cli.main(["add", "https://huggingface.co/stabilityai/sdxl-turbo", "--dry-run"])
    out = capsys.readouterr().out
    assert "image model, runtime diffusers" in out and "sdxl-turbo:" in out and common.MODELS_FILE.read_text(encoding="utf-8") == before
    cli.main(["add", "stabilityai/sdxl-turbo", "--name", "sdxl-t"])
    out = capsys.readouterr().out
    assert "added 'sdxl-t'" in out and "kllm up --model sdxl-t" in out and "/v1/images/generations" in out
    m = common.resolve_model("sdxl-t")
    assert media.kind(m) == "image" and media.runtime(m) == "diffusers"
    assert common.MODELS_FILE.read_text(encoding="utf-8").startswith(before)


def test_cli_add_errors_exit_nonzero(hf, registry_backup, capsys):
    repo(hf, "Qwen/Qwen3-0.6B", [("config.json", 0)])
    hf.add("/api/models?search=", [])
    with pytest.raises(SystemExit) as e:
        cli.main(["add", "Qwen/Qwen3-0.6B"])
    assert e.value.code == 1 and "neither a GGUF repo nor a diffusers pipeline" in capsys.readouterr().out
    _sdxl(hf)
    with pytest.raises(SystemExit):
        cli.main(["add", "stabilityai/sdxl-turbo", "--default"])             # media model cannot be the default
    assert "only a chat model" in capsys.readouterr().out


def test_cli_add_default_sets_config(hf, registry_backup, capsys):
    f = "m-Q4_K_M.gguf"
    repo(hf, "a/Fancy-GGUF", [(f, 4.7)], gguf_file=f)
    common.CONFIG_FILE.write_text("default_model: validate-3b\n", encoding="utf-8")
    try:
        cli.main(["add", "a/Fancy-GGUF", "--default"])
        assert common.load_config()["default_model"] == "fancy"
    finally:
        common.CONFIG_FILE.unlink(missing_ok=True)


def test_cli_models_and_model_handle_media(hf, registry_backup, capsys):
    _sdxl(hf)
    cli.main(["add", "stabilityai/sdxl-turbo", "--name", "sdxl-t"])
    common.CONFIG_FILE.write_text("default_model: validate-3b\n", encoding="utf-8")
    try:
        cli.main(["models"])
        out = capsys.readouterr().out
        assert "[image/diffusers] stabilityai/sdxl-turbo" in out and "validate-3b" in out
        with pytest.raises(SystemExit):
            cli.main(["model", "sdxl-t"])
    finally:
        common.CONFIG_FILE.unlink(missing_ok=True)


def test_panel_inspect_and_add(hf, registry_backup):
    _sdxl(hf)
    common.CONFIG_FILE.write_text("relay_host: 127.0.0.1\nrelay_port: 18199\ndefault_model: validate-3b\naccounts: []\n", encoding="utf-8")
    relay._cfg["mtime"] = 0
    try:
        c = TestClient(relay.app, client=("127.0.0.1", 50000))
        r = c.post("/admin/api/inspect", json={"source": "stabilityai/sdxl-turbo"})
        j = r.json()
        assert r.status_code == 200 and j["ok"] and j["kind"] == "image" and j["runtime"] == "diffusers" and j["key"] == "sdxl-turbo-2"   # shipped registry has sdxl-turbo
        assert "sdxl-turbo-2" not in common.load_models()                     # inspect never writes
        r = c.post("/admin/api/add-from-hf", json={"source": "stabilityai/sdxl-turbo", "name": "my-sdxl"})
        assert r.status_code == 200 and r.json()["key"] == "my-sdxl"
        assert media.kind(common.load_models()["my-sdxl"]) == "image"
        r = c.post("/admin/api/add-from-hf", json={"source": "stabilityai/sdxl-turbo", "name": "my-sdxl"})
        assert r.status_code == 400 and "already exists" in r.json()["msg"]
        r = c.post("/admin/api/inspect", json={"source": "garbage"})
        assert r.status_code == 400 and not r.json()["ok"]
        r = c.post("/admin/api/add-from-hf", json={"source": "stabilityai/sdxl-turbo", "name": "n2", "default": True})
        assert r.status_code == 400 and "default" in r.json()["msg"]
        r = c.post("/admin/api/add-model", json={"key": "manual-one", "hf": "a/b-GGUF", "quant": "Q4_K_M"})   # old endpoint intact
        assert r.status_code == 200 and "manual-one" in common.load_models()
        st = c.get("/admin/api/state").json()
        kinds = {m["key"]: m["kind"] for m in st["models"]}
        assert kinds["my-sdxl"] == "image" and kinds["validate-3b"] == "chat" and kinds["manual-one"] == "chat"
    finally:
        common.CONFIG_FILE.unlink(missing_ok=True); relay._cfg["mtime"] = 0
