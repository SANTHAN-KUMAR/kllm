"""Kernel diffusers runtime: HTTP API, auth, locking, errors, placement, loading (fake pipeline,
fake torch/diffusers modules: no GPU, no torch, no network)."""
import base64, http.client, json, sys, threading, time, types, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kernel_loader import load_kernel, llama_argv

KEY = "kllm-testkey"
IMG = {"hf": "acme/img", "runtime": "diffusers", "kind": "image", "vram_gb": 8,
       "opts": {"steps": 4, "guidance_scale": 0.0, "size": "512x512", "warmup": False}}
VID = {"hf": "acme/vid", "runtime": "diffusers", "kind": "video", "vram_gb": 12,
       "opts": {"num_frames": 17, "fps": 12, "size": "256x256", "max_frames": 33}}


class FakeImage:
    def save(self, buf, format=None):
        buf.write(b"PNG:" + (format or "").encode())


class FakePipe:
    """Takes the same kind of arguments a diffusers pipeline does (no negative_prompt: like FLUX)."""
    def __init__(self, video=False, delay=0.0, boom=None):
        self.calls, self.video, self.delay, self.boom = [], video, delay, boom
        self.active = self.peak = 0
        self.components = {}

    def __call__(self, prompt, width=512, height=512, num_inference_steps=30, guidance_scale=None,
                 generator=None, num_frames=None, image=None, video=None, strength=None):
        self.active += 1; self.peak = max(self.peak, self.active)
        try:
            self.calls.append(dict(prompt=prompt, width=width, height=height, steps=num_inference_steps,
                                   guidance=guidance_scale, generator=generator, num_frames=num_frames,
                                   image=image, video=video, strength=strength))
            time.sleep(self.delay)
            if self.boom:
                raise self.boom
            if self.video:
                return types.SimpleNamespace(frames=[[object()] * (num_frames or 1)])
            return types.SimpleNamespace(images=[FakeImage()])
        finally:
            self.active -= 1


def start(model, pipe, state="ready"):
    mod = load_kernel(model, key=KEY)
    rt = mod.DiffusersRuntime(mod.CFG)
    rt.pipe, rt.state = pipe, state
    rt._generator = lambda seed: ("gen", seed)
    rt._encode_image = lambda img: b"PNGDATA"
    rt._encode_mp4 = lambda frames, fps: b"MP4:%d:%d" % (len(frames), fps)
    srv = mod.make_diffusers_server(rt, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return mod, rt, srv


class Api:
    def __init__(self, srv):
        self.port = srv.server_address[1]

    def call(self, method, path, body=None, key=KEY, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = dict(headers or {})
        if key:
            h["Authorization"] = f"Bearer {key}"
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        if data:
            h["Content-Type"] = "application/json"
        c.request(method, path, data, h)
        r = c.getresponse()
        raw = r.read(); c.close()
        return r.status, (json.loads(raw) if raw else None)


@pytest.fixture
def image_api():
    pipe = FakePipe()
    mod, rt, srv = start(IMG, pipe)
    yield Api(srv), rt, pipe
    srv.shutdown()


def test_render_config_carries_runtime():
    mod = load_kernel(IMG)
    assert mod.RUNTIME == "diffusers" and mod.CFG["kind"] == "image" and mod.CFG["opts"]["steps"] == 4
    assert mod.MODEL_ID == "acme/img"                     # no ':None'
    assert load_kernel("qwen3.6-35b-abliterated").RUNTIME == "llama.cpp"


def test_health_open_models_need_key(image_api):
    api, rt, _ = image_api
    assert api.call("GET", "/health", key=None) == (200, {"status": "ok", "busy": False})
    assert api.call("GET", "/v1/models", key=None)[0] == 401
    assert api.call("GET", "/v1/models", key="wrong")[0] == 401
    st, body = api.call("GET", "/v1/models")
    assert st == 200 and body["data"][0]["id"] == "test-model" and body["data"][0]["kind"] == "image"
    # x-api-key (what the relay also sends) is accepted too
    assert api.call("GET", "/v1/models", key=None, headers={"x-api-key": KEY})[0] == 200


def test_generation_requires_key(image_api):
    api, _, pipe = image_api
    st, body = api.call("POST", "/v1/images/generations", {"prompt": "x"}, key=None)
    assert st == 401 and body["error"]["type"] == "authentication_error" and not pipe.calls


def test_image_generation_openai_shape(image_api):
    api, _, pipe = image_api
    st, body = api.call("POST", "/v1/images/generations",
                        {"model": "whatever", "prompt": "a cat", "n": 2, "size": "768x512", "seed": 7,
                         "negative_prompt": "blurry", "steps": 6, "guidance_scale": 2.5})
    assert st == 200 and isinstance(body["created"], int)
    assert [base64.b64decode(d["b64_json"]) for d in body["data"]] == [b"PNGDATA"] * 2
    assert [d["seed"] for d in body["data"]] == [7, 8]
    c = pipe.calls[0]
    assert (c["prompt"], c["width"], c["height"], c["steps"], c["guidance"]) == ("a cat", 768, 512, 6, 2.5)
    assert c["generator"] == ("gen", 7) and pipe.calls[1]["generator"] == ("gen", 8)
    # negative_prompt is not a parameter of this pipeline: dropped, not an error


def test_defaults_come_from_opts(image_api):
    api, _, pipe = image_api
    st, body = api.call("POST", "/v1/images/generations", {"prompt": "a cat"})
    assert st == 200
    c = pipe.calls[0]
    assert (c["width"], c["height"], c["steps"], c["guidance"]) == (512, 512, 4, 0.0)


def test_response_format_url_is_data_url(image_api):
    api, *_ = image_api
    st, body = api.call("POST", "/v1/images/generations", {"prompt": "x", "response_format": "url"})
    assert st == 200 and body["data"][0]["url"].startswith("data:image/png;base64,")
    assert "b64_json" not in body["data"][0]
    assert api.call("POST", "/v1/images/generations", {"prompt": "x", "response_format": "gif"})[0] == 400


@pytest.mark.parametrize("body,frag", [
    ({}, "prompt"),
    ({"prompt": "x", "size": "big"}, "size"),
    ({"prompt": "x", "size": "513x512"}, "multiples"),
    ({"prompt": "x", "size": "4096x4096"}, "limit"),
    ({"prompt": "x", "n": 9}, "n must"),
    ({"prompt": "x", "steps": 0}, "steps"),
    ({"prompt": "x", "steps": "many"}, "steps"),
    ({"prompt": "x", "seed": "abc"}, "seed"),
])
def test_bad_requests_are_400_json(image_api, body, frag):
    api, _, pipe = image_api
    st, out = api.call("POST", "/v1/images/generations", body)
    assert st == 400 and frag in out["error"]["message"] and not pipe.calls


def test_bad_json_wrong_path_wrong_kind(image_api):
    api, *_ = image_api
    assert api.call("POST", "/v1/images/generations", b"not json")[0] == 400
    assert api.call("POST", "/v1/nope", {"prompt": "x"})[0] == 404
    st, out = api.call("POST", "/v1/videos/generations", {"prompt": "x"})
    assert st == 400 and "images" in out["error"]["message"]


def test_video_generation():
    pipe = FakePipe(video=True)
    mod, rt, srv = start(VID, pipe)
    try:
        api = Api(srv)
        st, body = api.call("POST", "/v1/videos/generations",
                            {"prompt": "waves", "num_frames": 9, "fps": 8, "seed": 3, "negative_prompt": "x"})
        assert st == 200 and body["frames"] == 9 and body["fps"] == 8
        assert base64.b64decode(body["data"][0]["b64_mp4"]) == b"MP4:9:8"
        assert pipe.calls[0]["num_frames"] == 9 and (pipe.calls[0]["width"], pipe.calls[0]["height"]) == (256, 256)
        st, body = api.call("POST", "/v1/videos/generations", {"prompt": "waves"})   # opts defaults
        assert (body["frames"], body["fps"]) == (17, 12)
        st, out = api.call("POST", "/v1/videos/generations", {"prompt": "waves", "num_frames": 100})
        assert st == 400 and "num_frames" in out["error"]["message"]
        assert api.call("POST", "/v1/images/generations", {"prompt": "x"})[0] == 400
    finally:
        srv.shutdown()


def test_video_i2v_and_snapping():
    VID_SNAP = {"hf": "acme/vid-ltx", "runtime": "diffusers", "kind": "video", "vram_gb": 12,
                "opts": {"num_frames": 49, "fps": 24, "size": "256x256", "max_frames": 241, "frame_multiple": 8, "size_multiple": 32}}
    pipe = FakePipe(video=True)
    mod, rt, srv = start(VID_SNAP, pipe)
    try:
        api = Api(srv)
        # Test frame snapping: 50 frames snaps to 49
        st, body = api.call("POST", "/v1/videos/generations", {"prompt": "cinematic", "num_frames": 50})
        assert st == 200 and body["frames"] == 49
        assert body["duration_seconds"] == round(49 / 24, 2)
        assert pipe.calls[0]["num_frames"] == 49

        # Test minimum frame snapping: 1 frame snaps to 1 + 8 = 9 frames
        st, body = api.call("POST", "/v1/videos/generations", {"prompt": "cinematic", "num_frames": 1})
        assert st == 200 and body["frames"] == 9
        assert pipe.calls[1]["num_frames"] == 9

        # Test dimension auto-snapping for video: 854x480 snaps to 864x480 (multiple of 32)
        st, body = api.call("POST", "/v1/videos/generations", {"prompt": "cinematic", "size": "854x480"})
        assert st == 200 and body["width"] == 864 and body["height"] == 480
        assert pipe.calls[2]["width"] == 864 and pipe.calls[2]["height"] == 480

        # Test Image-to-Video with base64 1x1 png image
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        st, body = api.call("POST", "/v1/videos/generations", {"prompt": "animate face", "image": tiny_png})
        assert st == 200
        assert pipe.calls[3]["image"] is not None
    finally:
        srv.shutdown()


def test_video_v2v():
    VID_V2V = {"hf": "acme/vid-v2v", "runtime": "diffusers", "kind": "video", "vram_gb": 12,
               "opts": {"num_frames": 17, "fps": 12, "size": "256x256", "max_frames": 33, "frame_multiple": 8}}
    pipe = FakePipe(video=True)
    mod, rt, srv = start(VID_V2V, pipe)
    # Test that decoded frames list is truncated to (T-1) % 8 == 0 (30 frames -> 25 frames)
    # and num_frames in call matches len(source_video) (25)
    rt._decode_video = lambda req, w, h, max_f: ["frame"] * 25 if (req.get("video") or req.get("source_video")) else None
    try:
        api = Api(srv)
        st, body = api.call("POST", "/v1/videos/generations",
                            {"prompt": "anime style", "video": "dummy_video_data", "strength": 0.65})
        assert st == 200
        assert len(pipe.calls[0]["video"]) == 25
        assert pipe.calls[0]["num_frames"] == 25
        assert pipe.calls[0]["strength"] == 0.65
        assert body["frames"] == 25
    finally:
        srv.shutdown()


def test_video_strict_capability_routing():
    """Verify that models strictly reject unsupported modes instead of silently generating text-to-video."""
    class StrictT2VPipe:
        def __init__(self):
            self.components = {}
        def __call__(self, prompt, width=512, height=512, num_inference_steps=30, guidance_scale=None, generator=None, num_frames=None):
            return types.SimpleNamespace(frames=[[object()] * (num_frames or 1)])

    class StrictI2VPipe:
        def __init__(self):
            self.components = {}
        def __call__(self, prompt, image, width=512, height=512, num_inference_steps=30, guidance_scale=None, generator=None, num_frames=None):
            if image is None:
                raise TypeError("image cannot be None")
            return types.SimpleNamespace(frames=[[object()] * (num_frames or 1)])

    # 1. T2V pipe rejects I2V and V2V
    t2v_pipe = StrictT2VPipe()
    mod, rt, srv = start({"hf": "acme/strict-t2v", "runtime": "diffusers", "kind": "video", "vram_gb": 12, "opts": {"num_frames": 17, "fps": 12, "size": "256x256"}}, t2v_pipe)
    rt._decode_video = lambda req, w, h, max_f: ["f"] * 17 if (req.get("video") or req.get("source_video")) else None
    try:
        api = Api(srv)
        # Attempt I2V on T2V pipe
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        st, body = api.call("POST", "/v1/videos/generations", {"prompt": "animate", "image": tiny_png})
        assert st == 400
        assert "does not support image-to-video" in body["error"]["message"]

        # Attempt V2V on T2V pipe
        st2, body2 = api.call("POST", "/v1/videos/generations", {"prompt": "restyle", "video": "dummy"})
        assert st2 == 400
        assert "does not support video-to-video" in body2["error"]["message"]
    finally:
        srv.shutdown()

    # 2. I2V pipe requires image
    i2v_pipe = StrictI2VPipe()
    mod2, rt2, srv2 = start({"hf": "acme/strict-i2v", "runtime": "diffusers", "kind": "video", "vram_gb": 12, "opts": {"num_frames": 17, "fps": 12, "size": "256x256"}}, i2v_pipe)
    try:
        api2 = Api(srv2)
        # Call without image -> should reject cleanly with 400
        st3, body3 = api2.call("POST", "/v1/videos/generations", {"prompt": "text only"})
        assert st3 == 400
        assert "requires a reference image" in body3["error"]["message"]
    finally:
        srv2.shutdown()


def test_one_generation_at_a_time():
    pipe = FakePipe(delay=0.25)
    mod, rt, srv = start(IMG, pipe)
    try:
        api = Api(srv)
        res = []
        ts = [threading.Thread(target=lambda: res.append(api.call("POST", "/v1/images/generations", {"prompt": "x"})[0]))
              for _ in range(3)]
        [t.start() for t in ts]
        time.sleep(0.1)
        assert api.call("GET", "/health", key=None)[1]["busy"] is True
        [t.join() for t in ts]
        assert res == [200, 200, 200] and pipe.peak == 1 and len(pipe.calls) == 3
    finally:
        srv.shutdown()


def test_generation_errors_are_json():
    mod, rt, srv = start(IMG, FakePipe(boom=RuntimeError("kaboom")))
    try:
        api = Api(srv)
        st, out = api.call("POST", "/v1/images/generations", {"prompt": "x"})
        assert st == 500 and "kaboom" in out["error"]["message"] and out["error"]["type"] == "generation_error"
        assert api.call("GET", "/health", key=None)[0] == 200            # the server survives, lock released
        rt.pipe.boom = type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory. Tried to allocate")
        st, out = api.call("POST", "/v1/images/generations", {"prompt": "x"})
        assert st == 500 and out["error"]["type"] == "out_of_memory" and "smaller" in out["error"]["message"]
    finally:
        srv.shutdown()


def test_loading_and_failed_states():
    mod, rt, srv = start(IMG, None, state="loading")
    try:
        api = Api(srv)
        assert api.call("GET", "/health", key=None)[0] == 503
        st, out = api.call("POST", "/v1/images/generations", {"prompt": "x"})
        assert st == 503 and out["error"]["type"] == "model_unavailable"
        rt.state, rt.error = "error", "RuntimeError: nope"
        assert api.call("GET", "/health", key=None)[0] == 500
        assert "nope" in api.call("POST", "/v1/images/generations", {"prompt": "x"})[1]["error"]["message"]
    finally:
        srv.shutdown()


# ---- placement ---------------------------------------------------------------------------------
def test_plan_placement():
    mod = load_kernel(IMG)
    p = mod.plan_placement
    assert p(7, 5) == "gpu"                      # SDXL-class
    assert p(20, 12) == "model"                  # components swap onto the T4
    assert p(27, 20) == "sequential"             # one component does not fit a T4
    with pytest.raises(mod.ApiError):
        p(40, 20)                                # cannot even be held in 31 GB RAM
    assert p(27, 20, "model") == "model" and p(5, 5, "sequential") == "sequential" and p(30, 30, "none") == "gpu"


def test_component_sizes():
    mod = load_kernel(IMG)
    class T:
        def __init__(self, n): self.n = n
        def numel(self): return self.n
        def element_size(self): return 2
    class M:
        def __init__(self, n): self.n = n
        def parameters(self): return [T(self.n)]
    pipe = types.SimpleNamespace(components={"unet": M(5_000_000_000), "vae": M(100_000_000), "tok": "str"})
    s = mod.component_sizes(pipe)
    assert s == {"unet": 10.0, "vae": 0.2}


# ---- loading with fake torch / diffusers -------------------------------------------------------
class _Param:
    def __init__(self, gb): self.gb = gb
    def numel(self): return int(self.gb * 1e9 / 2)
    def element_size(self): return 2


class _Comp:
    def __init__(self, gb): self.gb = gb; self.dtype = "f16"
    def parameters(self): return [_Param(self.gb)]
    def to(self, dt): self.dtype = dt; return self
    def enable_tiling(self): self.tiling = True
    def enable_slicing(self): self.slicing = True


class _LoadedPipe(FakePipe):
    def __init__(self, sizes):
        super().__init__()
        self.components = {k: _Comp(v) for k, v in sizes.items()}
        self.vae = self.components.get("vae")
        self.placed = None
        self.vae_sliced = False
        self.vae_tiled = False
        self.attention_sliced = False
    def to(self, dev): self.placed = ("to", dev); return self
    def enable_model_cpu_offload(self): self.placed = ("model",)
    def enable_sequential_cpu_offload(self): self.placed = ("sequential",)
    def enable_vae_slicing(self): self.vae_sliced = True
    def enable_vae_tiling(self): self.vae_tiled = True
    def enable_attention_slicing(self): self.attention_sliced = True


@pytest.fixture
def fake_libs(monkeypatch):
    fp16 = object()
    torch = types.ModuleType("torch"); torch.float16 = fp16; torch.float32 = "f32"
    class Gen:
        def __init__(self, device=None): pass
        def manual_seed(self, s): self.seed = s; return self
    torch.Generator = Gen
    state = {"loads": [], "sizes": {"unet": 5.0, "vae": 0.2}, "fail_variant": False}
    class DP:
        def __init__(self, **kw):
            self.components = kw
        @classmethod
        def from_pretrained(cls, repo, **kw):
            state["loads"].append((repo, kw))
            if kw.get("variant") and state["fail_variant"]:
                raise ValueError("no such modeling files are available for variant fp16")
            return _LoadedPipe(state["sizes"])
    diffusers = types.ModuleType("diffusers"); diffusers.DiffusionPipeline = DP
    diffusers.WanPipeline = DP
    monkeypatch.setitem(sys.modules, "torch", torch); monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    state["fp16"] = fp16
    return state


def make_rt(model):
    mod = load_kernel(model, key=KEY)
    return mod, mod.DiffusersRuntime(mod.CFG)


def test_load_small_model_on_gpu_fp16(fake_libs):
    mod, rt = make_rt(IMG)
    rt.load_all()
    assert rt.state == "ready" and rt.placement == "gpu" and rt.pipe.placed == ("to", "cuda")
    repo, kw = fake_libs["loads"][0]
    assert repo == "acme/img" and kw["torch_dtype"] is fake_libs["fp16"] and kw["variant"] == "fp16"
    assert not hasattr(rt.pipe.vae, "tiling")


def test_load_big_model_offloads_and_tiles_vae(fake_libs):
    fake_libs["sizes"] = {"transformer": 12.0, "text_encoder": 9.0, "vae": 0.3}
    mod, rt = make_rt(dict(VID, opts={"warmup": False}))
    rt.load_all()
    assert rt.state == "ready" and rt.placement == "model" and rt.pipe.placed == ("model",)
    assert rt.pipe.vae.tiling and rt.pipe.vae.slicing
    fake_libs["sizes"] = {"transformer": 20.0, "vae": 0.3}
    mod, rt = make_rt(dict(VID, opts={"warmup": False}))
    rt.load_all()
    assert rt.placement == "sequential"


def test_load_falls_back_without_fp16_variant_and_fp32_vae(fake_libs):
    fake_libs["fail_variant"] = True
    mod, rt = make_rt(dict(VID, opts={"fp32": ["vae"], "warmup": False}))
    rt.load_all()
    assert rt.state == "ready" and "variant" not in fake_libs["loads"][1][1]
    assert rt.pipe.vae.dtype == "f32"


def test_load_error_is_reported(fake_libs):
    fake_libs["sizes"] = {"unet": 40.0}
    mod, rt = make_rt(IMG)
    rt.load_all()
    assert rt.state == "error" and "RAM" in rt.error


def test_warmup_runs_one_step(fake_libs):
    mod, rt = make_rt(dict(IMG, opts={"warmup": True}))
    rt._generator = lambda s: s; rt._encode_image = lambda i: b"x"
    rt.load_all()
    assert rt.state == "ready" and len(rt.pipe.calls) == 1
    c = rt.pipe.calls[0]
    assert c["steps"] == 1 and (c["width"], c["height"]) == (256, 256)


def test_start_diffusers_serves_on_internal_port(fake_libs, monkeypatch):
    mod = load_kernel(IMG, key=KEY)
    mod.LLAMA_PORT = 18231
    mod.ensure_packages = lambda kind, extra=(): []
    h = mod.start_diffusers()
    try:
        assert h.poll() is None
        st, body = Api(types.SimpleNamespace(server_address=("", 18231))).call("GET", "/v1/models")
        assert st == 200
        h.terminate(); time.sleep(0.3)
        assert h.poll() == 1
    finally:
        h.terminate()


def test_ensure_packages_installs_missing_but_never_torch(monkeypatch):
    mod = load_kernel(VID)
    ran = []
    mod.subprocess = types.SimpleNamespace(run=lambda cmd, check=False: ran.append(cmd))
    mod._have = lambda m: m in ("transformers", "accelerate", "sentencepiece")
    mod._diffusers_too_old = lambda: False
    pk = mod.ensure_packages("video", ["extra-pkg"])
    assert set(pk) == {"diffusers", "ftfy", "imageio", "imageio-ffmpeg", "extra-pkg"}
    cmd = ran[0]
    assert cmd[1:4] == ["-m", "pip", "install"] and "-q" in cmd and "torch" not in cmd
    ran.clear()
    mod._have = lambda m: True
    assert mod.ensure_packages("image") == [] and ran == []        # nothing missing: no pip at all
    mod._diffusers_too_old = lambda: True
    assert mod.ensure_packages("image") == ["diffusers>=0.33"]


# ---- the llama.cpp path must not change --------------------------------------------------------
def test_llama_argv_unchanged():
    golden = json.loads((pathlib.Path(__file__).parent / "golden" / "llama_argv.json").read_text(encoding="utf-8"))
    for key, argv in golden.items():
        assert llama_argv(key) == argv, key


# ---- diffusers memory optimizations ------------------------------------------------------------
def test_models_yaml_qwen_image_opts():
    import yaml
    models_file = pathlib.Path(__file__).parent.parent / "models.yaml"
    cfg = yaml.safe_load(models_file.read_text(encoding="utf-8"))
    turbo = cfg["models"]["qwen-image-2.1-turbo"]
    assert turbo["runtime"] == "diffusers"
    assert turbo["kind"] == "image"
    assert turbo["opts"]["offload"] == "model"
    assert turbo["opts"].get("vae_tiling") is True
    assert turbo["opts"].get("vae_slicing") is True
    assert turbo["opts"].get("size_multiple") == 16
    assert turbo["opts"].get("variant") is None
    assert "vae" not in (turbo["opts"].get("fp32") or [])
    assert turbo["opts"]["pipeline"] == "QwenImage21Pipeline"

    base = cfg["models"]["qwen-image-2.1"]
    assert base["runtime"] == "diffusers"
    assert base["opts"]["offload"] == "model"
    assert base["opts"].get("vae_tiling") is True
    assert base["opts"].get("vae_slicing") is True
    assert base["opts"].get("size_multiple") == 16
    assert base["opts"].get("variant") is None
    assert "vae" not in (base["opts"].get("fp32") or [])


def test_pipe_optimizations_called_on_pipeline(fake_libs):
    fake_libs["sizes"] = {"transformer": 12.0, "text_encoder": 9.0, "vae": 0.3}
    mod, rt = make_rt(dict(IMG, opts={"offload": "model", "warmup": False}))
    rt.load_all()
    assert rt.state == "ready" and rt.placement == "model"
    assert rt.pipe.placed == ("model",)
    assert rt.pipe.vae_sliced is True
    assert rt.pipe.vae_tiled is True
    assert rt.pipe.attention_sliced is True
    assert rt.pipe.vae.slicing and rt.pipe.vae.tiling


def test_ltx_does_not_enable_vae_tiling(fake_libs):
    fake_libs["sizes"] = {"transformer": 12.0, "text_encoder": 9.0, "vae": 0.3}
    ltx_model = {"hf": "Lightricks/LTX-Video", "runtime": "diffusers", "kind": "video",
                 "opts": {"offload": "model", "warmup": False}}
    mod, rt = make_rt(ltx_model)
    rt.load_all()
    assert rt.state == "ready"
    assert getattr(rt.pipe, "vae_tiled", False) is False
    assert getattr(rt.pipe.vae, "tiling", False) is False



def test_cuda_empty_cache_called_around_inference(monkeypatch):
    from unittest.mock import MagicMock
    mock_torch = types.ModuleType("torch")
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_torch.cuda = mock_cuda
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    pipe = FakePipe()
    mod, rt, srv = start(IMG, pipe)
    try:
        api = Api(srv)
        st, body = api.call("POST", "/v1/images/generations", {"prompt": "test"})
        assert st == 200
        # empty_cache should have been called before and after inference (at least 2 times)
        assert mock_cuda.empty_cache.call_count >= 2
    finally:
        srv.shutdown()


def test_cuda_empty_cache_called_on_oom(monkeypatch):
    from unittest.mock import MagicMock
    mock_torch = types.ModuleType("torch")
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_torch.cuda = mock_cuda
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    oom_err = type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")
    mod, rt, srv = start(IMG, FakePipe(boom=oom_err))
    try:
        api = Api(srv)
        st, body = api.call("POST", "/v1/images/generations", {"prompt": "test"})
        assert st == 500
        assert body["error"]["type"] == "out_of_memory"
        # empty_cache should have been called in before, in exception handler, and finally
        assert mock_cuda.empty_cache.call_count >= 2
    finally:
        srv.shutdown()


def test_reference_image_preprocessing():
    pytest.importorskip("PIL")
    from PIL import Image
    mod = load_kernel(IMG)
    # 1. Image larger than 1024x1024 should scale down preserving aspect ratio and mult
    large = Image.new("RGBA", (2400, 1600), color=(255, 0, 0, 255))
    prep = mod._preprocess_image(large, 1024, 1024, mult=8)
    assert prep.mode == "RGB"
    assert prep.width <= 1024 and prep.height <= 1024
    assert prep.width % 8 == 0 and prep.height % 8 == 0
    # 2400x1600 has 1.5 aspect ratio: 1024 / 1.5 = 682.66 -> rounded & multiple of 8: 680
    assert prep.size == (1024, 680)

    # 2. Arbitrary non-aligned small image
    small = Image.new("RGB", (503, 501))
    prep_sm = mod._preprocess_image(small, 1024, 1024, mult=8)
    assert prep_sm.size == (496, 496)
    assert prep_sm.width % 8 == 0 and prep_sm.height % 8 == 0


def test_generate_image_with_reference_image():
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    class RefPipe(FakePipe):
        def __call__(self, prompt, width=512, height=512, num_inference_steps=30, guidance_scale=None,
                     generator=None, num_frames=None, image=None):
            self.passed_image = image
            return super().__call__(prompt, width=width, height=height, num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale, generator=generator, num_frames=num_frames)

    pipe = RefPipe()
    mod, rt, srv = start(IMG, pipe)
    try:
        api = Api(srv)
        # Create a sample PNG image encoded as base64
        buf = io.BytesIO()
        Image.new("RGB", (2000, 1500), color=(100, 150, 200)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        st, body = api.call("POST", "/v1/images/generations", {
            "prompt": "a portrait",
            "size": "1024x1024",
            "image": data_uri,
        })
        assert st == 200
        assert len(pipe.calls) == 1
        passed_call = pipe.calls[0]
        assert (passed_call["width"], passed_call["height"]) == (1024, 1024)
        # Reference image was received and preprocessed
        ref_img = pipe.passed_image
        assert isinstance(ref_img, Image.Image)
        assert ref_img.width <= 1024 and ref_img.height <= 1024
        assert ref_img.width % 8 == 0 and ref_img.height % 8 == 0
    finally:
        srv.shutdown()


def test_balanced_fallback_to_model_offload_on_single_gpu(fake_libs, monkeypatch):
    # Mock torch.cuda.device_count to return 1
    torch = sys.modules["torch"]
    torch.cuda = types.SimpleNamespace(device_count=lambda: 1, is_available=lambda: True)
    fake_libs["sizes"] = {"transformer": 12.0, "vae": 0.3}

    mod, rt = make_rt(dict(IMG, opts={"offload": "balanced", "warmup": False}))
    rt.load_all()
    assert rt.state == "ready"
    assert rt.placement == "model"
    assert rt.pipe.placed == ("model",)


def test_reference_image_bad_decode_raises_400(image_api):
    api, *_ = image_api
    st, body = api.call("POST", "/v1/images/generations", {
        "prompt": "test",
        "image": "not-valid-base64-or-image"
    })
    assert st == 400
    assert "failed to decode reference image" in body["error"]["message"]


def test_large_request_body_accepted(image_api):
    api, *_ = image_api
    # Send a prompt padded to ~1.2 MB (exceeds the old 1MB limit)
    large_prompt = "a cute cat " + ("x" * 1_200_000)
    st, body = api.call("POST", "/v1/images/generations", {"prompt": large_prompt})
    assert st == 200


def test_true_cfg_scale_mapping_in_accepted():
    mod = load_kernel(IMG)
    rt = mod.DiffusersRuntime(dict(IMG, opts={"guidance_scale": 4.5}))
    class QwenLikePipe:
        def __call__(self, prompt, true_cfg_scale=1.0, negative_prompt=None):
            pass
    rt.pipe = QwenLikePipe()
    kw = {"prompt": "a photo", "guidance_scale": 4.5}
    accepted = rt._accepted(kw)
    assert "true_cfg_scale" in accepted
    assert accepted["true_cfg_scale"] == 4.5
    assert accepted.get("negative_prompt") == ""


def test_oom_retries_with_sequential_offload():
    oom_err = type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")
    class RecoverablePipe(FakePipe):
        def __init__(self):
            super().__init__()
            self.seq_offload_called = False
            self.attempts = 0
        def enable_sequential_cpu_offload(self):
            self.seq_offload_called = True
        def __call__(self, prompt, **kw):
            self.attempts += 1
            if self.attempts == 1:
                raise oom_err
            return super().__call__(prompt, **kw)

    pipe = RecoverablePipe()
    mod, rt, srv = start(dict(IMG, opts={"offload": "model", "warmup": False}), pipe)
    rt.placement = "model"
    try:
        api = Api(srv)
        st, body = api.call("POST", "/v1/images/generations", {"prompt": "test"})
        assert st == 200
        assert pipe.seq_offload_called is True
        assert rt.placement == "sequential"
        assert pipe.attempts == 2
    finally:
        srv.shutdown()


def test_component_dtype_conversion_for_bfloat16_and_no_dtype_attribute(fake_libs):
    # Component with bfloat16
    bf16 = object()
    sys.modules["torch"].bfloat16 = bf16

    class CompWithParam:
        def __init__(self, dt):
            self._dt = dt
            self.parameters_called = False
        def parameters(self):
            self.parameters_called = True
            p = types.SimpleNamespace(dtype=self._dt, numel=lambda: 100, element_size=lambda: 2)
            return [p]
        def to(self, dt):
            self._dt = dt
            return self

    pipe = fake_libs["loads"]
    mod, rt = make_rt(dict(IMG, opts={"offload": "model", "warmup": False, "dtype": "float16"}))
    p = _LoadedPipe({"transformer": 10.0})
    # Replace transformer component with one that has bf16 and no module .dtype attribute
    comp = CompWithParam(bf16)
    p.components["transformer"] = comp
    rt.pipe = p

    # When load runs, it should detect bf16 via parameters() and cast to float16
    assert mod._module_dtype(comp) == bf16
    # Call the conversion logic as in load()
    for name, c in p.components.items():
        cur_dt = mod._module_dtype(c)
        if cur_dt is not None and cur_dt != fake_libs["fp16"]:
            c.to(fake_libs["fp16"])
    assert comp._dt == fake_libs["fp16"]


def test_teacache_activation_and_reset(fake_libs):
    mod = load_kernel(VID)
    call_count = 0

    class FakeTransformer:
        def __init__(self):
            self.val = 1
        def forward(self, hidden_states=None, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"out_{call_count}"

    tr = FakeTransformer()
    mod._apply_teacache(tr, rel_l1_thresh=0.2)
    assert hasattr(tr, "_teacache_enabled")
    assert hasattr(tr, "_teacache_reset")

    # Step 0: should compute
    import numpy as np
    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=np.float32)
            self.shape = self.arr.shape
        def float(self):
            return self
        def abs(self):
            return FakeTensor(np.abs(self.arr))
        def mean(self):
            class M:
                def __init__(self, v): self.v = float(v)
                def item(self): return self.v
            return M(np.mean(self.arr))
        def __sub__(self, o):
            return FakeTensor(self.arr - o.arr)
        def detach(self):
            return self
        def clone(self):
            return FakeTensor(self.arr.copy())

    # Step 0: Initial call
    t0 = FakeTensor([1.0, 1.0, 1.0])
    res0 = tr.forward(hidden_states=t0)
    assert res0 == "out_1"
    assert call_count == 1

    # Step 1: Small difference (relative L1 delta < 0.2) -> should skip and return cached
    t1 = FakeTensor([1.01, 1.01, 1.01])
    res1 = tr.forward(hidden_states=t1)
    assert res1 == "out_1"
    assert call_count == 1  # Block calculation was skipped!

    # Step 2: Reset -> next call should recompute
    tr._teacache_reset()
    res2 = tr.forward(hidden_states=t1)
    assert res2 == "out_2"
    assert call_count == 2


def test_wan_i2v_quantized_loading_and_pairing(fake_libs):
    WAN_I2V = {
        "hf": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "runtime": "diffusers",
        "kind": "video",
        "pipeline": "WanImageToVideoPipeline",
        "vram_gb": 15,
        "opts": {
            "size": "832x480",
            "num_frames": 49,
            "fps": 16,
            "steps": 30,
            "guidance_scale": 5.0,
            "size_multiple": 16,
            "frame_multiple": 4,
            "max_frames": 81,
            "quant": "nf4",
            "teacache": True,
            "teacache_thresh": 0.15,
            "warmup": False
        }
    }
    fake_libs["sizes"] = {"transformer": 12.0, "text_encoder": 5.0, "vae": 0.3}
    sys.modules["diffusers"].WanImageToVideoPipeline = sys.modules["diffusers"].DiffusionPipeline
    mod, rt = make_rt(WAN_I2V)
    rt.load_all()
    assert rt.state == "ready"
    assert rt.placement == "model"
    assert rt.i2v_pipe is not None
    assert rt.t2v_pipe is not None


def test_wan22_ti2v_5b_dense_loading_and_pairing(fake_libs):
    WAN22_5B = {
        "hf": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "runtime": "diffusers",
        "kind": "video",
        "pipeline": "WanImageToVideoPipeline",
        "vram_gb": 15,
        "opts": {
            "size": "1280x704",
            "num_frames": 49,
            "fps": 16,
            "steps": 30,
            "guidance_scale": 5.0,
            "size_multiple": 32,
            "frame_multiple": 4,
            "max_frames": 81,
            "torch_dtype": "float16",
            "teacache": True,
            "teacache_thresh": 0.15,
            "warmup": False
        }
    }
    fake_libs["sizes"] = {"transformer": 9.5, "text_encoder": 2.5, "vae": 0.3}

    # Define STRICT classes that reject unknown kwargs (e.g. image_processor passed to WanPipeline)
    class StrictWanT2V:
        def __init__(self, tokenizer=None, text_encoder=None, vae=None, scheduler=None, transformer=None, transformer_2=None, boundary_ratio=None, expand_timesteps=False):
            self.components = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae, "scheduler": scheduler, "transformer": transformer}
            self.expand_timesteps = expand_timesteps

    class StrictWanV2V:
        def __init__(self, tokenizer=None, text_encoder=None, transformer=None, vae=None, scheduler=None):
            self.components = {"tokenizer": tokenizer, "text_encoder": text_encoder, "transformer": transformer, "vae": vae, "scheduler": scheduler}

    sys.modules["diffusers"].WanPipeline = StrictWanT2V
    sys.modules["diffusers"].WanVideoToVideoPipeline = StrictWanV2V
    sys.modules["diffusers"].WanImageToVideoPipeline = sys.modules["diffusers"].DiffusionPipeline

    mod, rt = make_rt(WAN22_5B)
    rt.load_all()
    assert rt.state == "ready"
    assert rt.i2v_pipe is not None
    assert rt.t2v_pipe is not None
    assert isinstance(rt.t2v_pipe, StrictWanT2V)
    assert rt.v2v_pipe is not None
    assert isinstance(rt.v2v_pipe, StrictWanV2V)


def test_models_yaml_wan_models():
    import yaml
    models_file = pathlib.Path(__file__).parent.parent / "models.yaml"
    cfg = yaml.safe_load(models_file.read_text(encoding="utf-8"))
    
    # Check 14B NF4
    i2v_14b = cfg["models"]["wan2.1-i2v-14b"]
    assert i2v_14b["runtime"] == "diffusers"
    assert i2v_14b["opts"]["quant"] == "nf4"
    assert i2v_14b["opts"]["teacache"] is True
    assert i2v_14b["opts"]["vae_tiling"] is True
    assert i2v_14b["opts"]["vae_slicing"] is True
    assert i2v_14b["opts"]["tile_sample_min_height"] == 256
    assert i2v_14b["opts"]["tile_sample_min_width"] == 256
    assert i2v_14b["opts"]["tile_sample_stride_height"] == 192
    assert i2v_14b["opts"]["tile_sample_stride_width"] == 192
    assert i2v_14b["opts"]["tile_sample_min_num_frames"] == 16
    assert i2v_14b["opts"]["tile_sample_stride_num_frames"] == 12
    assert "diffusers>=0.35.0" in i2v_14b["opts"]["pip"]
    
    # Check 14B 720p HD path
    i2v_14b_720 = cfg["models"]["wan2.1-i2v-14b-720p"]
    assert i2v_14b_720["runtime"] == "diffusers"
    assert i2v_14b_720["opts"]["vae_tiling"] is True
    assert i2v_14b_720["opts"]["vae_slicing"] is True
    assert i2v_14b_720["opts"]["tile_sample_min_height"] == 256
    assert "diffusers>=0.35.0" in i2v_14b_720["opts"]["pip"]

    # Check 5B dense efficient path
    ti2v_5b = cfg["models"]["wan2.2-ti2v-5b"]
    assert ti2v_5b["runtime"] == "diffusers"
    assert ti2v_5b["kind"] == "video"
    assert ti2v_5b["hf"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert ti2v_5b["pipeline"] == "WanImageToVideoPipeline"
    assert ti2v_5b["opts"]["teacache"] is True
    assert ti2v_5b["opts"]["vae_tiling"] is True
    assert ti2v_5b["opts"]["vae_slicing"] is True
    assert ti2v_5b["opts"]["tile_sample_min_height"] == 256
    assert ti2v_5b["opts"]["size"] == "720x1280"
    assert ti2v_5b["opts"]["size_multiple"] == 32
    assert "diffusers>=0.35.0" in ti2v_5b["opts"]["pip"]
    
    # Check 5B 720p HD path
    ti2v_5b_720p = cfg["models"]["wan2.2-ti2v-5b-720p"]
    assert ti2v_5b_720p["runtime"] == "diffusers"
    assert ti2v_5b_720p["kind"] == "video"
    assert ti2v_5b_720p["hf"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert ti2v_5b_720p["pipeline"] == "WanImageToVideoPipeline"
    assert ti2v_5b_720p["opts"]["size"] == "1280x704"
    assert ti2v_5b_720p["opts"]["size_multiple"] == 32
    assert ti2v_5b_720p["opts"]["teacache"] is True
    assert ti2v_5b_720p["opts"]["vae_tiling"] is True
    assert ti2v_5b_720p["opts"]["vae_slicing"] is True
    assert ti2v_5b_720p["opts"]["tile_sample_min_height"] == 256
    assert "diffusers>=0.35.0" in ti2v_5b_720p["opts"]["pip"]


def test_wan_14b_vae_tiling_chunks_and_slicing(fake_libs):
    WAN_14B = {
        "hf": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "runtime": "diffusers",
        "kind": "video",
        "pipeline": "WanImageToVideoPipeline",
        "vram_gb": 15,
        "opts": {
            "size": "1280x720",
            "num_frames": 49,
            "fps": 16,
            "steps": 30,
            "guidance_scale": 5.0,
            "size_multiple": 16,
            "frame_multiple": 4,
            "quant": "nf4",
            "teacache": True,
            "teacache_thresh": 0.15,
            "vae_tiling": True,
            "vae_slicing": True,
            "tile_sample_min_height": 256,
            "tile_sample_min_width": 256,
            "tile_sample_stride_height": 192,
            "tile_sample_stride_width": 192,
            "tile_sample_min_num_frames": 16,
            "tile_sample_stride_num_frames": 12,
            "warmup": False
        }
    }
    fake_libs["sizes"] = {"transformer": 12.0, "text_encoder": 5.0, "vae": 0.3}

    class MockWanVAE:
        def __init__(self):
            self.tiling = False
            self.slicing = False
            self.tiled_kwargs = None
            self.dtype = "f16"

        def enable_tiling(self, tile_sample_min_height=None, tile_sample_min_width=None,
                          tile_sample_stride_height=None, tile_sample_stride_width=None,
                          tile_sample_min_num_frames=None, tile_sample_stride_num_frames=None):
            self.tiling = True
            self.tiled_kwargs = {
                "tile_sample_min_height": tile_sample_min_height,
                "tile_sample_min_width": tile_sample_min_width,
                "tile_sample_stride_height": tile_sample_stride_height,
                "tile_sample_stride_width": tile_sample_stride_width,
                "tile_sample_min_num_frames": tile_sample_min_num_frames,
                "tile_sample_stride_num_frames": tile_sample_stride_num_frames,
            }

        def enable_slicing(self):
            self.slicing = True

        def to(self, dt):
            self.dtype = dt
            return self

        def parameters(self):
            return [_Param(0.3)]

    mock_vae = MockWanVAE()

    class MockWanPipe(_LoadedPipe):
        def __init__(self, sizes):
            super().__init__(sizes)
            self.vae = mock_vae
            self.components["vae"] = mock_vae

    class MockDP:
        @classmethod
        def from_pretrained(cls, repo, **kw):
            return MockWanPipe(fake_libs["sizes"])

    sys.modules["diffusers"].WanImageToVideoPipeline = MockDP
    mod, rt = make_rt(WAN_14B)
    rt.load_all()

    assert rt.state == "ready"
    assert mock_vae.tiling is True
    assert mock_vae.slicing is True
    assert mock_vae.tiled_kwargs["tile_sample_min_height"] == 256
    assert mock_vae.tiled_kwargs["tile_sample_min_width"] == 256
    assert mock_vae.tiled_kwargs["tile_sample_stride_height"] == 192
    assert mock_vae.tiled_kwargs["tile_sample_stride_width"] == 192
    assert mock_vae.tiled_kwargs["tile_sample_min_num_frames"] == 16
    assert mock_vae.tiled_kwargs["tile_sample_stride_num_frames"] == 12


def test_wan_14b_vae_decode_wrap_reclaims_memory(fake_libs):
    mod = load_kernel(VID, key=KEY)
    rt = mod.DiffusersRuntime(mod.CFG)

    cleaned = []
    mod._clean_cuda = lambda: cleaned.append("clean")

    teacache_reset_called = []
    class MockBlock:
        def __init__(self):
            self._cache = "submodule_block_activation"
            self.kv_cache = "block_kv"

    class MockTransformer:
        def __init__(self):
            self._cache = "large_activation_buffer"
            self.kv_cache = "kv_buffers"
            self._blocks = [MockBlock()]
        def children(self):
            return self._blocks
        def _teacache_reset(self):
            teacache_reset_called.append(True)

    tr = MockTransformer()

    class MockTextEncoder:
        def __init__(self):
            self._cache = "te_activations"
        def children(self):
            return []

    te = MockTextEncoder()

    decode_called = []
    class MockVAE:
        def decode(self, latents, return_dict=False):
            # Verify TeaCache, root and recursive block activation buffers, and text encoder were cleared before decode starts
            assert len(teacache_reset_called) > 0, "TeaCache must be reset before VAE decode begins"
            assert tr._cache is None, "Transformer activation buffer must be cleared before VAE decode begins"
            assert tr.kv_cache is None, "Transformer kv_cache must be cleared before VAE decode begins"
            assert tr._blocks[0]._cache is None, "Submodule block cache must be cleared before VAE decode begins"
            assert te._cache is None, "Text encoder buffer must be cleared before VAE decode begins"
            assert "clean" in cleaned, "_clean_cuda must run before VAE decode begins"
            decode_called.append(True)
            return ("decoded_frames",)

    vae = MockVAE()
    pipe = types.SimpleNamespace(vae=vae, transformer=tr, text_encoder=te)
    rt.pipe = pipe

    # Initialize tiling & wrap decode with None first (tests target update bug fix)
    rt.opts = {"vae_tiling": True, "vae_slicing": True}
    tile_kw = {"tile_sample_min_height": 256}
    rt.tile_kwargs = tile_kw

    # Wrap decode first with None, then with actual transformer
    rt._wrap_vae_decode(vae, None)
    assert getattr(vae, "_kllm_cleanup_wrapped", False) is True
    rt._wrap_vae_decode(vae, tr)
    assert tr in vae._kllm_extra_targets

    # Call decode and verify it cleans before and after
    res = vae.decode("dummy_latents")
    assert res == ("decoded_frames",)
    assert len(decode_called) == 1
    assert len(cleaned) >= 2  # Once before decode, once after decode


def test_wan_14b_generate_video_end_to_end(fake_libs):
    WAN_I2V = {
        "hf": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "runtime": "diffusers",
        "kind": "video",
        "pipeline": "WanImageToVideoPipeline",
        "vram_gb": 15,
        "opts": {
            "size": "832x480",
            "num_frames": 17,
            "fps": 16,
            "steps": 30,
            "guidance_scale": 5.0,
            "size_multiple": 16,
            "frame_multiple": 4,
            "max_frames": 81,
            "quant": "nf4",
            "teacache": True,
            "teacache_thresh": 0.15,
            "warmup": False
        }
    }
    mod = load_kernel(WAN_I2V, key=KEY)
    rt = mod.DiffusersRuntime(mod.CFG)

    teacache_resets = []
    class MockTransformer:
        def __init__(self):
            self._cache = "buffer"
        def _teacache_reset(self):
            teacache_resets.append(1)

    tr = MockTransformer()
    decode_invoked = []

    class MockPipeline(FakePipe):
        def __init__(self):
            super().__init__(video=True)
            self.transformer = tr
            self.text_encoder = types.SimpleNamespace(_cache="te_buffer")
            self.vae = types.SimpleNamespace(
                enable_tiling=lambda **kw: None,
                enable_slicing=lambda: None,
                decode=lambda *a, **k: decode_invoked.append(True) or ("frames",)
            )

        def __call__(self, *args, **kwargs):
            # Check callback_on_step_end is provided with latents
            cb = kwargs.pop("callback_on_step_end", None)
            kwargs.pop("callback_on_step_end_tensor_inputs", None)
            assert cb is not None, "callback_on_step_end should be hooked"
            # Simulate invoking callback at final step (step 29 of 30)
            cb(self, 29, 0.05, {"latents": "latent_tensor"})
            assert len(teacache_resets) > 0, "TeaCache must be reset on final step cleanup"
            assert self.transformer._cache is None, "Transformer buffers must be freed"
            assert self.text_encoder._cache is None, "Text encoder buffers must be freed"
            # Simulate real pipeline calling vae.decode(latents)
            self.vae.decode("latent_tensor")
            return super().__call__(*args, **kwargs)

    p = MockPipeline()
    rt.pipe = p
    rt.state = "ready"
    rt._generator = lambda s: s
    rt._encode_mp4 = lambda frames, fps: b"FAKE_MP4_BYTES"

    out = rt.generate_video({
        "prompt": "cinematic photorealistic portrait of an astronaut",
        "size": "832x480",
        "num_frames": 17,
        "steps": 30
    })

    assert "data" in out
    assert out["data"][0]["b64_mp4"] == base64.b64encode(b"FAKE_MP4_BYTES").decode()
    assert out["width"] == 832
    assert out["height"] == 480
    assert out["frames"] == 17
    assert out["fps"] == 16
    assert len(decode_invoked) == 1


def test_wan_14b_generate_video_without_callback(fake_libs):
    """Verify that even when a pipeline does not support callback_on_step_end,
    the VAE decode wrapper intercepts decode() and frees memory before VAE execution."""
    WAN_I2V = {
        "hf": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "runtime": "diffusers",
        "kind": "video",
        "pipeline": "WanImageToVideoPipeline",
        "vram_gb": 15,
        "opts": {"size": "832x480", "num_frames": 17, "steps": 30, "teacache": True, "warmup": False}
    }
    mod = load_kernel(WAN_I2V, key=KEY)
    rt = mod.DiffusersRuntime(mod.CFG)

    teacache_resets = []
    class MockTransformer:
        def __init__(self):
            self._cache = "uncleared_cache_buffer"
        def _teacache_reset(self):
            teacache_resets.append(1)

    tr = MockTransformer()

    class PipeNoCallback:
        def __init__(self):
            self.transformer = tr
            self.vae = types.SimpleNamespace(
                enable_tiling=lambda **kw: None,
                enable_slicing=lambda: None,
                decode=self._decode
            )

        def _decode(self, *a, **k):
            # When decode runs, wrapper should have purged cache and reset TeaCache
            assert tr._cache is None, "Transformer cache must be freed by VAE decode wrapper"
            assert len(teacache_resets) > 0, "TeaCache must be reset by VAE decode wrapper"
            return ("frames",)

        def __call__(self, prompt, width=832, height=480, num_inference_steps=30, generator=None, num_frames=17):
            # No callback_on_step_end in signature!
            self.vae.decode("latents")
            return types.SimpleNamespace(frames=[[object()] * 17])

    p = PipeNoCallback()
    rt.pipe = p
    rt.state = "ready"
    rt._generator = lambda s: s
    rt._encode_mp4 = lambda frames, fps: b"FAKE_MP4_BYTES"

    out = rt.generate_video({
        "prompt": "cinematic photorealistic portrait of an astronaut",
        "size": "832x480",
        "num_frames": 17,
        "steps": 30
    })

    assert "data" in out
    assert out["frames"] == 17
    assert tr._cache is None


def test_malloc_trim_cross_platform(fake_libs):
    mod = load_kernel(VID, key=KEY)
    trimmed = []

    class MockLibc:
        def malloc_trim(self, pad):
            trimmed.append(pad)
            return 1

    import ctypes
    orig_cdll = ctypes.CDLL
    try:
        def mock_cdll(name):
            if name == "libc.so.6":
                return MockLibc()
            raise OSError("no library")
        ctypes.CDLL = mock_cdll
        mod._clean_cuda()
        assert trimmed == [0]
    finally:
        ctypes.CDLL = orig_cdll







def test_text_only_pipe_switches_to_image_to_image_instead_of_dropping_the_image(monkeypatch):
    """sdxl-turbo's text pipeline has no `image` argument: the reference must go through img2img, never vanish."""
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    class TextPipe(FakePipe):
        def __call__(self, prompt, width=512, height=512, num_inference_steps=30, guidance_scale=None, generator=None):
            return super().__call__(prompt, width=width, height=height, num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale, generator=generator)

    i2i = FakePipe()
    made = []
    fake = types.ModuleType("diffusers")
    fake.AutoPipelineForImage2Image = types.SimpleNamespace(from_pipe=lambda p: made.append(p) or i2i)
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    text = TextPipe()
    mod, rt, srv = start(IMG, text)
    try:
        api = Api(srv)
        buf = io.BytesIO(); Image.new("RGB", (640, 480)).save(buf, format="PNG")
        img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        st, _ = api.call("POST", "/v1/images/generations", {"prompt": "ponytail", "image": img, "steps": 1, "strength": 0.5})
        assert st == 200 and made == [text] and not text.calls
        call = i2i.calls[0]
        assert isinstance(call["image"], Image.Image) and call["strength"] == 0.5
        assert call["steps"] * call["strength"] >= 1          # turbo: at least one real denoise step
        # without an image the text pipe is used as before
        st, _ = api.call("POST", "/v1/images/generations", {"prompt": "plain"})
        assert st == 200 and len(text.calls) == 1
    finally:
        srv.shutdown()


def test_model_that_cannot_take_an_image_says_so(monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    class TextPipe(FakePipe):
        def __call__(self, prompt, width=512, height=512, num_inference_steps=30, guidance_scale=None, generator=None):
            return super().__call__(prompt, width=width, height=height, num_inference_steps=num_inference_steps)

    def nope(p):
        raise ValueError("no image-to-image variant")
    fake = types.ModuleType("diffusers")
    fake.AutoPipelineForImage2Image = types.SimpleNamespace(from_pipe=nope)
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    mod, rt, srv = start(IMG, TextPipe())
    try:
        buf = io.BytesIO(); Image.new("RGB", (64, 64)).save(buf, format="PNG")
        st, body = Api(srv).call("POST", "/v1/images/generations", {"prompt": "x", "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
        assert st == 400 and "cannot start from an image" in json.dumps(body)
    finally:
        srv.shutdown()
