"""kllm ComfyUI video runtime (Wan2.2 on Kaggle 2x T4). Embedded into server_kernel.py by the orchestrator
when a models.yaml entry has `runtime: comfyui`; started by the kernel via start(cfg, port, log, ntfy).

Serves (behind the kernel's keep-alive proxy, so 20+ minute requests survive Cloudflare):
  GET  /health                      200 when ComfyUI is up
  GET  /v1/models
  POST /v1/videos/generations       JSON in, {"data":[{"b64_mp4":..}], fps, frames, width, height, duration_seconds, timings}

Request fields (all optional except prompt):
  mode        "i2v" | "replace" | "recreate"   (default: replace if a video is given, else i2v)
  prompt      text. For replace: who the new person is + what they do. For recreate: the whole scene.
  image       reference image (data URL / base64). i2v: first frame. replace/recreate: the character to cast.
  video       reference video (data URL / base64 mp4)               [replace / recreate]
  target      SAM3 text for the person to replace, e.g. "man in blue shirt"   [replace; optional for recreate = hybrid control]
  outfit      "long" when the reference wears a long garment (saree, long kurta/dress): widens the edit mask below the hips [replace]
  duration    seconds: 5 (default), 10 or 15  (or num_frames)
  size        "480p" (default) | "720p" | "WxH"   (orientation follows the input)
  quality     "best" (default: 3-phase CFG start, both T4s) | "fast" (4-step distilled) | "max" (5 s @480p only: most physical)
  hq          true / "smooth" -> RIFE x2 (32 fps) + lanczos to ~1080 wide;  "sharp" -> also Real-ESRGAN x2 (crisper, airbrushed skin)
  seed, steps, start_seconds (where to cut the reference video), negative_prompt

Recipe numbers come from experiments/wan22/RESULTS.md (measured + visually verified on Kaggle 2x T4).
"""
import base64
import signal, json, os, re, shutil, subprocess, sys, threading, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMFY_COMMIT = "0decb39"
NODES = {
    "gguf": "https://github.com/city96/ComfyUI-GGUF",
    "vhs": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    "kj": "https://github.com/kijai/ComfyUI-KJNodes",
    "wap": "https://github.com/kijai/ComfyUI-WanAnimatePreprocess",
    "rife": "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
    "ray": "https://github.com/komikndr/raylight",
}
RAY_PIP = ["ray>=2.48.0", "xfuser>=0.4.4,<0.5", "git+https://github.com/feifeibear/long-context-attention.git"]

M = {  # repo, file, models/<dir>
    "t5": ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    "vae": ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    "clipvis": ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision"),
    "i2v_high": ("QuantStack/Wan2.2-I2V-A14B-GGUF", "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", "unet"),
    "i2v_low": ("QuantStack/Wan2.2-I2V-A14B-GGUF", "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", "unet"),
    "i2v_high_q6": ("QuantStack/Wan2.2-I2V-A14B-GGUF", "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf", "unet"),
    "i2v_low_q6": ("QuantStack/Wan2.2-I2V-A14B-GGUF", "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf", "unet"),
    "lx_i2v_high": ("lightx2v/Wan2.2-Distill-Loras", "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors", "loras"),
    "lx_i2v_low": ("lightx2v/Wan2.2-Distill-Loras", "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors", "loras"),
    "svi_high": ("Kijai/WanVideo_comfy", "LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors", "loras"),
    "svi_low": ("Kijai/WanVideo_comfy", "LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors", "loras"),
    "animate": ("QuantStack/Wan2.2-Animate-14B-GGUF", "Wan2.2-Animate-14B-Q4_K_M.gguf", "unet"),
    "lx21_i2v": ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
    "relight": ("Kijai/WanVideo_comfy", "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors", "loras"),
    "sam31": ("Comfy-Org/sam3.1", "checkpoints/sam3.1_multiplex_fp16.safetensors", "checkpoints"),
    "yolo": ("Wan-AI/Wan2.2-Animate-14B", "process_checkpoint/det/yolov10m.onnx", "detection"),
    "vitpose": ("Kijai/vitpose_comfy", "onnx/vitpose_h_wholebody_model.onnx", "detection"),
    "vitpose_data": ("Kijai/vitpose_comfy", "onnx/vitpose_h_wholebody_data.bin", "detection"),
    "vace_high": ("QuantStack/Wan2.2-VACE-Fun-A14B-GGUF", "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q4_K_M.gguf", "unet"),
    "vace_low": ("QuantStack/Wan2.2-VACE-Fun-A14B-GGUF", "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_M.gguf", "unet"),
    "lx_t2v_high": ("lightx2v/Wan2.2-Distill-Loras", "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors", "loras"),
    "lx_t2v_low": ("lightx2v/Wan2.2-Distill-Loras", "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors", "loras"),
    "da3": ("Comfy-Org/Depth-Anything-3", "geometry_estimation/depth_anything_3_mono_large.safetensors", "geometry_estimation"),
    "esrgan": ("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth", "upscale_models"),
    # anchored recreate: Qwen-Image-Edit casts the reference person into frame 0 (experiments/wan22 e31c-e31i)
    "qie": ("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf", "unet"),
    "qie_light8": ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "loras"),
    "qwen_te": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
    "qwen_vae": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae"),
}
MODE_MODELS = {
    "i2v": ["t5", "vae", "i2v_high", "i2v_low", "lx_i2v_high", "lx_i2v_low", "svi_high", "svi_low"],
    "replace": ["t5", "vae", "clipvis", "animate", "lx21_i2v", "relight", "sam31", "yolo", "vitpose", "vitpose_data"],
    "recreate": ["t5", "vae", "vace_high", "vace_low", "lx_t2v_high", "lx_t2v_low", "da3", "sam31", "yolo", "vitpose", "vitpose_data",
                 "qie", "qie_light8", "qwen_te", "qwen_vae"],
    "i2v_real": ["i2v_high_q6", "i2v_low_q6"],
    "hq": ["esrgan"],
}
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
       "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
# "best": undistilled high-noise expert with real CFG for the first 2 of 6 steps (restores action semantics the
# distill LoRA flattens; e9), then LoRA steps. "fast": lightx2v 4-step (2 high + 2 low), CFG 1.
PHASES = {
    "best": (6, [{"m": "high", "lora": False, "cfg": 3.5, "end": 2}, {"m": "high", "lora": True, "cfg": 1.0, "end": 3},
                 {"m": "low", "lora": True, "cfg": 1.0, "end": 6}]),
    "fast": (4, [{"m": "high", "lora": True, "cfg": 1.0, "end": 2}, {"m": "low", "lora": True, "cfg": 1.0, "end": 4}]),
    # "max": undistilled high-noise expert with CFG for the whole high-noise half (e26: motion 13.5 = the 20-step
    # undistilled ceiling, cut hair visibly lands on the client's nape). ~23 min for 5 s @480p -> 5 s only.
    "max": (8, [{"m": "high", "lora": False, "cfg": 3.5, "end": 4}, {"m": "low", "lora": True, "cfg": 1.0, "end": 8}]),
    # "real": the only recipe that shows the action's consequence (hair actually shaved off, clumps on the floor) — every
    # distilled recipe above produced gesture without consequence on 20+ held-out scenes. Undistilled 20 steps, CFG 3.5 on
    # both experts, Q6_K, SVI conditioning even for one window (L1 PASS at 20 st; plain conditioning at 40 st only partial,
    # c1/c5b). Ceilings: c2 40 st 10 s 4 h. Cost: ~50-80 min per 5 s window on 2x T4.
    "real": (20, [{"m": "high", "lora": False, "cfg": 3.5, "end": 10}, {"m": "low", "lora": False, "cfg": 3.5, "end": 20}]),
}
WINDOW = 85            # frames per SVI window for 10/15 s (81 for a single 5 s window)
FPS = 16


class ApiError(Exception):
    def __init__(self, status, message, typ="invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.typ = status, message, typ


class Runtime:
    def __init__(self, cfg, log, ntfy):
        self.cfg, self.log, self.ntfy = cfg, log, ntfy
        self.opts = cfg.get("opts") or {}
        self.root = "/kaggle/tmp" if os.path.isdir("/kaggle") else "/tmp"
        self.comfy = os.path.join(self.root, "ComfyUI")
        self.out = os.path.join(self.root, "kllm_video_out")
        self.port = 8188
        self.proc = None
        self.lock = threading.Lock()          # one generation at a time (both GPUs are used by each job)
        self.busy, self.served, self.state, self.error = False, 0, "loading", None
        self.have_modes, self.mode_used = set(), {}
        self.use_ray = bool(self.opts.get("raylight", True))
        self.ray_gen, self.last_sig = 0, None   # Ray namespace generation; model-set signature of the last request
        # Raylight is used for i2v only. Replace (Animate + SAM3 + ViTPose + CLIP-vision) and recreate (VACE + DA3) on
        # Raylight exceed the 30 GB host RAM (P4/P6: Ray's OOM killer fired at 29.4/30 GB on every edit request, even
        # right after a worker reset); the edit modes were developed and verified on the native single-GPU path.
        self.ray_edit = bool(self.opts.get("raylight_edit", False))
        self.ray = self.use_ray

    # ------------------------------------------------------------------ setup
    def sh(self, cmd, check=True):
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
        if r.returncode and check:
            raise RuntimeError("%s failed: %s" % (cmd if isinstance(cmd, str) else " ".join(cmd), (r.stderr or r.stdout)[-1500:]))
        return r

    def install(self):
        if not os.path.isdir(self.comfy):
            self.sh(["git", "clone", "-q", "https://github.com/comfyanonymous/ComfyUI.git", self.comfy])
            self.sh(["git", "-C", self.comfy, "checkout", "-q", self.opts.get("comfy_commit", COMFY_COMMIT)], check=False)
        req = [l for l in open(os.path.join(self.comfy, "requirements.txt")).read().splitlines()
               if l.strip() and not l.startswith("#") and not re.match(r"^(torch|torchvision|torchaudio)\b", l)]
        pk = req + ["gguf>=0.13.0", "hf_xet", "huggingface_hub>=0.34", "imageio-ffmpeg", "opencv-python-headless", "onnxruntime-gpu"]
        if self.use_ray:
            pk += RAY_PIP
        self.sh([sys.executable, "-m", "pip", "install", "-q", "--no-input"] + pk, check=False)
        for key, url in NODES.items():
            if key == "ray" and not self.use_ray:
                continue
            d = os.path.join(self.comfy, "custom_nodes", url.rsplit("/", 1)[-1])
            if not os.path.isdir(d):
                self.sh(["git", "clone", "-q", "--depth", "1", url, d])
                rq = os.path.join(d, "requirements.txt")
                if os.path.exists(rq):
                    lines = [l for l in open(rq).read().splitlines() if l.strip() and not l.startswith("#")
                             and not re.match(r"^(torch|torchvision|torchaudio)\b", l)]
                    if lines:
                        self.sh([sys.executable, "-m", "pip", "install", "-q", "--no-input"] + lines, check=False)
        if self.use_ray:
            self.patch_raylight(os.path.join(self.comfy, "custom_nodes", "raylight"))
        nodes_src = self.cfg.get("_kllm_nodes_src")
        if nodes_src:
            open(os.path.join(self.comfy, "custom_nodes", "kllm_nodes.py"), "w").write(nodes_src)

    def free_gb(self):
        return shutil.disk_usage(self.root).free / 1e9

    def ensure_mode(self, mode):
        """Download a mode's weights on first use; evict the least-recently-used other mode if disk is short."""
        need = [k for k in MODE_MODELS[mode] if not os.path.exists(self._path(k))]
        size_gb = 0
        if need:
            from huggingface_hub import HfApi
            api = HfApi()
            for k in need:
                repo, f, _ = M[k]
                try:
                    size_gb += (api.get_paths_info(repo, [f])[0].size or 0) / 1e9
                except Exception:
                    size_gb += 12
            while self.free_gb() < size_gb + 4:
                victims = sorted((t, m) for m, t in self.mode_used.items() if m != mode and m in self.have_modes)
                if not victims:
                    break
                self.evict(victims[0][1])
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(self._download, need))
        self.have_modes.add(mode)
        self.mode_used[mode] = time.time()

    def evict(self, mode):
        keep = {k for m in self.have_modes if m != mode for k in MODE_MODELS[m]}
        for k in MODE_MODELS[mode]:
            if k not in keep and os.path.exists(self._path(k)):
                os.remove(self._path(k))
        self.have_modes.discard(mode)
        self.log("evicted", mode, "weights; free %.1f GB" % self.free_gb())

    def _path(self, k):
        repo, f, d = M[k]
        return os.path.join(self.comfy, "models", d, os.path.basename(f))

    def _download(self, k):
        from huggingface_hub import hf_hub_download
        repo, f, d = M[k]
        dst = self._path(k)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        t = time.time()
        p = hf_hub_download(repo, f, local_dir=os.path.join(self.root, "hf", repo.replace("/", "__")))
        shutil.move(os.path.realpath(p), dst)
        self.log("downloaded %s (%.0fs)" % (os.path.basename(f), time.time() - t))

    RAY_DROP_MARK = "# kllm: KLLM_RAY_DROP"

    def patch_raylight(self, d):
        """Make RayWorker.clear_sampling_vram drop the model instead of offloading it to host RAM when KLLM_RAY_DROP=1.
        P14 memtrace: after sampling, both workers run unload_all_models + unpatch_model(device_to=cpu), copying ~11 GB of
        GPU weights each into anonymous host RAM at the same moment -> 30 GB cgroup limit -> worker killed (staged replace)."""
        f = os.path.join(d, "src", "raylight", "distributed_worker", "ray_worker.py")
        if not os.path.exists(f):
            self.log("raylight patch: %s not found" % f)
            return False
        src = open(f).read()
        if self.RAY_DROP_MARK in src:
            return True
        m = re.search(r'\n    def clear_sampling_vram\(self\):\n(\s+"""[\s\S]*?"""\n)?', src)
        if not m or "def _free_current_model" not in src:
            self.log("raylight patch: anchor not found, leaving Raylight unpatched")
            return False
        block = ('        if os.environ.get("KLLM_RAY_DROP") == "1":   %s\n'
                 '            self._free_current_model()\n'
                 '            self.cached_base_model = None\n'
                 '            if hasattr(self, "_invalidate_non_fsdp_cache"):\n'
                 '                self._invalidate_non_fsdp_cache()\n'
                 '            gc.collect()\n'
                 '            torch.cuda.empty_cache()\n'
                 '            return\n') % self.RAY_DROP_MARK
        open(f, "w").write(src[:m.end()] + block + src[m.end():])
        self.log("raylight patched: clear_sampling_vram drops weights when KLLM_RAY_DROP=1")
        return True

    def comfy_mem_flags(self):
        """Host-RAM flags for the main ComfyUI process when Raylight workers hold the diffusion model.
        P12: staged replace OOM-killed at 31.8/32 GB: 2 workers x 9.1 GB + main 6.4 GB (default --cache-ram keeps
        every output resident and pins up to 14 GB). The main process only runs T5/VAE/CLIP-vision here."""
        # --cache-none always: the RAM-cache release sizes VHS_LoadVideo's lazy audio output, which runs ffmpeg and
        # raises on clips without an audio track, failing the NEXT prompt (e31c/e31d: every 2nd job died).
        flags = ["--cache-none"]
        if self.use_ray and self.opts.get("lean_main", True):
            flags.insert(0, "--disable-pinned-memory")
        return flags

    def _memtrace(self):
        """opts.memtrace: 1 s host-RAM trace (cgroup total + top-4 RSS). Logged every second above 24 GB, else every 10 s."""
        path, n = os.path.join(self.root, "memtrace.log"), 0
        def cg():
            for f in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                try:
                    return int(open(f).read()) / 2**30
                except Exception:
                    pass
            return -1.0
        while True:
            used, n = cg(), n + 1
            if used > 24 or n % 10 == 0:
                top = []
                for pid in os.listdir("/proc"):
                    if pid.isdigit():
                        try:
                            rss = int(open("/proc/%s/statm" % pid).read().split()[1]) * 4096 / 2**30
                            top.append((rss, pid, open("/proc/%s/cmdline" % pid).read().replace("\0", " ")[:40]))
                        except Exception:
                            pass
                top.sort(reverse=True)
                with open(path, "a") as f:
                    f.write("%.0f cg=%.2f %s\n" % (time.time(), used, " | ".join("%.1f %s %s" % t for t in top[:4])))
            time.sleep(1)

    def start_comfy(self):
        if self.opts.get("memtrace") and not getattr(self, "_mt", None):
            self._mt = threading.Thread(target=self._memtrace, daemon=True)
            self._mt.start()
        # Ray's OOM killer defaults to 95% of the 30 GB limit; Raylight's window/expert switches spike briefly above it.
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", RAY_memory_usage_threshold="0.985")
        # replace on Raylight: drop sampled weights instead of offloading them (i2v keeps them: same model across windows)
        env["KLLM_RAY_DROP"] = "1" if getattr(self, "next_mode", None) == "replace" and self.opts.get("ray_drop", True) else "0"
        self.comfy_log = os.path.join(self.root, "comfy.log")
        self.proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch",
                                      "--preview-method", "none", "--output-directory", self.out] + self.comfy_mem_flags(),
                                     cwd=self.comfy, env=env, stdout=open(self.comfy_log, "a"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
        t = time.time()
        while time.time() - t < 900:
            if self.proc.poll() is not None:
                raise RuntimeError("ComfyUI exited: " + open(self.comfy_log).read()[-2000:])
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/system_stats" % self.port, timeout=5).read()
                return
            except Exception:
                time.sleep(3)
        raise RuntimeError("ComfyUI did not start")

    def load(self):
        try:
            self.ntfy("downloading", note="comfyui: installing ComfyUI + Wan2.2 nodes")
            self.install()
            for m in self.opts.get("preload", ["i2v"]):
                self.ntfy("downloading", note="comfyui: fetching %s weights" % m)
                self.ensure_mode(m)
            self.start_comfy()
            if self.opts.get("warmup", True) and "i2v" in self.opts.get("preload", ["i2v"]):
                self.warmup()
            self.state = "ready"
        except Exception as e:
            import traceback
            self.state, self.error = "error", "%s: %s" % (type(e).__name__, e)
            self.log("comfy video runtime failed:", traceback.format_exc()[-2500:])

    def warmup(self):
        """Pay Ray actor start-up + GGUF load + LoRA patching once, before reporting ready (a cold first request
        otherwise spends ~15-20 min of its budget on it; e19). 9 frames at 320x576 through the default recipe."""
        from PIL import Image
        t = time.time()
        self.ntfy("downloading", note="comfyui: warming models (first load into both GPUs)")
        p = os.path.join(self.comfy, "input", "_warm.png")
        Image.new("RGB", (320, 576), (128, 120, 110)).save(p)
        g = self.g_i2v("_warm", p, "a person standing still", 320, 576, 1, self.opts.get("quality", "best"), 0, {"_frames": 9})
        try:
            self.run_graph(g, 3600)
            self.last_sig = ("i2v", False, self.opts.get("quality", "best"), False)
            self.log("warm-up done in %.0fs" % (time.time() - t))
        except ApiError as e:
            self.log("warm-up failed (continuing):", e.message[:500])
        shutil.rmtree(os.path.join(self.out, "_warm"), ignore_errors=True)

    # ------------------------------------------------------------------ comfy client
    def _req(self, path, data=None, timeout=120):
        r = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path),
                                   data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"})
        for attempt in range(20):
            try:
                raw = urllib.request.urlopen(r, timeout=timeout).read()
                return json.loads(raw) if raw.strip() else {}       # e.g. POST /free answers 200 with an empty body
            except urllib.error.HTTPError as e:
                raise ApiError(500, "ComfyUI rejected the graph: %s" % e.read().decode()[:3000], "generation_error")
            except (urllib.error.URLError, TimeoutError, OSError):
                if self.proc.poll() is not None:
                    raise ApiError(500, "ComfyUI died: " + open(self.comfy_log).read()[-1500:], "generation_error")
                time.sleep(10)
        raise ApiError(500, "ComfyUI unresponsive", "generation_error")

    def free(self):
        """Release everything the ComfyUI process holds on the GPUs (models, caches)."""
        try:
            self._req("/free", {"unload_models": True, "free_memory": True})
        except Exception as e:                       # best effort: never fail a request over cache release
            self.log("free failed:", str(e)[:300])

    def restart_comfy(self):
        """Fresh ComfyUI + Ray: the only reliable way back to a clean 30 GB host-RAM budget. P7/P8: after a native
        edit request the main process kept ~3-4 GB (caches/allocator) and the next long Raylight i2v was OOM-killed
        even after RayKillInit + /free; a fresh process peaks ~25 GB on the same job."""
        self.log("restarting ComfyUI + Ray for a clean memory state")
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(60)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
            except Exception:
                pass
        if self.use_ray:
            self.sh("ray stop --force", check=False)
        self.ray_gen += 1
        self.start_comfy()

    def reset_ray(self):
        """Kill the Ray workers (and their resident models) and start the next graph on a fresh namespace.
        Workers keep every model variant they ever loaded in VRAM; switching mode / SVI / recipe without this
        OOMs (P1: replace, recreate and 10 s i2v all OOM'd after a 5 s i2v)."""
        if self.proc is not None:
            return self.restart_comfy()
        if self.use_ray:
            g = {}
            self._ray_init(g)                     # same inputs as the live initializer -> ComfyUI returns the live actors
            g["kill"] = {"class_type": "RayKillInit", "inputs": {"ray_actors_init": ["ri", 0], "kill_mode": "Kill Entire Cluster"}}
            try:
                self.run_graph(g, 600)
            except ApiError as e:
                self.log("ray kill failed (continuing):", e.message[:300])
            self.ray_gen += 1
        self.free()

    def run_graph(self, g, timeout=7200):
        pid = self._req("/prompt", {"prompt": g, "client_id": "kllm"})["prompt_id"]
        t = time.time()
        while time.time() - t < timeout:
            h = self._req("/history/" + pid)
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    msgs = json.dumps(st.get("messages", []))
                    typ = "out_of_memory" if "out of memory" in msgs.lower() else "generation_error"
                    raise ApiError(500, "generation failed: " + msgs[-2500:], typ)
                return h[pid]
            time.sleep(4)
        self._req("/interrupt", {})
        raise ApiError(504, "generation timed out", "timeout")

    # ------------------------------------------------------------------ request handling
    def _media(self, req, key, name):
        v = req.get(key)
        if not v:
            return None
        if isinstance(v, str) and v.startswith("data:"):
            v = v.split(",", 1)[1]
        try:
            raw = base64.b64decode(v)
        except Exception:
            raise ApiError(400, "%s must be base64 or a data URL" % key)
        p = os.path.join(self.comfy, "input", name)
        open(p, "wb").write(raw)
        return p

    @staticmethod
    def _grid(size, aspect):
        """480p/720p buckets (multiples of 16) in the input's orientation."""
        s = str(size or "480p").lower()
        if "x" in s:
            w, h = [int(x) for x in s.split("x")]
            return w // 16 * 16, h // 16 * 16
        short, long_ = (720, 1280) if s.startswith("720") else (480, 832)
        return (short, long_) if aspect < 1 else (long_, short)

    def generate_video(self, req):
        if self.state != "ready":
            raise ApiError(503, "model is still loading" if self.state == "loading" else "runtime failed: %s" % self.error, "model_unavailable")
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiError(400, "prompt is required")
        with self.lock:
            self.busy = True
            try:
                return self._generate(req, prompt.strip())
            finally:
                self.busy = False
                self.served += 1

    def _generate(self, req, prompt):
        from PIL import Image
        tag = uuid.uuid4().hex[:10]
        img = self._media(req, "image", tag + "_ref.png")
        vid = self._media(req, "video", tag + "_src.mp4")
        mode = req.get("mode") or ("replace" if vid else "i2v")
        if mode not in ("i2v", "replace", "recreate"):
            raise ApiError(400, "mode must be i2v, replace or recreate")
        if mode == "i2v" and not img:
            raise ApiError(400, "i2v needs an image (first frame)")
        if mode in ("replace", "recreate") and not (img and vid):
            raise ApiError(400, "%s needs both a reference video and a reference image" % mode)
        if mode == "replace" and not req.get("target"):
            raise ApiError(400, "replace needs `target`: who to replace, e.g. \"man in blue shirt\"")
        dur = float(req.get("duration") or 5)
        if req.get("num_frames"):
            dur = (int(req["num_frames"]) - 1) / FPS
        windows = 1 if dur <= 5.5 else (2 if dur <= 10.5 else 3)
        if dur > 15.5:
            raise ApiError(400, "duration is capped at 15 s")
        aspect = 1.0
        if mode == "i2v":
            with Image.open(img) as im:
                aspect = im.width / im.height
        else:
            o = self.sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", vid]).stdout
            w0, h0 = [int(x) for x in o.strip().split(",")[:2]]
            aspect = w0 / h0
        W, H = self._grid(req.get("size") or self.opts.get("size"), aspect)
        big = W * H > 480 * 832
        warnings = []
        # long clips default to the distilled recipe: undistilled-CFG + SVI disintegrates on large body motion
        # (e29b/e29: walk-away story broke inside window 1; fast windows were clean in every long test)
        quality = req.get("quality") or ("fast" if (big or windows > 1) else self.opts.get("quality", "best"))
        if windows > 1 and quality == "best":
            warnings.append("quality=best for 10/15 s suits steady actions (e.g. shaving); large body motion (turning, walking) "
                            "can smear or ghost - use quality=fast for those")
        if quality not in PHASES:
            raise ApiError(400, "quality must be one of %s" % ", ".join(PHASES))
        if quality == "max" and (windows > 1 or big):
            raise ApiError(400, "quality=max is limited to 5 s at 480p (it takes ~23 min there); use best for longer/larger clips")
        if quality == "real" and big:
            raise ApiError(400, "quality=real is 480p only (720p undistilled would take many hours); use hq for 1080p delivery")
        if big and mode == "i2v" and (quality != "fast" or windows > 1):
            # measured on 2x T4 (e18/e12): 720p best = 64 min, 720p fast = ~17 min sampling per 5 s window
            raise ApiError(400, "720p is limited to quality=fast and 5 s (720p best measured 64 min); use 480p for best/10 s/15 s, or hq for 1080p delivery")
        seed = int(req.get("seed") if req.get("seed") is not None else int.from_bytes(os.urandom(4), "little"))
        t0 = time.time()
        self.ensure_mode(mode)
        if mode == "i2v" and quality == "real":
            self.ensure_mode("i2v_real")
            warnings.append("quality=real runs 20 undistilled steps with CFG: expect ~50-80 min per 5 s on 2x T4")
        hq = req.get("hq")
        sharp = hq == "sharp"
        if sharp:
            self.ensure_mode("hq")
        frames_total = 81 if windows == 1 else WINDOW + (windows - 1) * (WINDOW - 5)
        g_prep = None
        if mode == "i2v":
            g = self.g_i2v(tag, img, prompt, W, H, windows, quality, seed, req)
        else:
            src = os.path.join(self.comfy, "input", tag + "_prep.mp4")
            self._prep_video(vid, src, W, H, float(req.get("start_seconds") or 0), frames_total if mode == "replace" else 81)
            if mode == "replace":
                staged = os.path.join(self.out, tag + "_prep") if (self.use_ray and self.opts.get("replace_staged", True)) else None
                if staged:
                    g_prep = self.g_replace_prep(tag, os.path.basename(src), req["target"], W, H, windows, req)
                g = self.g_replace(tag, img, os.path.basename(src), prompt, req["target"], W, H, windows, seed, req, staged=staged)
            else:
                anchor = bool(req.get("target")) and req.get("anchor", self.opts.get("recreate_anchor", True)) is not False
                if anchor:
                    g_kf = self.g_keyframe(tag, img, os.path.basename(src), req["target"], W, H, seed, req)
                g = self.g_recreate(tag, img, os.path.basename(src), prompt, W, H, quality, seed, req,
                                    anchor_image=(tag + "_kf.png") if anchor else None)
        g_kf = locals().get("g_kf")

        def run_all():
            if g_kf:                # anchored recreate: cast the reference into frame 0 natively, drop Qwen, then animate
                ts = time.time()
                self.run_graph(g_kf)
                d = os.path.join(self.out, tag + "_kf")
                png = sorted(x for x in os.listdir(d) if x.endswith(".png"))[0]
                shutil.copy(os.path.join(d, png), os.path.join(self.comfy, "input", tag + "_kf.png"))
                self.free()
                self.log("recreate keyframe %.0fs" % (time.time() - ts))
            if g_prep:              # staged replace: preprocess natively, drop those models, then sample on both GPUs
                ts = time.time()
                self.run_graph(g_prep)
                self.free()
                self.log("replace preprocessing %.0fs" % (time.time() - ts))
            self.run_graph(self._retag_namespace(g))

        sig = (mode, windows > 1, quality, W * H > 480 * 832)
        self.next_mode = mode
        if (self.last_sig is not None and sig != self.last_sig) or (self.last_sig is None and mode == "replace" and g_prep):
            self.log("model set changes %s -> %s: resetting GPU workers" % (self.last_sig, sig))
            self.reset_ray()
        try:
            run_all()
        except ApiError as e:
            if e.typ == "out_of_memory" and not getattr(self, "_retrying", False):
                self.log("OOM: resetting GPU workers and retrying once")
                self.reset_ray()
                self._retrying = True
                try:
                    run_all()
                finally:
                    self._retrying = False
            else:
                self.last_sig = None
                raise
        finally:
            shutil.rmtree(os.path.join(self.out, tag + "_prep"), ignore_errors=True)
        self.last_sig = sig
        t_gen = time.time() - t0
        fdir = os.path.join(self.out, tag)
        fps = FPS
        if hq:
            fdir, fps = self.hq_pass(tag, fdir, W, H, sharp=sharp), 2 * FPS
        mp4 = os.path.join(self.out, tag + ".mp4")
        self.sh("ffmpeg -y -v error -framerate %d -pattern_type glob -i '%s/*.png' -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p '%s'" % (fps, fdir, mp4))
        n = len([x for x in os.listdir(fdir) if x.endswith(".png")])
        data = open(mp4, "rb").read()
        shutil.rmtree(os.path.join(self.out, tag), ignore_errors=True)
        shutil.rmtree(os.path.join(self.out, tag + "_hq"), ignore_errors=True)
        ow, oh = W, H
        if hq:
            o = self.sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", mp4]).stdout
            ow, oh = [int(x) for x in o.strip().split(",")[:2]]
        for f in (img, vid):
            if f and os.path.exists(f):
                os.remove(f)
        item = {"b64_mp4": base64.b64encode(data).decode(), "seed": seed}
        kf = os.path.join(self.comfy, "input", tag + "_kf.png")
        if os.path.exists(kf):   # anchored recreate: the cast frame 0, so a wrong target is visible to the caller
            item["keyframe_b64_png"] = base64.b64encode(open(kf, "rb").read()).decode()
            os.remove(kf)
            shutil.rmtree(os.path.join(self.out, tag + "_kf"), ignore_errors=True)
        return {"created": int(time.time()), "data": [item],
                "fps": fps, "frames": n, "width": ow, "height": oh,
                "duration_seconds": round(n / fps, 2), "mode": mode, "quality": quality, "warnings": warnings,
                "timings": {"generate_s": round(t_gen, 1), "total_s": round(time.time() - t0, 1)}}

    def _retag_namespace(self, g):
        if "ri" in g:
            g["ri"]["inputs"]["ray_cluster_namespace"] = "kllm%d" % self.ray_gen
        return g

    def _prep_video(self, src, dst, W, H, ss, frames):
        cd = subprocess.run(["ffmpeg", "-v", "info", "-ss", str(ss), "-t", "4", "-i", src, "-vf", "cropdetect=24:2:0", "-f", "null", "-"],
                            capture_output=True, text=True).stderr
        crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", cd)
        pre = ("crop=%s," % crops[-1]) if crops else ""
        vf = pre + "fps=16,scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos,crop=%d:%d" % (W, H, W, H)
        self.sh(["ffmpeg", "-y", "-v", "error", "-ss", str(ss), "-i", src, "-frames:v", str(frames), "-vf", vf, "-an",
                 "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", dst])

    # ------------------------------------------------------------------ graphs
    def _ray_init(self, g):
        g["ri"] = {"class_type": "RayInitializer", "inputs": {"ray_cluster_address": "local", "ray_cluster_namespace": "kllm%d" % self.ray_gen, "GPU": 2,
                   "ulysses_degree": 2, "ring_degree": 1, "cfg_degree": 1, "dp_degree": 1, "sync_ulysses": False,
                   "clear_vram_after_sampling": True, "FSDP": False, "FSDP_CPU_OFFLOAD": False,
                   "XFuser_attention": "TORCH_EFFICIENT", "skip_comm_test": True, "use_mmap": True}}

    def _ray_model(self, g, tag, unet, loras, shift=None):
        prev = None
        for i, (name, w) in enumerate(loras):
            nid = "rl_%s_%d" % (tag, i)
            g[nid] = {"class_type": "RayLoraLoader", "inputs": {"lora_name": name, "strength_model": w}}
            if prev:
                g[nid]["inputs"]["prev_ray_lora"] = prev
            prev = [nid, 0]
        ld = {"unet_name": unet, "dequant_dtype": "default", "patch_dtype": "default", "ray_actors_init": ["ri", 0]}
        if prev:
            ld["lora"] = prev
        g["u_" + tag] = {"class_type": "RayGGUFLoader", "inputs": ld}
        if shift:
            g["m_" + tag] = {"class_type": "RayModelSamplingSD3", "inputs": {"ray_actors": ["u_" + tag, 0], "shift": shift}}
            return ["m_" + tag, 0]
        return ["u_" + tag, 0]

    def _native_model(self, g, tag, unet, loras, shift=None, device=None):
        g["u_" + tag] = {"class_type": "KllmUnetLoaderGGUF", "inputs": {"unet_name": unet}}
        cur = ["u_" + tag, 0]
        for i, (name, w) in enumerate(loras):
            nid = "l_%s_%d" % (tag, i)
            g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": cur, "lora_name": name, "strength_model": w}}
            cur = [nid, 0]
        if shift:
            g["m_" + tag] = {"class_type": "ModelSamplingSD3", "inputs": {"model": cur, "shift": shift}}
            cur = ["m_" + tag, 0]
        return cur

    def _sampler(self, g, nid, model, seed, steps, cfg, pos, neg, lat, start, end, first, last):
        common = {"noise_seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                  "positive": pos, "negative": neg, "latent_image": lat, "start_at_step": start, "end_at_step": 10000 if last else end,
                  "add_noise": "enable" if first else "disable", "return_with_leftover_noise": "disable" if last else "enable"}
        if self.ray:
            g[nid] = {"class_type": "XFuserKSamplerAdvanced", "inputs": dict(common, ray_actors=model, denoise=1.0)}
        else:
            g[nid] = {"class_type": "KSamplerAdvanced", "inputs": dict(common, model=model)}
        return [nid, 0]

    def _expert_variants(self, g, high, low, lx, svi):
        cache = {}

        def get(m, lora):
            if (m, lora) not in cache:
                # `lora` toggles only the distillation LoRA. The SVI LoRA is part of the conditioning contract
                # (anchor + zero-padded motion latents) and must be on in EVERY phase: without it the base model
                # turns SVI conditioning into noise (e19/e20: 3-phase + SVI collapsed after frame 0).
                loras = ([[os.path.basename(M[lx[m]][1]), 1.0]] if lora else []) + ([[os.path.basename(M[svi[m]][1]), 1.0]] if svi else [])
                unet = os.path.basename(M[high if m == "high" else low][1])
                tag = "%s%d" % (m, lora)
                cache[(m, lora)] = (self._ray_model if self.ray else self._native_model)(g, tag, unet, loras, 5.0)
            return cache[(m, lora)]
        return get

    def g_i2v(self, tag, img, prompt, W, H, windows, quality, seed, req):
        steps, phases = PHASES[quality]
        self.ray = self.use_ray
        g = {"t5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
             "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": req.get("negative_prompt") or NEG}},
             "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
             "img": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(img)}},
             "rs": {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}}}
        if self.ray:
            self._ray_init(g)
        real = quality == "real"
        svi = windows > 1 or real
        get = self._expert_variants(g, "i2v_high_q6" if real else "i2v_high", "i2v_low_q6" if real else "i2v_low",
                                    {"high": "lx_i2v_high", "low": "lx_i2v_low"},
                                    {"high": "svi_high", "low": "svi_low"} if svi else None)
        prompts = req.get("prompts") if isinstance(req.get("prompts"), list) else [prompt] * windows
        prompts = (prompts + [prompts[-1]] * windows)[:windows]
        if svi:
            g["anchor"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["rs", 0], "vae": ["vae", 0]}}
        prev, acc = None, None
        for k in range(windows):
            g["pos%d" % k] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": prompts[k]}}
            if svi:
                c = {"positive": ["pos%d" % k, 0], "negative": ["neg", 0], "length": WINDOW, "anchor_samples": ["anchor", 0], "motion_latent_count": 1}
                if prev:
                    c["prev_samples"] = prev
                g["c%d" % k] = {"class_type": "WanImageToVideoSVIPro", "inputs": c}
            else:
                g["c%d" % k] = {"class_type": "WanImageToVideo", "inputs": {"positive": ["pos%d" % k, 0], "negative": ["neg", 0], "vae": ["vae", 0],
                                "width": W, "height": H, "length": int(req.get("_frames") or 81), "batch_size": 1, "start_image": ["rs", 0]}}
            lat, start = ["c%d" % k, 2], 0
            for pi, ph in enumerate(phases):
                lat = self._sampler(g, "k%d_%d" % (k, pi), get(ph["m"], ph["lora"]), seed + k, steps, ph["cfg"], ["c%d" % k, 0], ["c%d" % k, 1],
                                    lat, start, ph["end"], pi == 0, pi == len(phases) - 1)
                start = ph["end"]
            prev = lat
            g["d%d" % k] = self._decode(lat, W, H)
            if k == 0:
                acc = ["d0", 0]
                continue
            g["t%d" % k] = {"class_type": "ImageFromBatch", "inputs": {"image": ["d%d" % k, 0], "batch_index": 1, "length": 4096}}
            g["x%d" % k] = {"class_type": "ImageBatchExtendWithOverlap", "inputs": {"source_images": acc, "overlap": 4, "overlap_side": "source",
                            "overlap_mode": "linear_blend", "new_images": ["t%d" % k, 0]}}
            acc = ["x%d" % k, 2]
        g["save"] = {"class_type": "SaveImage", "inputs": {"images": acc, "filename_prefix": tag + "/f"}}
        return g

    def _decode(self, lat, W, H):
        if W * H > 480 * 832:
            return {"class_type": "VAEDecodeTiled", "inputs": {"samples": lat, "vae": ["vae", 0], "tile_size": 512, "overlap": 64,
                                                               "temporal_size": 64, "temporal_overlap": 8}}
        return {"class_type": "VAEDecode", "inputs": {"samples": lat, "vae": ["vae", 0]}}

    PREP_NODES = ("pm", "sam", "samtxt", "trk", "m0", "pose", "cm", "samp", "trkp", "prot", "skirt", "grow", "blk", "drw")

    def g_replace_prep(self, tag, video, target, W, H, windows, req):
        """Stage 1 of the staged (Raylight) replace: SAM3 + ViTPose + masks only, saved as PNG sequences.
        Run natively, then the main process drops these models before the 2-GPU sampling stage (P4: everything
        resident at once exceeded the 30 GB host RAM)."""
        full = self.g_replace(tag, "unused.png", video, "-", target, W, H, windows, 0, req)
        g = {k: v for k, v in full.items() if k in self.PREP_NODES or k == "v"}
        g["mimg"] = {"class_type": "MaskToImage", "inputs": {"mask": ["blk", 0]}}
        for nid, src, sub in (("s_pose", ["drw", 0], "pose"), ("s_face", ["pose", 1], "face"), ("s_mask", ["mimg", 0], "mask")):
            g[nid] = {"class_type": "SaveImage", "inputs": {"images": src, "filename_prefix": "%s_prep/%s/f" % (tag, sub)}}
        return g

    def g_replace(self, tag, img, video, prompt, target, W, H, windows, seed, req, staged=None):
        L, steps = 77, int(req.get("steps") or 6)
        self.ray = self.use_ray and (self.ray_edit or bool(staged))
        total = L + (windows - 1) * (L - 5)
        loras = [[os.path.basename(M["lx21_i2v"][1]), 1.2], [os.path.basename(M["relight"][1]), 1.0]]
        g = {
            "v": {"class_type": "VHS_LoadVideo", "inputs": {"video": video, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                                                            "frame_load_cap": total, "skip_first_frames": 0, "select_every_nth": 1}},
            "ref": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(img)}},
            "pm": {"class_type": "OnnxDetectionModelLoader", "inputs": {"vitpose_model": "vitpose_h_wholebody_model.onnx", "yolo_model": "yolov10m.onnx",
                                                                         "onnx_device": "CUDAExecutionProvider"}},
            "t5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": prompt}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": req.get("negative_prompt") or NEG}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "cv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
            "cve": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["cv", 0], "image": ["ref", 0], "crop": "none"}},
            "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
            "samtxt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": target}},
            "trk": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["v", 0], "model": ["sam", 0], "conditioning": ["samtxt", 0],
                                                                 "detection_threshold": 0.4, "max_objects": 1, "detect_interval": 1}},
            "m0": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["trk", 0], "object_indices": "0"}},
            "pose": {"class_type": "KllmPoseFromMask", "inputs": {"model": ["pm", 0], "images": ["v", 0], "masks": ["m0", 0], "width": W, "height": H, "face_padding": 0}},
            "cm": {"class_type": "KllmCompleteMask", "inputs": {"masks": ["m0", 0], "pose_data": ["pose", 0], "head_up": 1.4, "head_side": 0.9,
                                                                "limb_scale": 0.0, "min_score": 0.3}},
            # every other tracked person (client, bystanders) is subtracted from the edit mask (e14: limbs over the
            # client's head otherwise regenerate the client with the reference face)
            "samp": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": "person"}},
            "trkp": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["v", 0], "model": ["sam", 0], "conditioning": ["samp", 0],
                                                                  "detection_threshold": 0.4, "max_objects": 8, "detect_interval": 2}},
            "prot": {"class_type": "KllmProtectOthers", "inputs": {"mask": ["cm", 0], "target_mask": ["m0", 0], "others_track": ["trkp", 0],
                                                                    "iou_max": 0.3, "protect_grow": 6}},
            "grow": {"class_type": "GrowMaskWithBlur", "inputs": {"mask": ["skirt", 0] if req.get("outfit") == "long" else ["prot", 0], "expand": 10, "incremental_expandrate": 0.0, "tapered_corners": True,
                                                                  "flip_input": False, "blur_radius": 0.0, "lerp_alpha": 1.0, "decay_factor": 1.0, "fill_holes": False}},
            "blk": {"class_type": "BlockifyMask", "inputs": {"masks": ["grow", 0], "block_size": 32, "device": "cpu"}},
            "bg": {"class_type": "DrawMaskOnImage", "inputs": {"image": ["v", 0], "mask": ["blk", 0], "color": "0, 0, 0", "device": "cpu"}},
            "drw": {"class_type": "DrawViTPose", "inputs": {"pose_data": ["pose", 0], "width": W, "height": H, "retarget_padding": 16,
                                                             "body_stick_width": -1, "hand_stick_width": -1, "draw_head": True}},
        }
        if req.get("outfit") == "long":   # long garment in the reference: give the model a flared skirt region
            g["skirt"] = {"class_type": "KllmSkirtMask", "inputs": {"mask": ["prot", 0], "pose_data": ["pose", 0], "hip_scale": 1.5,
                                                                     "hem_scale": 2.6, "min_score": 0.3}}
        if staged:   # stage 2: load the preprocessed control videos instead of recomputing them
            for k in self.PREP_NODES:
                g.pop(k, None)
            total_cap = total
            for nid, sub in (("ldp", "pose"), ("ldf", "face"), ("ldm", "mask")):
                g[nid] = {"class_type": "VHS_LoadImagesPath", "inputs": {"directory": os.path.join(staged, sub), "image_load_cap": total_cap,
                                                                         "skip_first_images": 0, "select_every_nth": 1}}
            g["blk"] = {"class_type": "ImageToMask", "inputs": {"image": ["ldm", 0], "channel": "red"}}
            g["bg"]["inputs"]["mask"] = ["blk", 0]
        if self.ray:
            self._ray_init(g)
            model = self._ray_model(g, "anim", os.path.basename(M["animate"][1]), loras)
        else:
            model = self._native_model(g, "anim", os.path.basename(M["animate"][1]), loras)
        prev_dec, acc = None, None
        for k in range(windows):
            wa = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": W, "height": H, "length": L, "batch_size": 1,
                  "clip_vision_output": ["cve", 0], "reference_image": ["ref", 0],
                  "face_video": ["ldf", 0] if staged else ["pose", 1], "pose_video": ["ldp", 0] if staged else ["drw", 0],
                  "continue_motion_max_frames": 5, "video_frame_offset": 0 if k == 0 else ["wa%d" % (k - 1), 5],
                  "background_video": ["bg", 0], "character_mask": ["blk", 0]}
            if k > 0:
                wa["continue_motion"] = [prev_dec, 0]
            g["wa%d" % k] = {"class_type": "WanAnimateToVideo", "inputs": wa}
            lat = self._sampler(g, "ks%d" % k, model, seed, steps, 1.0, ["wa%d" % k, 0], ["wa%d" % k, 1], ["wa%d" % k, 2], 0, 10000, True, True)
            g["tr%d" % k] = {"class_type": "TrimVideoLatent", "inputs": {"samples": lat, "trim_amount": ["wa%d" % k, 3]}}
            g["dec%d" % k] = self._decode(["tr%d" % k, 0], W, H)
            if k == 0:
                acc = ["dec0", 0]
            else:
                g["cut%d" % k] = {"class_type": "ImageFromBatch", "inputs": {"image": ["dec%d" % k, 0], "batch_index": ["wa%d" % k, 4], "length": 4096}}
                g["cat%d" % k] = {"class_type": "ImageBatch", "inputs": {"image1": acc, "image2": ["cut%d" % k, 0]}}
                acc = ["cat%d" % k, 0]
            prev_dec = "dec%d" % k
        g["save"] = {"class_type": "SaveImage", "inputs": {"images": acc, "filename_prefix": tag + "/f"}}
        return g

    KF_PROMPT = ("Replace the {t} in image 1 with the person from image 2. The person wears exactly the outfit, hairstyle and accessories "
                 "from image 2.{desc} Put them in the same position, pose and action as the {t}, holding the same tool, at the same size. "
                 "Keep everything else in image 1 exactly unchanged: the other people, the background, the lighting and the camera framing. "
                 "Photorealistic.")

    def g_keyframe(self, tag, img, video, target, W, H, seed, req):
        """Frame 0 with the reference person cast in place of `target`: SAM3 finds the target, Qwen-Image-Edit-2511
        (Q4_K_M + Lightning 8-step) edits a crop around it, and the edit is pasted back only inside the grown target mask,
        so nobody else can change (e31d/e31g: full-frame edits recast the wrong person)."""
        # P16 H8: the generic prompt left the barber unedited; the lab prompts that worked spelled out the outfit -> reuse the
        # request's own description of the person (recreate prompt) inside the edit instruction
        desc = (" Description of the person: %s" % req["prompt"].strip()) if req.get("prompt") else ""
        prompt = req.get("keyframe_prompt") or self.KF_PROMPT.format(t=target, desc=desc)
        return {
            "v": {"class_type": "VHS_LoadVideo", "inputs": {"video": video, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                                                            "frame_load_cap": 1, "skip_first_frames": 0, "select_every_nth": 1}},
            "f0": {"class_type": "ImageScale", "inputs": {"image": ["v", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
            "ref": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(img)}},
            "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
            "samtxt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": target}},
            "trk": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["f0", 0], "model": ["sam", 0], "conditioning": ["samtxt", 0],
                                                                 "detection_threshold": 0.4, "max_objects": 1, "detect_interval": 1}},
            "m0": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["trk", 0], "object_indices": "0"}},
            "grow": {"class_type": "GrowMaskWithBlur", "inputs": {"mask": ["m0", 0], "expand": 40, "incremental_expandrate": 0.0, "tapered_corners": True,
                                                                   "flip_input": False, "blur_radius": 12.0, "lerp_alpha": 1.0, "decay_factor": 1.0, "fill_holes": True}},
            "crop": {"class_type": "KllmMaskBBoxCrop", "inputs": {"image": ["f0", 0], "mask": ["m0", 0], "margin": 0.35}},
            "qclip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
            "qvae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "qunet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"}},
            "qlora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["qunet", 0], "strength_model": 1.0,
                                                                       "lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"}},
            "qms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["qlora", 0], "shift": 3.1}},
            "qpos": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["qclip", 0], "vae": ["qvae", 0], "prompt": prompt,
                                                                             "image1": ["crop", 0], "image2": ["ref", 0]}},
            "qneg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["qclip", 0], "vae": ["qvae", 0], "prompt": "",
                                                                             "image1": ["crop", 0], "image2": ["ref", 0]}},
            "qlat": {"class_type": "VAEEncode", "inputs": {"pixels": ["crop", 0], "vae": ["qvae", 0]}},
            "qks": {"class_type": "KSampler", "inputs": {"model": ["qms", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                                          "scheduler": "simple", "positive": ["qpos", 0], "negative": ["qneg", 0],
                                                          "latent_image": ["qlat", 0], "denoise": 1.0}},
            "qdec": {"class_type": "VAEDecode", "inputs": {"samples": ["qks", 0], "vae": ["qvae", 0]}},
            "paste": {"class_type": "KllmPasteCrop", "inputs": {"dest": ["f0", 0], "src": ["qdec", 0], "bbox": ["crop", 1], "mask": ["grow", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"images": ["paste", 0], "filename_prefix": tag + "_kf/f"}},
        }

    def g_recreate(self, tag, img, video, prompt, W, H, quality, seed, req, anchor_image=None):
        steps, phases = PHASES[quality]
        self.ray = self.use_ray and self.ray_edit
        g = {
            "v": {"class_type": "VHS_LoadVideo", "inputs": {"video": video, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                                                            "frame_load_cap": 81, "skip_first_frames": 0, "select_every_nth": 1}},
            "ref": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(img)}},
            "da": {"class_type": "LoadDA3Model", "inputs": {"model_name": "depth_anything_3_mono_large.safetensors", "weight_dtype": "fp16"}},
            "di": {"class_type": "DA3Inference", "inputs": {"da3_model": ["da", 0], "image": ["v", 0], "resolution": 504, "resize_method": "upper_bound_resize", "mode": "mono"}},
            "dr": {"class_type": "DA3Render", "inputs": {"da3_geometry": ["di", 0], "output": "depth", "output.normalization": "v2_style", "output.apply_sky_clip": False}},
            "t5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": prompt}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": req.get("negative_prompt") or NEG}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "vc": {"class_type": "WanVaceToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": W, "height": H,
                   "length": 81, "batch_size": 1, "strength": float(req.get("control_strength") or 1.0), "control_video": ["dr", 0], "reference_image": ["ref", 0]}},
        }
        if req.get("target"):   # hybrid control: the target person is driven by pose only, the scene by depth
            g.update({
                "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
                "samtxt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": req["target"]}},
                "trk": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["v", 0], "model": ["sam", 0], "conditioning": ["samtxt", 0],
                                                                     "detection_threshold": 0.4, "max_objects": 1, "detect_interval": 1}},
                "m0": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["trk", 0], "object_indices": "0"}},
                "pm": {"class_type": "OnnxDetectionModelLoader", "inputs": {"vitpose_model": "vitpose_h_wholebody_model.onnx", "yolo_model": "yolov10m.onnx",
                                                                             "onnx_device": "CUDAExecutionProvider"}},
                "pose": {"class_type": "KllmPoseFromMask", "inputs": {"model": ["pm", 0], "images": ["v", 0], "masks": ["m0", 0], "width": W, "height": H, "face_padding": 0}},
                "cm": {"class_type": "KllmCompleteMask", "inputs": {"masks": ["m0", 0], "pose_data": ["pose", 0], "head_up": 1.4, "head_side": 0.9,
                                                                    "limb_scale": 0.0, "min_score": 0.3}},
                "sk": {"class_type": "KllmSkirtMask", "inputs": {"mask": ["cm", 0], "pose_data": ["pose", 0], "hip_scale": 1.5, "hem_scale": 2.6, "min_score": 0.3}},
                "drw": {"class_type": "DrawViTPose", "inputs": {"pose_data": ["pose", 0], "width": W, "height": H, "retarget_padding": 0,
                                                                 "body_stick_width": -1, "hand_stick_width": -1, "draw_head": True}},
                "hy": {"class_type": "KllmHybridControl", "inputs": {"depth": ["dr", 0], "pose": ["drw", 0], "mask": ["sk", 0], "feather": 6}},
            })
            g["vc"]["inputs"]["control_video"] = ["hy", 0]
        if anchor_image:   # frame 0 = the cast keyframe (kept, mask 0); frames 1.. = control. No VACE reference image:
            L = 81         # it is not person-bound and leaked the reference outfit onto other people (e31f H7 -> fixed in e31h)
            ctl = g["vc"]["inputs"]["control_video"]
            g.update({
                "kimg": {"class_type": "LoadImage", "inputs": {"image": anchor_image}},
                "key": {"class_type": "ImageScale", "inputs": {"image": ["kimg", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
                "crest": {"class_type": "ImageFromBatch", "inputs": {"image": ctl, "batch_index": 1, "length": L - 1}},
                "cvid": {"class_type": "ImageBatch", "inputs": {"image1": ["key", 0], "image2": ["crest", 0]}},
                "mk0": {"class_type": "EmptyImage", "inputs": {"width": W, "height": H, "batch_size": 1, "color": 0}},
                "mk1": {"class_type": "EmptyImage", "inputs": {"width": W, "height": H, "batch_size": L - 1, "color": 16777215}},
                "mkb": {"class_type": "ImageBatch", "inputs": {"image1": ["mk0", 0], "image2": ["mk1", 0]}},
                "cmask": {"class_type": "ImageToMask", "inputs": {"image": ["mkb", 0], "channel": "red"}},
            })
            g["vc"]["inputs"]["control_video"] = ["cvid", 0]
            g["vc"]["inputs"]["control_masks"] = ["cmask", 0]
            g["vc"]["inputs"].pop("reference_image", None)
        if self.ray:
            self._ray_init(g)
        get = self._expert_variants(g, "vace_high", "vace_low", {"high": "lx_t2v_high", "low": "lx_t2v_low"}, None)
        lat, start = ["vc", 2], 0
        for pi, ph in enumerate(phases):
            lat = self._sampler(g, "ks%d" % pi, get(ph["m"], ph["lora"]), seed, steps, ph["cfg"], ["vc", 0], ["vc", 1], lat, start, ph["end"],
                                pi == 0, pi == len(phases) - 1)
            start = ph["end"]
        g["tr"] = {"class_type": "TrimVideoLatent", "inputs": {"samples": lat, "trim_amount": ["vc", 3]}}
        g["dec"] = self._decode(["tr", 0], W, H)
        g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": tag + "/f"}}
        return g

    def hq_pass(self, tag, src, W, H, sharp=False, chunk=60):
        """RIFE x2 (16->32 fps) + lanczos to 1080 wide (portrait) / 1920 wide (landscape), in 61-frame chunks.
        sharp=True inserts Real-ESRGAN x2 before the resize."""
        n = len([x for x in os.listdir(src) if x.endswith(".png")])
        tw = 1080 if W < H else 1920
        th = int(round(H * tw / W / 16) * 16)
        for ci, s in enumerate(range(0, n - 1, chunk)):
            g = {"ld": {"class_type": "VHS_LoadImagesPath", "inputs": {"directory": src, "image_load_cap": min(chunk + 1, n - s), "skip_first_images": s, "select_every_nth": 1}},
                 "rife": {"class_type": "RIFE VFI", "inputs": {"ckpt_name": "rife49.pth", "frames": ["ld", 0], "clear_cache_after_n_frames": 10, "multiplier": 2,
                          "fast_mode": True, "ensemble": True, "scale_factor": 1.0, "dtype": "float16", "torch_compile": False, "batch_size": 1}},
                 }
            im = ["rife", 0]
            if ci > 0:
                g["cut"] = {"class_type": "ImageFromBatch", "inputs": {"image": im, "batch_index": 1, "length": 4096}}
                im = ["cut", 0]
            if sharp:   # Real-ESRGAN: crisper edges but airbrushed skin (e23) -> opt-in only
                g["um"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x2.pth"}}
                g["up"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["um", 0], "image": im}}
                im = ["up", 0]
            g["rs"] = {"class_type": "ImageScale", "inputs": {"image": im, "upscale_method": "lanczos", "width": tw, "height": th, "crop": "disabled"}}
            g["sv"] = {"class_type": "SaveImage", "inputs": {"images": ["rs", 0], "filename_prefix": "%s_hq/c%04d" % (tag, s)}}
            self.run_graph(g, 3600)
        return os.path.join(self.out, tag + "_hq")


def _handler(rt, api_key, model_key):
    import hmac

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, status, msg, typ="invalid_request_error"):
            self._send(status, {"error": {"message": msg, "type": typ}})

        def _authed(self):
            got = self.headers.get("Authorization", "")
            got = got[7:].strip() if got.lower().startswith("bearer ") else (self.headers.get("x-api-key") or "").strip()
            if api_key and hmac.compare_digest(got.encode(), str(api_key).encode()):
                return True
            self._err(401, "Invalid API Key", "authentication_error")
            return False

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                if rt.state == "ready":
                    return self._send(200, {"status": "ok", "busy": rt.busy, "served": rt.served})
                return self._send(503 if rt.state == "loading" else 500, {"status": rt.state, "error": {"message": rt.error or "loading"}})
            if not self._authed():
                return
            if path in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{"id": model_key, "object": "model", "owned_by": "kllm", "kind": "video",
                                                                     "runtime": "comfyui", "modes": ["i2v", "replace", "recreate"]}]})
            self._err(404, "unknown path %s" % path, "not_found")

        def do_POST(self):
            path = self.path.split("?")[0]
            if not self._authed():
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > 200_000_000:
                return self._err(413, "request body too large")
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._err(400, "body must be a JSON object")
            if path != "/v1/videos/generations":
                return self._err(404, "unknown path %s" % path, "not_found")
            try:
                self._send(200, rt.generate_video(req))
            except ApiError as e:
                self._err(e.status, e.message, e.typ)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                import traceback
                rt.log("video generation crashed:", traceback.format_exc()[-2000:])
                self._err(500, "%s: %s" % (type(e).__name__, e), "server_error")
    return H


class Handle:
    """Looks like the llama-server Popen to the kernel's main(): poll() is None while healthy."""

    def __init__(self, rt, srv, th):
        self.rt, self.srv, self.th = rt, srv, th

    def poll(self):
        if self.rt.state == "error" or not self.th.is_alive():
            return 1
        if self.rt.proc is not None and self.rt.proc.poll() is not None:
            return self.rt.proc.returncode or 1
        return None

    def terminate(self):
        try:
            self.srv.shutdown()
        except Exception:
            pass
        if self.rt.proc is not None:
            try:
                os.killpg(self.rt.proc.pid, signal.SIGTERM)   # own session: takes the Ray raylet/workers with it
            except Exception:
                pass
    kill = terminate


def start(cfg, port, log, ntfy):
    rt = Runtime(cfg, log, ntfy)

    class S(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    srv = S(("127.0.0.1", port), _handler(rt, cfg.get("api_key"), cfg.get("model_key") or "wan2.2-video"))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    rt.load()
    if rt.state != "ready":
        ntfy("error", stage="server_health", tail=(rt.error or "load failed")[-1500:])
        raise RuntimeError("comfyui runtime failed: %s" % rt.error)
    return Handle(rt, srv, th)
