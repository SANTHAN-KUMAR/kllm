# Models

kllm ships a registry of models in `models.yaml` and can add any GGUF or diffusers repo from Hugging Face.

## Which chat model should I use?
| key | what | context | measured on 2×T4 |
|---|---|---|---|
| **qwen3.8-27b** (+ `-vision`, `-abliterated`, `-256k` variants) | Qwen3.8-27B dense, thinking; highest answer quality | 128K (256K variants) | 16 to 19 t/s median over 6,889 real requests (99.3% success, up to 190K context) |
| **qwen3.6-35b-abliterated** (default) | Qwen3.6-35B-A3B MoE, thinking | **256K** | 70 t/s short · 54 t/s @ 90K · 43 t/s @ 209K |
| qwen3-coder-30b-abliterated | Qwen3-Coder-30B-A3B MoE, no thinking | 96K | 84 t/s short · 48 t/s @ 43K · 35 t/s @ 84K |
| gpt-oss-20b-abliterated | gpt-oss-20B MoE, reasoning | 128K | unverified |
| qwen3-30b-a3b / qwen3-32b / qwq-32b / r1-distill-32b / coder-32b (abliterated) | older dense/MoE | 32K | unverified |

Which one?
- **Best answers:** `qwen3.8-27b`. A dense 27B, so it decodes at 16 to 19 t/s instead of 40+, but it was the
  daily driver for over 6,800 requests. Pick `qwen3.8-27b-vision-abliterated-256k` to paste screenshots and keep 256K of context.
- **Quality first, long sessions:** `qwen3.6-35b-abliterated`. In a 4-turn Claude Code session that
  grew past 55K tokens it gave the most precise code summaries and perfect recall of early context.
  It thinks before answering, so short replies take a few seconds longer.
- **Fastest turnaround:** `qwen3-coder-30b-abliterated`. Same tool-call reliability, no thinking,
  but 96K context and slightly more prone to confident mistakes in long summaries.
- Run both at once on two accounts (`kllm up --model qwen3-coder-30b-abliterated`), then pick per
  session: `KLLM_MODEL=qwen3-coder-30b-abliterated claude-kllm`.

Add any GGUF from the panel (Models, then Add from Hugging Face) or with `kllm add <repo>`. Keep weights + KV under ~28 GB. Image and video models: see *Any model* below.

## How do I add a model, or run image and video models?
Every model in `models.yaml` has a **runtime** and a **kind**: `runtime: llama.cpp` (the default, all
the abliterated chat models above) or `runtime: diffusers` for image and video pipelines;
`kind: chat | image | video`. Existing entries need no edits. Image/video models run in the same
Kaggle kernel (fp16: the T4 has no bf16), behind the same tunnel keep-alive and relay, and are
autoscaled like chat models (a request for a model that is not running starts it).

**Add a model with one command** (or paste it in the panel: *Add a model* -> Inspect -> Add):
```bash
kllm add mradermacher/Some-Model-abliterated-i1-GGUF            # GGUF -> llama.cpp chat model
kllm add https://huggingface.co/org/name --file Q5_K_M --ctx 65536 --name my-model --default
kllm add stabilityai/sdxl-turbo                                 # diffusers repo -> image model
kllm add Lightricks/LTX-Video --dry-run                         # inspect only: prints the entry, writes nothing
kllm up --model sdxl-turbo                                      # or just send a request: autoscale starts it
```
`kllm add` asks the HuggingFace API (public repos need no token; `hf_token` in `config.yaml` for gated ones):
- **GGUF repo**: lists the `.gguf` files (split files grouped) and picks the best quant that fits 2xT4 with
  room for context (imatrix `i1-Q4_K_M`, then `Q4_K_M`, `Q4_K_S`/`IQ4_XS`; `Q5_K_M` when the model is small),
  reads the context length and KV shape from the GGUF header, bounds `ctx` by VRAM, estimates VRAM, and writes
  sampling flags by family (coder: temp 0.7/top-p 0.8; Qwen3/thinking: 0.6/0.95/top-k 20 + reasoning flags) plus
  `-sm tensor` and an f16 KV cache. Entries are appended to `models.yaml` (comments untouched, always valid YAML).
- **diffusers repo** (`model_index.json`): image or video from the pipeline class (Wan, LTX, CogVideoX,
  HunyuanVideo, Mochi are video), fp16 footprint from the safetensors sizes, per-family default size/steps/frames.
  Image-to-video pipelines (they need an input image) are refused.
- **anything else** (a transformers-format LLM, for example): a clear error and GGUF repos to look for.

**Calling them** (through the relay; `model` routes to the backend serving it, one request at a time per
backend; `model` may be omitted/any alias when only one model of that kind exists, or set
`default_image_model` / `default_video_model` in `config.yaml`):
```bash
# image: OpenAI Images API (b64_json by default; response_format "url" returns a data: URL)
curl -s http://127.0.0.1:8080/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "sdxl-turbo", "prompt": "a lighthouse at dusk, oil painting", "size": "512x512", "n": 1,
  "negative_prompt": "blurry", "steps": 4, "guidance_scale": 0, "seed": 42}' \
 | python3 -c "import sys,json,base64; d=json.loads(sys.stdin.read()); open('out.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))"

# video: kllm-specific, returns an mp4 as base64
curl -s http://127.0.0.1:8080/v1/videos/generations -H 'Content-Type: application/json' -d '{
  "model": "ltx-video", "prompt": "waves rolling onto a beach", "size": "704x480", "num_frames": 49, "fps": 24,
  "steps": 30, "guidance_scale": 3, "seed": 1}' \
 | python3 -c "import sys,json,base64; d=json.loads(sys.stdin.read()); open('out.mp4','wb').write(base64.b64decode(d['data'][0]['b64_mp4']))"
```
Responses: images `{"created", "data": [{"b64_json", "seed"}]}`; videos `{"created", "data": [{"b64_mp4", "seed"}], "fps", "frames"}`;
errors `{"error": {"message", "type"}}`. Chat clients (opencode, Claude Code, `claude-kllm`) are unchanged and never
pick an image/video model. The panel's model library shows chat/image/video badges and has an *Images* test box.

**How it works.** The kernel pip-installs what the Kaggle image lacks (diffusers, transformers, accelerate,
imageio-ffmpeg for video; never torch), loads the pipeline in fp16, and chooses placement from the loaded size:
the whole pipeline on one T4 (<= ~11.5 GB), model CPU offload (largest component <= ~13.5 GB), or sequential
offload; the VAE is tiled for video and offloaded models. It serves a small API on the port llama-server would
use, with the same API-key check. Generation takes minutes, so the kernel's keep-alive proxy commits a slow image/video
request to `200 application/json` after 60 s and sends a space every 10 s (valid JSON whitespace), then the
JSON body, or a JSON `error` object if it fails late (the HTTP status is then already 200: check for `error`).
The relay waits for a free backend slot without bytes, so a client behind the public tunnel should give image
requests a start-up allowance (~10 min cold) or pre-start the model with `kllm up`.

**What does not fit.** 2xT4 is 30 GB of VRAM and 31 GB of RAM: a ~230B model such as **MiniMax-M2** does not fit
(even at 2-bit it is far above 30 GB), and MiniMax's **Hailuo** video model is not open-weights, so neither can be
added. Large video models (Wan 14B, HunyuanVideo 13B, CogVideoX-5B, FLUX) exceed a T4 and fall back to very slow
sequential offload or are refused by `kllm add`. The shipped `sdxl-turbo`, `wan2.1-t2v-1.3b` and `ltx-video` entries are
marked `unverified`: repos and sizes were checked, they were never launched on a real GPU, and fp16 (instead of the
published bf16) can degrade or break some video models.
