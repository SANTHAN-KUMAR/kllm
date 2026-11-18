# kllm — open-weight LLMs on free Kaggle GPUs, as a local API for coding agents

kllm runs large open-weight models (including **abliterated** variants) on Kaggle's free **2× Tesla T4**
notebooks (30 GB VRAM) and makes them look like a local API to your tools: **Claude Code,
opencode, Cursor/Continue, Open WebUI, scripts** — OpenAI and Anthropic protocols both.
The model runs in the cloud; everything you talk to runs on your PC — **one small Python process
(~55 MB RAM)**. When a Kaggle session nears its 12-hour cap, kllm starts a replacement behind the
same stable endpoint, so your tools never notice.

Built and tuned for **agent-heavy harnesses**: big tool-laden prompts, long sessions, prompt-cache
reuse every turn, tool calls that must parse, and no runaway repetition.

```
YOUR PC (always up: `kllm serve`, one process)                   KAGGLE (2×T4, ≤12 h)
 Claude Code ── claude-kllm ─┐                                    kernel on your account:
 opencode / Cursor / scripts ┼─► relay :8080 ─────────────────────► llama-server (CUDA, -sm tensor)
 Open WebUI / OpenAI apps ───┘   - OpenAI + Anthropic APIs          cloudflared quick tunnel
 other devices (+gateway key)    - picks backend, loop guard          │ posts its URL, encrypted
 panel  http://127.0.0.1:8080/admin  - 524-proof                      │ and signed, to ntfy
 supervisor thread: launch, health, failover, rotation ◄── ntfy ◄─────┘
```

> **Use Kaggle fairly.** kllm uses your own Kaggle account and its free weekly GPU quota. Follow
> [Kaggle's Terms of Service](https://www.kaggle.com/terms): Kaggle allows one account per person.
> kllm can manage several accounts (for example a team where each member adds their own), but which
> accounts you add, and whether you may use them, is your responsibility.

## Quick start
The only thing you need is a Kaggle API token. Everything else is created for you.

1. **Kaggle token** (one time): on a phone-verified Kaggle account (needed for GPUs), open
   kaggle.com → Settings → API → *Create New Token*, and save the file as `~/.kaggle/kaggle.json`
   (or `~/.kaggle/access_token` for the newer token format; `KAGGLE_USERNAME`/`KAGGLE_KEY` or
   `KAGGLE_API_TOKEN` in your environment also work).
2. **Install and run:**
```bash
pipx install git+https://github.com/SANTHAN-KUMAR/kllm   # or: uv tool install ..., or pipx install . from a checkout
kllm up                    # first run writes the config; starts the default model, ready in ~8-10 min
claude-kllm                # Claude Code on it (model + context window picked automatically)
```
That's it. No GitHub token, no tunnel account, no API keys to copy: the first `kllm` command finds your
Kaggle credentials, generates a gateway key and the rendezvous secret, and writes `config.yaml`.

More commands:
```bash
kllm install-service       # relay starts at every logon (systemd / launchd / Windows Task Scheduler)
kllm doctor                # end-to-end check: accounts, rendezvous, relay, backends, context, tool calls, speed
opencode -m kllm/qwen3.6-35b-abliterated
kllm down                  # stop when done — saves your weekly GPU quota (~30 h)
kllm init                  # optional wizard: review accounts and settings, prerequisite checks
```
Zero-manual mode: with the service installed, the tiny relay is always up, so the first
request from any client (opencode, Claude Code, scripts) starts the GPU backend on demand
(~8–10 min cold; the request waits, kept alive) and `idle_stop_minutes` (25 min default)
stops it when nobody uses it. Even without the service, `claude-kllm` starts a missing
relay itself (`kllm ensure-serve` does the same for scripts/hooks).
Linux, macOS and Windows (native Python) are supported. Everything lives in one config dir:
`$KLLM_ROOT`, else the checkout, else `~/.config/kllm`.

### How the kernel finds your PC (and why it is safe)
The Kaggle kernel opens a free cloudflared quick tunnel and has to tell your PC its URL. By default it posts
a short status message to [ntfy.sh](https://ntfy.sh) (free, no account). Every message is **encrypted and
authenticated** with a random per-install secret (`rendezvous_secret`, HMAC-SHA256 encrypt-then-MAC, stdlib
only), on topic names derived from that secret. So:
- ntfy and anyone who guesses the topic see only random bytes;
- a forged or altered message (say, a fake URL meant to capture your prompts) fails the check and is dropped;
- prompts and replies never touch ntfy: they go straight through the tunnel, guarded by a per-kernel API key.

Messages are sent only on state changes plus a heartbeat every ~10 min (~90 per 11-hour session, well
inside ntfy.sh's free per-IP allowance). It adds about a second to a startup that takes minutes, and nothing
to requests. Prefer your own infrastructure? Set `ntfy_server:` to a self-hosted ntfy, or use a private
GitHub gist instead (`kllm init --github-token <gist-scoped token>`).

Or open the panel at **http://127.0.0.1:8080/admin** (local only, one static page, no external requests):
a control center with a left nav (`g d` dashboard, `g u` usage, `g s` sessions, `g b` backends, `g m` models,
`g a` accounts, `g o` autoscale, `g l` logs, `g p` playground, `g t` settings; `/` filters, `?` lists keys).
- **Usage**: requests, errors, loop cuts, tokens in (new vs cached) and out (thinking vs answer), cache-hit rate,
  TTFT / speed percentiles, speed vs prompt size, by model / backend / hour / day (24 h, 7 d, 14 d), plus GPU hours
  per account, quota left and projected exhaustion, standby estimate, launch/cancel/retire counts.
- **Control**: launch on a chosen account, rotate, stop, pin; move or forget a session; every autoscale setting;
  add (paste a Kaggle token), disable/enable and remove accounts; edit/delete/default models, add from HuggingFace
  (non-abliterated repos get a notice and abliterated alternatives); gateway key, remote tunnel, defaults.
- **Diagnostics**: `kllm doctor` from the page, service log tail, request-log explorer with CSV export, chat/image/video playground.
- `requests.jsonl` rotates at 50 MB (3 kept); an hourly rollup in `state/metrics_rollup.json` keeps metrics cheap.
- An account with `disabled: true` in `config.yaml` is never launched on; a backend can be pinned against idle-stop.

## Models (all abliterated)
| key | what | context | measured on 2×T4 |
|---|---|---|---|
| **qwen3.6-35b-abliterated** (default) | Qwen3.6-35B-A3B MoE, thinking | **256K** | 70 t/s short · 54 t/s @ 90K · 43 t/s @ 209K |
| qwen3-coder-30b-abliterated | Qwen3-Coder-30B-A3B MoE, no thinking | 96K | 84 t/s short · 48 t/s @ 43K · 35 t/s @ 84K |
| gpt-oss-20b-abliterated | gpt-oss-20B MoE, reasoning | 128K | unverified |
| qwen3-30b-a3b / qwen3-32b / qwq-32b / r1-distill-32b / coder-32b (abliterated) | older dense/MoE | 32K | unverified |

Which one?
- **Quality first, long sessions:** `qwen3.6-35b-abliterated`. In a 4-turn Claude Code session that
  grew past 55K tokens it gave the most precise code summaries and perfect recall of early context.
  It thinks before answering, so short replies take a few seconds longer.
- **Fastest turnaround:** `qwen3-coder-30b-abliterated`. Same tool-call reliability, no thinking,
  but 96K context and slightly more prone to confident mistakes in long summaries.
- Run both at once on two accounts (`kllm up --model qwen3-coder-30b-abliterated`), then pick per
  session: `KLLM_MODEL=qwen3-coder-30b-abliterated claude-kllm`.

Add any GGUF from the panel (or `models.yaml`, or `kllm add <repo>`). Keep weights + KV under ~28 GB. Image and video models: see *Any model* below.

## Any model: chat, image, video
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

## Using it from harnesses
| Harness | How | Notes |
|---|---|---|
| Claude Code | `claude-kllm [claude args]` | Picks your default model if running (else `$KLLM_MODEL`, else whatever runs), sets the context window from the live backend, lean tool set (`LEAN=0` keeps MCP). |
| opencode | `opencode -m kllm/<model key>` | Provider in `clients/opencode.jsonc` (merge into `~/.config/opencode/opencode.jsonc`). Its `limit.context` must match the model's `ctx`. |
| OpenAI-compatible | `http://127.0.0.1:8080/v1` (the relay itself) | model `kllm` = whatever runs, or a model key. No key from this machine. |
| Anthropic API | `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` | the relay; no key needed locally. |
| Other devices | `remote_enabled: true` (or `kllm remote`), or your LAN | public tunnel / LAN to the relay's model API only, gated by `gateway_api_key` (`Authorization: Bearer` or `x-api-key`). |

### Why Claude Code used to auto-compact after 1–2 prompts
Claude Code reserves an autocompact buffer of ~13K + its max-output setting (32K by default) out of
the context window. With a 60K window that left ~22K usable. `claude-kllm` now uses the backend's
real window (256K for the default model) and sizes max-output to the model: **~224K usable**
(Qwen3.6) / ~64K (Qwen3-Coder). Check any time with `claude-kllm -p /context`.

## What makes it fast and reliable (measured, model-agnostic where possible)
- **f16 KV cache.** On Turing GPUs a quantized (q8_0) KV cache sends llama.cpp's FlashAttention down a
  slow path: generation was 1.6–2.5× slower at 12–43K context. Context is instead gained by picking
  models with small KV (Qwen3.6's hybrid attention: ~20 KiB/token → 256K fits).
- **Tensor-parallel split (`-sm tensor`)** across both T4s instead of pipelining layers:
  +13–54% generation, prefill unchanged.
- **N-gram speculative decoding (`--spec-type ngram-mod`)**: lossless; agents copy a lot of text
  (Edit `old_string`, rewrites) → 120–300 t/s on copied spans.
- **Prompt cache that survives agents:** 2 slots on one shared KV pool, 16 GB host-RAM prompt cache
  (`--cache-ram`) so side requests don't evict the conversation, context checkpoints for hybrid
  models, per-conversation backend affinity in the relay, and Claude Code's
  `x-anthropic-billing-header` stripped by `claude-kllm`. Measured: **99%+ of each turn's prompt
  served from cache** at 55K context; agent steps take 1.5–3 s.
- **Warm-up before "ready"**: the first real request is fast (a cold server took 33 s).
- **Sampling = each model's published recommendation; no repeat/DRY penalties** — they penalise
  copying prompt text and corrupted Edit `old_string` reproduction in tests.

### Loops and long-context degeneration
Any model can fall into exact repetition as context grows (one Claude Code turn here burned ~8,000
tokens). The relay has a **generic loop guard** for every model and protocol: when output ends in
a block of ≥20 chars repeated exactly ≥8 times over ≥1,200 chars, it stops generation on the
backend and ends the response cleanly (OpenAI `finish_reason: stop` / Anthropic `end_turn` +
`message_stop`), and logs `loop_cut`. Tuned against 3,568 real source files: 1 false positive (a
generated Unicode data table). Short-period runs (`-----`, `None,` columns) never trip it.

### Long requests and the tunnel's 120 s limit
Cloudflare tunnels abort a response that sends nothing for 120 s (HTTP 524). llama-server sends
nothing while a request waits for a slot or prefills a long prompt. Three layers make this
impossible to hit:
- **Kernel keep-alive proxy** (in front of llama-server on Kaggle): transparent when llama-server
  answers within 60 s; otherwise it commits to a 200 SSE stream with keep-alive comments until the
  first token, and turns late errors into protocol-correct error events.
- **Relay admission control**: a backend never receives more concurrent requests than it has slots;
  the rest wait in the relay (keep-alive comments for streaming clients).
- **Relay retry**: 502/530/524 from the edge are retried before anything reaches the client; dead
  tunnels fail over to another backend, then to your local llama-server (:8081).
- Non-streaming requests are streamed upstream and rebuilt into JSON for the client.

## Resource manager: Kaggle GPUs as servers, sessions as clients
`gateway/scheduler.py` holds the whole policy (pure, unit-tested); the supervisor only executes it.
- **On demand.** A request for a model that is not running starts it: the request is held (streaming
  clients get keep-alive comments) and continues automatically when the backend is ready (~8–10 min
  cold). `claude-kllm` sizes Claude Code's window for the model that will start.
- **Parallel sessions.** Each conversation is placed once on the least-loaded backend of its model
  and stays there (its prompt cache lives there). More active sessions than slots → another backend
  is launched; when it is up, the overloaded backend sheds its *newest* session (cheapest to move).
  Side requests (titles, probes) never count as sessions.
- **Queued accounts.** A launch that is not running after 4 min is treated as queued: the next-best
  account is launched in parallel (at most 2 queued per model, so it never hops endlessly), and the
  loser is cancelled once a launch actually runs. Queued accounts cool down 30 min.
- **Account ranking**: weekly quota left, measured queue and ready times, failures (cooldown with
  backoff). Accounts with < 1.5 h quota are skipped. Add accounts in `config.yaml`.
- **Idle**: a backend nobody used for 45 min is stopped (above the always-on minimum).
- **Unlimited sessions despite Kaggle's 12 h cap.** 25 min before a backend's cap a successor is
  launched; the relay then *warm-hands* every session: it replays the session's latest prompt on the
  successor (1 output token) and only then moves it, so the next turn is a cache hit, not a full
  re-read. The old backend retires once empty (or at its hard cap). Sessions on a backend that dies
  are handed off the same way. `kllm autoscale` / panel show every decision and why.
- Settings: `autoscale:` in `config.yaml` (`kllm autoscale on|off`, `--max N`). Manual launches
  (`kllm up`, panel) are never cancelled for lack of demand.

## Long-horizon test (2026-10-06, qwen3.6-35b-abliterated)
Three agent sessions in parallel (2× Claude Code, 1× opencode), each implementing a ~3-page spec
(library + SQLite + recurrence + undo + import/export + CLI) from scratch, graded afterwards by a
hidden 38-case acceptance suite the model never saw:

| session | hidden tests | own tests | time |
|---|---|---|---|
| opencode | 37/38 (97%) | 121 | 28 min |
| Claude Code | 35/38 (92%) | 116 | 35 min |

Over 72 min of agent work and 10.2M prompt tokens: contexts grew to 165K, **95.9%** of prompt
tokens came from the prompt cache, generation stayed at **50–63 t/s from 0 to 175K context**,
median time to first token 2.3 s, 0 loops. The run exposed one failure (a request queued inside
llama-server hit the tunnel's 120 s limit and killed a third session); the keep-alive proxy and
admission control above were built from it. Most of the wall time is the model's thinking
(≈205K output tokens); the request log now separates thinking from answer tokens.

## Observability
- `logs/requests.jsonl` — one line per request: backend, prompt tokens, cached tokens, time to
  first token, output tokens (thinking vs answer), speed, loop cuts. Shown live in the panel.
- `kllm sessions` — live sessions, their backends, load and handoffs; `kllm autoscale` — decisions.
- `kllm status`, `kllm kstatus` (raw Kaggle state), `kllm doctor`, `journalctl --user -u kllm` (macOS: `<root>/logs/serve.log`).

## Gateway key, remote access and LiteLLM (optional)
- **Local clients need no key.** Requests from 127.0.0.1/::1 are trusted. Anything else (LAN, tunnel)
  must send `gateway_api_key` and may only use the model API (`/v1/*`); the panel, `/admin` and
  `/__relay` stay local-only. With no real key configured, non-local requests are refused. A tunnel
  connects from localhost but adds `X-Forwarded-For`/`CF-Connecting-IP`, so it is correctly treated as remote.
- **Remote tunnel** is off by default: `remote_enabled: true` makes `kllm serve` keep a cloudflared quick
  tunnel up (the binary is downloaded into the config dir on first use only); `kllm remote` runs it alone.
  The URL shows in the panel (Connect → Remote).
- **LiteLLM is optional and off.** The relay already speaks OpenAI and Anthropic. If you want LiteLLM's
  usage UI: `pip install 'kllm[litellm]'`, set `litellm_enabled: true`, `kllm litellm-config`, and run
  `litellm --config <root>/litellm/config.yaml --port 4000` (or `systemd/kllm-litellm.service`, edit its paths).
  `kllm doctor` then checks it too.
- **Accounts without wrapper scripts:** per account, `kaggle_json: /path/kaggle.json` or
  `kaggle_config_dir: /dir/with/kaggle.json` (credentials go to the kaggle CLI by environment, never on
  the command line). Old `kg1..kgN` wrapper scripts keep working when neither key is set.
- **Rendezvous:** sealed ntfy messages by default (`rendezvous_secret`, `ntfy_server`); optional private GitHub
  gist (`rendezvous: gist`, `gist_id`, `gist_token`), in which case the PC talks to the gist API directly.

### Three-process layout (legacy)
`kllm relay` and `kllm watch` still work separately (and `systemd/kllm-relay.service`,
`kllm-supervisor.service`): `kllm serve` is the same two in one process. Only one supervisor runs at a
time (a lock in `state/`), so mixing them is safe but pointless — don't run two relays on one port.

## Files
| path | role |
|---|---|
| `gateway/relay.py` | local endpoint: routing, failover, stream conversion, loop guard, metrics, admin API |
| `gateway/auth.py` | `gateway_api_key` check for non-local callers (replaces LiteLLM's key) |
| `gateway/rendezvous.py` | kernel status/URL handoff: sealed ntfy (default) or private gist |
| `gateway/sealbox.py` | encrypt-then-MAC for rendezvous messages (stdlib HMAC-SHA256) |
| `gateway/setup.py`, `gateway/service.py` | `kllm init` wizard; `kllm install-service` (systemd / launchd) |
| `gateway/tunnel.py` | optional cloudflared tunnel (downloaded on demand) |
| `gateway/claude_launcher.py` | `claude-kllm` as a console script (Python port of `bin/claude-kllm`) |
| `gateway/ui/` | the control panel: plain HTML, CSS and ES modules, no build step  |
| `gateway/admin_api.py`, `gateway/metrics.py`, `gateway/accounts.py`, `gateway/yamledit.py` | panel API, usage rollup + log rotation, account add/remove, comment-preserving YAML edits |
| `gateway/supervisor.py` | health, failover, rotation, keep-alive |
| `gateway/orchestrator.py` | render + push kernels, promote ready backends, stop |
| `gateway/inspect_model.py`, `gateway/admin_models.py` | `kllm add`: inspect a HF repo, write the `models.yaml` entry; panel API |
| `gateway/media.py` | runtimes/kinds, image/video routing helpers |
| `gateway/common.py` | config, model registry, locked state file |
| `kernel/server_kernel.tmpl.py` | what runs on Kaggle: llama.cpp CUDA binary or diffusers runtime, warm-up, keep-alive proxy, tunnel, heartbeats |
| `models.yaml` | model registry (sampling and llama-server flags per model) |
| `templates/` | patched chat templates shipped to kernels |
| `bin/claude-kllm` | Claude Code launcher (bash version; the installed `claude-kllm` is the Python one) |
| `pyproject.toml` | `pipx install .` / `uv tool install .`; deps: fastapi, uvicorn, httpx, pyyaml, kaggle |
| `clients/opencode.jsonc` | opencode provider |
| `tests/` | unit tests: loop guard, scheduler, rendezvous (ntfy + gist), accounts, gateway key, init, service files, diffusers kernel (fake pipeline), media relay, kllm add |

## Known limits
- First launch of a model on an account downloads ~20 GB: ~8–10 min until ready. Kaggle GPU queues
  occasionally add more.
- Free GPU quota is ~30 h/week per account. `kllm down` when idle.
- Claude Code's auto mode shows a one-time "classifier billing" notice through any gateway; press
  Enter (nothing breaks). Adding `"CLAUDE_CODE_AUTO_MODE_SERVER":"0"` to the `--settings` env in
  `bin/claude-kllm` (or `gateway/claude_launcher.py`) removes it.
- Qwen3.6 sometimes thinks at length even on trivial requests; that costs seconds, not quality.
- `--n-expert-used` reduction, expert dropping and quantized KV were evaluated and rejected: they buy
  speed with measurable quality loss (or, for KV on T4, lose speed too).

## Contributing
Issues and pull requests are welcome. Run the tests with `pip install -e ".[dev]"` and `python -m pytest`
(they never contact Kaggle, GitHub or ntfy, and never touch your home directory).

## License
[MIT](LICENSE)
