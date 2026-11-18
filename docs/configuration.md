# Configuration and reference

Everything lives in one folder: `$KLLM_ROOT`, else the checkout, else `~/.config/kllm` (Windows:
`%APPDATA%\kllm`). The panel edits it for you; this page is for when you want to know what it holds.

## How do I point each tool at kllm?
| Harness | How | Notes |
|---|---|---|
| Claude Code | `claude-kllm [claude args]` | Picks your default model if running (else `$KLLM_MODEL`, else whatever runs), sets the context window from the live backend, lean tool set (`LEAN=0` keeps MCP). |
| opencode | `opencode -m kllm/<model key>` | Provider in `clients/opencode.jsonc` (merge into `~/.config/opencode/opencode.jsonc`). Its `limit.context` must match the model's `ctx`. |
| OpenAI-compatible | `http://127.0.0.1:8080/v1` (the relay itself) | model `kllm` = whatever runs, or a model key. No key from this machine. |
| Anthropic API | `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` | the relay; no key needed locally. |
| Other devices | `remote_enabled: true` (or `kllm remote`), or your LAN | public tunnel / LAN to the relay's model API only, gated by `gateway_api_key` (`Authorization: Bearer` or `x-api-key`). |

## Why did Claude Code auto-compact after one or two prompts?
Claude Code reserves an autocompact buffer of ~13K + its max-output setting (32K by default) out of
the context window. With a 60K window that left ~22K usable. `claude-kllm` now uses the backend's
real window (256K for the default model) and sizes max-output to the model: **~224K usable**
(Qwen3.6) / ~64K (Qwen3-Coder). Check any time with `claude-kllm -p /context`.

## How do remote access, the gateway key and LiteLLM work?
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

## Can I run the relay and supervisor separately?
`kllm relay` and `kllm watch` still work separately (and `systemd/kllm-relay.service`,
`kllm-supervisor.service`): `kllm serve` is the same two in one process. Only one supervisor runs at a
time (a lock in `state/`), so mixing them is safe but pointless: don't run two relays on one port.

## Where is each part in the code?
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
| `pyproject.toml` | package metadata (PyPI name `kllm-gateway`; commands `kllm`, `claude-kllm`); deps: fastapi, uvicorn, httpx, pyyaml, kaggle |
| `clients/opencode.jsonc` | opencode provider |
| `tests/` | unit tests: loop guard, scheduler, rendezvous (ntfy + gist), accounts, gateway key, init, service files, diffusers kernel (fake pipeline), media relay, kllm add |
