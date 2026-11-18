# How kllm works

This page explains what happens between your tool and the GPU, and why each piece exists.
Start with the [README](../README.md) if you have not run kllm yet.

## What keeps generation fast on two T4s?
- **f16 KV cache.** On Turing GPUs a quantized (q8_0) KV cache sends llama.cpp's FlashAttention down a
  slow path: generation was 1.6 to 2.5× slower at 12 to 43K context. Context is instead gained by picking
  models with small KV (Qwen3.6's hybrid attention: ~20 KiB/token → 256K fits).
- **Tensor-parallel split (`-sm tensor`)** across both T4s instead of pipelining layers:
  +13 to 54% generation, prefill unchanged.
- **N-gram speculative decoding (`--spec-type ngram-mod`)**: lossless; agents copy a lot of text
  (Edit `old_string`, rewrites) → 120 to 300 t/s on copied spans.
- **Prompt cache that survives agents:** 2 slots on one shared KV pool, 16 GB host-RAM prompt cache
  (`--cache-ram`) so side requests don't evict the conversation, context checkpoints for hybrid
  models, per-conversation backend affinity in the relay, and Claude Code's
  `x-anthropic-billing-header` stripped by `claude-kllm`. Measured: **99%+ of each turn's prompt
  served from cache** at 55K context; agent steps take 1.5 to 3 s.
- **Warm-up before "ready"**: the first real request is fast (a cold server took 33 s).
- **Sampling = each model's published recommendation; no repeat/DRY penalties**: they penalise
  copying prompt text and corrupted Edit `old_string` reproduction in tests.

## What stops a model from looping forever?
Any model can fall into exact repetition as context grows (one Claude Code turn here burned ~8,000
tokens). The relay has a **generic loop guard** for every model and protocol: when output ends in
a block of ≥20 chars repeated exactly ≥8 times over ≥1,200 chars, it stops generation on the
backend and ends the response cleanly (OpenAI `finish_reason: stop` / Anthropic `end_turn` +
`message_stop`), and logs `loop_cut`. Tuned against 3,568 real source files: 1 false positive (a
generated Unicode data table). Short-period runs (`-----`, `None,` columns) never trip it.

## Why don't long requests hit the tunnel's 120 s limit?
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

## How are GPUs started, shared and rotated?
`gateway/scheduler.py` holds the whole policy (pure, unit-tested); the supervisor only executes it.
- **On demand.** A request for a model that is not running starts it: the request is held (streaming
  clients get keep-alive comments) and continues automatically when the backend is ready (~8 to 10 min
  cold). `claude-kllm` sizes Claude Code's window for the model that will start.
- **Parallel sessions.** Each conversation is placed once on the least-loaded backend of its model
  and stays there (its prompt cache lives there). More active sessions than slots → another backend
  is launched; when it is up, the overloaded backend sheds its *newest* session (cheapest to move).
  Side requests (titles, probes) never count as sessions.
- **Queued accounts.** A launch that is not running after 4 min is treated as queued: the next-best
  account is launched in parallel (at most 2 queued per model, so it never hops endlessly), and the
  loser is cancelled once a launch actually runs. Queued accounts cool down 30 min.
- **Account ranking**: weekly quota left, measured queue and ready times, failures (cooldown with
  backoff). Accounts with < 1.5 h quota are skipped. Add accounts on the panel's Accounts page.
- **Idle**: a backend nobody used for 45 min is stopped (above the always-on minimum).
- **Unlimited sessions despite Kaggle's 12 h cap.** 25 min before a backend's cap a successor is
  launched; the relay then *warm-hands* every session: it replays the session's latest prompt on the
  successor (1 output token) and only then moves it, so the next turn is a cache hit, not a full
  re-read. The old backend retires once empty (or at its hard cap). Sessions on a backend that dies
  are handed off the same way. `kllm autoscale` / panel show every decision and why.
- Settings: the panel's Autoscale page, or `kllm autoscale on|off` and `--max N`. Manual launches
  (`kllm up`, panel) are never cancelled for lack of demand.

## How does the Kaggle kernel tell your PC where it is?
The kernel opens a free cloudflared quick tunnel and posts its URL to [ntfy.sh](https://ntfy.sh) (free, no
account). Every message is encrypted and authenticated with a random per-install secret
(`rendezvous_secret`, HMAC-SHA256 encrypt-then-MAC, standard library only), on topic names derived from that
secret. ntfy, and anyone who guesses the topic, see only random bytes. A forged or altered message (say, a fake
URL meant to capture your prompts) fails the check and is dropped. Prompts and replies never touch ntfy: they go
straight through the tunnel, guarded by a per-kernel API key.

Messages are sent on state changes plus a heartbeat every ~10 min, about 90 per 11-hour session. ntfy.sh allows
250 messages a day per IP for anonymous use (its [`/v1/account`](https://ntfy.sh/v1/account) endpoint reports the limit). A publish
takes about 0.2 to 0.8 s, which only affects startup, never requests. Prefer your own infrastructure? Set
`ntfy_server:` to a self-hosted ntfy, or use a private GitHub gist (`kllm init --github-token <gist-scoped token>`).
The code is in `gateway/rendezvous.py` and `gateway/sealbox.py`.

## How well does it hold up in a long agent session?
Three agent sessions in parallel (2× Claude Code, 1× opencode), each implementing a ~3-page spec
(library + SQLite + recurrence + undo + import/export + CLI) from scratch, graded afterwards by a
hidden 38-case acceptance suite the model never saw:

| session | hidden tests | own tests | time |
|---|---|---|---|
| opencode | 37/38 (97%) | 121 | 28 min |
| Claude Code | 35/38 (92%) | 116 | 35 min |

Over 72 min of agent work and 10.2M prompt tokens: contexts grew to 165K, **95.9%** of prompt
tokens came from the prompt cache, generation stayed at **50 to 63 t/s from 0 to 175K context**,
median time to first token 2.3 s, 0 loops. The run exposed one failure (a request queued inside
llama-server hit the tunnel's 120 s limit and killed a third session); the keep-alive proxy and
admission control above were built from it. Most of the wall time is the model's thinking
(≈205K output tokens); the request log now separates thinking from answer tokens.

## Where do I see what it is doing?
- `logs/requests.jsonl`: one line per request: backend, prompt tokens, cached tokens, time to
  first token, output tokens (thinking vs answer), speed, loop cuts. Shown live in the panel.
- `kllm sessions`: live sessions, their backends, load and handoffs; `kllm autoscale`: decisions.
- `kllm status`, `kllm kstatus` (raw Kaggle state), `kllm doctor`, `journalctl --user -u kllm` (macOS: `<root>/logs/serve.log`).
