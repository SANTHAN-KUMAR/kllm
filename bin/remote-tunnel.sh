#!/usr/bin/env bash
# Legacy helper. Prefer `kllm remote` (downloads cloudflared on demand) or `remote_enabled: true`
# in config.yaml with `kllm serve`. Exposes the relay (or LiteLLM) on a public https URL.
exec "${KLLM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/kllm" remote "$@"
