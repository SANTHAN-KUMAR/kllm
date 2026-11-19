# Changelog

## 0.3.2 (2026-11-19)

- Opening `http://127.0.0.1:8080/` in a browser now goes to the panel instead of showing an `all_down` error.
- With no Kaggle account set up, requests fail at once with a pointer to the panel, instead of waiting up to
  20 minutes for a GPU that cannot start.
- `local_fallback` is off by default. Configs written by 0.3.0 and 0.3.1 had it on for a server most machines do
  not run; kllm switches it off once on the next `kllm serve` (a fallback you enabled yourself is left alone).
- `kllm serve` prints the panel address.
- Update checks: kllm looks for a newer version on PyPI once a day in the background and says so in the terminal
  and the panel. `kllm update` upgrades with the tool that installed it (uv, pipx, pip or git).
  Turn checks off with `update_check: false`.

## 0.3.1 (2026-11-18)

- PyPI page: images load from absolute URLs, richer metadata and project links.
- Docs: `qwen3.8-27b` family and real usage numbers; community files, issue templates and a landing page.

## 0.3.0 (2026-11-18)

First public release, published to PyPI as `kllm-gateway`.

- One-command install: `pipx install kllm-gateway`, `uv tool install kllm-gateway`, or the one-line
  installers for machines without Python.
- Setup in the browser: `kllm` opens the panel; the first-run page takes a Kaggle token or a dropped
  `kaggle.json`. Existing `~/.kaggle` credentials are picked up automatically.
- No GitHub token needed: kernels report their tunnel URL over ntfy, encrypted and signed with a per-install
  secret. A private gist remains optional.
- Linux, macOS and Windows, Python 3.10 to 3.13, with CI on all three.
