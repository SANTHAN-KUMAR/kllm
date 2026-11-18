# Changelog

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
