# Changelog

## 0.3.5 (2026-11-19)

- Fix: valid `KGAT_` tokens were rejected on clean installs because `kaggle` 2.2.5 imports `six` without declaring it,
  and kllm misreported that import crash as invalid credentials. `six` is now declared, and environment errors say
  what actually failed instead of blaming your token.
- Fix: adding the first account to a fresh starter config (`accounts: []`) failed to save valid YAML; `yamledit` now
  converts `accounts: []` (and `null`) into a block list and cleans up temporary credential files if saving fails.
- Fix: Windows UTF-16LE / BOM `access_token` or `kaggle.json` files and non-ASCII Kaggle CLI output no longer raise
  `UnicodeDecodeError` during setup or account checks.
- Clean-room smoke tests now check the `kaggle` library import, `kaggle` CLI, credential validation on `accounts: []`,
  and adding an account to the real starter config on Linux, macOS and Windows.

## 0.3.4 (2026-11-19)

- Fix: `kllm serve` crashed in clean installs with `No module named 'numpy'`. The image-analysis code imported
  numpy and Pillow, which were never declared as dependencies. They are now declared, and loaded only when an image
  needs analysing, so a missing image library can no longer stop the relay from starting.
- `kllm update` and `kllm uninstall` work when uv or pipx is no longer on the machine: updates fall back to the
  environment's own pip, and uninstall deletes the environment and its launchers directly. On Windows the final
  step runs from cmd.exe, so it no longer tries to delete the Python it is running from.
- Every release is now installed into a clean, isolated environment and `kllm serve` is started there, on Linux,
  macOS and Windows, before it can be published. A unit test also fails on any undeclared import.

## 0.3.3 (2026-11-19)

- `kllm uninstall` removes kllm completely, after listing what it will do: stops running Kaggle GPUs and the
  relay, removes the background service and the desktop icon, undoes the tool settings the Connect page wrote,
  deletes kllm's config, state and stored token copies, and uninstalls the package with the tool that installed
  it. Your own `~/.kaggle` credentials are never touched; `--keep-config`, `--dry-run` and `--yes` are available.
- `kllm stop` stops the local relay.

## 0.3.2 (2026-11-19)

- Opening `http://127.0.0.1:8080/` in a browser now goes to the panel instead of showing an `all_down` error.
- With no Kaggle account set up, requests fail at once with a pointer to the panel, instead of waiting up to
  20 minutes for a GPU that cannot start.
- `local_fallback` is off by default. Configs written by 0.3.0 and 0.3.1 had it on for a server most machines do
  not run; kllm switches it off once on the next `kllm serve` (a fallback you enabled yourself is left alone).
- `kllm serve` opens the panel in your browser (the welcome page on a first run). It never does this for
  the background service, over SSH or on a headless machine; `--no-browser` turns it off.
- Desktop icon: `kllm shortcut` adds a kllm icon to the desktop and app menu (Windows, macOS, Linux) that
  starts kllm and opens the panel with no terminal window. The one-line installers add it automatically.
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
