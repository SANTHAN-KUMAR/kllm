# Security policy

## How do I report a vulnerability?

Please report it privately through
[GitHub's private vulnerability reporting](https://github.com/SANTHAN-KUMAR/kllm/security/advisories/new),
not in a public issue. Include the version (`pip show kllm-gateway`), what an attacker can do, and steps to
reproduce. You will get a reply within a week.

## What is in scope?

- The relay accepting requests it should refuse (non-local callers without the gateway key, or reaching the
  panel or `/admin` from another machine).
- Credentials leaking: Kaggle tokens, the gateway key or `rendezvous_secret` appearing in logs, URLs, kernels
  or panel responses.
- Forged rendezvous messages being accepted (see `gateway/sealbox.py`).
- The panel's Connect page writing outside the files it says it changes.

## Which versions get fixes?

The latest release on [PyPI](https://pypi.org/project/kllm-gateway/). Upgrade with
`uv tool upgrade kllm-gateway` or `pipx upgrade kllm-gateway`.
