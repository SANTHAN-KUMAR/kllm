# Contributing to kllm

Thanks for helping. Bug reports, model results on real T4s, docs fixes and code are all welcome.

## How do I report a bug?

Open an [issue](https://github.com/SANTHAN-KUMAR/kllm/issues/new/choose) with:

- what you ran (`kllm ...` or the request you sent),
- what you expected and what happened,
- the output of `kllm doctor`, with tokens and URLs removed.

## How do I share a model that works (or does not)?

Use the **Model report** issue template. Include the model key or Hugging Face repo, the context size you ran,
tokens per second at short and long prompts, and anything you changed in `models.yaml`. Measured results are
how an entry moves from `unverified` to `verified`.

## How do I set up a development copy?

```bash
git clone https://github.com/SANTHAN-KUMAR/kllm
cd kllm
pip install -e ".[dev]"
python -m pytest
```

The tests never contact Kaggle, GitHub or ntfy and never touch your home directory. They run on Linux, macOS
and Windows with Python 3.10 to 3.13; CI checks all three.

## What makes a pull request easy to merge?

- One change per pull request, with a test that fails without it.
- `python -m pytest` passes locally.
- No new runtime dependency unless it is essential: the relay deliberately depends on five packages.
- Panel changes pass `python scripts/ui_lint.py gateway/ui` (no external requests, no em dashes in UI text).

## Where do I ask questions?

[Discussions](https://github.com/SANTHAN-KUMAR/kllm/discussions) is the place for setup questions, ideas and
showing what you built.

By contributing you agree that your work is released under the [MIT license](LICENSE), and that you follow the
[code of conduct](CODE_OF_CONDUCT.md).
