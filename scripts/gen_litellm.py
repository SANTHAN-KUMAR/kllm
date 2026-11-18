#!/usr/bin/env python3
"""Compatibility wrapper: same as `kllm litellm-config`."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gateway import litellm_cfg

out, names = litellm_cfg.generate()
print("wrote", out, "with models:", names)
