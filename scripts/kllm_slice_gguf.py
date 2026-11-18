"""Shim for backward compatibility with scripts.kllm_slice_gguf.

The implementation is modularized in `swarm.scripts.kllm_slice_gguf`.
"""

from swarm.scripts.kllm_slice_gguf import *

if __name__ == "__main__":
    import sys
    from swarm.scripts.kllm_slice_gguf import main
    main()
