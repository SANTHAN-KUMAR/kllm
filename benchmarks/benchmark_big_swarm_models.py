"""Shim for backward compatibility with benchmarks.benchmark_big_swarm_models.

The implementation is modularized in `swarm.benchmarks.benchmark_big_swarm_models`.
"""

from swarm.benchmarks.benchmark_big_swarm_models import *

if __name__ == "__main__":
    import asyncio
    from swarm.benchmarks.benchmark_big_swarm_models import main
    asyncio.run(main())
