"""Run one sweep step in a fresh Python session, for checking the kernel's on-disk Numba cache."""

import json
import os
import subprocess
import sys
import textwrap

# One short run in a fresh interpreter; reports the kernel's compile/cache counters and the run's wall time.
SESSION = textwrap.dedent(
    """
    import json, time
    import numpy as np
    from anb.agents import ANBAgent, _sweep
    from anb.model import ModelSpec
    from anb.networks import ErdosRenyiNetwork

    model = ModelSpec(ANBAgent(), ErdosRenyiNetwork, {"link_prob": 0.1}, n_agents=20).build(np.random.default_rng(0))
    start = time.perf_counter()
    model.step(1, 0.0, np.random.default_rng(1))
    elapsed = time.perf_counter() - start
    print(json.dumps({
        "hits": sum(_sweep.stats.cache_hits.values()),
        "misses": sum(_sweep.stats.cache_misses.values()),
        "seconds": elapsed,
    }))
    """
)


def run_session(cache_dir) -> dict:
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    out = subprocess.run([sys.executable, "-c", SESSION], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])
