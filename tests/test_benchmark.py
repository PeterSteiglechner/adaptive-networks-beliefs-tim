"""On-demand benchmark: one full run (N = 100, M = 10, T = 200, all 14 metrics), new versus legacy.

The update loop and the metrics are timed separately. Run with `pytest -m slow -s tests/test_benchmark.py`
to see the report; ADR-001's target is a 50–200× speed-up of the update loop.
"""

import os
import time
from contextlib import contextmanager

import numpy as np
import pytest

from anb.agents import ANBAgent
from anb.metrics import MetricsRecorder
from anb.model import ANBModel, ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig
from legacy_oracle import load_legacy
from numba_session import run_session

pytestmark = pytest.mark.slow

# The paper's fully adaptive configuration at pressure 4.
PARAMS = dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=1 / 3, eps=1.0, mu=0.0, fixedBNat100=False, ext_strength=4)
SEEDS = range(3)


@contextmanager
def timed(owner, name, totals, key):
    """Replace `owner.name` with a wrapper that adds its wall time to `totals[key]`."""
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            totals[key] += time.perf_counter() - start

    setattr(owner, name, wrapper)
    try:
        yield
    finally:
        setattr(owner, name, original)


def time_new(seed: int) -> dict:
    p = PARAMS
    spec = ModelSpec(
        ANBAgent(beta=p["beta"], eps=p["eps"], mu=p["mu"], init_w=p["init_w"]),
        ErdosRenyiNetwork,
        {"link_prob": p["link_prob"]},
        n_agents=100,
        rho=p["rho"],
    )
    testbed = Testbed(spec, TestbedConfig(PressureSchedule.single_event(p["ext_strength"])))
    totals = {"update": 0.0, "metrics": 0.0}
    start = time.perf_counter()
    with timed(ANBModel, "step", totals, "update"), timed(MetricsRecorder, "record", totals, "metrics"):
        testbed.run(seed)
    return {**totals, "total": time.perf_counter() - start}


def time_legacy(legacy, seed: int) -> dict:
    """Legacy's update loop is everything in `run_simulation` except setup, `fill_metrics` and `get_output`."""
    totals = {"setup": 0.0, "metrics": 0.0, "output": 0.0}
    start = time.perf_counter()
    with (
        timed(legacy, "initialise_agents", totals, "setup"),
        timed(legacy, "initialise_network", totals, "setup"),
        timed(legacy, "fill_metrics", totals, "metrics"),
        timed(legacy, "get_output", totals, "output"),
    ):
        legacy.run_simulation({**PARAMS, "seed": seed})
    total = time.perf_counter() - start
    return {"update": total - sum(totals.values()), "metrics": totals["metrics"], "total": total}


def median(runs: list[dict], key: str) -> float:
    return float(np.median([r[key] for r in runs]))


def test_one_run_new_versus_legacy():
    legacy = load_legacy()
    time_new(0)  # pay (or load from cache) the kernel compile before timing

    new = [time_new(seed) for seed in SEEDS]
    old = [time_legacy(legacy, seed) for seed in SEEDS]

    print(f"\nOne run, N = 100, M = 10, T = 200, 14 metrics; median of {len(SEEDS)} seeds")
    print(f"{'':10}{'legacy [s]':>12}{'new [s]':>12}{'speed-up':>10}")
    for key in ("update", "metrics", "total"):
        o, n = median(old, key), median(new, key)
        print(f"{key:10}{o:12.3f}{n:12.4f}{o / n:9.0f}×")
    speedup = median(old, "update") / median(new, "update")
    verdict = "within" if 50 <= speedup <= 200 else ("above" if speedup > 200 else "below")
    print(f"Update loop: {speedup:.0f}× ({verdict} ADR-001's 50–200× target)")

    assert speedup > 1


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="nothing is compiled without the JIT")
def test_compile_once_then_load_from_the_disk_cache(tmp_path):
    cold = run_session(tmp_path)
    cached = run_session(tmp_path)

    print(f"\nFirst sweep step: {cold['seconds']:.2f} s compiling, {cached['seconds']:.3f} s from the disk cache")
    assert cached["seconds"] < cold["seconds"]
