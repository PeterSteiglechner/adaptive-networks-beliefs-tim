"""On-demand benchmark, new versus legacy: one run (N = 100, M = 10, T = 200, all 14 metrics) and the full
240-run ensemble (4 configurations × 6 pressures × 10 seeds, legacy `__main__`'s default sweep).

The update loop and the metrics are timed separately. Run with `pytest -m slow -s tests/test_benchmark.py`
to see the report; ADR-001's target is a 50–200× speed-up of the update loop.
"""

import os
import time
from contextlib import contextmanager

import numpy as np
import pytest
from joblib import Parallel, delayed

from anb.metrics import MetricsRecorder
from anb.model import ANBModel
from legacy_oracle import load_legacy
from numba_session import run_session
from validation import CONFIGS, PRESSURES, anb_testbed, legacy_run_params

pytestmark = pytest.mark.slow

# One run: the paper's fully adaptive configuration at pressure 4.
CELL = ("adaptive", 4)
SEEDS = range(3)
ENSEMBLE_SEEDS = range(10)
N_JOBS = max(1, (os.cpu_count() or 2) - 1)  # as legacy `__main__`


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


def time_new(seed: int, cell=CELL) -> dict:
    testbed = anb_testbed(*cell)
    totals = {"update": 0.0, "metrics": 0.0}
    start = time.perf_counter()
    with timed(ANBModel, "step", totals, "update"), timed(MetricsRecorder, "record", totals, "metrics"):
        testbed.run(seed)
    return {**totals, "total": time.perf_counter() - start}


def time_legacy(legacy, seed: int, cell=CELL) -> dict:
    """Legacy's update loop is everything in `run_simulation` except setup, `fill_metrics` and `get_output`."""
    totals = {"setup": 0.0, "metrics": 0.0, "output": 0.0}
    start = time.perf_counter()
    with (
        timed(legacy, "initialise_agents", totals, "setup"),
        timed(legacy, "initialise_network", totals, "setup"),
        timed(legacy, "fill_metrics", totals, "metrics"),
        timed(legacy, "get_output", totals, "output"),
    ):
        legacy.run_simulation(legacy_run_params(*cell, seed))
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


def time_ensemble(timer) -> dict:
    """Wall time of all 240 runs on `N_JOBS` workers, plus each part summed over runs (CPU seconds)."""
    jobs = [(seed, (config, pressure)) for config in CONFIGS for pressure in PRESSURES for seed in ENSEMBLE_SEEDS]
    start = time.perf_counter()
    runs = Parallel(n_jobs=N_JOBS)(delayed(timer)(seed, cell) for seed, cell in jobs)
    wall = time.perf_counter() - start
    return {"runs": len(runs), "wall": wall, **{k: sum(r[k] for r in runs) for k in ("update", "metrics", "total")}}


def legacy_run(seed, cell):
    return time_legacy(load_legacy(), seed, cell)  # a fresh legacy module per run (known issue #2)


def test_full_ensemble_new_versus_legacy():
    Parallel(n_jobs=N_JOBS)(delayed(time_new)(0) for _ in range(N_JOBS))  # warm every worker's kernels

    new = time_ensemble(time_new)
    old = time_ensemble(legacy_run)

    print(f"\nFull ensemble: {new['runs']} runs on {N_JOBS} workers")
    print(f"{'':22}{'legacy':>10}{'new':>10}{'speed-up':>10}")
    print(f"{'wall time [s]':22}{old['wall']:10.1f}{new['wall']:10.1f}{old['wall'] / new['wall']:9.0f}×")
    for key in ("update", "metrics", "total"):
        o, n = old[key], new[key]
        print(f"{key + ' [CPU s]':22}{o:10.1f}{n:10.1f}{o / n:9.0f}×")
    print(f"Metrics share of the new run time: {new['metrics'] / new['total']:.0%}")

    assert new["wall"] < old["wall"]


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="nothing is compiled without the JIT")
def test_compile_once_then_load_from_the_disk_cache(tmp_path):
    cold = run_session(tmp_path)
    cached = run_session(tmp_path)

    print(f"\nFirst sweep step: {cold['seconds']:.2f} s compiling, {cached['seconds']:.3f} s from the disk cache")
    assert cached["seconds"] < cold["seconds"]
