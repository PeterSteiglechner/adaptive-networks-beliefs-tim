"""Smoke tests for the test scaffold: the package imports and the legacy oracle loads."""

from pathlib import Path

import numpy as np

from legacy_oracle import load_legacy

REPO = Path(__file__).resolve().parent.parent


def test_anb_package_imports_from_the_repo():
    import anb

    assert Path(anb.__file__).resolve().parent == REPO / "anb"


def test_legacy_oracle_injects_main_only_names():
    legacy = load_legacy()

    assert legacy.detail is False
    assert list(legacy.track_times) == (
        [0, 1] + list(range(91, 101)) + [100] + list(range(141, 151)) + list(range(191, 201)) + [200]
    )
    assert list(load_legacy(detail=True).track_times) == list(range(201))


def test_legacy_oracle_loads_are_independent():
    assert load_legacy() is not load_legacy()


def test_legacy_get_output_runs_through_the_oracle(legacy):
    np.random.seed(0)
    agents = legacy.initialise_agents(0.2)
    # One snapshot per tracked step, as run_simulation records them (metrics left NaN).
    times = sorted(set(legacy.track_times))
    snapshots = np.concatenate([np.where(np.arange(agents.shape[1]) == 0, t, agents) for t in times])

    out = legacy.get_output(snapshots)

    assert set(out.columns) == set(legacy.columns)
    assert sorted(out.t.unique()) == [0, 95.5, 100, 145.5, 195.5, 200]
    assert len(out) == 6 * legacy.n_agents


def test_numba_accepts_a_generator_argument():
    from numba import njit

    @njit
    def draw(rng):
        return rng.random()

    assert draw(np.random.default_rng(1)) == draw(np.random.default_rng(1))
