"""Seam 2 (betweenness): `FocalBetweenness.compute` equals legacy `bc_foc` on the same state."""

import time

import numpy as np
import pytest

from anb.metrics import FocalBetweenness
from legacy_oracle import legacy_and_anb, legacy_metrics, legacy_params, random_legacy_state, to_context

STATES = [(0, 50), (1, 120), (2, 180), (3, 120), (4, 50)]
TOL = {"rtol": 1e-12, "atol": 1e-12}


@pytest.mark.parametrize("seed, t", STATES)
def test_focal_betweenness_equals_legacy(legacy, seed, t):
    expected, ctx = legacy_and_anb(legacy, seed, t, ext_strength=4.0)

    np.testing.assert_allclose(FocalBetweenness().compute(ctx), expected["bc_foc"], **TOL)


def test_states_have_varied_focal_betweenness(legacy):
    expected, _ = legacy_and_anb(legacy, seed=0, t=50, ext_strength=4.0)

    assert (expected["bc_foc"] > 0).any()
    assert len(np.unique(expected["bc_foc"])) > 5


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # legacy `clust` divides by zero on all-zero networks
@pytest.mark.parametrize("seed", [10, 11])
def test_zero_and_negative_edges_are_treated_like_legacy(legacy, seed):
    """Negative edges count by |w|. Zero edges stay in the graph with length 1e9, as in legacy."""
    agents, nb_list = random_legacy_state(legacy, seed)
    rng = np.random.default_rng(seed)
    edges = agents[:, legacy.edgeids]
    edges[rng.random(edges.shape) < 0.4] = 0.0
    edges[:5] = 0.0  # all-zero belief networks, as with init_w = 0
    edges[5:10] = -np.abs(edges[5:10])  # all-negative belief networks
    agents[:, legacy.edgeids] = edges
    params = legacy_params()
    expected = legacy_metrics(legacy, agents, nb_list, 50, params)
    ctx = to_context(legacy, agents, nb_list, 50, params)

    assert (expected["bc_foc"][5:10] > 0).any()
    np.testing.assert_allclose(FocalBetweenness().compute(ctx), expected["bc_foc"], **TOL)


def test_follows_each_agents_focal_dimension(legacy):
    """Swapping belief dimensions 0 and k and setting focal = k leaves the betweenness unchanged."""
    _, ctx = legacy_and_anb(legacy, seed=7, t=120, ext_strength=4.0)
    before = FocalBetweenness().compute(ctx)
    state = ctx.model.state
    focal = np.random.default_rng(0).integers(state.n_beliefs, size=state.n_agents)
    for i, k in enumerate(focal):
        perm = np.arange(state.n_beliefs)
        perm[[0, k]] = perm[[k, 0]]
        state.beliefs[i] = state.beliefs[i, perm]
        state.edge_weights[i] = state.edge_weights[i][np.ix_(perm, perm)]
    state.focal[:] = focal

    assert (focal != 0).any()
    np.testing.assert_allclose(FocalBetweenness().compute(ctx), before, **TOL)


@pytest.mark.slow
def test_wall_time_per_tracked_step(legacy):
    """Baseline for ticket 11's profiling: one `compute` call (all N agents) at a tracked step."""
    _, ctx = legacy_and_anb(legacy, seed=0, t=50, ext_strength=4.0)
    metric = FocalBetweenness()
    metric.compute(ctx)
    repeats = 20
    start = time.perf_counter()
    for _ in range(repeats):
        metric.compute(ctx)
    per_step = (time.perf_counter() - start) / repeats
    print(f"\nFocalBetweenness: {per_step * 1e3:.2f} ms per tracked step (N = {ctx.model.state.n_agents})")
