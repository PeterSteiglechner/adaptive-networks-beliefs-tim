"""Seam 2 (betweenness): `FocalBetweenness.compute` equals legacy `bc_foc`, and igraph, on the same state."""

import time
from itertools import combinations

import igraph as ig
import numpy as np
import pytest

from anb.agents import ANBAgent, AgentState
from anb.metrics import FocalBetweenness, StepContext
from anb.model import ANBModel
from anb.networks import ErdosRenyiNetwork
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


def igraph_focal_betweenness(edge_weights, focal):
    """The reference: igraph betweenness of each agent's focal node, with legacy's edge lengths."""
    m = edge_weights.shape[1]
    pairs = list(combinations(range(m), 2))
    graph = ig.Graph(n=m, edges=pairs, directed=False)
    out = []
    for w, f in zip(np.abs(edge_weights), focal):
        lengths = [1.0 / (w[i, j] + 1e-6) if w[i, j] > 0 else 1e9 for i, j in pairs]
        out.append(graph.betweenness(vertices=int(f), weights=lengths, directed=False))
    return np.array(out)


def context_for(edge_weights, focal):
    """A `StepContext` whose state carries `edge_weights`; nothing else matters to betweenness."""
    n, m, _ = edge_weights.shape
    state = AgentState(np.zeros((n, m)), edge_weights, np.zeros((n, 1, m)), focal.astype(np.int64))
    model = ANBModel(ANBAgent(n_beliefs=m), ErdosRenyiNetwork(np.zeros((n, n)), link_prob=0.0), state, 1 / 3)
    return StepContext(model, 0, 0.0)


def random_edge_matrices(rng, n, m, kind):
    """Symmetric, zero-diagonal `(n, m, m)` edges. `grid` puts |w| on a coarse grid, so shortest paths tie."""
    if kind == "uniform":
        upper = rng.uniform(-2, 2, size=(n, m, m))
    elif kind == "grid":
        upper = rng.integers(-4, 5, size=(n, m, m)) * 0.25
    elif kind == "sparse":
        upper = rng.uniform(-1, 1, size=(n, m, m)) * (rng.random((n, m, m)) < 0.3)
    elif kind == "equal":
        upper = np.full((n, m, m), 0.2) * rng.choice([-1, 1], size=(n, m, m))
    upper = np.triu(upper, k=1)
    return upper + upper.transpose(0, 2, 1)


KINDS = ["uniform", "grid", "sparse", "equal"]


@pytest.mark.parametrize("m", [4, 10])
@pytest.mark.parametrize("kind", KINDS)
def test_equals_igraph_on_random_edge_matrices(kind, m):
    rng = np.random.default_rng([KINDS.index(kind), m])
    edges = random_edge_matrices(rng, 200, m, kind)
    focal = rng.integers(m, size=200)

    np.testing.assert_allclose(FocalBetweenness().compute(context_for(edges, focal)),
                               igraph_focal_betweenness(edges, focal), **TOL)


def test_tied_shortest_paths_split_the_betweenness():
    """A 4-cycle 0-1-2-3 of equal edges (the chords are zero, length 1e9): two shortest 1-3 paths, one via 0."""
    w = np.zeros((1, 4, 4))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        w[0, i, j] = w[0, j, i] = 0.5

    assert FocalBetweenness().compute(context_for(w, np.array([0]))) == pytest.approx([0.5])


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
