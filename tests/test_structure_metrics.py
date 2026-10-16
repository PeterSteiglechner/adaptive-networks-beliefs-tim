"""Seam 2 (BN structure): each structure `Metric.compute(StepContext)` equals legacy `fill_metrics` on the same state."""

import warnings

import numpy as np
import pytest

from anb.metrics import (
    ExpectedInfluence,
    FocalMeanAbsEdge,
    FocalTriangleBalance,
    MeanAbsEdge,
    TriangleBalance,
    WeightedClustering,
)
from legacy_oracle import legacy_and_anb, refocus_randomly

SEEDS = [0, 1, 2, 3, 4]
METRICS = [
    TriangleBalance(),
    FocalTriangleBalance(),
    MeanAbsEdge(),
    FocalMeanAbsEdge(),
    WeightedClustering(),
    ExpectedInfluence(),
]
TOL = {"rtol": 1e-12, "atol": 1e-12}
EMPTY_BN_AGENTS = list(range(5))
# Structure metrics read only the BN, so time and pressure don't matter. t = 50 lies outside any event.
LEGACY_EMPTY_BN_WARNINGS = pytest.mark.filterwarnings("ignore::RuntimeWarning")  # legacy clust: 0/0 and nanmean


def both_sides(legacy, seed, edit=None):
    return legacy_and_anb(legacy, seed, t=50, ext_strength=0.0, edit=edit)


def sparsify(legacy, frac):
    """Zero a fraction `frac` of each agent's edges, so BNs have nodes of degree 0 and 1."""

    def edit(agents):
        keep = np.random.default_rng(0).random((len(agents), len(legacy.edgeids))) > frac
        agents[:, legacy.edgeids] *= keep

    return edit


def empty_bn(legacy):
    """Agents in `EMPTY_BN_AGENTS` get all-zero edge weights."""

    def edit(agents):
        agents[np.ix_(EMPTY_BN_AGENTS, legacy.edgeids)] = 0.0

    return edit


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_metric_equals_legacy(legacy, metric, seed):
    expected, ctx = both_sides(legacy, seed)

    np.testing.assert_allclose(metric.compute(ctx), expected[metric.name], **TOL)


@LEGACY_EMPTY_BN_WARNINGS
@pytest.mark.parametrize("frac", [0.5, 0.8, 0.95])
@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_sparse_belief_networks_equal_legacy(legacy, metric, frac):
    expected, ctx = both_sides(legacy, seed=8, edit=sparsify(legacy, frac))

    np.testing.assert_allclose(metric.compute(ctx), expected[metric.name], **TOL)


@LEGACY_EMPTY_BN_WARNINGS
@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_all_zero_edges_give_the_legacy_result(legacy, metric):
    expected, ctx = both_sides(legacy, seed=9, edit=empty_bn(legacy))

    np.testing.assert_allclose(metric.compute(ctx), expected[metric.name], **TOL)


@LEGACY_EMPTY_BN_WARNINGS
def test_empty_belief_network_has_nan_clustering_without_warnings(legacy):
    _, ctx = both_sides(legacy, seed=9, edit=empty_bn(legacy))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clust = WeightedClustering().compute(ctx)

    assert np.isnan(clust[EMPTY_BN_AGENTS]).all()
    assert np.isfinite(np.delete(clust, EMPTY_BN_AGENTS)).all()


def test_focal_balance_counts_only_triangles_through_focal(legacy):
    _, ctx = both_sides(legacy, seed=1)
    m = ctx.model.state.n_beliefs

    tot = TriangleBalance().compute(ctx)
    foc = FocalTriangleBalance().compute(ctx)

    assert (foc <= tot).all()
    assert (foc <= (m - 1) * (m - 2) / 2).all()


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_metrics_follow_each_agents_focal_dimension(legacy, metric):
    """Swapping belief dimensions 0 and k and setting focal = k leaves every metric unchanged."""
    _, ctx = both_sides(legacy, seed=7)
    before = metric.compute(ctx)
    refocus_randomly(ctx.model.state, np.random.default_rng(0))

    np.testing.assert_allclose(metric.compute(ctx), before, **TOL)
