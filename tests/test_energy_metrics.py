"""Seam 2 (energies): each `Metric.compute(StepContext)` equals legacy `fill_metrics` on the same state."""

import numpy as np
import pytest

from anb.metrics import ExternalEnergy, FocalPersonalEnergy, NonFocalExtremity, PersonalEnergy, SocialEnergy
from legacy_oracle import legacy_and_anb, refocus_randomly

ISOLATED = (3, 41, 77)
# (seed, t, ext_strength): t = 120 lies inside legacy's pressure event (t = 101…150), t = 50 and t = 180 don't.
STATES = [(0, 50, 4.0), (1, 120, 4.0), (2, 120, 16.0), (3, 180, 8.0), (4, 120, 0.0)]
METRICS = [PersonalEnergy(), FocalPersonalEnergy(), SocialEnergy(), ExternalEnergy(), NonFocalExtremity()]
TOL = {"rtol": 1e-12, "atol": 1e-12}


def both_sides(legacy, seed, t, ext_strength, rho=1 / 3):
    return legacy_and_anb(legacy, seed, t, ext_strength, rho=rho, isolate=ISOLATED)


@pytest.mark.parametrize("seed, t, ext_strength", STATES)
@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_metric_equals_legacy(legacy, metric, seed, t, ext_strength):
    expected, ctx = both_sides(legacy, seed, t, ext_strength)

    np.testing.assert_allclose(metric.compute(ctx), expected[metric.name], **TOL)


def test_states_cover_nonzero_pressure_and_isolated_agents(legacy):
    expected, ctx = both_sides(legacy, seed=1, t=120, ext_strength=4.0)

    assert ctx.ext == 4.0
    assert (expected["Hext"] != 0).any()
    assert (ctx.model.network.degree()[list(ISOLATED)] == 0).all()


def test_isolated_agents_have_zero_social_energy_like_legacy(legacy):
    expected, ctx = both_sides(legacy, seed=0, t=120, ext_strength=4.0)

    assert (expected["Hsoc"][list(ISOLATED)] == 0).all()
    assert (SocialEnergy().compute(ctx)[list(ISOLATED)] == 0).all()
    assert (SocialEnergy(aggregate="sum").compute(ctx)[list(ISOLATED)] == 0).all()


def test_social_energy_sum_is_the_neighbour_sum_of_eq_1(legacy):
    rho = 0.4
    _, ctx = both_sides(legacy, seed=5, t=120, ext_strength=0.0, rho=rho)
    x = ctx.model.state.focal_beliefs()
    expected = np.array([-rho * x[i] * x[ctx.model.network.neighbours(i)].sum() for i in range(len(x))])

    np.testing.assert_allclose(SocialEnergy(aggregate="sum").compute(ctx), expected, **TOL)


def test_social_energy_mean_is_the_sum_divided_by_degree(legacy):
    expected, ctx = both_sides(legacy, seed=6, t=50, ext_strength=0.0)
    degree = ctx.model.network.degree()
    has_nbs = degree > 0

    np.testing.assert_allclose(
        SocialEnergy("mean").compute(ctx)[has_nbs],
        SocialEnergy("sum").compute(ctx)[has_nbs] / degree[has_nbs],
    )


def test_social_energy_rejects_unknown_aggregate():
    with pytest.raises(ValueError, match="aggregate"):
        SocialEnergy(aggregate="median")


@pytest.mark.parametrize("metric", METRICS, ids=lambda m: m.name)
def test_metrics_follow_each_agents_focal_dimension(legacy, metric):
    """Swapping belief dimensions 0 and k and setting focal = k leaves every metric unchanged."""
    _, ctx = both_sides(legacy, seed=7, t=120, ext_strength=4.0)
    before = metric.compute(ctx)
    refocus_randomly(ctx.model.state, np.random.default_rng(0))

    np.testing.assert_allclose(metric.compute(ctx), before, **TOL)
