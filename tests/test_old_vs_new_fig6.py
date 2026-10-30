"""Old vs new (seam 3): `fig6` and `figA7` against `analyse_socialAdaptation.py` (11 μ values × 100 seeds, s = 4).

Run with `pytest -m slow tests/test_old_vs_new_fig6.py`. Skipped when `simOut_old_model` is missing.
"""

import numpy as np
import pytest

from anb.analysis import Loader, fig6, figA7, index_folder
from anb.analysis.social import FIG6_METRICS, PAPER_MUS, PLOTTED
from old_scripts import OLD_SIM_DIR, png_difference

pytestmark = pytest.mark.slow

SCRIPT = "analyse_socialAdaptation.py"
# Fig. 6: the D y-tick labels differ on purpose (the bug fix), the rest is drawn the same, with the same strip jitter.
FIG6_MAX_PIXEL_DIFF = 0.001  # measured: 0.05 %
FIGA7_MAX_PIXEL_DIFF = 0.001  # measured: 0 %


@pytest.fixture(scope="module")
def old(old_script):
    result = old_script(SCRIPT)
    if "df_plot" not in result.globals:
        pytest.fail(result.report())
    return result


@pytest.fixture(scope="module")
def shared():
    return dict(index=index_folder(OLD_SIM_DIR), loader=Loader())


@pytest.fixture(scope="module")
def new(shared):
    return fig6(OLD_SIM_DIR, **shared)


def test_fig6_uses_the_papers_eleven_mu_values(new):
    assert sorted(new.data.mu.unique()) == list(PAPER_MUS)


def test_heterogeneity_and_its_z_score_match(old, new):
    expected = old.globals["res2"].assign(mu=lambda f: f.mu.astype(float)).set_index(["mu", "seed"])
    got = new.data[new.data.quantity == "H"].set_index(["mu", "seed"])
    assert len(got) == 11 * 100
    np.testing.assert_allclose(got.value, expected.loc[got.index, "mean"], rtol=1e-12)
    np.testing.assert_allclose(got.z, expected.loc[got.index, "meanR"], rtol=1e-10, atol=1e-12)


def test_metric_z_scores_match(old, new):
    """The old `df_plot` has no seed or id: rows are compared in the old order (μ, seed, agent row)."""
    expected = old.globals["df_plot"].assign(mu=lambda f: f.mu.astype(float))
    got = new.data[new.data.quantity.isin(FIG6_METRICS)]
    for metric in FIG6_METRICS:
        for mu in PAPER_MUS:
            old_rows = expected[(expected.metric == metric) & (expected.mu == mu)]
            new_rows = got[(got.quantity == metric) & (got.mu == mu)]
            assert len(new_rows) == len(old_rows) == 100 * 100
            np.testing.assert_allclose(new_rows.value, old_rows.value, rtol=1e-12, equal_nan=True)
            np.testing.assert_allclose(new_rows.z, old_rows.value_norm, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_response_frequencies_match(old, new):
    expected = old.globals["relres"].set_index(["mu", "seed"])
    got = new.data[new.data.quantity == "frequency"]
    for response in PLOTTED:
        rows = got[got.response == response].set_index(["mu", "seed"])
        assert len(rows) == 11 * 100
        np.testing.assert_allclose(rows.value, expected.loc[rows.index, response], rtol=1e-12, equal_nan=True)


def test_fig6_png_differs_only_by_the_tick_fix(old, new, tmp_path):
    [path] = new.save(tmp_path)
    diff = png_difference(path, old.pngs[f"{new.name}.png"])
    print(f"\nfig6: {diff:.3%} of the pixels differ")
    assert diff <= FIG6_MAX_PIXEL_DIFF, f"{diff:.2%} of the pixels differ"


def test_figA7_png_matches(old, shared, tmp_path):
    result = figA7(OLD_SIM_DIR, **shared)
    [path] = result.save(tmp_path)
    diff = png_difference(path, old.pngs[f"{result.name}.png"])
    print(f"\nfigA7: {diff:.3%} of the pixels differ")
    assert diff <= FIGA7_MAX_PIXEL_DIFF, f"{diff:.2%} of the pixels differ"
