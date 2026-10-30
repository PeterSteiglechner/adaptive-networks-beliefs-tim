"""Old vs new (seam 3): `fig4` against `analyse_BNmetrics.py` (μ = 0, all 100 seeds).

Run with `pytest -m slow tests/test_old_vs_new_fig4.py`. Skipped when `simOut_old_model` is missing.
"""

import numpy as np
import pytest

from anb.analysis import Loader, fig4, index_folder
from anb.analysis.bnmetrics import PREDICTORS
from anb.analysis.style import METRIC_LABELS
from old_scripts import OLD_SIM_DIR, png_difference

pytestmark = pytest.mark.slow

SCRIPT = "analyse_BNmetrics.py"
# Same numbers and the same drawing calls (the pressure labels are placed from the data instead of from the sorted
# bar patches, at the same positions), so the PNGs match (measured: 0 %).
MAX_PIXEL_DIFF = 0.001
METRIC_OF_LABEL = {label: metric for metric, label in METRIC_LABELS.items()}


@pytest.fixture(scope="module")
def old(old_script):
    result = old_script(SCRIPT)
    if "combined" not in result.globals:
        pytest.fail(result.report())
    return result


@pytest.fixture(scope="module")
def new():
    return fig4(OLD_SIM_DIR, index=index_folder(OLD_SIM_DIR), loader=Loader())


def test_cohens_d_matches(old, new):
    expected = old.globals["cohensd_df"].assign(metric=lambda f: f.metric.map(METRIC_OF_LABEL))
    expected = expected.set_index(["type", "s", "metric"]).d
    got = new.data.set_index(["contrast", "s", "metric"]).d
    assert len(got) == 5 * 12 + 3 * 12  # rR-C at all five levels, R-r at s = 1, 2, 4
    np.testing.assert_allclose(got, expected.loc[got.index], rtol=1e-10, atol=1e-12)


def test_logit_coefficients_and_cis_match(old, new):
    coef = old.globals["coef_df"].drop(columns="metric").rename_axis("metric").reset_index()  # its metric: words
    expected = coef[coef.metric.isin(PREDICTORS)].set_index(["type", "s", "metric"])
    got = new.data[new.data.metric.isin(PREDICTORS)].set_index(["contrast", "s", "metric"])
    assert got.coefficient.notna().all()
    for column in ("coefficient", "ci_low", "ci_high"):
        np.testing.assert_allclose(got[column], expected.loc[got.index, column], rtol=1e-6, atol=1e-8)


def test_png_matches(old, new, tmp_path):
    [path] = new.save(tmp_path)
    diff = png_difference(path, old.pngs[f"{new.name}.png"])
    assert diff <= MAX_PIXEL_DIFF, f"{diff:.2%} of the pixels differ"
