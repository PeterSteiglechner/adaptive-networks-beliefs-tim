"""Old vs new (seam 3): `fig5` and `figA4` against `analyse_BNmetrics.py` (μ = 0, all 100 seeds).

Run with `pytest -m slow tests/test_old_vs_new_fig5.py`. Skipped when `simOut_old_model` is missing.
"""

import numpy as np
import pandas as pd
import pytest

from anb.analysis import Loader, index_folder
from anb.analysis.bnmetrics import METRICS, NEGATIVE, REFERENCE, fig5, figA4
from anb.analysis.style import METRIC_LABELS
from old_scripts import OLD_SIM_DIR, png_difference

pytestmark = pytest.mark.slow

SCRIPT = "analyse_BNmetrics.py"
# Same runs and the same seaborn calls, so the Fig. 5 PNGs differ only by anti-aliasing. Fig. A4's title changed
# (the bug fix), so its PNG is reported, not asserted.
MAX_PIXEL_DIFF_FIG5 = 0.001


@pytest.fixture(scope="module")
def old(old_script):
    result = old_script(SCRIPT)
    if "res" not in result.globals or "a" not in result.globals:
        pytest.fail(result.report())
    return result


@pytest.fixture(scope="module")
def shared():
    return dict(index=index_folder(OLD_SIM_DIR), loader=Loader())


@pytest.fixture(scope="module")
def new_fig5(shared):
    return fig5(OLD_SIM_DIR, **shared)


@pytest.fixture(scope="module")
def new_figA4(shared):
    return figA4(OLD_SIM_DIR, **shared)


def test_old_script_ran(old):
    assert old.repo_changes == []
    print("\n" + old.report())  # later cells (the logit fits) may fail under newer statsmodels; fig5/A4 don't need them


def test_fig5_means_and_sds_match(old, new_fig5):
    res = old.globals["res"]
    pressured = res[(res.s == 4) & res.response.isin(NEGATIVE)]
    reference = res[res.s == 0].assign(response=REFERENCE)
    long = pd.concat([pressured, reference]).melt(id_vars=["response", "t_eval"], value_vars=list(METRICS),
                                                  var_name="metric")
    expected = long.groupby(["metric", "response", "t_eval"])["value"].agg(["mean", "std", "count"])
    got = new_fig5.data.set_index(["metric", "response", "t"]).sort_index()
    expected = expected.sort_index()
    assert list(got.index) == list(expected.index)
    np.testing.assert_allclose(got["mean"], expected["mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["sd"], expected["std"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(got["n"], expected["count"])


def test_figA4_correlation_matrix_matches(old, new_figA4):
    a = old.globals["a"]  # the last (only, for μ = 0) panel: s = 0, labelled with the metric words
    words = {label: metric for metric, label in METRIC_LABELS.items()}
    expected = a.rename(index=words, columns=words).loc[list(METRICS), list(METRICS)]
    got = new_figA4.data.pivot(index="metric", columns="other", values="r").loc[list(METRICS), list(METRICS)]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12)


def test_fig5_png_matches(old, new_fig5, tmp_path):
    [path] = new_fig5.save(tmp_path)
    diff = png_difference(path, old.pngs[f"{new_fig5.name}.png"])
    assert diff <= MAX_PIXEL_DIFF_FIG5, f"{diff:.2%} of the pixels differ"


def test_figA4_png_is_close(old, new_figA4, tmp_path):
    [path] = new_figA4.save(tmp_path)
    print(f"\nfigA4: {png_difference(path, old.pngs[f'{new_figA4.name}.png']):.2%} of the pixels differ (title)")
