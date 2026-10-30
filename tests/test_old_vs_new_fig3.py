"""Old vs new (seam 3): `fig3`, `figA3` and `polarisation_summary` against `analyse_responseFrequencies.py`.

Run with `pytest -m slow tests/test_old_vs_new_fig3.py`. Skipped when `simOut_old_model` is missing. The old script
uses seeds 0…19, so the package is called with `seeds=range(20)`.
"""

import re

import numpy as np
import pandas as pd
import pytest

from anb.analysis import CONFIGS, Loader, index_folder
from anb.analysis.responses import PAPER_PRESSURES, fig3, figA3, polarisation_summary
from old_scripts import OLD_SIM_DIR, png_difference

pytestmark = pytest.mark.slow

SCRIPT = "analyse_responseFrequencies.py"
SEEDS = range(20)
CONFIG_OF = {values: name for name, values in CONFIGS.items()}  # (init_w, eps, mu, fixedBNat100) → config
RESPONSES = ["compliant", "resilient", "resistant", "latecompliant"]
# Fig. A3 is drawn from the same runs with the same agents, so the PNGs differ only by anti-aliasing. Fig. 3's
# strip plots jitter with the global RNG, unseeded in the old script, so its dots move: reported, not asserted.
MAX_PIXEL_DIFF_A3 = 0.001


def config_column(frame: pd.DataFrame) -> pd.Series:
    keys = zip(frame.init_w, frame.eps, frame.mu, frame.fixedBNat100)
    return pd.Series([CONFIG_OF[tuple(k)] for k in keys], index=frame.index)


@pytest.fixture(scope="module")
def old(old_script):
    result = old_script(SCRIPT)
    if "relres" not in result.globals:
        pytest.fail(result.report())
    return result


@pytest.fixture(scope="module")
def shared():
    return dict(index=index_folder(OLD_SIM_DIR), loader=Loader())


@pytest.fixture(scope="module")
def new_fig3(shared):
    return fig3(OLD_SIM_DIR, seeds=SEEDS, **shared)


@pytest.fixture(scope="module")
def new_figA3(shared):
    return figA3(OLD_SIM_DIR, **shared)


def test_old_script_ran(old):
    assert old.ok, old.report()
    assert old.repo_changes == []


def test_fig3_frequencies_match_exactly(old, new_fig3):
    relres = old.globals["relres"]  # already restricted to s > 0 by the script
    expected = relres.assign(config=config_column(relres), s=relres.s_ext.astype(int))
    expected["seed"] = relres.index % len(SEEDS)  # `res` rows run over config, s, then seed (innermost)
    expected = expected.melt(id_vars=["config", "s", "seed"], value_vars=RESPONSES, var_name="response",
                             value_name="frequency")
    key = ["config", "s", "seed", "response"]
    expected = expected.sort_values(key, ignore_index=True)
    got = new_fig3.data.sort_values(key, ignore_index=True)
    pd.testing.assert_frame_equal(got[key], expected[key], check_dtype=False)
    np.testing.assert_array_equal(got.frequency.to_numpy(), expected.frequency.to_numpy())


def test_figA3_highlights_the_same_agents(old, new_figA3):
    # The script prints "type id" for Fig. 3A (ids given), then the filename, then "type [id.]" per Fig. A3 draw.
    after = old.stdout.split("figs/fig3_responseFrequencies", 1)[1]
    drawn = [(name, int(float(agent))) for name, agent in re.findall(r"^(\w+) \[([\d.]+)\]$", after, re.M)]
    rows = new_figA3.data[new_figA3.data.highlight != ""][["s", "highlight", "agent"]].drop_duplicates()
    order = {name: k for k, name in enumerate(("resistant", "resilient", "compliant"))}
    rows = rows.sort_values(["s", "highlight"], key=lambda c: c.map(order) if c.name == "highlight" else c)
    assert list(zip(rows.highlight, rows.agent)) == drawn
    assert sorted(rows.s.unique()) == list(PAPER_PRESSURES)


def test_figA3_png_matches(old, new_figA3, tmp_path):
    [path] = new_figA3.save(tmp_path)
    diff = png_difference(path, old.pngs[f"{new_figA3.name}.png"])
    assert diff <= MAX_PIXEL_DIFF_A3, f"{diff:.2%} of the pixels differ"


def test_fig3_png_is_close(old, new_fig3, tmp_path):
    [path] = new_fig3.save(tmp_path)
    print(f"\nfig3: {png_difference(path, old.pngs[f'{new_fig3.name}.png']):.2%} of the pixels differ (jitter)")


def test_polarisation_summary_matches_the_old_summary_cells(old, shared):
    summary = polarisation_summary(OLD_SIM_DIR, seeds=SEEDS, **shared)
    res = old.globals["res"].assign(config=lambda d: config_column(d))
    edges = pd.DataFrame(old.globals["mean_absedges"], columns=["init_w", "eps", "mu", "s", "fixedBNat100", "name",
                                                                "absOm_tot", "Om_tot"])
    edges["config"] = config_column(edges)
    for column, metric in (("std_focal", "std_focal"), ("groupishness", "groupishness")):
        for stat in ("mean", "std"):
            expected = res.groupby("config")[column].agg(stat)
            np.testing.assert_allclose(summary[(metric, stat)].sort_index(), expected.sort_index(), rtol=1e-12)
    for column, metric in (("absOm_tot", "abs_edge"), ("Om_tot", "mean_edge")):
        expected = edges.groupby("config")[column].mean()
        np.testing.assert_allclose(summary[(metric, "mean")].sort_index(), expected.sort_index(), rtol=1e-12)
    assert (summary[("n_runs", "")] == len(SEEDS) * len(PAPER_PRESSURES)).all()
    # nr_negs differs on purpose: the old script reads the t = 100 snapshot (ensemble files have no t = 91…99),
    # the package the 95.5 before-window row.
