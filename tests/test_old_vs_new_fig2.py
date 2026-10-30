"""Old vs new (seam 3): `fig2` and its summary numbers against the unchanged `analyse_timeseries.py`.

Run with `pytest -m slow tests/test_old_vs_new_fig2.py`. Skipped when `simOut_old_model` is missing.
"""

import subprocess

import numpy as np
import pandas as pd
import pytest

from anb.analysis.timeseries import example_agent_stats, fig2, focal_sign_counts
from old_scripts import OLD_FIGS, OLD_SIM_DIR, REPO, png_difference

pytestmark = pytest.mark.slow

SCRIPT = "analyse_timeseries.py"
AGENTS = [61, 79]
# The two PNGs (600 dpi, same size) may differ in at most this fraction of pixels (any RGB channel off by > 0.05).
# Both are drawn by the same matplotlib/networkx with the same layout seeds; measured: 0.004 % (matplotlib 3.11.2),
# a few anti-aliased edge pixels. 0.1 % leaves room for other versions but catches a moved or recoloured element.
MAX_PIXEL_DIFF = 0.001


@pytest.fixture(scope="module")
def old(old_script):
    result = old_script(SCRIPT)
    if "df_pivot" not in result.globals:  # the cells that build the figure didn't run; nothing left to compare
        pytest.fail(result.report())
    return result


@pytest.fixture(scope="module")
def new():
    return fig2(OLD_SIM_DIR)


def test_old_script_runs_unchanged_and_writes_nothing_into_the_repo(old):
    assert old.ok, old.report()
    changed = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "analyse_*.py"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    assert changed == ""
    assert old.repo_changes == []


def test_focal_trajectories_match(old, new):
    expected = old.globals["df_pivot"].copy()  # the globals are cached for the session; don't modify them
    expected.columns = expected.columns.astype(int)
    focal = new.data[new.data.quantity == "focal"].pivot(index="t", columns="agent", values="value")
    pd.testing.assert_frame_equal(focal, expected, check_names=False, check_index_type=False,
                                  check_column_type=False)


@pytest.mark.parametrize("quantity", ["belief", "edge"])
def test_example_agent_networks_match(old, new, quantity):
    df = old.globals["df"]
    columns = old.globals["belief_columns" if quantity == "belief" else "edges_columns"]
    expected = df[df.t.isin([0, 100]) & df.id.isin(AGENTS)].melt(id_vars=["t", "id"], value_vars=columns)
    expected = expected.set_index(["id", "t", "variable"]).value.sort_index()
    got = new.data[new.data.quantity == quantity].set_index(["agent", "t", "item"]).value.sort_index()
    np.testing.assert_array_equal(got.index.to_numpy(), expected.index.to_numpy())
    np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())


def test_example_agent_stats_and_sign_counts_match(old):
    df = old.globals["df"]
    at100 = df.loc[df.t == 100].set_index("id")
    stats = example_agent_stats(OLD_SIM_DIR)
    for column in ("tb_tot", "tb_foc", "Hpers"):
        np.testing.assert_array_equal(stats[column].to_numpy(), at100.loc[AGENTS, column].to_numpy())
    old_signs = np.sign(old.globals["final_focal_values"]["belief_smooth"]).value_counts()
    counts = focal_sign_counts(OLD_SIM_DIR)["count"]
    assert {int(k): int(v) for k, v in old_signs.items()} == {k: v for k, v in counts.items() if v}


def test_png_matches_the_old_one(old, new, tmp_path):
    [path] = new.save(tmp_path)
    old_png = old.pngs[f"{new.name}.png"]
    diff = png_difference(path, old_png)
    assert diff <= MAX_PIXEL_DIFF, f"{diff:.2%} of the pixels differ; compare {path} with {old_png}"
    if (OLD_FIGS / old_png.name).is_file():  # the committed paper figure: reported, not asserted
        print(f"\n{old_png.name}: {png_difference(path, OLD_FIGS / old_png.name):.2%} of the pixels differ from "
              f"figs/figsold/")
