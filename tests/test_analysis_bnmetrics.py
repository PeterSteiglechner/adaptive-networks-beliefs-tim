"""`anb.analysis` Fig. 5, Fig. A4 and the agent-metric frame, on the synthetic folder of `sim_folder.py`.

The fixture's paper base has the adaptive config at s ∈ {0, 1, 4} × seeds {0, 1, 2} (μ = 0 only). The `link_prob 0.2`
base has adaptive at s = 4 only, so it lacks the s = 0 reference.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim_folder  # noqa: E402
from anb.analysis import (  # noqa: E402
    PAPER, Baseline, FewSeedsWarning, FigureNotApplicable, FigureResult, Loader, fig5, figA4, index_folder,
    metric_frame, runs,
)
from anb.analysis.bnmetrics import METRICS, NEGATIVE, WINDOWS  # noqa: E402
from anb.io import legacy_filename  # noqa: E402

RESPONSE_COLUMNS = ["persistPos", "nonpersistPos", "compliant", "resilient", "resistant", "latecompliant"]


def expected_warnings(func, *args, **kwargs):
    """`func(...)` and the categories it warned with (the stray-file warning of the index left out)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, {w.category for w in caught if "legacy naming scheme" not in str(w.message)}


def adaptive_csv(sim_dir, s, seed) -> pd.DataFrame:
    stem = legacy_filename(dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=1 / 3, eps=1.0, mu=0.0, fixedBNat100=False,
                                ext_strength=s, seed=seed, lam=0.0, n_events=1), False)
    return pd.read_csv(sim_dir / f"{stem}.csv")


def old_style_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The three window rows of one CSV, each agent labelled with its 95.5-row response (the last flag wins)."""
    labels = df.loc[df.t == 95.5].set_index("id")[RESPONSE_COLUMNS]
    response = pd.Series(None, index=labels.index, dtype=object)
    for col in RESPONSE_COLUMNS:
        response[labels[col] == 1] = col
    rows = df[df.t.isin(WINDOWS)].copy()
    rows["response"] = rows.id.map(response)
    rows["Hpersnonfoc"] = rows.Hpers - rows.Hpersfoc
    return rows


@pytest.fixture(scope="module")
def paper_fig5(sim_dir):
    result, categories = expected_warnings(fig5, sim_dir)
    yield result, categories
    plt.close(result.figure)


@pytest.fixture(scope="module")
def paper_figA4(sim_dir):
    result, categories = expected_warnings(figA4, sim_dir)
    yield result, categories
    plt.close(result.figure)


# The agent-metric frame

def test_metric_frame_has_one_row_per_agent_run_and_window(sim_dir):
    selected = runs(index_folder(sim_dir), PAPER, s=4)
    frame = metric_frame(selected, Loader())
    assert len(frame) == 3 * 3 * 100  # 3 seeds × 3 windows × N
    assert {"s", "seed", "t", "id", "response", *METRICS} <= set(frame.columns)
    assert "Hpers" not in frame.columns
    for seed in (0, 1, 2):
        expected = old_style_rows(adaptive_csv(sim_dir, 4, seed)).sort_values(["t", "id"], ignore_index=True)
        got = frame[frame.seed == seed].sort_values(["t", "id"], ignore_index=True)
        np.testing.assert_allclose(got.Hpersnonfoc, expected.Hpersnonfoc)
        assert got.response.fillna("").tolist() == expected.response.fillna("").tolist()


# Fig. 5

def test_fig5_returns_a_figure_result_with_the_old_name(paper_fig5):
    result, categories = paper_fig5
    assert isinstance(result, FigureResult)
    assert result.name == "fig5_metricChange_mu0.0"
    assert len(result.figure.axes) == len(METRICS)
    assert FewSeedsWarning in categories  # 3 seeds, not the paper's 100
    assert FigureNotApplicable not in categories


def test_fig5_data_holds_mean_and_sd_per_metric_response_and_window(paper_fig5, sim_dir):
    data = paper_fig5[0].data
    assert list(data.columns) == ["metric", "response", "s", "t", "mean", "sd", "n"]
    assert set(data.metric) == set(METRICS)
    assert set(data.t) == set(WINDOWS)
    assert set(data.response) <= {"reference", *NEGATIVE}
    assert set(data[data.response == "reference"].s) == {0}
    pressured = pd.concat([old_style_rows(adaptive_csv(sim_dir, 4, k)) for k in (0, 1, 2)])
    reference = pd.concat([old_style_rows(adaptive_csv(sim_dir, 0, k)) for k in (0, 1, 2)])
    for _, row in data.iterrows():
        rows = reference if row.response == "reference" else pressured[pressured.response == row.response]
        values = rows.loc[rows.t == row.t, row.metric]
        assert row.n == len(values)
        np.testing.assert_allclose([row["mean"], row.sd], [values.mean(), values.std()], equal_nan=True)


def test_fig5_at_another_pressure(sim_dir):
    result, _ = expected_warnings(fig5, sim_dir, s=1)
    assert set(result.data[result.data.response != "reference"].s) == {1}
    plt.close(result.figure)


def test_fig5_without_the_s0_reference_is_not_applicable(sim_dir):
    """The `link_prob 0.2` base has adaptive runs at s = 4 only."""
    with pytest.warns(FigureNotApplicable, match="s=0"):
        assert fig5(sim_dir, base=Baseline(link_prob=0.2)) is None


def test_fig5_without_the_pressure_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="s=8"):
        assert fig5(sim_dir, s=8) is None


def test_fig5_at_a_missing_mu_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable):
        assert fig5(sim_dir, mu=0.005) is None


# Fig. A4

def test_figA4_returns_a_figure_result_with_the_old_name(paper_figA4):
    result, categories = paper_figA4
    assert isinstance(result, FigureResult)
    assert result.name == "AppendixFig_metricCorrelations"
    assert FewSeedsWarning in categories


def test_figA4_title_names_the_pooled_windows(paper_figA4):
    title = paper_figA4[0].figure.axes[0].get_title()
    for window in ("91-100", "141-150", "191-200"):
        assert window in title
    assert "pooled" in title


def test_figA4_data_is_the_correlation_matrix_of_the_pooled_s0_rows(paper_figA4, sim_dir):
    data = paper_figA4[0].data
    assert list(data.columns) == ["metric", "other", "r"]
    pooled = pd.concat([old_style_rows(adaptive_csv(sim_dir, 0, k)) for k in (0, 1, 2)])
    expected = pooled[list(METRICS)].corr()
    got = data.pivot(index="metric", columns="other", values="r").loc[list(METRICS), list(METRICS)]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_figA4_without_the_s0_runs_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="s=0"):
        assert figA4(sim_dir, base=Baseline(link_prob=0.2)) is None


def test_figA4_for_mu_above_zero_needs_s4_at_that_mu(sim_dir):
    with pytest.warns(FigureNotApplicable, match="s=4"):
        assert figA4(sim_dir, mu=0.005) is None


def test_figA4_name_for_mu_above_zero_and_a_non_paper_base(sim_dir, monkeypatch):
    """The fixture has no μ > 0 runs: pretend its adaptive s = 4 runs are μ = 0.005 ones."""
    index = index_folder(sim_dir)
    index.loc[(index.eps == 1) & ~index.fixedBNat100.astype(bool), "mu"] = 0.005
    result, _ = expected_warnings(figA4, sim_dir, mu=0.005, index=index)
    assert result.name == "AppendixFig_metricCorrelations_mu0.005"
    plt.close(result.figure)
    result, _ = expected_warnings(figA4, sim_dir, base=Baseline(link_prob=0.2), mu=0.005, index=index)
    assert result.name == "AppendixFig_metricCorrelations_mu0.005_link_prob0.20"
    plt.close(result.figure)


def test_the_figures_write_nothing_and_print_nothing(sim_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before = sorted(sim_dir.rglob("*"))
    loader = Loader()
    for func in (fig5, figA4):
        result, _ = expected_warnings(func, sim_dir, loader=loader)
        plt.close(result.figure)
    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())
    assert sorted(sim_dir.rglob("*")) == before
