"""`anb.analysis` Fig. 4, Fig. 5, Fig. A4 and the agent-metric frame, on the synthetic folder of `sim_folder.py`.

The fixture's paper base has the adaptive config at s ∈ {0, 1, 4} × seeds {0, 1, 2} for μ = 0, and μ ∈ {0.005, 0.2}
at s = 4 for seed 1 only. The `link_prob 0.2` base has adaptive at s = 4 only, so it lacks the s = 0 reference.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim_folder  # noqa: E402
from analysis_helpers import expected_warnings  # noqa: E402
from anb.analysis import (  # noqa: E402
    PAPER, Baseline, FewSeedsWarning, FigureNotApplicable, FigureResult, FitWarning, Loader, MissingLevelsWarning,
    cohens_d, fig4, fig5, figA4, index_folder, logit_fit, metric_frame, runs,
)
from anb.analysis.bnmetrics import CONTROL, METRICS, NEGATIVE, PREDICTORS, WINDOWS  # noqa: E402
from anb.io import legacy_filename  # noqa: E402

RESPONSE_COLUMNS = ["persistPos", "nonpersistPos", "compliant", "resilient", "resistant", "latecompliant"]


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
        assert figA4(sim_dir, mu=0.05) is None


def test_figA4_name_for_mu_above_zero_and_a_non_paper_base(sim_dir, monkeypatch):
    """Pretend every adaptive run is a μ = 0.005 one (the fixture has no μ > 0 for `link_prob 0.2`)."""
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
    for func in (fig4, fig5, figA4):
        result, _ = expected_warnings(func, sim_dir, loader=loader)
        plt.close(result.figure)
    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())
    assert sorted(sim_dir.rglob("*")) == before


# Fig. 4: Cohen's d and the logit model

def before_rows(sim_dir, s) -> pd.DataFrame:
    rows = pd.concat([old_style_rows(adaptive_csv(sim_dir, s, k)) for k in (0, 1, 2)])
    return rows[rows.t == 95.5]


def old_cohens_d(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """`analyse_BNmetrics.py`'s `cohens_d`."""
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    pooled[pooled < 1e-10] = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (a.mean() - b.mean()) / pooled**0.5
    return d.where(pooled > 0, other=np.nan)


def synthetic_before(n=400, seed=0) -> pd.DataFrame:
    """A before-window frame with every metric random and the response depending on `tb_foc` and `x_focal`."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({m: rng.normal(size=n) for m in METRICS})
    p = 1 / (1 + np.exp(-(1.5 * frame.tb_foc - frame.x_focal)))
    noncompliant = rng.random(n) < p
    frame["response"] = np.where(noncompliant, np.where(rng.random(n) < 0.5, "resistant", "resilient"), "compliant")
    frame.loc[frame.index[:10], "response"] = "latecompliant"
    frame.loc[frame.index[10:20], "response"] = None
    return frame


@pytest.fixture(scope="module")
def paper_fig4(sim_dir):
    result, categories = expected_warnings(fig4, sim_dir)
    yield result, categories
    plt.close(result.figure)


def test_cohens_d_matches_the_old_formula():
    frame = synthetic_before()
    noncompliant = frame[frame.response.isin(["resistant", "resilient"])][list(METRICS)]
    compliant = frame[frame.response == "compliant"][list(METRICS)]  # latecompliant left out, as in the old script
    pd.testing.assert_series_equal(cohens_d(frame, "rR-C"), old_cohens_d(noncompliant, compliant), check_names=False)
    resistant = frame[frame.response == "resistant"][list(METRICS)]
    resilient = frame[frame.response == "resilient"][list(METRICS)]
    pd.testing.assert_series_equal(cohens_d(frame, "R-r"), old_cohens_d(resistant, resilient), check_names=False)


def test_cohens_d_of_a_constant_metric_is_nan():
    frame = synthetic_before().assign(clust=1.0)
    assert np.isnan(cohens_d(frame, "rR-C")["clust"])


def test_logit_fit_matches_statsmodels_on_the_standardised_subset():
    import statsmodels.api as sm

    frame = synthetic_before()
    fit = logit_fit(frame, "rR-C")
    subset = frame[frame.response.isin(["resistant", "resilient", "compliant", "latecompliant"])]
    y = subset.response.isin(["resistant", "resilient"]).astype(float).to_numpy()
    X = subset[[*PREDICTORS, *CONTROL]]
    X = (X - X.mean()) / X.std(ddof=0)
    model = sm.Logit(y, sm.add_constant(X.reset_index(drop=True))).fit(disp=0)
    np.testing.assert_allclose(fit.coefficients.coefficient, model.params[1:], rtol=1e-8)
    ci = model.conf_int(alpha=0.05)
    np.testing.assert_allclose(fit.coefficients.ci_low, ci[0][1:], rtol=1e-8)
    np.testing.assert_allclose(fit.coefficients.ci_high, ci[1][1:], rtol=1e-8)
    assert fit.coefficients.loc["tb_foc", "coefficient"] > 0 > fit.coefficients.loc["x_focal", "coefficient"]
    assert fit.n == len(subset)
    assert (fit.n_group1, fit.n_group0) == (int(y.sum()), int(len(y) - y.sum()))
    assert fit.pseudo_r2 == pytest.approx(model.prsquared)
    null = sm.Logit(y, sm.add_constant(X[list(CONTROL)].reset_index(drop=True))).fit(disp=0)
    assert fit.lr_stat == pytest.approx(2 * (model.llf - null.llf))


def test_logit_fit_with_an_empty_group_warns_and_returns_none():
    frame = synthetic_before()
    frame.loc[frame.response == "resistant", "response"] = "resilient"
    with pytest.warns(FitWarning, match="R-r"):
        assert logit_fit(frame, "R-r") is None


def test_logit_fit_of_a_perfectly_separated_subset_warns_and_returns_none():
    frame = synthetic_before()
    frame["response"] = np.where(frame.tb_foc > 0, "resistant", "compliant")  # perfectly separated
    with pytest.warns(FitWarning):
        assert logit_fit(frame, "rR-C") is None


def test_fig4_renders_with_the_available_pressures(paper_fig4):
    result, categories = paper_fig4
    assert isinstance(result, FigureResult)
    assert result.name == "fig4_mu0.0"
    assert len(result.figure.axes) == 4  # two panels, each with its regression twin
    assert MissingLevelsWarning in categories  # the fixture has s = 1 and 4, not 2, 8, 16
    assert FewSeedsWarning in categories
    assert FigureNotApplicable not in categories
    data = result.data
    assert list(data.columns) == ["contrast", "s", "metric", "d", "coefficient", "ci_low", "ci_high"]
    assert set(data.contrast) == {"rR-C", "R-r"}
    assert set(data.s) == {1, 4}
    assert len(data) == 2 * 2 * len(METRICS)


def test_fig4_data_holds_cohens_d_and_the_logit_coefficients_of_each_subset(paper_fig4, sim_dir):
    data = paper_fig4[0].data.set_index(["contrast", "s", "metric"])
    for s in (1, 4):
        before = before_rows(sim_dir, s)
        for contrast in ("rR-C", "R-r"):
            np.testing.assert_allclose(data.loc[(contrast, s), "d"].loc[list(METRICS)],
                                       cohens_d(before, contrast).loc[list(METRICS)], equal_nan=True)
            got = data.loc[(contrast, s)]
            assert got.loc[list(set(METRICS) - set(PREDICTORS)), "coefficient"].isna().all()  # control: not drawn
            fit = logit_fit(before, contrast)
            if fit is not None:
                np.testing.assert_allclose(got.loc[list(PREDICTORS), "coefficient"],
                                           fit.coefficients.loc[list(PREDICTORS), "coefficient"])


def test_fig4_restricts_the_second_contrast_to_its_pressures(sim_dir):
    result, _ = expected_warnings(fig4, sim_dir, contrast2_pressures=(4,))
    assert set(result.data[result.data.contrast == "R-r"].s) == {4}
    assert set(result.data[result.data.contrast == "rR-C"].s) == {1, 4}
    plt.close(result.figure)


def test_fig4_leaves_out_the_dots_of_a_failed_fit(sim_dir, monkeypatch):
    import anb.analysis.bnmetrics as bnmetrics

    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(bnmetrics.sm, "Logit", failing)
    result, categories = expected_warnings(fig4, sim_dir)
    assert FitWarning in categories
    assert result.data.coefficient.isna().all()
    assert result.data.d.notna().any()
    plt.close(result.figure)


def test_fig4_at_a_missing_mu_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="s=4"):
        assert fig4(sim_dir, mu=0.05) is None


def test_fig4_without_any_listed_pressure_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable):
        assert fig4(sim_dir, pressures=(2, 8)) is None


def test_fig4_for_a_non_paper_base_and_mu_above_zero(sim_dir):
    result, categories = expected_warnings(fig4, sim_dir, base=Baseline(link_prob=0.2))
    assert result.name == "fig4_mu0.0_link_prob0.20"
    assert set(result.data.s) == {4}
    plt.close(result.figure)
    index = index_folder(sim_dir)
    index.loc[(index.eps == 1) & ~index.fixedBNat100.astype(bool), "mu"] = 0.005
    result, categories = expected_warnings(fig4, sim_dir, mu=0.005, index=index)
    assert result.name == "fig4_mu0.005"
    assert set(result.data.s) == {4}  # the paper's rule for μ > 0: s = 4 only, so s = 1 is not used
    assert MissingLevelsWarning not in categories
    plt.close(result.figure)
