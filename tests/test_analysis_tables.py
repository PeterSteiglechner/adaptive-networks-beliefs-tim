"""`anb.analysis` Tables 3, A1–A7, `save_table` and the greedy metric subset, on the folder of `sim_folder.py`.

The fixture's paper base has the adaptive config at s ∈ {0, 1, 4} × seeds {0, 1, 2} for μ = 0, and μ ∈ {0.005, 0.2}
at s = 4 for seed 1 only. The `link_prob 0.2` base has adaptive at s = 4 only, so it lacks the s = 0 reference. The
expected LaTeX is built here the way the old script printed it, from the CSVs read directly.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from anb.analysis import (
    Baseline, FewSeedsWarning, FigureNotApplicable, FitWarning, LatexTable, Loader, MissingLevelsWarning,
    greedy_uncorrelated_subset, index_folder, logit_fit, save_table, table3, tableA5, tableA6, tableA7,
)
from anb.analysis.bnmetrics import METRICS, NEGATIVE, PREDICTORS, WINDOWS
from anb.analysis.style import METRIC_LABELS
from anb.io import legacy_filename
from analysis_helpers import expected_warnings, squash

RESPONSE_COLUMNS = ["persistPos", "nonpersistPos", "compliant", "resilient", "resistant", "latecompliant"]
GREEDY_CANDIDATES = ["x_focal", "n_nbs", "tb_tot", "tb_foc", "absOm_tot", "absOm_foc", "clust", "bc_foc",
                     "extr_nonfoc"]


def res(sim_dir, levels=(0, 1, 4)) -> pd.DataFrame:
    """The old script's `res` for the fixture's adaptive runs: every window row, labelled from the 95.5 row."""
    blocks = []
    for s in levels:
        for seed in (0, 1, 2):
            stem = legacy_filename(dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=1 / 3, eps=1.0, mu=0.0,
                                        fixedBNat100=False, ext_strength=s, seed=seed, lam=0.0, n_events=1), False)
            df = pd.read_csv(sim_dir / f"{stem}.csv")
            labels = df.loc[df.t == 95.5].set_index("id")[RESPONSE_COLUMNS]
            response = pd.Series(None, index=labels.index, dtype=object)
            for col in RESPONSE_COLUMNS:
                response[labels[col] == 1] = col
            rows = df[df.t.isin(WINDOWS)].copy()
            rows["response"] = rows.id.map(response)
            rows["Hpersnonfoc"] = rows.Hpers - rows.Hpersfoc
            rows["s"], rows["seed"] = s, seed
            blocks.append(rows)
    return pd.concat(blocks, ignore_index=True)


def old_cell(a: pd.Series) -> str:
    if len(a) == 0:
        return "  & "
    if abs(a.mean()) > 10:
        return f"${a.mean():.1f} \\pm {a.std():.1f}$ & "
    return f"${a.mean():.2f} \\pm {a.std():.2f}$ & "


def old_shares(res, s) -> list[str]:
    counts = res.loc[(res.s == s) & (res.t == 95.5), "response"].value_counts()
    counts = counts.reindex(list(NEGATIVE), fill_value=0)
    return [rf"${100 * c / counts.sum():.1f}\,\%$" for c in counts]


def old_table3(res, s) -> str:
    """The old script's Table 3 print cell for one pressure."""
    rule = "% " + "#" * 50
    header = "  & compliant & resilient & resistant & \\\\ \\hline"
    out = [rule, "% time = 91.0 - 100.0", rule, f"% s_ext = {s}", rule, header]
    for metric in METRICS:
        out.append(f"{METRIC_LABELS[metric]} &  " + "".join(
            old_cell(res.loc[(res.response == r) & (res.s == s) & (res.t == 95.5), metric]) for r in NEGATIVE)
            + "\\\\")
    out.append("proportion &  " + "".join(share + " & " for share in old_shares(res, s)) + "\\\\")
    return "\n".join(out)


def expected_tableA5(res) -> str:
    """Table A5: the old print cell, but with the s = 0 reference columns after the resistant ones (the paper's
    layout; the old cell interleaved them with the resistant columns)."""
    out = ["% Change over time", "%  & compliant & resilient & resistant & \\\\ \\hline"]
    for metric in METRICS:
        cells = [old_cell(res.loc[(res.response == r) & (res.s == 4) & (res.t == t), metric])
                 for r in NEGATIVE for t in WINDOWS]
        cells += [old_cell(res.loc[(res.s == 0) & (res.t == t), metric]) for t in WINDOWS]
        out.append(f"  {METRIC_LABELS[metric]} &  " + "".join(cells) + "\\\\")
    out.append("proportion &   " + "".join(share + " & & &  " for share in old_shares(res, 4)) + "\\\\")
    return "\n".join(out)


@pytest.fixture(scope="module")
def old_res(sim_dir):
    return res(sim_dir)


# Table 3 / A1–A4

def test_table3_returns_the_latex_of_the_old_print_cell(sim_dir, old_res):
    for s in (1, 4):
        table, categories = expected_warnings(table3, sim_dir, s=s)
        assert isinstance(table, str) and isinstance(table, LatexTable)
        assert table.name == f"table3_s{s}"
        assert FewSeedsWarning in categories  # 3 seeds, not the paper's 100
        assert squash(table) == squash(old_table3(old_res, s))


def test_table3_has_a_row_per_metric_and_the_proportions(sim_dir):
    table, _ = expected_warnings(table3, sim_dir, s=4)
    rows = [line for line in table.splitlines() if not line.startswith("%")]
    assert [row.split(" &")[0] for row in rows[1:]] == [METRIC_LABELS[m] for m in METRICS] + ["proportion"]
    assert all(row.count("&") == len(NEGATIVE) + 1 for row in rows)


def test_table3_without_the_pressure_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="s=8"):
        assert table3(sim_dir, s=8) is None


def test_table3_at_a_missing_mu_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable):
        assert table3(sim_dir, s=4, mu=0.05) is None


def test_table3_at_s0_is_an_error(sim_dir):
    with pytest.raises(ValueError):
        table3(sim_dir, s=0)


def test_table3_names_for_mu_above_zero_and_a_non_paper_base(sim_dir):
    table, _ = expected_warnings(table3, sim_dir, s=4, base=Baseline(link_prob=0.2))
    assert table.name == "table3_s4_link_prob0.20"
    index = index_folder(sim_dir)
    index.loc[(index.eps == 1) & ~index.fixedBNat100.astype(bool), "mu"] = 0.005  # pretend these are μ > 0 runs
    table, _ = expected_warnings(table3, sim_dir, s=4, mu=0.005, index=index)
    assert table.name == "table3_s4_mu0.005"


# Table A5

def test_tableA5_holds_before_during_after_and_the_s0_reference_last(sim_dir, old_res):
    table, categories = expected_warnings(tableA5, sim_dir)
    assert table.name == "tableA5_metricChange"
    assert FewSeedsWarning in categories
    assert squash(table) == squash(expected_tableA5(old_res))
    rows = [line for line in table.splitlines() if not line.startswith("%")]
    assert all(row.count("&") == 4 * len(WINDOWS) + 1 for row in rows[:-1])  # the proportions: 3 per type


def test_tableA5_without_the_s0_reference_is_not_applicable(sim_dir):
    """The `link_prob 0.2` base has adaptive runs at s = 4 only."""
    with pytest.warns(FigureNotApplicable, match="s=0"):
        assert tableA5(sim_dir, base=Baseline(link_prob=0.2)) is None


def test_tableA5_at_a_missing_mu_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable):
        assert tableA5(sim_dir, mu=0.005) is None


# Tables A6 / A7

def fit_rows(table: str) -> dict:
    """The body rows of a fit table by s: N, proportions, McFadden, Tjur, LR χ², LR p (as strings)."""
    lines = table.splitlines()
    body = lines[lines.index("\\midrule") + 1:lines.index("\\bottomrule")]
    rows = [[cell.strip() for cell in line.rstrip(" \\").split("&")] for line in body]
    return {int(row[0]): row[1:] for row in rows}


def expected_fit_row(before, contrast) -> list[str]:
    fit = logit_fit(before, contrast)
    p = f"{fit.lr_p:.2e}" if fit.lr_p > 0.001 else "<0.001"
    return [str(fit.n), f"{fit.n_group1 / fit.n:.2f} / {fit.n_group0 / fit.n:.2f}", f"{fit.pseudo_r2:.3f}",
            f"{fit.tjur_r2:.3f}", f"{fit.lr_stat:.2f}", p]


@pytest.mark.parametrize("func, contrast, name, label", [
    (tableA6, "rR-C", "tableA6_logit_rRC", "tab:fit_rRC"),
    (tableA7, "R-r", "tableA7_logit_Rr", "tab:fit_Rr"),
])
def test_the_fit_tables_hold_one_row_per_available_pressure(sim_dir, old_res, func, contrast, name, label):
    table, categories = expected_warnings(func, sim_dir)
    assert table.name == name
    assert table.startswith("\\begin{table}[ht]") and table.endswith("\\end{table}")
    assert f"\\label{{{label}}}" in table
    assert MissingLevelsWarning in categories  # the fixture has s = 1 and 4 only
    assert FewSeedsWarning in categories
    rows = fit_rows(table)
    assert list(rows) == [1, 4]
    for s, row in rows.items():
        before = old_res[(old_res.s == s) & (old_res.t == 95.5)]
        assert row == expected_fit_row(before, contrast)


def test_tableA7_uses_only_the_second_contrasts_pressures(sim_dir):
    table, _ = expected_warnings(tableA7, sim_dir, contrast2_pressures=(4,))
    assert list(fit_rows(table)) == [4]


def test_tableA6_uses_only_the_given_pressures(sim_dir):
    table, _ = expected_warnings(tableA6, sim_dir, pressures=(4,))
    assert list(fit_rows(table)) == [4]


@pytest.mark.parametrize("func", [tableA6, tableA7])
def test_the_fit_tables_warn_at_the_callers_line(sim_dir, func):
    """The level warnings point at the call of `tableA6`/`tableA7`, not into the package."""
    for kwargs, category in (({}, MissingLevelsWarning), (dict(mu=0.05), FigureNotApplicable)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            func(sim_dir, **kwargs)
        assert [w.filename for w in caught if w.category is category] == [__file__]


def test_tableA7_without_any_pressure_of_the_second_contrast_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable):
        assert tableA7(sim_dir, pressures=(8, 16)) is None


def test_the_fit_tables_at_a_missing_mu_are_not_applicable(sim_dir):
    for func in (tableA6, tableA7):
        with pytest.warns(FigureNotApplicable):
            assert func(sim_dir, mu=0.05) is None


def test_a_failed_fit_leaves_out_its_row(sim_dir, monkeypatch):
    import anb.analysis.bnmetrics as bnmetrics

    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(bnmetrics.sm, "Logit", failing)
    table, categories = expected_warnings(tableA6, sim_dir)
    assert FitWarning in categories
    assert fit_rows(table) == {}


def test_the_fit_table_names_for_mu_above_zero_and_a_non_paper_base(sim_dir):
    table, _ = expected_warnings(tableA6, sim_dir, base=Baseline(link_prob=0.2))
    assert table.name == "tableA6_logit_rRC_link_prob0.20"
    assert list(fit_rows(table)) == [4]
    index = index_folder(sim_dir)
    index.loc[(index.eps == 1) & ~index.fixedBNat100.astype(bool), "mu"] = 0.005
    table, categories = expected_warnings(tableA7, sim_dir, mu=0.005, index=index)
    assert table.name == "tableA7_logit_Rr_mu0.005"
    assert list(fit_rows(table)) == [4]  # the paper's rule for μ > 0: s = 4 only
    assert MissingLevelsWarning not in categories


# save_table, and that nothing is printed or written before it

def test_save_table_writes_tables_name_tex_and_creates_the_folder(sim_dir, tmp_path):
    table, _ = expected_warnings(table3, sim_dir, s=4)
    out_dir = tmp_path / "new" / "figs"
    path = save_table(table, out_dir)
    assert path == out_dir / "tables" / "table3_s4.tex"
    assert path.read_text() == table + "\n"
    assert save_table(table, out_dir, "other") == out_dir / "tables" / "other.tex"
    assert save_table("x \\\\", out_dir, "plain").read_text() == "x \\\\\n"
    assert sorted(p.name for p in (out_dir / "tables").iterdir()) == ["other.tex", "plain.tex", "table3_s4.tex"]


def test_save_table_needs_a_name_for_a_plain_string(tmp_path):
    with pytest.raises(ValueError):
        save_table("x", tmp_path)


def test_the_tables_print_nothing_and_write_nothing(sim_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before = sorted(sim_dir.rglob("*"))
    loader = Loader()
    for func, kwargs in ((table3, dict(s=4)), (tableA5, {}), (tableA6, {}), (tableA7, {}),
                         (greedy_uncorrelated_subset, {})):
        result, _ = expected_warnings(func, sim_dir, loader=loader, **kwargs)
        assert result is not None
    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())
    assert sorted(sim_dir.rglob("*")) == before


# The greedy subset of weakly correlated metrics

def old_greedy(corr_matrix: pd.DataFrame, n: int) -> list[str]:
    """`analyse_BNmetrics.py`'s `greedy_uncorrelated_subset`."""
    abs_corr = corr_matrix.abs()
    selected = [abs_corr.mean().idxmin()]
    while len(selected) < n:
        remaining = [c for c in corr_matrix.columns if c not in selected]
        max_corr = {c: abs_corr.loc[c, selected].max() for c in remaining}
        selected.append(min(max_corr, key=max_corr.get))
    return selected


def test_greedy_subset_is_the_old_order_on_the_figA4_correlations(sim_dir, old_res):
    order, categories = expected_warnings(greedy_uncorrelated_subset, sim_dir)
    assert FewSeedsWarning in categories
    assert list(order.columns) == ["rank", "metric", "mean_abs_r", "max_abs_r", "predictor"]
    corr = old_res[old_res.s == 0][list(METRICS)].corr().loc[GREEDY_CANDIDATES, GREEDY_CANDIDATES]
    assert order.metric.tolist() == old_greedy(corr, len(GREEDY_CANDIDATES))
    assert order["rank"].tolist() == list(range(1, len(GREEDY_CANDIDATES) + 1))
    assert order.predictor.tolist() == [m in PREDICTORS for m in order.metric]
    first = order.metric[0]
    assert order.mean_abs_r[0] == pytest.approx(corr.abs().mean()[first])
    assert np.isnan(order.max_abs_r[0])
    for k in range(1, len(order)):
        chosen = order.metric[:k].tolist()
        assert order.max_abs_r[k] == pytest.approx(corr.abs().loc[order.metric[k], chosen].max())


def test_greedy_subset_of_n_given_metrics(sim_dir, old_res):
    metrics = ["tb_tot", "tb_foc", "clust", "n_nbs"]
    order, _ = expected_warnings(greedy_uncorrelated_subset, sim_dir, metrics=metrics, n=2)
    corr = old_res[old_res.s == 0][list(METRICS)].corr().loc[metrics, metrics]
    assert order.metric.tolist() == old_greedy(corr, 2)


def test_greedy_subset_without_the_s0_runs_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="s=0"):
        assert greedy_uncorrelated_subset(sim_dir, base=Baseline(link_prob=0.2)) is None
