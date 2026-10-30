"""Tables 3, A1–A4 (metrics by response), A5 (their change over time) and A6/A7 (the logit fit statistics).

Reproduces the LaTeX that `analyse_BNmetrics.py` printed to stdout, from `metric_frame` and Fig. 4's `logit_fit`.
Each function returns a `LatexTable` (a `str` with a `name`) and prints nothing; `save_table` writes it. Table A5
puts the s = 0 reference columns after the three resistant ones, as in the paper (the old print cell interleaved
them with the resistant columns), and its proportion row keeps three cells per response when a type is absent.
"""

import warnings

import pandas as pd

from .bnmetrics import (BEFORE, CONTRAST2_PRESSURES, HALF_WINDOW, METRICS, NEGATIVE, PAPER_S, WINDOWS, fig4_before,
                        fig4_levels, logit_fit, mu_name, selected_metric_frame)
from .index import index_folder
from .output import LatexTable
from .select import PAPER, Baseline, FigureNotApplicable
from .style import METRIC_LABELS

_RULE = "% " + "#" * 50
_HEADER = " & ".join(["", *NEGATIVE]) + " & \\\\ \\hline"
_FIT_CAPTIONS = {
    "rR-C": r"Model fit statistics across pressure levels --- \textit{Resistant/Resilient-Compliant} classification "
            r"(Resistant+Resilient vs.\ Compliant)",
    "R-r": r"Model fit statistics across pressure levels --- \textit{Resistant-Resilient} classification "
           r"(Resistant vs.\ Resilient)",
}
_FIT_LABELS = {"rR-C": "tab:fit_rRC", "R-r": "tab:fit_Rr"}


def _cell(values: pd.Series) -> str:
    """Mean ± sd as the old script wrote it (one decimal when |mean| > 10, two otherwise), blank without values."""
    if not len(values):
        return " "
    mean, sd = values.mean(), values.std()
    digits = 1 if abs(mean) > 10 else 2
    return f"${mean:.{digits}f} \\pm {sd:.{digits}f}$"


def _shares(before: pd.DataFrame) -> list[str] | None:
    """Each negative response type's share among the negative responders, in %; `None` without any."""
    counts = before.response.value_counts().reindex(list(NEGATIVE), fill_value=0)
    total = counts.sum()
    return None if not total else [rf"${100 * c / total:.1f}\,\%$" for c in counts]


def table3(sim_dir, *, s, base: Baseline = PAPER, mu=0.0, seeds=None, min_seeds=1, loader=None,
           index=None) -> LatexTable | None:
    """Table 3 (s = 4) and Tables A1–A4 (s = 1, 2, 8, 16): each metric's mean ± sd per response type before pressure.

    One row per metric (`METRICS`) and a last row with the proportions of the compliant, resilient and resistant
    agents among the three, from the before window (t = 91–100) of the base's runs at `s`; `mu` replaces the base's
    μ. A type without agents gets blank cells (and 0.0 %). The name is `table3_s{s}`, plus `_mu{mu}` for μ > 0 and the
    base suffix.
    """
    if s == 0:
        raise ValueError("table3 needs a pressure s > 0: there are no response types without one")
    index = index_folder(sim_dir) if index is None else index
    frame = selected_metric_frame(index, base, mu, (s,), item="table3", seeds=seeds, min_seeds=min_seeds,
                                  loader=loader)
    if frame is None:
        return None
    before = frame[frame.t == BEFORE]
    lines = [_RULE, f"% time = {BEFORE - HALF_WINDOW} - {BEFORE + HALF_WINDOW}", _RULE, f"% s_ext = {s:g}", _RULE,
             " " + _HEADER]
    for metric in METRICS:
        cells = [_cell(before.loc[before.response == r, metric]) for r in NEGATIVE]
        lines.append(f"{METRIC_LABELS[metric]} &  " + "".join(f"{c} & " for c in cells) + "\\\\")
    shares = _shares(before) or [" "] * len(NEGATIVE)
    lines.append("proportion &  " + "".join(f"{c} & " for c in shares) + "\\\\")
    return LatexTable("\n".join(lines), mu_name(f"table3_s{s:g}", base, mu))


def tableA5(sim_dir, *, base: Baseline = PAPER, mu=0.0, seeds=None, min_seeds=1, loader=None,
            index=None) -> LatexTable | None:
    """Table A5: each metric's mean ± sd per response type before, during and after pressure at s = 4.

    Per metric row: compliant, resilient and resistant at each of the three windows, then every agent of the s = 0
    runs at each window (the paper's reference columns), so both levels are needed; `mu` replaces the base's μ. The
    last row holds the proportions of the three types (as in Table 3). The column header stays a comment, as in the
    old output. The name is `tableA5_metricChange`, plus `_mu{mu}` for μ > 0 and the base suffix.
    """
    index = index_folder(sim_dir) if index is None else index
    frame = selected_metric_frame(index, base, mu, (PAPER_S, 0), item="tableA5", seeds=seeds, min_seeds=min_seeds,
                                  loader=loader)
    if frame is None:
        return None
    pressured, reference = frame[frame.s == PAPER_S], frame[frame.s == 0]
    lines = ["% Change over time", "% " + _HEADER]
    for metric in METRICS:
        cells = [_cell(pressured.loc[(pressured.response == r) & (pressured.t == t), metric])
                 for r in NEGATIVE for t in WINDOWS]
        cells += [_cell(reference.loc[reference.t == t, metric]) for t in WINDOWS]
        lines.append(f"  {METRIC_LABELS[metric]} &  " + "".join(f"{c} & " for c in cells) + "\\\\")
    shares = _shares(pressured[pressured.t == BEFORE]) or [" "] * len(NEGATIVE)
    lines.append("proportion &   " + "".join(f"{c} & & &  " for c in shares) + "\\\\")
    return LatexTable("\n".join(lines), mu_name("tableA5_metricChange", base, mu))


def tableA6(sim_dir, *, base: Baseline = PAPER, mu=0.0, pressures=None, seeds=None, min_seeds=1, loader=None,
            index=None) -> LatexTable | None:
    """Table A6: the fit statistics of Fig. 4's logit of resistant/resilient vs compliant (`logit_fit`, `"rR-C"`).

    One row per pressure: N, the proportions of group 1 / group 0, McFadden and Tjur R², and the LR test against
    the control model (χ², p; `<0.001` below 0.001). `pressures` and `mu` as in `fig4`: the paper's levels that
    have runs, the missing ones warned about. A fit that fails warns (`FitWarning`) and its row is left out. The
    name is `tableA6_logit_rRC`, plus `_mu{mu}` for μ > 0 and the base suffix.
    """
    index = index_folder(sim_dir) if index is None else index
    selected = fig4_before(index, base, mu, fig4_levels(pressures, mu), item="tableA6", seeds=seeds,
                           min_seeds=min_seeds, loader=loader)
    return None if selected is None else _fit_table("rR-C", *selected, mu_name("tableA6_logit_rRC", base, mu))


def tableA7(sim_dir, *, base: Baseline = PAPER, mu=0.0, pressures=None, contrast2_pressures=CONTRAST2_PRESSURES,
            seeds=None, min_seeds=1, loader=None, index=None) -> LatexTable | None:
    """Table A7: as `tableA6`, for the logit of resistant vs resilient (`"R-r"`).

    Uses only those of the pressures that are in `contrast2_pressures`, as Fig. 4's second panel does. The name is
    `tableA7_logit_Rr`, plus `_mu{mu}` for μ > 0 and the base suffix.
    """
    levels = tuple(s for s in fig4_levels(pressures, mu) if s in contrast2_pressures)
    if not levels:
        warnings.warn("tableA7 not applicable: none of the pressures is in contrast2_pressures", FigureNotApplicable,
                      stacklevel=2)
        return None
    index = index_folder(sim_dir) if index is None else index
    selected = fig4_before(index, base, mu, levels, item="tableA7", seeds=seeds, min_seeds=min_seeds, loader=loader)
    return None if selected is None else _fit_table("R-r", *selected, mu_name("tableA7_logit_Rr", base, mu))


def _fit_table(contrast, before: pd.DataFrame, present, name) -> LatexTable:
    """The fit table of `contrast` on `before` (from `fig4_before`), a row per level of `present` whose fit works."""
    lines = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{_FIT_CAPTIONS[contrast]}}}",
             f"\\label{{{_FIT_LABELS[contrast]}}}", "\\begin{tabular}{ccccccc}", "\\toprule",
             r"$s$ & $N$ & Proportions & McFadden $R^2$ & Tjur $R^2$ & LR $\chi^2$ & LR $p$ \\",
             r"     &     & (group 1 / group 0) &          &            &             &        \\",
             "\\midrule"]
    for s in present:
        fit = logit_fit(before[before.s == s], contrast, label=f"s={s:g}")
        if fit is None:
            continue
        p = f"{fit.lr_p:.2e}" if fit.lr_p > 0.001 else "<0.001"
        lines.append(f"  {s:g} & {fit.n} & {fit.n_group1 / fit.n:.2f} / {fit.n_group0 / fit.n:.2f} & "
                     f"{fit.pseudo_r2:.3f} & {fit.tjur_r2:.3f} & {fit.lr_stat:.2f} & {p} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return LatexTable("\n".join(lines), name)
