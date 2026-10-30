"""Fig. 4, Fig. 5, Fig. A4 and the agent-metric frame they share: each agent's BN metrics before, during and after
pressure.

Reproduces `analyse_BNmetrics.py`. Its `res` (one row per agent × seed × s × window, labelled with the agent's
response) is `metric_frame`; the runs come from the index instead of the script's fixed seed and pressure lists. The
Fig. A4 title states the three windows it pools (the old title named only the first). Fig. 4's Cohen's d and logit
fits are the plain functions `cohens_d` and `logit_fit`, which the tables (`tables.py`) reuse, as they do the
run selection (`selected_metric_frame`, `fig4_levels`, `fig4_before`) and `mu_name`; `anb.analysis` doesn't export
those. The old script's `greedy_uncorrelated_subset` works on Fig. A4's correlations.
"""

import string
import warnings
from dataclasses import dataclass, replace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import patches
from scipy import stats
from statsmodels.tools.sm_exceptions import (ConvergenceWarning, HessianInversionWarning, PerfectSeparationError,
                                             PerfectSeparationWarning)

from .index import index_folder
from .load import Loader
from .output import FigureResult
from .select import PAPER, Baseline, Cell, FigureNotApplicable, FitWarning, MissingLevelsWarning, runs, select
from .style import METRIC_LABELS, PRESSURE_COLORS, RESPONSE_COLORS, SCRIPT_RC, figure_style

WINDOWS = (95.5, 145.5, 195.5)  # the averaged before / during / after rows
BEFORE = WINDOWS[0]  # the row that holds the response classification
HALF_WINDOW = 4.5
RESPONSE_FLAGS = ("persistPos", "nonpersistPos", "compliant", "resilient", "resistant", "latecompliant")
PAPER_S = 4  # the pressure of Fig. 5, and of Fig. A4 for μ > 0
NEGATIVE = ("compliant", "resilient", "resistant")
METRICS = tuple(METRIC_LABELS)  # the old `metrics_table_sort`
REFERENCE = "reference"  # Fig. 5's s = 0 curve, every agent pooled (the old script's "RRC")
PAPER_SEEDS = 100

# Fig. 4's two contrasts: the response types of group 1 and group 0. Cohen's d compares exactly these; the logit also
# counts `latecompliant` as compliant (`_LOGIT_GROUP0`), as the old script does.
CONTRASTS = {"rR-C": (("resistant", "resilient"), ("compliant",)), "R-r": (("resistant",), ("resilient",))}
_LOGIT_GROUP0 = {"rR-C": ("compliant", "latecompliant"), "R-r": ("resilient",)}
CONTRAST_TITLES = {"rR-C": "Resistant/Resilient vs. Compliant", "R-r": "Resistant vs. Resilient"}
PREDICTORS = ("n_nbs", "tb_foc", "absOm_foc", "clust", "bc_foc")  # the old `dependent_vars`
CONTROL = ("x_focal",)
FIG4_PRESSURES = (1, 2, 4, 8, 16)  # the paper's levels for μ = 0; for μ > 0 only `PAPER_S`
CONTRAST2_PRESSURES = (1, 2, 4)  # the second contrast has too few resistant agents above s = 4
# The metrics the greedy search of `greedy_uncorrelated_subset` chooses from (the old `metrics_considered`).
GREEDY_CANDIDATES = ("x_focal", "n_nbs", "tb_tot", "tb_foc", "absOm_tot", "absOm_foc", "clust", "bc_foc", "extr_nonfoc")

# The CSV metric columns `res` keeps; `Hpers` is replaced by `Hpersnonfoc` = `Hpers` − `Hpersfoc`.
_RAW = ("n_nbs", "Hpers", "Hpersfoc", "Hsoc", "Hext", "tb_tot", "tb_foc", "absOm_tot", "absOm_foc", "clust", "bc_foc",
        "expI", "x_focal", "extr_nonfoc")
_FS, _SMALL_FS = SCRIPT_RC["font.size"], SCRIPT_RC["legend.fontsize"]
_COLORS = {**RESPONSE_COLORS, REFERENCE: "darkgrey"}


def metric_frame(runs: pd.DataFrame, loader: Loader | None = None) -> pd.DataFrame:
    """The BN metrics of every agent in `runs` (rows of the index) at each of the three `WINDOWS`.

    One row per run × window × agent: `s`, `seed`, `t` (the window), `id`, `response` (the agent's type from the
    95.5 row on every window, `None` without one; with several flags the last of `RESPONSE_FLAGS` wins, as in the old
    script), the CSV metrics and `Hpersnonfoc` (`Hpers` itself is dropped).
    """
    loader = loader or Loader()
    blocks = []
    for s, seed, path in zip(runs.s, runs.seed, runs.path):
        frame = loader.load(path, t=WINDOWS, columns=(*_RAW, *RESPONSE_FLAGS)).frame
        labels = frame[frame.t == BEFORE].set_index("id")[list(RESPONSE_FLAGS)]
        response = pd.Series(None, index=labels.index, dtype=object)
        for name in RESPONSE_FLAGS:
            response[labels[name] == 1] = name
        block = frame[["t", "id", *_RAW]].copy()
        block.insert(0, "seed", int(seed))
        block.insert(0, "s", s)
        block.insert(4, "response", block.id.map(response))
        block["Hpersnonfoc"] = block.Hpers - block.Hpersfoc
        blocks.append(block.drop(columns="Hpers"))
    return pd.concat(blocks, ignore_index=True)


def selected_metric_frame(index, base: Baseline, mu, levels, *, item, seeds, min_seeds,
                          loader) -> pd.DataFrame | None:
    """`metric_frame` of the base at `mu` over each of `levels`, or `None` (after a `FigureNotApplicable`)."""
    config = replace(base, mu=mu)
    cells = [Cell(f"μ={mu:g} s={s}", config, s, seeds=seeds) for s in levels]
    selected = select(index, cells, item=item, paper_seeds=PAPER_SEEDS, min_seeds=min_seeds)
    if selected is None:
        return None
    return metric_frame(pd.concat(selected.values(), ignore_index=True), loader)


def fig5(sim_dir, *, base: Baseline = PAPER, mu=0.0, s=PAPER_S, seeds=None, min_seeds=1, loader=None,
         index=None) -> FigureResult | None:
    """Fig. 5: each BN metric before, during and after pressure, per response type at `s`, against s = 0.

    One panel per metric (`METRICS`): mean ± sd over agents and runs of the compliant, resilient and resistant
    agents at `s`, and of every agent at s = 0 (`REFERENCE`). Both levels are needed. `mu` replaces the base's μ.
    The name is the old one, `fig5_metricChange_mu{mu}`, plus `_s{s}` for an `s` other than the paper's 4.
    The response labels in panel A sit where the paper's s = 4 curves are.

    `data`: metric, response, s, t (the window), mean, sd, n.
    """
    if s == 0:
        raise ValueError("fig5 compares a pressure s > 0 with the s = 0 reference")
    index = index_folder(sim_dir) if index is None else index
    frame = selected_metric_frame(index, base, mu, (s, 0), item="fig5", seeds=seeds, min_seeds=min_seeds,
                                  loader=loader)
    if frame is None:
        return None
    plotted = _fig5_rows(frame, s)
    with figure_style(SCRIPT_RC):
        fig = _draw_fig5(plotted)
    name = f"fig5_metricChange_mu{float(mu)}" + ("" if s == PAPER_S else f"_s{s}") + base.suffix(ignore=("mu",))
    return FigureResult(fig, _fig5_data(plotted), name)


def _fig5_rows(frame: pd.DataFrame, s) -> pd.DataFrame:
    """The plotted rows: the negative responders at `s`, and every agent at s = 0 as `REFERENCE`."""
    pressured = frame[(frame.s == s) & frame.response.isin(NEGATIVE)]
    reference = frame[frame.s == 0].assign(response=REFERENCE)
    return pd.concat([pressured, reference], ignore_index=True)[["response", "s", "t", "seed", *METRICS]]


def _fig5_data(plotted: pd.DataFrame) -> pd.DataFrame:
    long = plotted.melt(id_vars=["response", "s", "t", "seed"], value_vars=list(METRICS), var_name="metric")
    data = (long.groupby(["metric", "response", "s", "t"], sort=False)["value"]
            .agg(mean="mean", sd="std", n="count").reset_index())
    order = {"metric": {m: k for k, m in enumerate(METRICS)},
             "response": {r: k for k, r in enumerate((REFERENCE, *NEGATIVE))}}
    data = data.sort_values(["metric", "response", "t"], key=lambda c: c.map(order[c.name]) if c.name in order else c,
                            ignore_index=True)
    return data[["metric", "response", "s", "t", "mean", "sd", "n"]]


def _draw_fig5(plotted: pd.DataFrame):
    long = plotted.melt(id_vars=["response", "t", "seed"], value_vars=list(METRICS), var_name="metric")
    long["metric"] = long.metric.map(METRIC_LABELS)
    g = sns.FacetGrid(long, col="metric", col_wrap=4, sharey=False, sharex=True, height=4 / 2.54, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="t", y="value", hue="response", palette=_COLORS,
                    hue_order=[REFERENCE, *NEGATIVE], errorbar="sd", marker="o", markersize=5, linewidth=2,
                    err_kws={"alpha": 0.1})
    g.set_titles(col_template="{col_name}", size=_SMALL_FS, pad=2)
    g.set_axis_labels("", "", fontsize=_SMALL_FS)
    g.tick_params(labelsize=_SMALL_FS)
    for ax in g.axes.flat:
        ax.grid(True, linestyle=":", alpha=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks(list(WINDOWS))
        ax.set_xticklabels(["before", "during", "after"])
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, min_n_ticks=3))
    fig = g.figure
    fig.set_size_inches(16 / 2.54, 10 / 2.54)
    fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.06, right=0.99, top=0.95, bottom=0.06)
    ax = g.axes.flat[0]  # the focal-belief panel, labelled as in the paper (positions are for s = 4)
    ax.text(153, 0.58, "compliant", color=_COLORS["compliant"], fontsize=_SMALL_FS, ha="left")
    ax.text(150, -0.5, "resilient", color=_COLORS["resilient"], fontsize=_SMALL_FS, ha="left", rotation=-55)
    ax.text(140, -0.7, "resistant", color=_COLORS["resistant"], fontsize=_SMALL_FS, ha="center", va="bottom")
    for n, ax in enumerate(g.axes.flat):
        ax.text(0.03, 0.975, string.ascii_uppercase[n], fontsize=12, fontdict={"weight": "bold"}, va="top",
                ha="left", transform=ax.transAxes)
    return fig


def figA4(sim_dir, *, base: Baseline = PAPER, mu=0.0, seeds=None, min_seeds=1, loader=None,
          index=None) -> FigureResult | None:
    """Fig. A4: the correlations between the BN metrics across agents, runs and the three windows pooled.

    Uses the runs at s = 0 for μ = 0 and at s = 4 for μ > 0, as the paper does; `mu` replaces the base's μ. The
    title names the pooled windows. The name is the old one, with `_mu{mu}` for μ > 0.

    `data`: metric, other, r (the full symmetric matrix; the figure shows the part below the diagonal).
    """
    index = index_folder(sim_dir) if index is None else index
    corr = _correlations(index, base, mu, item="figA4", seeds=seeds, min_seeds=min_seeds, loader=loader)
    if corr is None:
        return None
    with figure_style(SCRIPT_RC):
        fig = _draw_figA4(corr)
    data = corr.rename_axis(index="metric", columns=None).reset_index().melt(id_vars="metric", var_name="other",
                                                                             value_name="r")
    return FigureResult(fig, data, mu_name("AppendixFig_metricCorrelations", base, mu))


def mu_name(stem, base: Baseline, mu) -> str:
    """`stem`, plus `_mu{mu}` for μ > 0 and the base suffix (μ left out of it): Fig. A4's and the tables' names."""
    return stem + (f"_mu{float(mu)}" if mu > 0 else "") + base.suffix(ignore=("mu",))


def _correlations(index, base: Baseline, mu, *, item, seeds, min_seeds, loader) -> pd.DataFrame | None:
    """Fig. A4's correlation matrix of `METRICS` (s = 0 for μ = 0, s = 4 for μ > 0, three windows pooled), or `None`."""
    s = 0 if mu == 0 else PAPER_S
    frame = selected_metric_frame(index, base, mu, (s,), item=item, seeds=seeds, min_seeds=min_seeds, loader=loader)
    return None if frame is None else frame[list(METRICS)].corr()


def greedy_uncorrelated_subset(sim_dir, *, base: Baseline = PAPER, mu=0.0, metrics=GREEDY_CANDIDATES, n=None,
                               seeds=None, min_seeds=1, loader=None, index=None) -> pd.DataFrame | None:
    """The order in which a greedy search picks weakly correlated `metrics`, on Fig. A4's correlations.

    The old script's helper, which backs the paper's choice of `PREDICTORS`: it starts with the metric of the lowest
    mean |r| to all of `metrics` (itself included), then adds the one whose largest |r| to those already picked is
    smallest, until it has `n` (default: all). Ties go to the earlier of `metrics`. Same runs as `figA4`.

    Columns: rank (1…n), metric, mean_abs_r (to all of `metrics`), max_abs_r (to the earlier picks, NaN for the
    first) and predictor (one of the hand-chosen `PREDICTORS`).
    """
    metrics = list(metrics)
    n = len(metrics) if n is None else n
    index = index_folder(sim_dir) if index is None else index
    corr = _correlations(index, base, mu, item="greedy_uncorrelated_subset", seeds=seeds, min_seeds=min_seeds,
                         loader=loader)
    if corr is None:
        return None
    abs_corr = corr.loc[metrics, metrics].abs()
    mean_abs = abs_corr.mean()
    picked, max_abs = [mean_abs.idxmin()], [np.nan]
    while len(picked) < n:
        to_picked = abs_corr.loc[[m for m in metrics if m not in picked], picked].max(axis=1)
        picked.append(to_picked.idxmin())
        max_abs.append(to_picked.min())
    return pd.DataFrame({"rank": range(1, n + 1), "metric": picked, "mean_abs_r": mean_abs[picked].to_numpy(),
                         "max_abs_r": max_abs, "predictor": [m in PREDICTORS for m in picked]})


def _draw_figA4(corr: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 12 / 2.54))
    shown = corr.rename(columns=METRIC_LABELS, index=METRIC_LABELS)
    sns.heatmap(shown, ax=ax, cbar=False, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 6}, mask=np.triu(np.ones_like(shown)))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=_SMALL_FS - 1)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=_SMALL_FS - 1)
    ax.set_aspect("equal")
    before, during, after = (f"{w - HALF_WINDOW:.0f}-{w + HALF_WINDOW:.0f}" for w in WINDOWS)
    ax.set_title("correlation of variables\nacross agents &\nsimulations, pooled\n"
                 f"over $t={before}$, ${during}$\nand ${after}$", y=0.9, x=0.8, va="top", fontsize=_FS, ha="right")
    for i in range(len(shown)):
        ax.add_patch(plt.Circle((i + 0.5, i + 0.5), 0.1, color="gainsboro"))
    fig.subplots_adjust(hspace=0.01, wspace=0.02, top=1, right=1, left=0.27, bottom=0.27)
    return fig


def cohens_d(before: pd.DataFrame, contrast: str) -> pd.Series:
    """Cohen's d of each of `METRICS` between group 1 and group 0 of `contrast` (a key of `CONTRASTS`).

    `before` holds one row per agent (`metric_frame` rows of one window and pressure) with a `response` column. The
    pooled variance uses n − 1; a metric with a pooled variance below 1e-10 (or an empty group) gets NaN, as in the
    old script.
    """
    group1, group0 = CONTRASTS[contrast]
    a = before.loc[before.response.isin(group1), list(METRICS)]
    b = before.loc[before.response.isin(group0), list(METRICS)]
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    pooled = pooled.mask(pooled < 1e-10, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (a.mean() - b.mean()) / pooled**0.5
    return d.where(pooled > 0).rename("d")


@dataclass(frozen=True)
class LogitFit:
    """A logit of `contrast`'s group 1 against group 0 on the standardised `PREDICTORS` and `CONTROL`.

    `coefficients` is indexed by `(*PREDICTORS, *CONTROL)` (NaN for a constant, dropped column) with `coefficient`,
    `odds_ratio`, `pvalue`, `ci_low` and `ci_high` (95 %). `lr_stat`/`lr_p` test the full model against the control
    alone, with one degree of freedom per predictor kept.
    """

    contrast: str
    coefficients: pd.DataFrame
    pseudo_r2: float  # McFadden
    tjur_r2: float
    lr_stat: float
    lr_p: float
    n: int
    n_group1: int
    n_group0: int


def logit_fit(before: pd.DataFrame, contrast: str, *, label: str = "") -> LogitFit | None:
    """The logit of `contrast` (a key of `CONTRASTS`) on `before` (as for `cohens_d`), or `None` with a `FitWarning`.

    Group 0 of `rR-C` includes `latecompliant` here. The columns are standardised within the subset, and those with
    a std of at most 1e-6 are dropped, as in the old script. An empty group, a perfect separation or a fit that
    doesn't converge (or leaves no standard errors) warns, naming `contrast` and `label`, and gives `None`.
    """
    group1, group0 = CONTRASTS[contrast][0], _LOGIT_GROUP0[contrast]
    what = contrast + (f" {label}" if label else "")
    subset = before[before.response.isin(group1 + group0)]
    y = subset.response.isin(group1).to_numpy(float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        empty = "/".join(group0 if n1 else group1)
        warnings.warn(f"logit {what} left out: no {empty} agents", FitWarning, stacklevel=2)
        return None
    X = subset[[*PREDICTORS, *CONTROL]]
    X = X.loc[:, X.std() > 1e-6]
    X = ((X - X.mean()) / X.std(ddof=0)).reset_index(drop=True)
    control = [c for c in CONTROL if c in X.columns]
    try:
        with warnings.catch_warnings(), np.errstate(over="ignore"):  # exp overflows harmlessly when separating
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", HessianInversionWarning)  # no standard errors, so no CIs
            full = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
            null = sm.Logit(y, sm.add_constant(X[control]) if control else np.ones((len(y), 1))).fit(disp=0)
    except (ConvergenceWarning, PerfectSeparationWarning, HessianInversionWarning, PerfectSeparationError,
            np.linalg.LinAlgError) as err:
        warnings.warn(f"logit {what} left out: {err}", FitWarning, stacklevel=2)
        return None
    if not (full.mle_retvals["converged"] and null.mle_retvals["converged"]):
        warnings.warn(f"logit {what} left out: the fit did not converge", FitWarning, stacklevel=2)
        return None
    ci = full.conf_int(alpha=0.05)
    coefficients = pd.DataFrame({"coefficient": full.params, "odds_ratio": np.exp(full.params),
                                 "pvalue": full.pvalues, "ci_low": ci[0], "ci_high": ci[1]})
    coefficients = coefficients.drop(index="const").reindex([*PREDICTORS, *CONTROL])
    y_hat = full.predict()
    lr_stat = 2 * (full.llf - null.llf)
    return LogitFit(contrast, coefficients, pseudo_r2=float(full.prsquared),
                    tjur_r2=float(y_hat[y == 1].mean() - y_hat[y == 0].mean()), lr_stat=float(lr_stat),
                    lr_p=float(stats.chi2.sf(lr_stat, df=len(X.columns) - len(control))), n=len(y), n_group1=n1,
                    n_group0=n0)


def fig4_levels(pressures, mu) -> tuple:
    """`pressures`, or Fig. 4's rule: `FIG4_PRESSURES` for μ = 0, s = 4 for μ > 0."""
    return tuple(pressures) if pressures is not None else FIG4_PRESSURES if mu == 0 else (PAPER_S,)


def fig4_before(index, base: Baseline, mu, levels, *, item, seeds, min_seeds,
                loader) -> tuple[pd.DataFrame, tuple] | None:
    """The before-window `metric_frame` of the base at `mu` over the `levels` that have runs, and those levels.

    Missing levels get a `MissingLevelsWarning`; with none left, `None` (after a `FigureNotApplicable`).
    """
    config = replace(base, mu=mu)
    present = tuple(s for s in levels if len(runs(index, config, s=s, seeds=seeds)) >= min_seeds)
    missing = ", ".join(f"s={s}" for s in levels if s not in present)
    if not present:
        warnings.warn(f"{item} not applicable: no level has at least {min_seeds} run(s) of μ={mu:g} ({missing})",
                      FigureNotApplicable, stacklevel=3)
        return None
    if missing:
        warnings.warn(f"{item}: leaving out {missing}, which have fewer than {min_seeds} run(s) of μ={mu:g}",
                      MissingLevelsWarning, stacklevel=3)
    frame = selected_metric_frame(index, base, mu, present, item=item, seeds=seeds, min_seeds=min_seeds,
                                  loader=loader)
    return None if frame is None else (frame[frame.t == BEFORE], present)


def fig4(sim_dir, *, base: Baseline = PAPER, mu=0.0, pressures=None, contrast2_pressures=CONTRAST2_PRESSURES,
         seeds=None, min_seeds=1, loader=None, index=None) -> FigureResult | None:
    """Fig. 4 (μ = 0) and Fig. A5/A6 (μ > 0): what tells the response types apart before pressure.

    Two panels, one per contrast (`CONTRASTS`): resistant/resilient vs compliant, and resistant vs resilient. Bars
    are Cohen's d of each metric (`cohens_d`), dots the logit coefficients with 95 % CIs (`logit_fit`; the control
    `x_focal` is fitted but not drawn), both on the before window, per pressure. `pressures` defaults to the paper's
    rule, `FIG4_PRESSURES` for μ = 0 and s = 4 for μ > 0; the levels without runs are warned about. The second
    contrast uses only those of them in `contrast2_pressures`. A fit that fails warns (`FitWarning`) and its dots
    are left out. `mu` replaces the base's μ; the name is `fig4_mu{mu}` plus the base suffix.

    `data`: contrast, s, metric, d, coefficient, ci_low, ci_high (NaN where not drawn).
    """
    levels = fig4_levels(pressures, mu)
    index = index_folder(sim_dir) if index is None else index
    selected = fig4_before(index, base, mu, levels, item="fig4", seeds=seeds, min_seeds=min_seeds, loader=loader)
    if selected is None:
        return None
    before, present = selected
    panels = {"rR-C": present, "R-r": tuple(s for s in present if s in contrast2_pressures)}
    data = _fig4_data(before, panels)
    with figure_style(SCRIPT_RC):
        fig = _draw_fig4(data, panels, mu)
    return FigureResult(fig, data, f"fig4_mu{float(mu)}" + base.suffix(ignore=("mu",)))


def _fig4_data(before: pd.DataFrame, panels: dict) -> pd.DataFrame:
    blocks = []
    no_fit = pd.DataFrame(columns=["coefficient", "ci_low", "ci_high"], dtype=float)
    for contrast, levels in panels.items():
        for s in levels:
            rows = before[before.s == s]
            fit = logit_fit(rows, contrast, label=f"s={s}")
            drawn = no_fit if fit is None else fit.coefficients.loc[list(PREDICTORS), no_fit.columns]
            block = pd.DataFrame({"contrast": contrast, "s": s, "metric": list(METRICS),
                                  "d": cohens_d(rows, contrast).loc[list(METRICS)].to_numpy()})
            blocks.append(block.join(drawn, on="metric"))
    return pd.concat(blocks, ignore_index=True)


def _dot_offset(s, hue_order) -> float:
    """Where the old script puts the dot of `s` relative to the metric's row (not quite on its bar)."""
    return 0.0 if len(hue_order) == 1 else -0.4 + hue_order.index(s) / len(hue_order)


def _bar_offset(s, hue_order) -> float:
    """The centre of seaborn's dodged bar of `s` relative to the metric's row."""
    width = 0.8 / len(hue_order)
    return -0.4 + width * (hue_order.index(s) + 0.5)


def _draw_fig4(data: pd.DataFrame, panels: dict, mu):
    hue_order = list(panels["rR-C"])[::-1]
    palette = {s: PRESSURE_COLORS.get(s, "grey") for s in hue_order}
    bar_colors = {s: mpl.colors.to_rgba(sns.desaturate(c, 0.75), 0.8) for s, c in palette.items()}  # as seaborn draws
    metric_order = [METRIC_LABELS[m] for m in METRICS][::-1]
    position = {label: k for k, label in enumerate(metric_order)}
    nmetric = len(METRICS)
    labelled = nmetric - 2  # the row the pressure labels point at (the second from the top)
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16 / 2.54, 8 / 2.54))
    dot = mpl.lines.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="lightgrey",
                           markeredgecolor="grey", markeredgewidth=0.7, markersize=4, label="regression\ncoefficient")
    bar = patches.Patch(color="grey", label=r"Cohen's $d$")
    axs[0].legend(handles=[bar, dot], fontsize=_SMALL_FS, borderpad=0.25, handlelength=1, handleheight=0.5,
                  handletextpad=0.7, facecolor="gainsboro", loc="lower right", edgecolor="none", framealpha=0.5)
    for n, (contrast, levels) in enumerate(panels.items()):
        ax = axs[n]
        shown = data[data.contrast == contrast].assign(metric=lambda f: f.metric.map(METRIC_LABELS))
        shown = shown.sort_values("s", ascending=False)
        if levels:
            sns.barplot(shown, x="d", y="metric", hue="s", hue_order=hue_order, dodge=True, ax=ax, palette=palette,
                        legend=False, zorder=10, order=metric_order, errorbar=None, alpha=0.8, lw=0)
        ax2 = ax.twiny()
        dots = shown[shown.metric != METRIC_LABELS["x_focal"]].dropna(subset=["coefficient"])
        for s in levels[::-1]:
            for row in dots[dots.s == s].itertuples():
                ax2.errorbar(x=row.coefficient, y=position[row.metric] + _dot_offset(s, hue_order),
                             xerr=[[row.coefficient - row.ci_low], [row.ci_high - row.coefficient]], fmt="o",
                             color="k", markersize=3, markeredgecolor="k", markeredgewidth=0.4,
                             markerfacecolor=palette[s], elinewidth=0.8, capsize=2, capthick=0.8, zorder=20)
        ax2.set_xlabel("Regression Coefficient (dots)", fontsize=_SMALL_FS)
        ax2.set_xlim(-3, 3)
        ax.set_xlabel(r"Cohen's $d$ (bars)", fontsize=_SMALL_FS)
        ax.set_title("\n" + CONTRAST_TITLES[contrast], fontsize=_FS + 1)
        ax.set_ylabel("")
        d = shown[shown.metric == metric_order[labelled]].set_index("s").d
        for kk, s in enumerate(levels[::-1]):
            if np.isnan(d[s]):
                continue
            y = labelled + _bar_offset(s, hue_order)
            if mu == 0:
                ax.annotate(rf"$s={s}$", xy=(d[s] + 0.1, y),
                            xytext=(1.8, y + (1.3 if n == 0 else 1) - (0.56 if n == 0 else 0.46) * (len(levels) - kk)),
                            ha="left", va="center", rotation=0, fontsize=_SMALL_FS - 1,
                            arrowprops=dict(arrowstyle="-", lw=1, connectionstyle="arc,rad=0.1", color=bar_colors[s]),
                            bbox={"pad": 0.02, "fc": "white", "ec": "none"}, color=bar_colors[s], zorder=20)
            else:
                ax.text(1.8, labelled - 0.15 - 0.5 * kk, rf"$s={s}$", color=palette[s])
        xlim = 2.5
        ax.vlines([0], -0.5, nmetric + 0.5, lw=0.5, color="grey")
        ax.hlines([0.5 + k for k in range(nmetric)], -xlim, xlim, lw=0.5, color="grey")
        ax.hlines([nmetric - 2.5, nmetric - 3.5, nmetric - 9.5], -xlim, xlim, lw=1.5, color="grey")  # metric groups
        ax.set_ylim(-0.5, nmetric - 0.5)
        ax.set_xlim(-xlim, xlim)
        ax2.set_ylim(ax.get_ylim())
    fig.canvas.draw()  # so the tick labels exist
    for ax in fig.axes:  # the tick count depends on the rc tick-label size at draw time: keep this style's ticks
        lim = ax.get_xlim()
        ax.set_xticks(ax.get_xticks())
        ax.set_xlim(lim)
    if data.coefficient.notna().any():
        fitted = {METRIC_LABELS[m] for m in (*PREDICTORS, *CONTROL)}
        for label in axs[0].get_yticklabels():
            if label.get_text() in fitted:
                label.set_fontweight("bold")
    fig.subplots_adjust(left=0.23, top=0.82, right=0.98, bottom=0.12)
    return fig
