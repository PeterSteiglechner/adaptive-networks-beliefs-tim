"""Fig. 5, Fig. A4 and the agent-metric frame they share: each agent's BN metrics before, during and after pressure.

Reproduces `analyse_BNmetrics.py`. Its `res` (one row per agent × seed × s × window, labelled with the agent's
response) is `metric_frame`; the runs come from the index instead of the script's fixed seed and pressure lists. The
Fig. A4 title states the three windows it pools (the old title named only the first).
"""

import string
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .index import index_folder
from .load import Loader
from .output import FigureResult
from .select import PAPER, Baseline, Cell, select
from .style import METRIC_LABELS, RESPONSE_COLORS, SCRIPT_RC, figure_style

WINDOWS = (95.5, 145.5, 195.5)  # the averaged before / during / after rows
BEFORE = WINDOWS[0]  # the row that holds the response classification
HALF_WINDOW = 4.5
RESPONSE_FLAGS = ("persistPos", "nonpersistPos", "compliant", "resilient", "resistant", "latecompliant")
PAPER_S = 4  # the pressure of Fig. 5, and of Fig. A4 for μ > 0
NEGATIVE = ("compliant", "resilient", "resistant")
METRICS = tuple(METRIC_LABELS)  # the old `metrics_table_sort`
REFERENCE = "reference"  # Fig. 5's s = 0 curve, every agent pooled (the old script's "RRC")
PAPER_SEEDS = 100

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


def _selected_metric_frame(index, base: Baseline, mu, levels, *, item, seeds, min_seeds,
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
    frame = _selected_metric_frame(index, base, mu, (s, 0), item="fig5", seeds=seeds, min_seeds=min_seeds,
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
    s = 0 if mu == 0 else PAPER_S
    index = index_folder(sim_dir) if index is None else index
    frame = _selected_metric_frame(index, base, mu, (s,), item="figA4", seeds=seeds, min_seeds=min_seeds, loader=loader)
    if frame is None:
        return None
    corr = frame[list(METRICS)].corr()
    with figure_style(SCRIPT_RC):
        fig = _draw_figA4(corr)
    data = corr.rename_axis(index="metric", columns=None).reset_index().melt(id_vars="metric", var_name="other",
                                                                             value_name="r")
    name = "AppendixFig_metricCorrelations" + (f"_mu{float(mu)}" if mu > 0 else "") + base.suffix(ignore=("mu",))
    return FigureResult(fig, data, name)


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
