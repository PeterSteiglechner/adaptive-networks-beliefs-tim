"""Fig. 3, Fig. A3 and the polarisation summary: how agents respond to pressure, per BN config and pressure level.

Reproduces `analyse_responseFrequencies.py`. The configs come from the base and the pressure levels from the
folder. The old script's implicit choices are keyword arguments: Fig. 3A's run (leftover loop variables there) and
Fig. A3's agent draws (the global NumPy seed there, a local `RandomState` here).
"""

import contextlib
import string
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy.spatial.distance import pdist

from .index import index_folder
from .load import Loader, Run
from .output import FigureResult
from .select import (CONFIG_FIELDS, CONFIGS, PAPER, Baseline, Cell, ExampleAgentFallback, FigureNotApplicable,
                     MissingLevelsWarning, runs, select)
from .style import CONFIG_LABELS, FOCAL, RESPONSE_COLORS, figure_style, pressure_alpha

RESPONSES = ("compliant", "resilient", "resistant", "latecompliant")  # Fig. 3's denominator includes latecompliant
PLOTTED = ("compliant", "resilient", "resistant")
HIGHLIGHT_ORDER = ("resistant", "resilient", "compliant")  # the order the old script draws (and samples) agents in
PAPER_PRESSURES = (0, 1, 2, 4, 8, 16)
PAPER_SEEDS = 100
BEFORE, ONSET, T_END = 95.5, 100, 200  # the averaged before-window row, the pressure-onset snapshot, the horizon
SHADE = (100, 150)  # this script shades t = 100…150 (Fig. 2 shades 101…150)
FALLBACK_SEED = 0
JITTER_SEED = 0  # seaborn's strip-plot jitter draws from the global NumPy RNG; seeded (and restored) per figure
METRICS = ("std_focal", "abs_edge", "mean_edge", "groupishness", "nr_negs")

_ENSEMBLE_COLUMNS = (*RESPONSES, str(FOCAL))
_DETAILED_COLUMNS = (*PLOTTED, str(FOCAL))
_FS, _SMALL_FS = 9, 7
_RC = {"font.size": _FS, "axes.titlesize": _FS, "axes.labelsize": _FS, "legend.fontsize": _SMALL_FS,
      "xtick.labelsize": _SMALL_FS, "ytick.labelsize": _SMALL_FS}  # the old script's, on matplotlib's default style


def _grid(index, base: Baseline, levels, *, item, seeds, min_seeds) -> dict | None:
    """The ensemble runs of the four configs at each of `levels` that all four have, by `(config, s)`.

    Levels some config lacks are left out with a `MissingLevelsWarning`. `None` (after a `FigureNotApplicable`) if
    no level is complete.
    """
    configs = {name: base.config(name) for name in CONFIGS}
    lacking = {s: [name for name, cfg in configs.items() if len(runs(index, cfg, s=s, seeds=seeds)) < min_seeds]
               for s in levels}
    complete = [s for s in levels if not lacking[s]]
    gaps = "; ".join(f"s={s}: no {', '.join(names)}" for s, names in lacking.items() if names)
    if not complete:
        warnings.warn(f"{item} not applicable: no pressure level s ∈ {{{', '.join(map(str, levels))}}} has at least "
                      f"{min_seeds} run(s) of all four configs ({gaps})", FigureNotApplicable, stacklevel=3)
        return None
    if gaps:
        missing = ", ".join(str(s) for s in levels if lacking[s])
        warnings.warn(f"{item}: leaving out s = {missing}, which not all four configs have ({gaps})",
                      MissingLevelsWarning, stacklevel=3)
    cells = {(name, s): Cell(f"{name} s={s}", cfg, s, seeds=seeds) for s in complete for name, cfg in configs.items()}
    selected = select(index, cells.values(), item=item, paper_seeds=PAPER_SEEDS, min_seeds=min_seeds)
    return None if selected is None else {key: selected[cell.label] for key, cell in cells.items()}


def _per_run(grid: dict, base: Baseline, loader: Loader) -> pd.DataFrame:
    """One row per run: config, s, seed, the response counts (95.5 row) and the polarisation metrics.

    `std_focal`, `abs_edge`, `mean_edge` and `groupishness` (std / mean of the pairwise Manhattan distances of the
    agents' edge vectors, 0 for static BNs) are taken at t = 100, `nr_negs` (agents with a negative focal belief)
    from the 95.5 before-window row.
    """
    rows = []
    for (config, s), frame in grid.items():
        static = base.config(config).eps == 0
        for seed, path in zip(frame.seed, frame.path):
            run = loader.load(path, t=(BEFORE, ONSET), columns=_ENSEMBLE_COLUMNS, edges=True)
            before, onset = run.frame[run.frame.t == BEFORE], run.frame[run.frame.t == ONSET]
            W = onset[run.layout.edges].to_numpy(float)
            dists = None if static else pdist(W, metric="cityblock")
            rows.append({
                "config": config, "s": s, "seed": int(seed), **before[list(RESPONSES)].sum().to_dict(),
                "std_focal": onset[str(FOCAL)].std(), "abs_edge": np.abs(W).mean(), "mean_edge": W.mean(),
                "groupishness": 0.0 if dists is None else dists.std() / dists.mean(),
                "nr_negs": int((before[str(FOCAL)] < 0).sum()),
            })
    return pd.DataFrame(rows)


def _frequencies(table: pd.DataFrame, denominator) -> pd.DataFrame:
    """Tidy relative response frequencies per run (config, s, seed, response, frequency) of the `denominator` types."""
    counts = table[list(denominator)].astype(float)
    freq = counts.div(counts.sum(axis=1), axis=0)
    freq = pd.concat([table[["config", "s", "seed"]], freq], axis=1)
    long = freq.melt(id_vars=["config", "s", "seed"], value_vars=list(denominator), var_name="response",
                     value_name="frequency")
    return long.sort_values(["config", "s", "seed"], kind="stable", ignore_index=True)


@contextlib.contextmanager
def _seeded_jitter(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def fig3(sim_dir, *, base: Baseline = PAPER, seeds=None, example_seed=1, example_s=4,
         example_agents=None, denominator=RESPONSES, min_seeds=1, loader=None,
         index=None) -> FigureResult | None:
    """Fig. 3: relative response frequencies over pressure for the four BN configs, and an example simulation.

    B–E: per config, one dot per run (`seeds`, default all) at each of the paper's pressures s > 0 that all four
    configs have; missing levels are warned about (`MissingLevelsWarning`). The `denominator` types (default
    compliant, resilient, resistant and latecompliant, as in the paper; it must include the three plotted ones) are
    counted. A: the detailed adaptive run (`example_seed`,
    `example_s`) with one agent per response type highlighted, `example_agents` (default resistant 4, resilient
    53, compliant 16; `None` for a type draws one). A default agent that isn't of its type is replaced by a seeded
    random one (`ExampleAgentFallback`); a type no agent has is left out.

    `data`: config, s, seed, response (each `denominator` type), frequency.
    """
    if not set(PLOTTED) <= set(denominator) <= set(RESPONSES):
        raise ValueError(f"denominator must hold {', '.join(PLOTTED)} and may add latecompliant, got {denominator}")
    example_agents = {"resistant": 4, "resilient": 53, "compliant": 16} if example_agents is None else example_agents
    index = index_folder(sim_dir) if index is None else index
    loader = loader or Loader()
    grid = _grid(index, base, [s for s in PAPER_PRESSURES if s > 0], item="fig3", seeds=seeds, min_seeds=min_seeds)
    if grid is None:
        return None
    cell = Cell(f"detailed adaptive s={example_s} seed={example_seed}", base.config("adaptive"), example_s,
                detailed=True, seeds=(example_seed,))
    example = select(index, [cell], item="fig3", paper_seeds=1)
    if example is None:
        return None
    path = example[cell.label].path.iloc[0]
    run = loader.load(path, columns=_DETAILED_COLUMNS)
    agents = _example_agents(run, example_agents)
    data = _frequencies(_per_run(grid, base, loader), denominator)
    with figure_style(_RC), _seeded_jitter(JITTER_SEED):
        fig = _draw_fig3(data, run, agents, int(example[cell.label].s.iloc[0]))
    return FigureResult(fig, data, "fig3_responseFrequencies" + base.suffix(ignore=CONFIG_FIELDS))


def _example_agents(run: Run, wanted: dict) -> dict[str, int]:
    """The highlighted agent per response type (in `HIGHLIGHT_ORDER`); types no agent has are left out."""
    rng = np.random.default_rng(FALLBACK_SEED)
    chosen = {}
    for name in HIGHLIGHT_ORDER:
        ids = np.sort(run.frame.loc[run.frame[name] == 1, "id"].unique())
        if len(ids) == 0:
            continue
        agent = wanted.get(name)
        if agent is not None and agent in ids:
            chosen[name] = int(agent)
            continue
        chosen[name] = int(rng.choice(ids))
        if agent is not None:
            warnings.warn(f"example {name} agent {agent} isn't {name} in {run.path.name}; using agent "
                          f"{chosen[name]} instead", ExampleAgentFallback, stacklevel=3)
    return chosen


def _focal(run: Run) -> pd.DataFrame:
    focal = run.frame.pivot_table(index="t", columns="id", values=str(FOCAL))
    return focal.loc[focal.index <= T_END]


def _trajectories(ax, focal: pd.DataFrame, N, s, highlights: dict, *, alpha, lw):
    """Every agent's focal belief (grey by id), the `highlights` (type → agent) on top and the pressure window."""
    for agent in focal.columns:
        grey = 0.8 - 0.5 * agent / N
        focal[agent].plot(ax=ax, lw=0.6, alpha=alpha, legend=False, color=(grey, grey, grey, 1))
    for name, agent in highlights.items():
        focal[agent].plot(ax=ax, lw=lw, ls="-", color=RESPONSE_COLORS[name], alpha=0.8, legend=False, label="_")
    ax.set_xlabel("time")
    ax.set_ylabel(r"focal belief $x_\mathrm{foc}$", va="center")
    ax.set_clip_on(False)
    ax.set_xlim(0, T_END)
    ax.set_ylim(-1.02, 1.02)
    ax.set_yticks([-1, 0, 1])
    if s > 0:
        y0, y1 = ax.get_ylim()
        ax.fill_between(list(SHADE), [y0] * 2, [y1] * 2, color="red", alpha=pressure_alpha(s), zorder=-1, lw=0)


def _misaligned_label(ax):
    ax.text(90, -0.2, "agents with\n" + r"$x_\mathrm{foc}<0$", va="center", ha="right", fontsize=_FS)
    ax.add_patch(patches.Arc((BEFORE, -0.7), 10, 1.4, angle=0, linewidth=1, fill=False, zorder=10, color="k"))


def _panel_letters(axes):
    for n, ax in enumerate(axes):
        ax.text(0, 1.02, string.ascii_uppercase[n], fontsize=12, fontdict={"weight": "bold"}, va="bottom",
                ha="left", transform=ax.transAxes)


def _draw_fig3(data: pd.DataFrame, run: Run, agents: dict, s_example):
    levels = sorted(data.s.unique())
    fig, axs = plt.subplot_mosaic([["t"] * 4, ["1", "2", "3", "4"]], figsize=(16 / 2.54, 10 / 2.54))
    for k in range(2, 5):
        axs[str(k)].sharex(axs["1"])
        axs[str(k)].sharey(axs["1"])
    plotted = data[data.response.isin(PLOTTED)].assign(s_ext_log2=lambda d: np.log2(d.s))
    style = dict(x="s_ext_log2", hue="response", y="normalized_count", palette=RESPONSE_COLORS,
                 hue_order=list(PLOTTED), legend=False)
    for k, config in enumerate(CONFIGS, start=1):
        ax = axs[str(k)]
        subset = plotted[plotted.config == config][["s_ext_log2", "response", "frequency"]]
        subset = subset.rename(columns={"frequency": "normalized_count"})
        sns.boxplot(subset, ax=ax, **style, fliersize=0, fill=True, linewidth=0.0, whis=0)
        sns.stripplot(subset, ax=ax, **style, jitter=True, size=2, alpha=0.4, dodge=True)
        for coll in ax.collections:
            coll.set_clip_on(False)
        medians = subset.groupby(["response", "s_ext_log2"])["normalized_count"].median().reset_index()
        sns.stripplot(medians, ax=ax, **style, jitter=True, size=4, alpha=0.8, dodge=True, marker="s")
        for coll in ax.collections:
            coll.set_clip_on(False)
        for x in range(1, len(levels), 2):  # every other level on grey (s = 2, 8 for the paper's five)
            ax.fill_between([x - 0.5, x + 0.5], [-0.05] * 2, [1.05] * 2, color="gainsboro", zorder=-1)
        ax.set_ylabel("relative response frequency", fontsize=_FS, va="center")
        ax.set_title(CONFIG_LABELS[config], fontsize=_FS, x=0.98, y=0.99, ha="right")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([f"${int(s):0d}$" for s in levels], rotation=0)
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(rf"${int(x * 100)}\,\%$" for x in [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs["2"].set_xlabel(r"external pressure $s$", x=1)
    axs["1"].text(len(levels) - 0.5, 0.24, "1 dot = 1 simulation", ha="right", va="center", fontsize=_SMALL_FS,
                  color="k")

    ax = axs["t"]
    _trajectories(ax, _focal(run), run.layout.N, s_example, agents, alpha=0.3, lw=2)
    for (x, y), name in zip([(127, -0.6), (172, 0.5), (155, -0.2)], ["resistant", "compliant", "resilient"]):
        ax.text(x, y, name, color="white", va="center", ha="left", fontsize=_FS,
                bbox=dict(boxstyle="round", facecolor=RESPONSE_COLORS[name], edgecolor="white", alpha=0.8))
    ax.text(125, -0.1, "external\npressure", ha="center", fontsize=_FS)
    ax.text(135, -0.1, r"$\uparrow$", ha="center", fontsize=_FS + 15)
    ax.text(0.5, 1.02, rf"example simulation with {CONFIG_LABELS['adaptive']} belief network, $s={s_example}$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=_FS)
    _misaligned_label(ax)
    _panel_letters([axs[k] for k in ("t", "1", "2", "3", "4")])
    fig.subplots_adjust(hspace=0.45, top=0.94, left=0.09, right=0.98, bottom=0.1)
    return fig


def figA3(sim_dir, *, base: Baseline = PAPER, seed=1, rng_seed=14, loader=None, index=None) -> FigureResult | None:
    """Fig. A3: the detailed adaptive run `seed` at each pressure s ∈ {0, 1, 2, 4, 8, 16}, one panel each.

    Each panel highlights one random agent per response type. The draws come from one local
    `np.random.RandomState(rng_seed)` passed to `Series.sample` in panel order, so `rng_seed=14` picks the agents of
    the paper's figure (the old script seeded the global RNG). The global RNG isn't touched.

    `data`: s, t, agent, value (the focal belief) and highlight (the response type of a highlighted agent, else "").
    """
    index = index_folder(sim_dir) if index is None else index
    loader = loader or Loader()
    cells = [Cell(f"detailed adaptive s={s} seed={seed}", base.config("adaptive"), s, detailed=True, seeds=(seed,))
             for s in PAPER_PRESSURES]
    selected = select(index, cells, item="figA3", paper_seeds=1)
    if selected is None:
        return None
    rs = np.random.RandomState(rng_seed)
    panels = []
    for s, cell in zip(PAPER_PRESSURES, cells):
        run = loader.load(selected[cell.label].path.iloc[0], columns=_DETAILED_COLUMNS)
        drawn = {}
        for name in HIGHLIGHT_ORDER:
            ids = run.frame.loc[run.frame[name] == 1, "id"]
            if len(ids) > 0:
                drawn[name] = int(ids.sample(random_state=rs).iloc[0])
        panels.append((s, run, drawn))
    with figure_style(_RC):
        fig = _draw_figA3(panels)
    name = f"AppendixFig_exampleSim_adaptive_seed{seed}_overPressure" + base.suffix(ignore=CONFIG_FIELDS)
    return FigureResult(fig, _figA3_data(panels), name)


def _figA3_data(panels) -> pd.DataFrame:
    blocks = []
    for s, run, drawn in panels:
        long = _focal(run).reset_index().melt(id_vars=["t"], var_name="agent", value_name="value")
        long["agent"] = long.agent.astype(int)
        long["highlight"] = long.agent.map({agent: name for name, agent in drawn.items()}).fillna("")
        long.insert(0, "s", s)
        blocks.append(long)
    return pd.concat(blocks, ignore_index=True)[["s", "t", "agent", "value", "highlight"]]


def _draw_figA3(panels):
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16 / 2.54, 8 / 2.54))
    for (s, run, drawn), ax in zip(panels, axs.flatten()):
        _trajectories(ax, _focal(run), run.layout.N, s, drawn, alpha=0.2, lw=1)
        if s == PAPER_PRESSURES[-1]:
            ax.text(125, -0.7, "external\npressure", ha="center", fontsize=_FS)
        if s > 0:
            ax.text(125, -0.1, r"$\uparrow$", ha="center", fontsize=_FS + 12)
        ax.text(0.5, 1.02, rf"$s={s}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=_FS)
        if s == 4:
            _misaligned_label(ax)
    _panel_letters(axs.flatten())
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.93)
    return fig


def polarisation_summary(sim_dir, *, base: Baseline = PAPER, seeds=None, min_seeds=1, loader=None,
                         index=None) -> pd.DataFrame | None:
    """Mean and std over runs of the polarisation metrics per BN config, over the paper's pressures s ≥ 0.

    Uses the levels all four configs have (missing ones warned about, as in `fig3`). Metrics: `std_focal` (σ_foc
    at t = 100), `abs_edge` / `mean_edge` (mean |ω| / mean ω at t = 100), `groupishness` and `nr_negs` (agents
    with a negative focal belief in the 95.5 before-window row; the old script averaged t = 91…100, which in
    ensemble files is only the t = 100 snapshot). Columns
    are `(metric, "mean" | "std")` plus `("n_runs", "")`, indexed by config.
    """
    index = index_folder(sim_dir) if index is None else index
    grid = _grid(index, base, PAPER_PRESSURES, item="polarisation_summary", seeds=seeds, min_seeds=min_seeds)
    if grid is None:
        return None
    table = _per_run(grid, base, loader or Loader())
    by_config = table.groupby("config", sort=False)
    summary = by_config[list(METRICS)].agg(["mean", "std"])
    summary[("n_runs", "")] = by_config.size()
    return summary.reindex(list(CONFIGS)).rename_axis("config")
