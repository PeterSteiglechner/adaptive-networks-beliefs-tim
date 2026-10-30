"""Fig. 6 and Fig. A7: how social adaptation μ shapes the belief networks and the responses to pressure.

Reproduces `analyse_socialAdaptation.py`. The μ values come from the folder instead of the script's fixed list. Its
implicit choices are keyword arguments: Fig. A7's example simulation (leftover loop variables there) and Fig. 6A/B's
agents (`np.random.randint(100)` after the global `np.random.seed(42)` there, a local `RandomState` over N here).
Fig. 6D's y-ticks are fixed before their labels are set (the old labels read 1.25× too large).
"""

import contextlib
import string
import warnings
from dataclasses import replace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist

from .index import index_folder
from .load import Loader, Run
from .output import FigureResult
from .select import PAPER, Baseline, Cell, FigureNotApplicable, MissingLevelsWarning, runs, select
from .style import EDGE_CMAP, EDGE_NORM, FOCAL, METRIC_LABELS, RESPONSE_COLORS, SCRIPT_RC, figure_style

PAPER_MUS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
PAPER_S = 4
PAPER_SEEDS = 100
BEFORE, ONSET = 95.5, 100  # the averaged before-window row (responses, metrics), the pressure-onset snapshot (edges)
PLOTTED = ("compliant", "resilient", "resistant")  # Fig. 6's denominator: latecompliant is left out, as in the paper
RESPONSES = (*PLOTTED, "latecompliant")
FIG6_METRICS = ("absOm_foc", "tb_foc", "clust")  # the old `selected_metrics`
N_EXAMPLES = 8  # example belief networks per example μ
HETEROGENEITY_COLOR = "#FF69B4"
FREQUENCY_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DATA_COLUMNS = ["panel", "quantity", "mu", "seed", "id", "response", "value", "z"]

_COLUMNS = (*RESPONSES, *FIG6_METRICS)
_FS, _SMALL_FS = SCRIPT_RC["font.size"], SCRIPT_RC["legend.fontsize"]


def _load(loader: Loader, path) -> Run:
    """The before-window and onset rows of a run, the same request for both figures (so `make_all` reads it once)."""
    return loader.load(path, t=(BEFORE, ONSET), columns=_COLUMNS, edges=True)


def _present_mus(index, base: Baseline, levels, *, item, seeds, min_seeds) -> tuple | None:
    """The `levels` with at least `min_seeds` runs of the base at s = 4; the rest get a `MissingLevelsWarning`.

    `None` (after a `FigureNotApplicable`) if fewer than two are left.
    """
    present = tuple(mu for mu in levels if len(runs(index, replace(base, mu=mu), s=PAPER_S, seeds=seeds)) >= min_seeds)
    missing = ", ".join(f"{mu:g}" for mu in levels if mu not in present)
    if len(present) < 2:
        found = ", ".join(f"{mu:g}" for mu in present) or "none"
        warnings.warn(f"{item} not applicable: needs at least two μ values with at least {min_seeds} run(s) at "
                      f"s={PAPER_S}, found {found}", FigureNotApplicable, stacklevel=3)
        return None
    if missing:
        warnings.warn(f"{item}: leaving out μ = {missing}, which have fewer than {min_seeds} run(s) at s={PAPER_S}",
                      MissingLevelsWarning, stacklevel=3)
    return present


def _matching(mu, levels):
    """The value among `levels` within 1e-9 of `mu`, or `None`."""
    return next((level for level in levels if np.isclose(level, mu, rtol=0, atol=1e-9)), None)


def fig6(sim_dir, *, base: Baseline = PAPER, mus=None, example_seed=1, example_mus=(0.005, 0.2), agent_rng_seed=42,
         denominator=PLOTTED, seeds=None, min_seeds=1, loader=None, index=None) -> FigureResult | None:
    """Fig. 6: the belief networks and the response frequencies over social adaptation μ, at s = 4.

    The x-axis holds the `mus` (default the paper's 11, `PAPER_MUS`) that have runs of the base at s = 4 (`seeds`,
    default all); missing ones are warned about (`MissingLevelsWarning`), and at least two are needed. C: the
    heterogeneity H of the BNs (the mean Euclidean distance between the agents' edge vectors at t = 100, per run)
    and the metrics `FIG6_METRICS` (per agent, 95.5 row), each z-scored over everything plotted, mean ± sd per μ.
    D: the relative response frequencies per run, counted over `denominator` (default compliant, resilient and
    resistant, as in the paper; it may add latecompliant). A/B: the BNs at t = 100 of `N_EXAMPLES` agents in the
    run `example_seed` at each of the two `example_mus`, which must be among the plotted μ values (else
    `FigureNotApplicable`) and have that run. The agents are `RandomState(agent_rng_seed).randint(N)` draws (42
    picks the paper's); the strip-plot jitter continues that RNG, as the old script's global one did. The global RNG
    is left as it was. Each μ plotted replaces the base's μ.

    `data` (`DATA_COLUMNS`): panel, quantity, mu, seed, id, response, value, z. Panel C has quantity "H" (one row
    per run, no id) and each metric (one row per agent), with the z-score; D has "frequency" (per run × response);
    A/B the example agents' edge weights (quantity: the edge column), 45 rows per drawn agent in draw order.
    """
    if not set(PLOTTED) <= set(denominator) <= set(RESPONSES):
        raise ValueError(f"denominator must hold {', '.join(PLOTTED)} and may add latecompliant, got {denominator}")
    levels = PAPER_MUS if mus is None else tuple(float(mu) for mu in mus)
    requested = tuple(float(mu) for mu in example_mus)
    example_mus = tuple(_matching(mu, levels) for mu in requested)  # as the `levels` write them
    if len(example_mus) != 2 or None in example_mus:
        raise ValueError(f"example_mus must be two of the μ values plotted ({levels}), got {requested}")
    index = index_folder(sim_dir) if index is None else index
    loader = loader or Loader()
    present = _present_mus(index, base, levels, item="fig6", seeds=seeds, min_seeds=min_seeds)
    if present is None:
        return None
    if dropped := [mu for mu in example_mus if mu not in present]:
        warnings.warn(f"fig6 not applicable: example μ = {', '.join(f'{mu:g}' for mu in dropped)} isn't among the "
                      f"plotted μ values ({', '.join(f'{mu:g}' for mu in present)})", FigureNotApplicable,
                      stacklevel=2)
        return None
    example_cells = [Cell(f"μ={mu:g} s={PAPER_S} seed={example_seed}", replace(base, mu=mu), PAPER_S,
                          seeds=(example_seed,)) for mu in example_mus]
    examples = select(index, example_cells, item="fig6", paper_seeds=1)
    if examples is None:
        return None
    cells = [Cell(f"μ={mu:g} s={PAPER_S}", replace(base, mu=mu), PAPER_S, seeds=seeds) for mu in present]
    selected = select(index, cells, item="fig6", paper_seeds=PAPER_SEEDS, min_seeds=min_seeds)
    if selected is None:
        return None
    per_run, per_agent = _fig6_rows(present, cells, selected, denominator, loader)
    example_runs = [_load(loader, examples[cell.label].path.iloc[0]) for cell in example_cells]
    rs = np.random.RandomState(agent_rng_seed)
    agents = [int(rs.randint(example_runs[0].layout.N)) for _ in range(N_EXAMPLES)]
    data = _fig6_data(per_run, per_agent, example_mus, example_runs, agents, example_seed)
    with figure_style(SCRIPT_RC), _global_rng(rs.get_state()):
        fig = _draw_fig6(data, list(present), example_mus, example_runs[0].layout.M)
    return FigureResult(fig, data, "fig6_socialAdaptation" + base.suffix(ignore=("mu",)))


def _fig6_rows(present, cells, selected, denominator, loader) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per run (mu, seed, H, the plotted frequencies) and one per agent (mu, seed, id, `FIG6_METRICS`)."""
    per_run, per_agent = [], []
    for mu, cell in zip(present, cells):
        frame = selected[cell.label]
        for seed, path in zip(frame.seed, frame.path):
            run = _load(loader, path)
            before, onset = run.frame[run.frame.t == BEFORE], run.frame[run.frame.t == ONSET]
            counts = before[list(denominator)].astype(float).sum()
            dists = pdist(onset[run.layout.edges].to_numpy(float), metric="euclidean")
            per_run.append({"mu": mu, "seed": int(seed), "H": dists.mean(),
                            **(counts[list(PLOTTED)] / counts.sum()).to_dict()})
            per_agent.append(before[["id", *FIG6_METRICS]].assign(mu=mu, seed=int(seed)))
    return pd.DataFrame(per_run), pd.concat(per_agent, ignore_index=True)


def _zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def _fig6_data(per_run, per_agent, example_mus, example_runs, agents, example_seed) -> pd.DataFrame:
    blocks = [per_run.assign(panel="C", quantity="H", value=per_run.H, z=_zscore(per_run.H))]
    for metric in FIG6_METRICS:
        blocks.append(per_agent.assign(panel="C", quantity=metric, value=per_agent[metric],
                                       z=_zscore(per_agent[metric])))
    # response-major (then μ, seed), as the old script's melt: seaborn's strip jitter follows the row order
    freq = per_run.melt(id_vars=["mu", "seed"], value_vars=list(PLOTTED), var_name="response")
    blocks.append(freq.assign(panel="D", quantity="frequency"))
    for panel, mu, run in zip("AB", example_mus, example_runs):
        onset = run.frame[run.frame.t == ONSET].set_index("id")
        for agent in agents:
            edges = onset.loc[agent, run.layout.edges]
            blocks.append(pd.DataFrame({"panel": panel, "quantity": edges.index, "mu": mu, "seed": example_seed,
                                        "id": agent, "value": edges.to_numpy(float)}))
    data = pd.concat([block.reindex(columns=DATA_COLUMNS) for block in blocks], ignore_index=True)
    return data.astype({"id": "Int64", "seed": int, "mu": float, "value": float, "z": float})


@contextlib.contextmanager
def _global_rng(state):
    """NumPy's global RNG set to `state` (seaborn's strip-plot jitter draws from it) and restored afterwards."""
    previous = np.random.get_state()
    np.random.set_state(state)
    try:
        yield
    finally:
        np.random.set_state(previous)


def _bracket(fig, axs_list, mu_ax, mu_x, color="black"):
    """A dashed arc under a row of example BNs and a line from its middle to `mu_x` on `mu_ax` (the old helper)."""
    fig.canvas.draw()  # for the final axes positions
    xs = [ax.get_position().x0 + ax.get_position().width / 2 for ax in axs_list]
    brace_y = axs_list[0].get_position().y0
    x_arr = np.linspace(xs[0] - 0.2 * (xs[1] - xs[0]), xs[-1] + 0.2 * (xs[1] - xs[0]), 300)
    y_arr = brace_y - 0.02 * np.sin(np.pi * (x_arr - xs[0]) / (xs[-1] - xs[0]))
    fig.add_artist(plt.Line2D(x_arr, y_arr, transform=fig.transFigure, color=color, lw=0.8, linestyle="--",
                              clip_on=False))
    mu_bb, mu_xlim = mu_ax.get_position(), mu_ax.get_xlim()
    mu_fig_x = mu_bb.x0 + (mu_x - mu_xlim[0]) / (mu_xlim[1] - mu_xlim[0]) * mu_bb.width
    fig.add_artist(plt.Line2D([(xs[0] + xs[-1]) / 2, mu_fig_x], [brace_y - 0.02, mu_bb.y1], transform=fig.transFigure,
                              color=color, lw=0.8, clip_on=False, linestyle="--"))


def _example_bn(ax, weights: np.ndarray, pairs, pos):
    ax.axis("off")
    G = nx.complete_graph(len(pos))
    nx.set_edge_attributes(G, dict(zip(pairs, weights)), name="weight")
    nx.draw_networkx_edges(G=G, pos=pos, edgelist=pairs, edge_color=[EDGE_CMAP(EDGE_NORM(w)) for w in weights],
                           width=[min(5, 0.8 * w) for w in weights], ax=ax)
    nx.draw_networkx_nodes(G=G, pos=pos, node_color="grey", node_size=1, ax=ax)
    ax.plot(*pos[FOCAL], marker="o", ms=3, markerfacecolor="None", markeredgecolor="k", markeredgewidth=1, zorder=-1)


def _draw_fig6(data: pd.DataFrame, mus: list, example_mus, M):
    row = lambda letter, ks: [f"{letter}{i}" for i in ks]  # noqa: E731
    fig, axs = plt.subplot_mosaic(
        [row("a", range(4)) + ["."] + row("b", range(4)), row("a", range(4, 8)) + ["."] + row("b", range(4, 8)),
         [".."] * 9, ["metric"] * 9, ["mu"] * 9],
        height_ratios=[1, 1, 0.0005, 2.5, 2.5], width_ratios=[1] * 9, figsize=(16 / 2.54, 12 / 2.54))
    pos = nx.circular_layout(nx.complete_graph(M))
    pairs = [(int(m), int(n)) for m, n in zip(*np.triu_indices(M, k=1))]
    for panel, letter in zip("AB", "ab"):
        rows = data[data.panel == panel]
        for i in range(N_EXAMPLES):
            _example_bn(axs[f"{letter}{i}"], rows.value.to_numpy()[i * len(pairs):(i + 1) * len(pairs)], pairs, pos)

    mu_idx = {mu: k for k, mu in enumerate(mus)}
    ax_metric, ax_mu = axs["metric"], axs["mu"]
    plt.setp(ax_metric.get_xticklabels(), visible=False)
    ax_metric.sharex(ax_mu)
    panel_c = data[data.panel == "C"].assign(mu_idx=lambda d: d.mu.map(mu_idx))
    style = dict(x="mu_idx", y="z", marker="o", errorbar="sd", ax=ax_metric, err_kws={"alpha": 0.1})
    sns.lineplot(panel_c[panel_c.quantity == "H"], color=HETEROGENEITY_COLOR, label="heterogeneity BNs", **style)
    for metric in FIG6_METRICS:
        sns.lineplot(panel_c[panel_c.quantity == metric], label=METRIC_LABELS[metric], **style)
    ax_metric.legend(loc="lower left", ncols=4)
    ax_metric.set_ylim(-2, 2)
    ax_metric.set_yticks([-2, -1, 0, 1, 2])
    ax_metric.set_ylabel("metric value (z-score)")

    freq = data[data.panel == "D"].rename(columns={"value": "count"})
    freq = freq.assign(mu=pd.Categorical(freq.mu, categories=mus, ordered=True))[["mu", "response", "count"]]
    style = dict(x="mu", y="count", hue="response", hue_order=list(PLOTTED), palette=RESPONSE_COLORS, ax=ax_mu)
    sns.boxplot(freq, fliersize=0, fill=True, linewidth=0.0, whis=0, **style)
    sns.stripplot(freq, dodge=True, size=1.5, edgecolor="white", linewidth=0.2, legend=False, **style)
    ax_mu.get_legend().set_title("")
    ax_mu.get_legend().set_loc("center right")
    ax_mu.set_ylabel("relative response frequency")
    ax_mu.set_xlabel(r"social adaptation $\mu$ (log-scale)")
    ax_mu.set_ylim(-0.05, 1.05)
    ax_mu.set_xlim(-0.5, len(mus) - 0.5)
    ax_mu.set_yticks(FREQUENCY_TICKS)  # first the positions, then their labels (the old script only set labels)
    ax_mu.set_yticklabels([rf"${round(x * 100):d}\,\%$" for x in FREQUENCY_TICKS])
    for x in range(1, len(mus), 2):  # every other μ on grey
        ax_mu.fill_between([x - 0.5, x + 0.5], [-0.05] * 2, [1.05] * 2, color="gainsboro", zorder=-1)
        ax_metric.fill_between([x - 0.5, x + 0.5], [-2] * 2, [2] * 2, color="gainsboro", zorder=-1)

    axs[".."].axis("off")
    fig.subplots_adjust(left=0.1, bottom=0.085, top=0.99, right=0.99)
    for letter, mu in zip("ab", example_mus):
        _bracket(fig, [axs[f"{letter}{i}"] for i in range(4, 8)], ax_metric, mu_x=mu_idx[mu], color="darkgrey")
    bold = {"size": _FS, "weight": "bold"}
    fig.text(0.54, 0.88, "example\nbelief\nnetworks", fontdict=bold, ha="center", va="center")
    for x, mu in zip((0.29, 0.795), example_mus):
        fig.text(x, 0.87, rf"with $\mu={mu}$", fontdict=bold, ha="center", va="center")
    fig.text(0.49, 0.8, "focal\nbelief", fontdict={"size": _SMALL_FS, "weight": "normal"}, ha="left", va="center")
    for n, key in enumerate(["a0", "b0", "metric", "mu"]):
        axs[key].text(-0.1 if n < 2 else 0.01, 0.975, string.ascii_uppercase[n], fontsize=12,
                      fontdict={"weight": "bold"}, va="top", ha="left", transform=axs[key].transAxes)
    return fig


def figA7(sim_dir, *, base: Baseline = PAPER, seed=99, s=PAPER_S, mus=(0.0, 0.005, 0.2), loader=None,
          index=None) -> FigureResult | None:
    """Fig. A7: the distribution over agents of each edge weight at t = 100, one row per μ.

    One example simulation: the ensemble run `seed` at pressure `s` of the base at each of `mus` (`mu` replaces
    the base's μ); every one is needed. The dashed line is the base's `init_w`. The name is the old one plus the
    base suffix.

    `data`: mu, edge (the edge column), id, value (the weight at t = 100).
    """
    index = index_folder(sim_dir) if index is None else index
    loader = loader or Loader()
    mus = tuple(float(mu) for mu in mus)
    cells = [Cell(f"μ={mu:g} s={s} seed={seed}", replace(base, mu=mu), s, seeds=(seed,)) for mu in mus]
    selected = select(index, cells, item="figA7", paper_seeds=1)
    if selected is None:
        return None
    blocks = []
    for mu, cell in zip(mus, cells):
        run = _load(loader, selected[cell.label].path.iloc[0])
        onset = run.frame[run.frame.t == ONSET]
        long = onset.melt(id_vars="id", value_vars=run.layout.edges, var_name="edge")
        blocks.append(long.assign(mu=mu)[["mu", "edge", "id", "value"]])
    data = pd.concat(blocks, ignore_index=True)
    with figure_style(SCRIPT_RC):
        fig = _draw_figA7(data, mus, base.init_w)
    return FigureResult(fig, data, "AppendixFig_edgeweight_distributions" + base.suffix(ignore=("mu",)))


def _draw_figA7(data: pd.DataFrame, mus, init_w):
    fig, axs = plt.subplots(len(mus), 1, sharex=True, sharey=False, figsize=(12 / 2.54, 6 / 2.54), squeeze=False)
    axs = axs[:, 0]
    for row, (mu, ax) in enumerate(zip(mus, axs)):
        wide = data[data.mu == mu].pivot(index="id", columns="edge", values="value")
        wide = wide[list(dict.fromkeys(data.edge))]  # the CSV's edge order (the colours follow it)
        sns.kdeplot(wide, legend=False, ax=ax, alpha=0.4)
        ax.set_title(rf"$\mu={mu}$", x=0.99, ha="right", y=0.8, va="top")
        ax.set_xlabel(r"edge weights $\omega_{mn}$ at $t=100$")
        ax.set_ylabel("Density" if row == len(mus) // 2 else "")
        ax.set_yticks([])
        ax.vlines(init_w, 0, ax.get_ylim()[1], color="grey", zorder=-1, linestyles="--")
        ax.set_xlim(-6, 6)
    axs[-1].text(0.02, 0.95, r"each color = one pair of beliefs $(m,n)$", fontsize=_SMALL_FS, va="top", ha="left",
                 transform=axs[-1].transAxes)
    fig.subplots_adjust(bottom=0.15, left=0.05, right=0.97, top=0.97)
    return fig
