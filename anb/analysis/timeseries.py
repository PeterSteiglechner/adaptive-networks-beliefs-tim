"""Fig. 2 (and A1, A2): an example simulation's focal-belief trajectories and two agents' belief networks.

Reproduces `analyse_timeseries.py`. The figure-name part, the title and the grey levels come from the config and
the data instead of hard-coded values.
"""

import string
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches

from .index import index_folder
from .load import Loader, Run
from .output import FigureResult
from .select import CONFIG_FIELDS, PAPER, Baseline, Cell, ExampleAgentFallback, select
from .style import (CONFIG_LABELS, EDGE_CMAP, EDGE_NORM, EXAMPLE_AGENT_COLORS, FOCAL, PRESSURE_WINDOW, SMALL_FS,
                    BELIEF_CMAP, draw_bn, figure_style, pressure_alpha)

T0, T1 = 0, 100  # the two BN snapshots; T1 is also the end of the plotted trajectories
STATS = ("tb_tot", "tb_foc", "Hpers")
FALLBACK_SEED = 0


def _example_run(sim_dir, *, item, base, config, seed, s, agents, loader, index, columns=()):
    """The detailed run of `config` derived from `base`, and the example agents (missing ones replaced).

    `None` (after a `FigureNotApplicable`) if the run isn't in the folder.
    """
    index = index_folder(sim_dir) if index is None else index
    cell = Cell(f"detailed {config} s={s} seed={seed}", base.config(config), s, detailed=True, seeds=(seed,))
    selected = select(index, [cell], item=item, paper_seeds=1)
    if selected is None:
        return None
    path = selected[cell.label].path.iloc[0]
    run = (loader or Loader()).load(path, t=range(T0, T1 + 1), columns=columns, beliefs=True, edges=True)
    return run, _resolve_agents(run, agents, seed=FALLBACK_SEED)


def _resolve_agents(run: Run, agents, *, seed) -> list[int]:
    """`agents`, with each one that's not in the run replaced by a seeded random other agent, with a warning."""
    ids = np.sort(run.frame.id.unique())
    rng = np.random.default_rng(seed)
    resolved = [int(a) if a in ids else None for a in agents]
    for k, agent in enumerate(agents):
        if resolved[k] is None:
            free = [i for i in ids if i not in resolved]
            resolved[k] = int(rng.choice(free))
            warnings.warn(f"example agent {agent} isn't in {run.path.name} (N = {run.layout.N}); using agent "
                          f"{resolved[k]} instead", ExampleAgentFallback, stacklevel=4)
    return resolved


def fig2(sim_dir, *, base: Baseline = PAPER, config="adaptive", seed=2, s=4, agents=(61, 79), loader=None,
         index=None) -> FigureResult | None:
    """Fig. 2 (A1 with `config="staticlow"`, A2 with `"statichigh"`): the detailed run of `config` at `s`, `seed`.

    Left, every agent's focal belief up to t = 100 (grey by agent id) with the two `agents` highlighted and the
    focal-belief histogram at t = 100. Right, the two agents' belief networks at t = 0 and t = 100. An example agent
    missing from the run is replaced by a seeded random one (`ExampleAgentFallback`).

    `data` is tidy: `quantity` (`focal` for every agent and t ≤ 100, `belief`/`edge` for the example agents at
    t = 0, 100), `agent`, `t`, `item` (the CSV column) and `value`. `index` is a ready `index_folder(sim_dir)`.
    """
    if len(agents) != 2:
        raise ValueError(f"fig2 shows two example agents, got {len(agents)}")
    example = _example_run(sim_dir, item="fig2", base=base, config=config, seed=seed, s=s, agents=agents,
                           loader=loader, index=index)
    if example is None:
        return None
    run, agents = example
    with figure_style():
        fig = _draw(run, agents, config, s)
    name = f"fig2_{config}_t{T1}" + base.suffix(ignore=CONFIG_FIELDS)
    return FigureResult(fig, _data(run, agents), name)


def _data(run: Run, agents) -> pd.DataFrame:
    frame = run.frame
    focal = frame[["t", "id", str(FOCAL)]].rename(columns={"id": "agent", str(FOCAL): "value"})
    focal.insert(0, "quantity", "focal")
    focal.insert(3, "item", str(FOCAL))
    blocks = [focal]
    snapshots = frame[frame.t.isin([T0, T1]) & frame.id.isin(agents)]
    for quantity, columns in (("belief", run.layout.beliefs), ("edge", run.layout.edges)):
        long = snapshots.melt(id_vars=["t", "id"], value_vars=columns, var_name="item", value_name="value")
        long = long.rename(columns={"id": "agent"})
        long.insert(0, "quantity", quantity)
        blocks.append(long)
    data = pd.concat(blocks, ignore_index=True)[["quantity", "agent", "t", "item", "value"]]
    return data.sort_values(["quantity", "agent", "t"], kind="stable", ignore_index=True)


def _draw(run: Run, agents, config, s):
    frame, N = run.frame, run.layout.N
    focal = frame.pivot_table(index="t", columns="id", values=str(FOCAL))
    fig, axs = plt.subplot_mosaic(
        [["t", "hist", ".", f"a{T0}", ".", f"a{T1}"],
         ["t", "hist", ".", f"b{T0}", ".", f"b{T1}"],
         [".", ".", ".", f"b{T0}", ".", f"b{T1}"]],
        width_ratios=[0.45, 0.1, 0.04, 0.2, 0.06, 0.2],
        height_ratios=[1, 0.7, 0.3],
        figsize=(16 / 2.54, 7 / 2.54),
    )
    ax_t, ax_hist = axs["t"], axs["hist"]
    for agent in focal.columns:
        grey = 0.8 - 0.5 * agent / N
        focal[agent].plot(ax=ax_t, lw=0.4, alpha=0.2, legend=False, color=(grey, grey, grey, 1))
    ax_t.set_clip_on(False)
    ax_t.set_xlim(T0, T1)
    ax_t.set_ylim(-1.0, 1.0)
    ax_t.set_yticks([-1, 0, 1])
    ax_t.set_ylabel(r"focal belief $x_\mathrm{foc}$", va="center")
    if s > 0:
        ax_t.fill_between(list(PRESSURE_WINDOW), [-1] * 2, [1] * 2, color="red", alpha=pressure_alpha(s), zorder=-1,
                          lw=0)

    ax_hist.sharey(ax_t)
    final = pd.DataFrame({"focal": focal.loc[T1]})
    sns.histplot(final, y="focal", bins=np.linspace(-1.001, 1.001, 21), color="grey", edgecolor="black", ax=ax_hist)
    y0, y1 = ax_t.get_ylim()
    hist_norm = plt.Normalize(vmin=y0, vmax=y1)
    for patch in ax_hist.patches:
        patch.set_facecolor(BELIEF_CMAP(hist_norm(patch.get_y() + patch.get_height() / 2)))
    ax_hist.set_ylabel("")
    ax_hist.grid(False)
    ax_hist.set_clip_on(False)
    ax_hist.axis("off")
    ax_hist.text(0.5, 0.5, "focal belief distribution\n" + rf"at $t={T1}$", ha="center", va="center", rotation=270,
                 fontsize=SMALL_FS, transform=ax_hist.transAxes)

    for n, agent in enumerate(agents):
        focal[agent].plot(ax=ax_t, lw=3, ls="-", color=EXAMPLE_AGENT_COLORS[n], alpha=1, legend=False, clip_on=False)
        for t in (T0, T1):
            row = frame[(frame.id == agent) & (frame.t == t)].iloc[0]
            # The old script draws t = 0 on a circle and later snapshots with a spring layout, for every config.
            draw_bn(axs[f"{string.ascii_lowercase[n]}{t}"], row[run.layout.beliefs], row[run.layout.edges], t=t,
                    layout="spring" if t > 0 else "circle", seed=1 + n + 3 * 32, scale_e=1.4, scale_n=80)

    for row in ("a", "b"):
        start, end = axs[f"{row}{T0}"], axs[f"{row}{T1}"]
        fig.patches.append(patches.ConnectionPatch((1.1, 0.5), (0, 0.5), coordsA=start.transAxes,
                                                   coordsB=end.transAxes, color="black", arrowstyle="-|>",
                                                   mutation_scale=10, linewidth=1))
        for ax, t in ((start, T0), (end, T1)):
            ax.text(1.07, 0.02, rf"$t={t}$", ha="right", va="bottom", transform=ax.transAxes, fontsize=SMALL_FS,
                    rotation=40)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.91, bottom=0.02, left=0.07, right=0.99)

    for n, (agent, y) in enumerate(zip(agents, (0.93, 0.43))):
        colour = EXAMPLE_AGENT_COLORS[n]
        fig.text(0.77, y, f"agent {agent}", fontdict={"fontsize": SMALL_FS, "color": "white",
                                                        "bbox": dict(boxstyle="round", facecolor=colour,
                                                                     edgecolor=colour, alpha=1)})
    for n, ax in enumerate((ax_t, axs[f"a{T0}"], axs[f"b{T0}"])):
        ax.text(-0.12 if n == 0 else 0, 0.975, string.ascii_uppercase[n], fontsize=12, fontdict={"weight": "bold"},
                va="top", ha="left", transform=ax.transAxes)
    for k, sign in enumerate((-1, 1)):
        axs[f"b{T0}"].text(-0.2, 0.11 * k, r"$\omega_{ij}>0$" if sign == 1 else r"$\omega_{ij}<0$", ha="left",
                           va="bottom", fontsize=SMALL_FS, transform=axs[f"b{T0}"].transAxes,
                           color=EDGE_CMAP(EDGE_NORM(sign)))
    # The old title keys on (init_w, eps, mu, False), so adaptive→static runs are titled "adaptive" (up to t = 100
    # they are).
    title = CONFIG_LABELS["adaptive" if config == "adaptive2static" else config]
    ax_t.set_title(f"example simulation with {title} belief networks", fontsize=SMALL_FS)
    ax_t.set_xlabel("time $t$")  # last: the pandas plots above relabel the x axis with the index name
    return fig


def example_agent_stats(sim_dir, *, base: Baseline = PAPER, config="adaptive", seed=2, s=4, agents=(61, 79),
                        loader=None, index=None) -> pd.DataFrame | None:
    """`tb_tot`, `tb_foc` and `Hpers` of Fig. 2's example agents at t = 100, indexed by agent."""
    example = _example_run(sim_dir, item="example_agent_stats", base=base, config=config, seed=seed, s=s,
                           agents=agents, loader=loader, index=index, columns=STATS)
    if example is None:
        return None
    run, agents = example
    at = run.frame[run.frame.t == T1].set_index("id")
    return at.loc[agents, list(STATS)].rename_axis("agent")


def focal_sign_counts(sim_dir, *, base: Baseline = PAPER, config="adaptive", seed=2, s=4, loader=None,
                      index=None) -> pd.DataFrame | None:
    """How many agents hold a negative, zero or positive focal belief at t = 100 in Fig. 2's run (`count` by sign)."""
    example = _example_run(sim_dir, item="focal_sign_counts", base=base, config=config, seed=seed, s=s, agents=(),
                           loader=loader, index=index)
    if example is None:
        return None
    run, _ = example
    signs = np.sign(run.frame.loc[run.frame.t == T1, str(FOCAL)]).astype(int)
    counts = signs.value_counts().reindex([-1, 0, 1], fill_value=0)
    return counts.rename_axis("sign").to_frame("count")
