"""Shared look of the figures: rc settings, colours, labels and the belief-network drawing helper."""

import math
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

SMALL_FS = 8
BIG_FS = 9

# The old scripts' rc settings, applied only while a figure is drawn (`figure_style`), not globally.
RC = {
    **sns.axes_style("ticks", {"axes.linewidth": 0.5}),
    "font.family": "sans-serif",
    "font.size": BIG_FS,
    "axes.labelsize": BIG_FS,
    "axes.titlesize": BIG_FS,
    "legend.fontsize": SMALL_FS,
}


@contextmanager
def figure_style():
    """The paper's rc settings, with pyplot's interactive mode off so a figure function shows nothing."""
    with mpl.rc_context(RC), plt.ioff():
        yield


# BN configs: the file-name token (also the figure-name part) → the label in titles and legends.
CONFIG_LABELS = {
    "staticlow": r"static ($\omega_0=0.2$)",
    "statichigh": r"static ($\omega_0=0.8$)",
    "adaptive2static": r"adaptive$\rightarrow$static",
    "adaptive": r"adaptive",
}

# Edge weights: teal (−1) → light grey (0) → amber (+1). Beliefs: coolwarm on [−1, 1].
EDGE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "edge_diverge", [(0.00, "#2ec4b6"), (0.50, "#d0d0d0"), (1.00, "#f4a139")])
EDGE_NORM = mpl.colors.Normalize(vmin=-1, vmax=1)
BELIEF_CMAP = mpl.colormaps["coolwarm"]
BELIEF_NORM = mpl.colors.Normalize(vmin=-1, vmax=1)

EXAMPLE_AGENT_COLORS = ("#27ae60", "#9b59b6")  # Fig. 2's two highlighted agents
PRESSURE_WINDOW = (101, 150)
FOCAL = 0


def pressure_alpha(s: float) -> float:
    """Shading of the pressure window: 0.1, 0.175, 0.25, 0.325, 0.4 for s = 1, 2, 4, 8, 16 (log-linear in s)."""
    return float(np.clip(0.1 + 0.075 * math.log2(s), 0.05, 0.6))


def draw_bn(ax, beliefs: pd.Series, edges: pd.Series, *, t=None, layout="spring", seed=1, scale_e=1.0,
            scale_n=100, highlight_focal=True):
    """One agent's belief network on `ax`: nodes coloured by belief, edges by weight (the old `plot_BN_ax`).

    `beliefs` is indexed by the belief columns `"0"…"M-1"`, `edges` by the edge columns `w{m}{n}`. `layout` is
    `"spring"` (weighted, seeded with `seed`) or `"circle"`. A `t` inside the pressure window adds a label.
    """
    M = len(beliefs)
    pairs = list(zip(*np.triu_indices(M, k=1)))
    pairs = [(int(m), int(n)) for m, n in pairs]
    G = nx.complete_graph(M)
    nx.set_edge_attributes(G, dict(zip(pairs, edges.to_numpy(float))), name="weight")
    weights = np.array([G.edges[e]["weight"] for e in pairs])
    if layout == "circle":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.25)
    nx.draw_networkx_edges(G, pos=pos, edgelist=pairs, width=list(scale_e * weights),
                           edge_color=[EDGE_CMAP(EDGE_NORM(w)) for w in weights],
                           alpha=list(np.clip(0.5 + 0.45 * np.abs(weights) / 5, 0, 1)), ax=ax)
    values = beliefs.to_numpy(float)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=list(range(M)),
                           node_color=[BELIEF_CMAP(BELIEF_NORM(v)) for v in values], node_size=scale_n, ax=ax)
    label = {m: ("foc" if m == FOCAL else m) for m in range(M)}
    extreme = np.abs(values) >= 0.7
    for nodes, colour in (([m for m in range(M) if extreme[m]], "whitesmoke"),
                          ([m for m in range(M) if not extreme[m]], "k")):
        nx.draw_networkx_labels(G, pos=pos, labels={m: label[m] for m in nodes}, font_size=7, font_color=colour, ax=ax)
    if highlight_focal:
        ax.plot(*pos[FOCAL], marker="o", ms=12, markerfacecolor="None", markeredgecolor="gold", markeredgewidth=4,
                zorder=-1)
    if t is not None and PRESSURE_WINDOW[0] <= t <= PRESSURE_WINDOW[1]:
        ax.text(0.5, 1, "pressure", va="top", ha="center", transform=ax.transAxes, color="orange")
    ax.axis(False)
    ax.set_facecolor((0, 0, 0, 0))
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 * 1.1, y1 * 1.1)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 * 1.1, x1 * 1.1)
    return ax
