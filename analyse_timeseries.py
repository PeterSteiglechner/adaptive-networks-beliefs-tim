# %%
import matplotlib as mpl
import numpy as np
import pandas as pd
from itertools import combinations
import string
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
import string
import os

sns.set_style("ticks", {"axes.linewidth": 0.5})
smallfs = 8
bigfs = 9
plt.rc("font", family="sans-serif")
plt.rc("font", size=smallfs)  # Ticklabels, legend labels, etc.
plt.rc("axes", labelsize=bigfs)  # Axis labels
plt.rc("axes", titlesize=bigfs)  # Titles
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})

cmapE = mpl.colors.LinearSegmentedColormap.from_list(
    "edge_diverge",
    [
        (0.00, "#2ec4b6"),  # -1 : teal
        (0.50, "#d0d0d0"),  #  0 : light grey
        (1.00, "#f4a139"),  # +1 : amber-orange
    ],
)
normE = plt.Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap("coolwarm")
norm = plt.Normalize(vmin=-1, vmax=1)

names = {
    (0.2, 0.0, 0.0, False): r"staticlow",
    (0.8, 0.0, 0.0, False): r"statichigh",
    (0.2, 1.0, 0.0, False): r"adaptive",
    (0.2, 1.0, 0.0, True): r"adaptive2static",
}
namesTex = {
    (0.2, 0.0, 0.0, False): r"static ($\omega_0=0.2$)",
    (0.8, 0.0, 0.0, False): r"static ($\omega_0=0.8$)",
    (0.2, 1.0, 0.0, False): r"adaptive",
    (0.2, 1.0, 0.0, True): r"adaptive$\rightarrow$static",
}


# %%
init_w = 0.2
eps = 1.0
mu = 0.0
fixedBNat100 = False
seed = 2
s = 4
df = pd.read_csv(
    f"simOut/detailed/sim_link_prob0.10_init_w{init_w:.2f}_beta3.00_rho0.33_eps{eps:.2f}_mu{mu:.3f}_ext_strength{s}_seed{seed}_detailed.csv"
)

# %%
belief_dimensions = [int(a) for a in range(10)]
belief_columns = [str(int(a)) for a in belief_dimensions]
edgelist = list(combinations(belief_dimensions, 2))
edges_columns = [f"w{a}{b}" for a, b in edgelist]
focal = 0

# beliefs = df[["id", "t"] + ["0"]].pivot_table(index="t", columns="id", values="0")
# edges = df.loc[df.t == 100, ["id"] + edges_columns].set_index("id")


# beliefs.plot(color="grey", legend=False, lw=0.2)

# for i in [9, 98]:
#     beliefs[i].plot(legend=False, lw=1)
#     ag_b = df.query(f"id=={i} & t==100.")[belief_columns]
#     ag_e = edges.loc[i]
#     G = nx.complete_graph(10)
#     G.nodes()
#     nx.set_node_attributes(G, ag_b, name="value")
#     nx.set_edge_attributes(G, dict(zip(edgelist, ag_e.values)), name="weight")


# %%
def plot_BN_ax(
    ag_b,
    ag_e,
    ax,
    intStart=100,
    intEnd=150,
    scaleE=1,
    scaleN=100,
    highlightfocal=True,
    layout="spring",
    seed=1,
):
    G = nx.complete_graph(10)
    nx.set_node_attributes(G, ag_b, name="value")
    nx.set_edge_attributes(G, dict(zip(edgelist, ag_e.values)), name="weight")

    widths = [scaleE * G.edges[e]["weight"] for e in edgelist]
    edge_colors = [cmapE(normE(G.edges[e]["weight"])) for e in edgelist]
    amin, amax = (0.5, 0.95)
    alphas = [
        np.clip(amin + (amax - amin) * np.abs(G.edges[e]["weight"]) / 5, 0, 1)
        for e in edgelist
    ]
    if layout == "circle":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.25)
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        edge_color=edge_colors,
        width=widths,
        alpha=alphas,
        edgelist=edgelist,
        ax=ax,
    )
    node_colors = [cmap(norm(ag_b[dim])) for dim in belief_columns]
    nx.draw_networkx_nodes(
        G=G,
        pos=pos,
        nodelist=belief_dimensions,
        node_color=node_colors,
        node_size=scaleN,
        ax=ax,
    )
    nodelistMod = [int(dim) for dim in belief_columns if abs(ag_b[dim]).values < 0.7]
    nodelistEx = [int(dim) for dim in belief_columns if abs(ag_b[dim]).values >= 0.7]
    nodelabelsMod = [dim if not dim == focal else r"foc" for dim in nodelistMod]
    nodelabelsEx = [dim if not dim == focal else r"foc" for dim in nodelistEx]
    nx.draw_networkx_labels(
        G=G,
        pos=pos,
        labels=dict(zip(nodelistEx, nodelabelsEx)),
        font_size=7,
        font_color="whitesmoke",
        ax=ax,
    )
    nx.draw_networkx_labels(
        G=G,
        pos=pos,
        labels=dict(zip(nodelistMod, nodelabelsMod)),
        font_size=7,
        font_color="k",
        ax=ax,
    )
    if highlightfocal:
        ax.plot(
            pos[focal][0],
            pos[focal][1],
            marker="o",
            ms=12,
            markerfacecolor="None",
            markeredgecolor="gold",
            markeredgewidth=4,
            zorder=-1,
        )
    if intStart is not None:
        if t > intStart and t <= intEnd:
            ax.text(
                0.5,
                1,
                f"pressure",
                va="top",
                ha="center",
                transform=ax.transAxes,
                color="orange",
            )
    ax.axis(False)
    ax.set_facecolor((0, 0, 0, 0))
    y0, yh = ax.get_ylim()
    ax.set_ylim(y0 * 1.1, yh * 1.1)
    x0, xh = ax.get_xlim()
    ax.set_xlim(x0 * 1.1, xh * 1.1)
    return ax


# %%
beliefs = df[["id", "t"] + ["0"]].pivot_table(index="t", columns="id", values="0")
T0 = 0
Tfinal = 100
for T in [Tfinal]:
    window = 0
    for adaptive, w0 in zip([True], [0.2]):
        fig, axs = plt.subplot_mosaic(
            [
                [
                    "t",
                    "hist",
                    ".",
                    f"a{T0}",
                    ".",
                    f"a{T}",
                ],
                ["t", "hist", ".", f"b{T0}", ".", f"b{T}"],
                [".", ".", ".", f"b{T0}", ".", f"b{T}"],
            ],
            width_ratios=[0.45, 0.1, 0.04, 0.2, 0.06, 0.2],
            height_ratios=[1, 0.7, 0.3],
            figsize=(16 / 2.54, 7 / 2.54),
        )

        dim = 0
        final_values = df.loc[df.t == T, ["id"] + belief_columns]
        dff = beliefs.reset_index().melt(
            id_vars=["t"],
        )
        if window > 0:
            dff["belief_smooth"] = dff.groupby("id")["value"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        else:
            dff["belief_smooth"] = dff["value"]
        df_pivot = dff.pivot(index="t", columns="id", values="belief_smooth")

        colors = ["#27ae60", "#9b59b6"]  # plt.get_cmap("Dark2")

        final_focal_values = dff.loc[dff.t == T, ["id", "belief_smooth"]]
        tb = (
            df.loc[df.t == 100]
            .set_index("id")
            .absOm_tot[
                final_focal_values.loc[final_focal_values.belief_smooth < 0, "id"]
            ]
        )
        ags = [
            61,
            79,
        ]
        for n, ag_i in enumerate(ags):
            df_pivot = df_pivot.loc[df_pivot.index <= T]
            if n == 0:
                for aaa in df_pivot.columns:
                    df_pivot[aaa].plot(
                        ax=axs["t"],
                        lw=0.4,
                        alpha=0.2,
                        legend=False,
                        color=(
                            0.8 - 0.5 * aaa / 100,
                            0.8 - 0.5 * aaa / 100,
                            0.8 - 0.5 * aaa / 100,
                            1,
                        ),
                    )
                axs["t"].set_xlabel("time")
                axs["t"].set_ylabel("focal belief $x_\mathrm{foc}$", va="center")
                axs["t"].set_clip_on(False)
                axs["t"].set_xlim(0, Tfinal)
                axs["t"].set_ylim(-1.0, 1.0)
                axs["t"].set_yticks([-1, 0, 1])

                if s > 0:
                    y0, y1 = axs["t"].get_ylim()
                    int_colors = dict(
                        zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4])
                    )

                    axs["t"].fill_between(
                        [101, 150],
                        [-1] * 2,
                        [1] * 2,
                        color="red",
                        alpha=int_colors[s],
                        zorder=-1,
                        lw=0,
                    )
                ax_hist = axs["hist"]
                ax_hist.sharey(axs["t"])
                y0, y1 = axs["t"].get_ylim()
                bins = np.linspace(-1.001, 1.001, 21)
                sns.histplot(
                    final_focal_values,
                    y="belief_smooth",
                    bins=bins,
                    orientation="horizontal",
                    color="grey",
                    edgecolor="black",
                    ax=ax_hist,
                )
                cmap = mpl.cm.coolwarm
                norm = mpl.colors.Normalize(vmin=y0, vmax=y1)
                for patch in ax_hist.patches:
                    y = patch.get_y() + patch.get_height() / 2
                    patch.set_facecolor(cmap(norm(y)))
                ax_hist.set_ylabel("")
                ax_hist.grid(False)
                ax_hist.set_clip_on(False)
                ax_hist.axis("off")
                # ax_hist.text(
                #     1,
                #     0.5,
                #     f"Histogram\nat $t={t}$",
                #     ha="right",
                #     va="center",
                #     # rotation=270,
                #     fontsize=smallfs,
                #     transform=ax_hist.transAxes,
                # )
            df_pivot.T.loc[ag_i].plot(
                ax=axs["t"],
                lw=3,
                ls="-",
                color=colors[n],
                alpha=1,
                legend=False,
                clip_on=False,
            )
            for t in [T0, T]:
                ag_b_t = df.query(f"id=={ag_i} & t=={t}")[belief_columns]
                edges = df.loc[df.t == t, ["id"] + edges_columns].set_index("id")
                ag_e_t = edges.loc[ag_i]
                ax_net = axs[f"{string.ascii_lowercase[n]}{t}"]
                plot_BN_ax(
                    ag_b_t,
                    ag_e_t,
                    ax_net,
                    intStart=100,
                    intEnd=150,
                    scaleN=80,
                    scaleE=1.4,
                    highlightfocal=True,
                    layout="spring" if int(adaptive) and t > 0 else "circle",
                    seed=1 + 3 * 32 if ag_i == 61 else 2 + 3 * 32,
                )
        for l in ["a", "b"]:
            arrow = patches.ConnectionPatch(
                (1.1, 0.5),
                (0, 0.5),
                coordsA=axs[l + f"{T0}"].transAxes,
                coordsB=axs[l + f"{T}"].transAxes,
                color="black",
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1,
            )
            axs[l + f"{T0}"].text(
                1.07,
                0.02,
                rf"$t={T0}$",
                ha="right",
                va="bottom",
                transform=axs[l + f"{T0}"].transAxes,
                fontsize=smallfs,
                rotation=40,
            )
            axs[l + f"{T}"].text(
                1.07,
                0.02,
                rf"$t={T}$",
                ha="right",
                va="bottom",
                transform=axs[l + f"{T}"].transAxes,
                fontsize=smallfs,
                rotation=40,
            )
            fig.patches.append(arrow)

        fig.subplots_adjust(
            wspace=0.05, hspace=0.05, top=0.91, bottom=0.02, left=0.07, right=0.99
        )

        for n, (i, y) in enumerate(zip(ags, [0.93, 0.43])):
            bboxprops = dict(
                boxstyle="round",
                facecolor=colors[n],
                edgecolor=colors[n],
                alpha=1,
            )
            fig.text(
                0.77,
                y,
                f"agent {i}",
                fontdict={"fontsize": smallfs, "bbox": bboxprops, "color": "white"},
            )
        # fig.set_facecolor("pink")
        for n, ax in enumerate([axs["t"], axs[f"a{T0}"], axs[f"b{T0}"]]):
            ax.text(
                -0.12 if n == 0 else 0,
                0.975,
                string.ascii_uppercase[n],
                fontsize=12,
                fontdict={"weight": "bold"},
                va="top",
                ha="left",
                transform=ax.transAxes,
            )
        axs["t"].set_xlabel("time $t$")
        print(ags, df.loc[df.t == 100].set_index("id").tb_tot[ags])
        print(ags, df.loc[df.t == 100].set_index("id").tb_foc[ags])
        for nn, n in enumerate([-1, 1]):
            axs["b0"].text(
                -0.2,
                0.0 + 0.11 * nn,
                r"$\omega_{ij}>0$" if n == 1 else r"$\omega_{ij}<0$",
                ha="left",
                va="bottom",
                rotation=0,
                fontsize=smallfs,
                transform=axs["b0"].transAxes,
                color=cmapE(normE(n)),
            )
        ax_hist.text(
            0.5,
            0.5,
            "focal belief distribution\n" + rf"at $t={t}$",
            ha="center",
            va="center",
            rotation=270,
            fontsize=smallfs,
            transform=ax_hist.transAxes,
        )
        axs["t"].set_title(
            f"example simulation with {namesTex[(init_w,eps,mu,False)]} belief networks",
            fontsize=smallfs,
        )
    fname = (
        f"figs/fig2"
        + ("_" + names[(init_w, eps, mu, fixedBNat100)])
        + f"_t{T}"
    )
    if not os.path.isdir(fname.split("/")[0]):
        os.mkdir(fname.split("/")[0])
    print(fname)
    plt.savefig(fname + ".png", dpi=600)
    plt.savefig(fname + ".pdf")

    print(
        f"t={T}",
        np.sign(final_focal_values["belief_smooth"]).value_counts().sort_index(),
    )
    print(f"t={95.5}", np.sign(final_focal_values).value_counts().sort_index())
    print("t=100: Hpers", df.loc[df.t == 100].set_index("id").loc[ags]["Hpers"])

# %%


