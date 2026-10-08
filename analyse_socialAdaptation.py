# %%
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew
import networkx as nx
import matplotlib.path as mpath
import string
import os

plt.rcParams.update({"font.size": 10})
bigfs = 9
smallfs = 7
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})


cmapS = dict(
    zip(
        [1, 2, 4, 8, 16],
        [
            (0.0000, 0.4470, 0.7410),  # | deep blue         |
            (0.8500, 0.3250, 0.0980),  # | orange            |
            (0.9290, 0.6940, 0.1250),
            (0.4940, 0.1840, 0.5560),
            (0.4660, 0.6740, 0.1880),
            (0.3010, 0.7450, 0.9330),
            (0.6350, 0.0780, 0.1840),
        ],
    )
)


def add_bracket_with_tick(fig, axs_list, mu_ax, mu_x, color="black"):
    fig.canvas.draw()  # needed to get correct positions

    def bot_center(ax):
        bb = ax.get_position()
        return bb.x0 + bb.width / 2, bb.y0

    points = [bot_center(ax) for ax in axs_list]
    xs = [p[0] for p in points]
    y_bot = points[0][1]  # all same row
    gap = -0.0  # small gap below axes
    brace_y = y_bot - gap
    tr = fig.transFigure
    x_arr = np.linspace(
        xs[0] - 0.2 * (xs[1] - xs[0]), xs[-1] + 0.2 * (xs[1] - xs[0]), 300
    )
    y_arr = brace_y - 0.02 * np.sin(np.pi * (x_arr - xs[0]) / (xs[-1] - xs[0]))
    fig.add_artist(
        plt.Line2D(
            x_arr,
            y_arr - gap,
            transform=tr,
            color=color,
            lw=0.8,
            linestyle="--",
            clip_on=False,
        )
    )
    x_mid = (xs[0] + xs[-1]) / 2
    mu_bb = mu_ax.get_position()
    mu_xlim = mu_ax.get_xlim()
    mu_fig_x = mu_bb.x0 + (mu_x - mu_xlim[0]) / (mu_xlim[1] - mu_xlim[0]) * mu_bb.width

    fig.add_artist(
        plt.Line2D(
            [x_mid, mu_fig_x],
            [brace_y - gap - 0.02, mu_bb.y1],
            transform=tr,
            color=color,
            lw=0.8,
            clip_on=False,
            linestyle="--",
        )
    )


metric2title = dict(
    x_focal=r"$x_{foc}$",
    extr_nonfoc=r"$|X_{non\text{-}foc}|$",
    n_nbs=r"$|\mathcal{K}|$",
    absOm_tot=r"BN-$|\Omega|$",
    absOm_foc=r"BN-$|\Omega_{foc}|$",
    tb_tot=r"BN-$\alpha$",
    tb_foc=r"BN-$\alpha_{foc}$",
    clust=r"BN-clust",
    bc_foc=r"BN-centr$_{foc}$",
    # bn_expected_influence = r"$\langle\delta x_{foc}\rangle$",
    Hpersfoc=r"$D_\mathrm{BN\text{-}foc}$",
    Hpersnonfoc=r"$D_\mathrm{BN\text{-}non\text{-}foc}$",
    Hsoc=r"$D_{social}$",
    # external_energy = r"$D_{ext}$",
    # energy = r"$D_{tot}$",
)
metric2titleVerb = dict(
    x_focal=r"focal belief",
    extr_nonfoc=r"extremity non-focal beliefs",
    n_nbs=r"nr social contacts",
    absOm_tot=r"connectedness BN",
    absOm_foc=r"connectedness focal BN",
    tb_tot=r"balance BN",
    tb_foc=r"balance focal BN",
    clust=r"clustering BN",
    bc_foc=r"centrality focal BN",
    # bn_expected_influence = r"expected influence focal",
    Hpersfoc=r"focal BN dissonance",
    Hpersnonfoc=r"non-focal BN dissonance",
    Hsoc=r"focal social dissonance",
    # external_energy = r"focal external dissonance",
    # energy = r"total dissonance",
)

# %%
response_cols = [
    "persistPos",
    "nonpersistPos",
    "compliant",
    "resilient",
    "resistant",
    "latecompliant",
]
belief_dimensions = [int(a) for a in range(10)]
belief_columns = [str(int(a)) for a in belief_dimensions]
edgelist = list(combinations(belief_dimensions, 2))
edges_columns = [f"w{a}{b}" for a, b in edgelist]
metric_cols = [
    "n_nbs",
    "Hpers",
    "Hpersfoc",
    "Hsoc",
    "Hext",
    "tb_tot",
    "tb_foc",
    "absOm_tot",
    "absOm_foc",
    "clust",
    "bc_foc",
    "expI",
    "x_focal",
    "extr_nonfoc",
]
meta_cols = ["t", "id"]

cmap = dict(
    zip(
        response_cols + ["NA"],
        ["#4CAF50", "#AED581", "#2196F3", "#9C27B0", "#F44336", "#90CAF9", "#9E9E9E"],
    )
)
pressures = [4]
res = []
metrics = pd.DataFrame()

link_prob = 0.1
beta = 3.0
rho = 1.0 / 3.0
init_w = 0.2
fixedBNat100 = False
param_combis = [
    [link_prob, init_w, beta, rho, eps, mu, fixedBNat100]
    for mu in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    for eps in [1.0]
]


for link_prob, init_w, beta, rho, eps, mu, fixedBNat100 in param_combis:
    print(init_w, eps, mu, fixedBNat100)
    for s in pressures:
        for seed in range(100):
            df = pd.read_csv(
                f"simOut/sim_link_prob0.10_init_w{init_w:.2f}_beta3.00_rho0.33_eps{eps:.2f}_mu{mu:.3f}{'_fixedBNat100' if fixedBNat100 else ''}_ext_strength{s}_seed{seed}.csv"
            )
            #df = df.rename(columns={"bc":"bc_foc"})
            W = df.loc[df.t == 100, edges_columns].values
            nr_neg_edges = np.sum(W.flatten() < 0)
            dists = pdist(W, metric="euclidean")
            groupishness = 0 if eps == 0 else dists.var() / dists.mean()
            var = dists.std()
            mean = dists.mean()
            k = -kurtosis(dists)

            responses = (
                df.loc[df.t == 95.5][["id"] + response_cols]
                .melt(id_vars=["id"])
                .replace({False: np.nan})
                .dropna()
            )

            vals = df.loc[df.t == 95.5][meta_cols + metric_cols]
            vals["response"] = None
            for col in response_cols:
                vals.loc[df.loc[df.t == 95.5, col] == 1, "response"] = col

            for x in ["init_w", "eps", "mu", "s", "seed"]:
                vals[x] = eval(x)
            metrics = pd.concat([metrics, vals])

            res.append(
                [seed, init_w, eps, mu, s, fixedBNat100, f"eps{eps}_mu{mu}"]
                + df.loc[df.t == 95.5][response_cols]
                .sum(axis=0)[response_cols]
                .to_list()
                + [groupishness, mean, var, k, nr_neg_edges]
                + [np.abs(df.loc[df.t == 100, edges_columns].values).mean()]
            )
res = pd.DataFrame(
    res,
    columns=["seed", "init_w", "eps", "mu", "s_ext", "fixedBNat100", "name"]
    + response_cols
    + [
        "groupishness",
        "mean",
        "var",
        "kurtosis",
        "nr_negative_edges",
        "mean_abs_edges",
    ],
)
negcols = ["compliant", "resilient", "resistant"]
relres = res[negcols].div(res[negcols].sum(axis=1), axis=0)
for c in [
    "eps",
    "mu",
    "init_w",
    "s_ext",
    "fixedBNat100",
    "seed",
    "groupishness",
    "nr_negative_edges",
]:
    relres[c] = res[c]


metrics["Hpersnonfoc"] = metrics.Hpers - metrics.Hpersfoc
metric_cols[1] = "Hpersnonfoc"
metrics = metrics.drop(columns=["Hpers"])


# %%

fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(12 / 2.54, 6 / 2.54))
for row, mu in enumerate([0.0, 0.005, 0.2]):
    df = pd.read_csv(
        f"simOut/sim_link_prob{link_prob:.2f}_init_w{init_w:.2f}_beta{beta:.2f}_rho{rho:.2f}_eps{eps:.2f}_mu{mu:.3f}{'_fixedBNat100' if fixedBNat100 else ''}_ext_strength{s}_seed{seed}.csv"
    ).query("t==100")
    sns.kdeplot(
        df[edges_columns],
        legend=False,
        ax=axs[row],
        alpha=0.4,
    )
    axs[row].set_title(
        rf"$\mu={mu}$",
        x=0.99,
        ha="right",
        y=0.8,
        va="top",
    )
    axs[row].set_xlabel(r"edge weights $\omega_{mn}$ at $t=100$")

    axs[row].set_ylabel(r"Density" if row == 1 else "")
    axs[row].set_yticks([])
    axs[row].vlines(
        init_w, 0, axs[row].get_ylim()[1], color="grey", zorder=-1, linestyles="--"
    )
    axs[row].set_xlim(-6,6)
axs[-1].text(0.02,0.95, r"each color = one pair of beliefs $(m,n)$", fontsize=smallfs, va="top", ha="left", transform=axs[-1].transAxes)
fig.subplots_adjust(bottom=0.15, left=0.05, right=0.97, top=0.97)
plt.savefig("figs/AppendixFig_edgeweight_distributions.png", dpi=600)
plt.savefig("figs/AppendixFig_edgeweight_distributions.pdf")
# %%
# %%

fig, axs = plt.subplot_mosaic(
    [
        [f"a{i}" for i in range(4)] + ["."] + [f"b{i}" for i in range(4)],
        [f"a{i}" for i in range(4, 8)] + ["."] + [f"b{i}" for i in range(4, 8)],
        [".."] * 9,
        ["metric"] * 9,
        ["mu"] * 9,
    ],
    height_ratios=[1, 1, 0.0005, 2.5, 2.5],
    width_ratios=[1] * 9,
    figsize=(16 / 2.54, 12 / 2.54),
)

G0 = nx.complete_graph(len(belief_dimensions))
pos0 = nx.circular_layout(G0)

eps = 1.0
mus_networks = [0.005, 0.2]
np.random.seed(42)
seed = 1

edge_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "edge_diverge",
    [(0.00, "#2ec4b6"), (0.50, "#d0d0d0"), (1.00, "#f4a139")],
)
norm_edge = plt.Normalize(vmin=-1.0, vmax=1.0)

# ── Example belief network panels ─────────────────────────────────────────────

relResLong = relres.melt(
    id_vars=["eps", "mu"], value_vars=negcols, value_name="count", var_name="response"
)
mus = relResLong.mu.unique().tolist()

for i in range(8):
    agent_id = np.random.randint(100)
    for mu, letter in zip(mus_networks, ["a", "b"]):
        ax = axs[letter + str(i)]
        ax.axis("off")

        df = pd.read_csv(
            f"simOut/sim_link_prob{link_prob:.2f}_init_w{init_w:.2f}_beta{beta:.2f}_rho{rho:.2f}"
            f"_eps{eps:.2f}_mu{mu:.3f}{'_fixedBNat1001.00' if fixedBNat100 else ''}"
            f"_ext_strength{s}_seed{seed}.csv"
        )

        edges = df.query(f"t == 100 and id == {agent_id}")[edges_columns].iloc[0]
        G = nx.complete_graph(10)
        nx.set_edge_attributes(G, dict(zip(edgelist, edges.values)), name="weight")

        edge_colors = [edge_cmap(norm_edge(G.edges[e]["weight"])) for e in edgelist]
        widths = [min(5, 0.8 * G.edges[e]["weight"]) for e in edgelist]

        nx.draw_networkx_edges(
            G=G,
            pos=pos0,
            edgelist=edgelist,
            edge_color=edge_colors,
            width=widths,
            ax=ax,
        )
        nx.draw_networkx_nodes(G=G, pos=pos0, node_color="grey", node_size=1, ax=ax)

        # Highlight focal node
        focal_pos = pos0[0]
        ax.plot(
            *focal_pos,
            marker="o",
            ms=3,
            markerfacecolor="None",
            markeredgecolor="k",
            markeredgewidth=1,
            zorder=-1,
        )

# ── Metric line plot ───────────────────────────────────────────────────────────

ax_metric = axs["metric"]
plt.setp(ax_metric.get_xticklabels(), visible=False)
ax_metric.sharex(axs["mu"])

res2 = res.loc[res.eps == eps].copy()
res2["mu"] = pd.Categorical(res2["mu"], categories=mus, ordered=True)
res2["mu_idx"] = res2["mu"].cat.codes
res2.loc[res2.eps == eps, "meanR"] = res2.loc[res2.eps == eps, "mean"].transform(
    lambda x: (x - x.mean()) / x.std()
)

sns.lineplot(
    res2.loc[res2.eps == eps],
    x="mu_idx",
    y="meanR",
    color="#FF69B4",
    marker="o",
    errorbar="sd",
    ax=ax_metric,
    label="heterogeneity BNs",
    err_kws={"alpha": 0.1},
)

selected_metrics = ["absOm_foc", "tb_foc", "clust"]

metricsByMu = (
    metrics.loc[
        (metrics.eps == eps) & (metrics.init_w == init_w) & (metrics.s == s),
        ["mu"] + metric_cols + ["seed", "id"],
    ]
    .assign(mu=lambda df: pd.Categorical(df["mu"], categories=mus, ordered=True))
    .melt(id_vars=["mu"], value_vars=metric_cols, var_name="metric", value_name="value")
)
metricsByMu["mu"] = pd.Categorical(metricsByMu["mu"], categories=mus, ordered=True)

df_plot = metricsByMu[metricsByMu["metric"].isin(selected_metrics)].copy()
df_plot["value_norm"] = df_plot.groupby("metric")["value"].transform(
    lambda x: (x - x.mean()) / x.std()
)
df_plot["mu_idx"] = df_plot["mu"].cat.codes

for m in selected_metrics:
    sns.lineplot(
        data=df_plot[df_plot["metric"] == m],
        x="mu_idx",
        y="value_norm",
        marker="o",
        errorbar="sd",
        ax=ax_metric,
        label=metric2titleVerb[m],
        err_kws={"alpha": 0.1},
    )

ax_metric.legend(loc="lower left", ncols=4)
ax_metric.set_ylim(-2, 2)
ax_metric.set_yticks([-2, -1, 0, 1, 2])
ax_metric.set_ylabel("metric value (z-score)")

# ── Response frequency box/strip plot ─────────────────────────────────────────

ax_mu = axs["mu"]

relResLong_filtered = relResLong.loc[relResLong.eps == eps].copy()
relResLong_filtered["mu"] = pd.Categorical(
    relResLong_filtered["mu"], categories=mus, ordered=True
)


sns.boxplot(
    relResLong_filtered,
    x="mu",
    y="count",
    hue="response",
    palette=cmap,
    fliersize=0,
    fill=True,
    linewidth=0.0,
    whis=0,
    ax=ax_mu,
)

sns.stripplot(
    relResLong_filtered,
    x="mu",
    y="count",
    hue="response",
    palette=cmap,
    dodge=True,
    size=1.5,
    edgecolor="white",
    linewidth=0.2,
    legend=False,
    ax=ax_mu,
)


ax_mu.get_legend().set_title("")
ax_mu.get_legend().set_loc("center right")
ax_mu.set_ylabel("relative response frequency")
ax_mu.set_xlabel(r"social adaptation $\mu$ (log-scale)")
ax_mu.set_ylim(-0.05, 1.05)
ax_mu.set_xlim(-0.5, len(mus) - 0.5)
ax_mu.set_yticklabels([rf"${int(x * 100):d}\,\%$" for x in ax_mu.get_yticks()])

for i in [0.5, 2.5, 4.5, 6.5, 8.5]:
    ax_mu.fill_between(
        [i, i + 1], [-0.05, -0.05], [1.05, 1.05], color="gainsboro", zorder=-1
    )
for i in [0.5, 2.5, 4.5, 6.5, 8.5]:
    ax_metric.fill_between([i, i + 1], [-2, -2], [2, 2], color="gainsboro", zorder=-1)
# ── Bracket annotations ────────────────────────────────────────────────────────

axs[".."].axis("off")

fig.subplots_adjust(left=0.1, bottom=0.085, top=0.99, right=0.99)

add_bracket_with_tick(
    fig,
    [axs[f"a{i}"] for i in range(4, 8)],
    ax_metric,
    mu_x=mus.index(mus_networks[0]),
    color="darkgrey",
)
add_bracket_with_tick(
    fig,
    [axs[f"b{i}"] for i in range(4, 8)],
    ax_metric,
    mu_x=mus.index(mus_networks[1]),
    color="darkgrey",
)

# ── Figure text labels ─────────────────────────────────────────────────────────

fig.text(
    0.54,
    0.88,
    "example\nbelief\nnetworks",
    fontdict={"size": bigfs, "weight": "bold"},
    transform=fig.transFigure,
    ha="center",
    va="center",
)
fig.text(
    0.29,
    0.87,
    rf"with $\mu={mus_networks[0]}$",
    fontdict={"size": bigfs, "weight": "bold"},
    transform=fig.transFigure,
    ha="center",
    va="center",
)
fig.text(
    0.795,
    0.87,
    rf"with $\mu={mus_networks[1]}$",
    fontdict={"size": bigfs, "weight": "bold"},
    transform=fig.transFigure,
    ha="center",
    va="center",
)
fig.text(
    0.49,
    0.8,
    "focal\nbelief",
    fontdict={"size": smallfs, "weight": "normal"},
    transform=fig.transFigure,
    ha="left",
    va="center",
)

# ── Panel letter labels ────────────────────────────────────────────────────────

for n, key in enumerate(["a0", "b0", "metric", "mu"]):
    axs[key].text(
        -0.1 if n < 2 else 0.01,
        0.975,
        string.ascii_uppercase[n],
        fontsize=12,
        fontdict={"weight": "bold"},
        va="top",
        ha="left",
        transform=axs[key].transAxes,
    )

fname = f"figs/fig6_socialAdaptation"
if not os.path.isdir(fname.split("/")[0]):
    os.mkdir(fname.split("/")[0])
plt.savefig(fname+".png", dpi=600)
plt.savefig(fname+".pdf", )


# %%
plt.figure()
sns.lineplot(
    res.loc[res.eps == eps],
    x="mu",
    color="blue",
    y="compliant",
    marker="o",
    errorbar="sd",
    label="compliant",
)
sns.lineplot(
    res.loc[res.eps == eps],
    x="mu",
    color="purple",
    y="resilient",
    marker="o",
    errorbar="sd",
    label="resilient",
)
sns.lineplot(
    res.loc[res.eps == eps],
    x="mu",
    color="red",
    y="resistant",
    marker="o",
    errorbar="sd",
    label="resistant",
)

sns.lineplot(
    metrics.loc[metrics.eps == eps],
    x="mu",
    color="k",
    y="tb_foc",
    marker="o",
    errorbar="sd",
    label="tb_tot",
)

plt.legend()
plt.xlim(0, 0.1)


# %%
