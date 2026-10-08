# %%
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.spatial.distance import pdist
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
cmap = dict(
    zip(
        response_cols + ["NA"],
        ["#4CAF50", "#AED581", "#2196F3", "#9C27B0", "#F44336", "#90CAF9", "#9E9E9E"],
    )
)
names = {
    (0.2, 0.0, 0.0, False): r"staticlow",
    (0.8, 0.0, 0.0, False): r"statichigh",
    (0.2, 1.0, 0.0, True): r"adaptive2static",
    (0.2, 1.0, 0.0, False): r"adaptive",
}
namesTex = {
    (0.2, 0.0, 0.0, False): r"static ($\omega_0=0.2$)",
    (0.8, 0.0, 0.0, False): r"static ($\omega_0=0.8$)",
    (0.2, 1.0, 0.0, True): r"adaptive$\rightarrow$static",
    (0.2, 1.0, 0.0, False): r"adaptive",
}
# %%

s_exts = [0, 1, 2, 4, 8, 16]
seeds = list(range(20))
res = []
mean_absedges = []
param_combis = [
    (0.2, 0.0, 0.0, False),
    (0.8, 0.0, 0.0, False),
    (0.2, 1.0, 0.0, False),
    (0.2, 1.0, 0.0, True),
]  # init_w, eps, mu, fixedBNatt=100
for init_w, eps, mu, fixedBNat100 in param_combis:
    for s in s_exts:
        for seed in seeds:
            df = pd.read_csv(
                f"simOut/sim_link_prob0.10_init_w{init_w:.2f}_beta3.00_rho0.33_eps{eps:.2f}_mu{mu:.3f}{'_fixedBNat100' if fixedBNat100 else ''}_ext_strength{s}_seed{seed}.csv"
            )
            W = df.loc[df.t == 100, edges_columns].values
            dists = pdist(W, metric="cityblock")
            groupishness = 0 if eps == 0 else dists.std() / dists.mean()
            std_focal = df.loc[df.t == 100, "0"].std()
            nr_negs = sum(
                df.loc[df.t.isin(range(91, 101)), ["0", "id", "t"]]
                .pivot_table(index="id", values="0", columns="t")
                .mean(axis=1)
                < 0
            )
            res.append(
                [
                    init_w,
                    eps,
                    mu,
                    s,
                    fixedBNat100,
                    namesTex[(init_w, eps, mu, fixedBNat100)],
                ]
                + df.loc[df.t == 95.5][response_cols]
                .sum(axis=0)[response_cols]
                .to_list()
                + [groupishness, std_focal, nr_negs]
            )

            mean_absedges.append(
                [
                    init_w,
                    eps,
                    mu,
                    s,
                    fixedBNat100,
                    namesTex[(init_w, eps, mu, fixedBNat100)],
                ]
                + [np.abs(df.loc[df.t == 100, edges_columns].values).mean()]
                + [df.loc[df.t == 100, edges_columns].values.mean()]
            )

res = pd.DataFrame(
    res,
    columns=["init_w", "eps", "mu", "s_ext", "fixedBNat100", "name"]
    + response_cols
    + ["groupishness", "std_focal", "nr_negs"],
)
# %%
sns.barplot(
    pd.DataFrame(
        mean_absedges,
        columns=[
            "init_w",
            "eps",
            "mu",
            "s_ext",
            "fixedBNat100",
            "name",
            "mean_abs_edge",
            "mean_edge",
        ],
    )
    .groupby(["name"])
    .mean()
    .reset_index(),
    hue="name",
    x="name",
    y="mean_abs_edge",
    palette="plasma",
)
# %%

eps = 1.0
mu = 0.0
init_w = 0.2
fixedBNat100 = False
name = namesTex[(init_w, eps, mu, fixedBNat100)]
fig, axs = plt.subplots(1, 2, sharex=True)
sns.stripplot(
    res.query(f"name=='{name}'")[response_cols + ["s_ext"]].melt(
        id_vars=["s_ext"], var_name="response", value_name="count"
    ),
    hue="response",
    x="s_ext",
    y="count",
    dodge=True,
    palette=cmap,
    ax=axs[0],
)

relres = res[["compliant", "resilient", "resistant", "latecompliant"]].div(
    res[["compliant", "resilient", "resistant", "latecompliant"]].sum(axis=1), axis=0
)
for c in ["eps", "mu", "init_w", "s_ext", "fixedBNat100"]:
    relres[c] = res[c]
axs[1].set_title("only negative")
sns.stripplot(
    relres.query(f"eps=={eps} and mu=={mu} and init_w=={init_w}")
    .drop(columns=["mu", "eps", "init_w", "fixedBNat100"])
    .melt(id_vars=["s_ext"], var_name="response", value_name="count"),
    hue="response",
    x="s_ext",
    y="count",
    dodge=True,
    palette=cmap,
    ax=axs[1],
)
fig.suptitle(
    f"eps={eps}, mu={mu}, init_w={init_w}, {'fixedBNat100' if fixedBNat100 else ''}"
)


# %%
relres["s_ext_log2"] = np.log2(relres["s_ext"])
relres = relres.loc[relres.s_ext > 0]

seedExample = 1
pressurestrengthExample = 4
T = 200
fig, axs = plt.subplot_mosaic(
    [["t"] * 4, ["1", "2", "3", "4"]], figsize=(16 / 2.54, 10 / 2.54)
)
for ax in range(2, 5):
    axs[str(ax)].sharex(axs["1"])
    axs[str(ax)].sharey(axs["1"])
for ax, eps, init_w, fixedBNat100 in zip(
    range(1, 5),
    [0.0, 0.0, 1.0, 1.0],
    [0.2, 0.8, 0.2, 0.2],
    [False, False, True, False],
):
    ax = axs[str(ax)]
    subset = relres.query(
        f"eps == {eps} and init_w=={init_w} and mu=={mu} and fixedBNat100=={fixedBNat100}"
    )
    subset = subset[["compliant", "resilient", "resistant", "s_ext_log2"]].melt(
        id_vars="s_ext_log2", value_name="normalized_count", var_name="response"
    )
    ax = sns.boxplot(
        subset,
        ax=ax,
        x="s_ext_log2",
        hue="response",
        y="normalized_count",
        palette=cmap,
        hue_order=["compliant", "resilient", "resistant"],
        legend=False,
        fliersize=0,
        fill=True,
        linewidth=0.0,
        whis=0,
    )
    ax = sns.stripplot(
        subset,
        ax=ax,
        x="s_ext_log2",
        hue="response",
        y="normalized_count",
        jitter=True,
        palette=cmap,
        hue_order=["compliant", "resilient", "resistant"],
        legend=False,
        size=2,
        alpha=0.4,
        dodge=True,
    )

    for coll in ax.collections:
        coll.set_clip_on(False)
    avgs = (
        subset.groupby(["response", "s_ext_log2"])["normalized_count"]
        .median()
        .reset_index()
    )
    ax = sns.stripplot(
        avgs,
        ax=ax,
        x="s_ext_log2",
        hue="response",
        y="normalized_count",
        jitter=True,
        palette=cmap,
        hue_order=["compliant", "resilient", "resistant"],
        legend=False,
        size=4,
        alpha=0.8,
        dodge=True,
        marker="s",
    )
    for coll in ax.collections:
        coll.set_clip_on(False)
    for i in [0.5, 2.5]:
        ax.fill_between(
            [i, i + 1], [-0.05, -0.05], [1.05, 1.05], color="gainsboro", zorder=-1
        )
    ax.set_ylabel("relative response frequency", fontsize=bigfs, va="center")
    ax.set_title(
        namesTex[(init_w, eps, mu, fixedBNat100)],
        fontsize=bigfs,
        x=0.98,
        y=0.99,
        ha="right",
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [f"${2**int(float(v)):0d}$" for v in relres.s_ext_log2.unique()], rotation=0
    )
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([-0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(rf"${int(x*100)}\,\%$" for x in [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ax in range(1, 5):
        axs[str(ax)].set_yticklabels([])
        axs[str(ax)].set_ylabel("")
axs["2"].set_xlabel(r"external pressure $s$", x=1)
axs["1"].text(
    4.5,
    0.24,
    "1 dot = 1 simulation",
    ha="right",
    va="center",
    fontsize=smallfs,
    color="k",
)
ax_main = axs["t"]
examplesimadaptive = pd.read_csv(
    f"simOut/detailed/sim_link_prob0.10_init_w{init_w:.2f}_beta3.00_rho0.33_eps{eps:.2f}_mu{mu:.3f}_{'fixedBNat100_' if fixedBNat100 else ''}ext_strength{pressurestrengthExample}_seed{seedExample}_detailed.csv"
)
beliefs = examplesimadaptive[["id", "t"] + ["0"]].pivot_table(
    index="t", columns="id", values="0"
)
s_ext = examplesimadaptive.ext_strength.unique()[0]
window = 0
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
df_pivot = df_pivot.loc[df_pivot.index <= T]
for aaa in df_pivot.columns:
    df_pivot[aaa].plot(
        ax=ax_main,
        lw=0.6,
        alpha=0.3,
        legend=False,
        color=(
            0.8 - 0.5 * aaa / 100,
            0.8 - 0.5 * aaa / 100,
            0.8 - 0.5 * aaa / 100,
            1,
        ),
    )
agents = dict(zip(["resistant", "resilient", "compliant"], [4, 53, 16]))
for name in ["resistant", "resilient", "compliant"]:
    i = examplesimadaptive.loc[examplesimadaptive[name] == 1, "id"]
    if len(i) > 0:
        i = agents[name] if agents[name] else i.sample().values
        print(name, i)
        df_pivot.loc[:, i].plot(
            ax=ax_main,
            lw=2,
            ls="-",
            color=cmap[name],
            alpha=0.8,
            legend=False,
            label="_",
        )
ax_main.set_xlabel("time")
ax_main.set_ylabel(r"focal belief $x_\mathrm{foc}$", va="center")
ax_main.set_clip_on(False)
ax_main.set_xlim(0, T)
ax_main.set_ylim(-1.02, 1.02)
ax_main.set_yticks([-1, 0, 1])
if s_ext > 0:
    # ["#640000", "#850000", "#B20000", "#DE0000", "#FF0000"]
    int_colors = dict(zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4]))
    y0, y1 = ax_main.get_ylim()
    xx = [100, 150]
    xx = [ttt for ttt in xx if ttt <= T]
    if len(xx) > 0:
        ax_main.fill_between(
            xx,
            [y0] * len(xx),
            [y1] * len(xx),
            color="red",
            alpha=int_colors[s_ext],
            zorder=-1,
            lw=0,
        )
bboxprops = dict(
    boxstyle="round",
    facecolor="white",
    edgecolor="white",
    alpha=0.8,
)
ap = dict(
    arrowstyle="-",
    connectionstyle="arc3,rad=0",
    color="black",
    shrinkA=0,
    shrinkB=0,
)
for (x, y), type in zip(
    [(127, -0.6), (172, 0.5), (155, -0.2)], ["resistant", "compliant", "resilient"]
):
    bboxprops = dict(
        boxstyle="round",
        facecolor=cmap[type],
        edgecolor="white",
        alpha=0.8,
    )
    axs["t"].text(
        x,
        y,
        type,
        color="white",
        va="center",
        ha="left",
        bbox=bboxprops,
        fontsize=bigfs,
    )
axs["t"].text(125, -0.1, "external\npressure", ha="center", fontsize=bigfs)
axs["t"].text(135, -0.1, r"$\uparrow$", ha="center", fontsize=bigfs + 15)
axs["t"].text(
    0.5,
    1.02,
    rf"example simulation with {namesTex[((0.2, 1.0, 0.0, False))]} belief network, $s="
    + f"{s_ext}{'$, smoothed' if window>0 else '$'}",
    transform=axs["t"].transAxes,
    ha="center",
    va="bottom",
    fontsize=bigfs,
)
# fig.set_facecolor("pink")

axs["t"].text(
    90,
    -0.2,
    "agents with\n" + r"$x_\mathrm{foc}<0$",
    va="center",
    ha="right",
    fontsize=bigfs,
)
e1 = mpl.patches.Arc(
    (95.5, -0.7), 10, 1.4, angle=0, linewidth=1, fill=False, zorder=10, color="k"
)
axs["t"].add_patch(e1)
# axs["t"].fill_between([90,100],[-1,-1], [0,0], color="#ffda6799", edgecolor='none')

import string

for n, ax in enumerate([axs["t"]] + [axs[f"{i}"] for i in ["1", "2", "3", "4"]]):
    ax.text(
        0.0 if n == 0 else 0,
        1.02,
        string.ascii_uppercase[n],
        fontsize=12,
        fontdict={"weight": "bold"},
        va="bottom",
        ha="left",
        transform=ax.transAxes,
    )

filename = "figs/fig3_responseFrequencies"
print(filename)
if not os.path.isdir(filename.split("/")[0]):
    os.mkdir(filename.split("/")[0])
fig.subplots_adjust(hspace=0.45, top=0.94, left=0.09, right=0.98, bottom=0.1)
plt.savefig(filename + ".png", dpi=600)
plt.savefig(filename + ".pdf")


# %%

# %%
res.groupby("name").std_focal.mean(),
res.groupby("name").std_focal.std()
# %%
pd.DataFrame(
    mean_absedges,
    columns=["init_w", "eps", "mu", "s", "fixedBNat100", "name", "absOm_tot", "Om_tot"],
).groupby("name").absOm_tot.mean()

# %%

seed_for_agents = 14
np.random.seed(seed_for_agents)
seedExample = 1
pressurestrengthExample = 4
T = 200

fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(16/2.54, 8/2.54))
for pressurestrengthExample, ax in zip([0,1,2,4,8,16], axs.flatten()):
    ax_main = ax
    examplesimadaptive = pd.read_csv(
        f"simOut/detailed/sim_link_prob0.10_init_w{init_w:.2f}_beta3.00_rho0.33_eps{eps:.2f}_mu{mu:.3f}_{'fixedBNat100_' if fixedBNat100 else ''}ext_strength{pressurestrengthExample}_seed{seedExample}_detailed.csv"
    )
    beliefs = examplesimadaptive[["id", "t"] + ["0"]].pivot_table(
        index="t", columns="id", values="0"
    )
    s_ext = examplesimadaptive.ext_strength.unique()[0]
    window = 0
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
    df_pivot = df_pivot.loc[df_pivot.index <= T]
    for aaa in df_pivot.columns:
        df_pivot[aaa].plot(
            ax=ax_main,
            lw=0.6,
            alpha=0.2,
            legend=False,
            color=(
                0.8 - 0.5 * aaa / 100,
                0.8 - 0.5 * aaa / 100,
                0.8 - 0.5 * aaa / 100,
                1,
            ),
        )
    agents = dict(zip(["resistant", "resilient", "compliant"], [None, None, None]))
    for name in ["resistant", "resilient", "compliant"]:
        i = examplesimadaptive.loc[examplesimadaptive[name] == 1, "id"]
        if len(i) > 0:
            i = agents[name] if agents[name] else i.sample().values
            print(name, i)
            df_pivot.loc[:, i].plot(
                ax=ax_main,
                lw=1,
                ls="-",
                color=cmap[name],
                alpha=0.8,
                legend=False,
                label="_",
            )
    ax_main.set_xlabel("time")
    ax_main.set_ylabel(r"focal belief $x_\mathrm{foc}$", va="center")
    ax_main.set_clip_on(False)
    ax_main.set_xlim(0, T)
    ax_main.set_ylim(-1.02, 1.02)
    ax_main.set_yticks([-1, 0, 1])
    if s_ext > 0:
        # ["#640000", "#850000", "#B20000", "#DE0000", "#FF0000"]
        int_colors = dict(zip([1, 2, 4, 8, 16], [0.1, 0.175, 0.25, 0.325, 0.4]))
        y0, y1 = ax_main.get_ylim()
        xx = [100, 150]
        xx = [ttt for ttt in xx if ttt <= T]
        if len(xx) > 0:
            ax_main.fill_between(
                xx,
                [y0] * len(xx),
                [y1] * len(xx),
                color="red",
                alpha=int_colors[s_ext],
                zorder=-1,
                lw=0,
            )
    if pressurestrengthExample==16:
        ax_main.text(125, -0.7, "external\npressure", ha="center", fontsize=bigfs)
    if pressurestrengthExample>0:
        ax_main.text(125, -0.1, r"$\uparrow$", ha="center", fontsize=bigfs + 12)
    ax_main.text(
        0.5,
        1.02,
        rf"$s="
        + f"{s_ext}{'$, smoothed' if window>0 else '$'}",
        transform=ax_main.transAxes,
        ha="center",
        va="bottom",
        fontsize=bigfs,
    )
    # fig.set_facecolor("pink")

    if pressurestrengthExample==4:
        ax_main.text(
            90,
            -0.2,
            "agents with\n" + r"$x_\mathrm{foc}<0$",
            va="center",
            ha="right",
            fontsize=bigfs,
        )
        e1 = mpl.patches.Arc(
            (95.5, -0.7), 10, 1.4, angle=0, linewidth=1, fill=False, zorder=10, color="k"
        )
        ax_main.add_patch(e1)
    # axs["t"].fill_between([90,100],[-1,-1], [0,0], color="#ffda6799", edgecolor='none')

import string

for n, ax in enumerate(axs.flatten()):
    ax.text(
        0.0 if n == 0 else 0,
        1.02,
        string.ascii_uppercase[n],
        fontsize=12,
        fontdict={"weight": "bold"},
        va="bottom",
        ha="left",
        transform=ax.transAxes,
    )
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.93)
plt.savefig(f"figs/AppendixFig_exampleSim_adaptive_seed{seedExample}_overPressure.png", dpi=600)
plt.savefig(f"figs/AppendixFig_exampleSim_adaptive_seed{seedExample}_overPressure.pdf")
# %%
