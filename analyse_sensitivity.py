# %%
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
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
eps = 1.0
mu = 0.0
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
# s_exts = [4]
names = {
    (0.2, 0.0, 0.0, False): r"static ($\omega_0=0.2$)",
    (0.2, 1.0, 0.0, False): r"adaptive",
    (0.8, 0.0, 0.0, False): r"static ($\omega_0=0.8$)",
    (0.2, 1.0, 0.0, True): r"adaptive$\rightarrow$static",
    (0.1, 1.0, 0.0, False): r"adaptive",
    (0.4, 1.0, 0.0, False): r"adaptive",
    (0.2, 0.5, 0.0, False): r"adaptive",
    (0.2, 2.0, 0.0, False): r"adaptive",
}
experiments = {
    "beta":   r"attention to" + "\n" + r"dissonance $\boldsymbol{\beta}$",
    "M":      r"nr of beliefs $\boldsymbol{M}$",
    "N":      r"nr of agents $\boldsymbol{N}$" + "\n" + r"(with $\boldsymbol{p}=10/N$)",
    "p":      r"average nr of " + "\n" + r"social contacts" + "\n" + r"$\boldsymbol{p} \cdot N$",
    "tau":    r"activation" + "\n" + r"memory $\boldsymbol{\tau}$",
    "eps":    r"internal" + "\n" + r"adaptation" + "\n" + r"rate $\boldsymbol{\epsilon}$",
    "init_w": r"initial edge" + "\n" + r"weight $\boldsymbol{\omega_0}$",
}
res = []
# M N initw eps mu beta p s responsFreq

mean_absedges = []
pressure = 4
param_combis = (
    [
        ["base", 10, 100, w, eps, mu, 3.0, 0.1, 1, pressure]
        for w, eps, mu in [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
    ]
    + [
        ["M", M, 100, w, eps, mu, 3.0, 0.1, 1, pressure]
        for w, eps, mu in [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
        for M in [5, 15]
    ]
    + [
        ["N", 10, N, w, eps, mu, 3.0, p, 1, pressure]
        for w, eps, mu in [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
        for N, p in zip([50, 200], [0.2, 0.05])
    ]
    + [
        ["beta", 10, 100, w, eps, mu, beta, 0.1, 1, pressure]
        for w, eps, mu in [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
        for beta in [1.5, 6.0]
    ]
    + [
        ["p", 10, 100, w, eps, mu, 3.0, p, 1, pressure]
        for w, eps, mu in [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
        for p in [0.05, 0.2]
    ]
    + [
        ["tau", 10, 100, w, eps, mu, 3.0, 0.1, tau, pressure]
        for w, eps, mu in [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
        for tau in [2.0, 10.0]
    ]
    + [
        ["eps", 10, 100, w, eps, mu, 3.0, 0.1, 1, pressure]
        for eps2 in [0.5,1.0,2.0]
        for w, eps, mu in [(0.2, eps2, 0.0)]
        
    ]
    + [
        ["init_w", 10, 100, w, eps, mu, 3.0, 0.1, 1, pressure]
        for w2 in [0.1,0.2,0.4]
        for w, eps, mu in [ (w2, 1.0, 0.0)]
    ]
)
fixedBNatt100 = False

for exp, M, N, init_w, eps, mu, beta, p, tau, s in param_combis:
    print(exp, end=",")
    for seed in range(100):
        addon = (
            f"_M{M:.2f}"
            if exp == "M"
            else (
                f"_N{N:.2f}"
                if exp == "N"
                else (f"_tau{tau:.2f}" if exp == "tau" else "")
            )
        )
        df = pd.read_csv(
            f"simOut/sim_link_prob{p:.2f}_init_w{init_w:.2f}_beta{beta:.2f}_rho0.33_eps{eps:.2f}_mu{mu:.3f}{'_fixedBNatt100' if fixedBNatt100 else ''}_ext_strength{s}_seed{seed}{addon}.csv"
        )
        res.append(
            [
                exp,
                M,
                N,
                init_w,
                eps,
                mu,
                beta,
                p,
                tau,
                s,
                names[(init_w, eps, mu, fixedBNatt100)],
            ]
            + df.loc[df.t == 95.5][response_cols].sum(axis=0)[response_cols].to_list()
        )
# %%
res = pd.DataFrame(
    res,
    columns=[
        "exp",
        "M",
        "N",
        "init_w",
        "eps",
        "mu",
        "beta",
        "p",
        "tau",
        "s_ext",
        "name",
    ]
    + response_cols,
)
# %%
relres = res[["compliant", "resilient", "resistant", "latecompliant"]].div(
    res[["compliant", "resilient", "resistant", "latecompliant"]].sum(axis=1), axis=0
)
for c in ["exp", "M", "N", "init_w", "eps", "mu", "beta", "p", "tau", "s_ext", "name"]:
    relres[c] = res[c]
relres["p"] = (relres["p"] * relres["N"]).astype(int)
relres["tau"] = (relres["tau"]).astype(int)
# %%

relres = relres.loc[relres.s_ext > 0]
fig, axs = plt.subplots(7, 3, sharex=False, sharey=True, figsize=(18 / 2.54, 16 / 2.54))
T = 200

for n, name in enumerate(["M", "N", "beta", "p", "tau"]):
    for nn, (init_w, eps, mu) in enumerate(
        [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
    ):
        ax = axs[n, nn]
        subset = relres.query(
            f"exp=='{name}' and eps == {eps} and init_w=={init_w} and mu=={mu}"
        )
        subset = subset[["compliant", "resilient", "resistant", name]].melt(
            id_vars=name, value_name="normalized_count", var_name="response"
        )
        base = relres.query(
            f"exp=='base' and eps == {eps} and init_w=={init_w} and mu=={mu}"
        )
        base = base[["compliant", "resilient", "resistant", name]].melt(
            id_vars=name, value_name="normalized_count", var_name="response"
        )
        subset = pd.concat([subset, base])
        subset = subset.reset_index()
        # ax = sns.boxplot(subset, ax=ax, x=name, hue="response", y="normalized_count", palette=cmap, hue_order=["compliant", "resilient", "resistant"], legend=False, fliersize=0, fill=True, linewidth=0., whis=0)
        ax = sns.stripplot(
            subset,
            ax=ax,
            x=name,
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
        avgs = (
            subset.groupby(["response", name])["normalized_count"]
            .median()
            .reset_index()
        )
        ax = sns.stripplot(
            avgs,
            ax=ax,
            x=name,
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
for n2, name in enumerate(["eps", "init_w"]):
    (init_w, eps, mu) = ([0.2] if name=='eps' else [0.1,0.2,0.4], [0.5,1.0,2.0] if name=='eps' else [1.0], 0.0)
    ax = axs[n+n2+1, 2]
    subset = relres.query(
        f"exp=='{name}' and eps in {eps} and init_w in {init_w} and mu=={mu}"
    )
    subset = subset[["compliant", "resilient", "resistant", name]].melt(
        id_vars=name, value_name="normalized_count", var_name="response"
    )
    base = relres.query(
        f"exp=='base' and eps == {1.0} and init_w=={0.2} and mu=={mu}"
    )
    base = base[["compliant", "resilient", "resistant", name]].melt(
        id_vars=name, value_name="normalized_count", var_name="response"
    )
    subset = pd.concat([subset, base])
    subset = subset.reset_index()
    # ax = sns.boxplot(subset, ax=ax, x=name, hue="response", y="normalized_count", palette=cmap, hue_order=["compliant", "resilient", "resistant"], legend=False, fliersize=0, fill=True, linewidth=0., whis=0)
    ax = sns.stripplot(
        subset,
        ax=ax,
        x=name,
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
    avgs = (
        subset.groupby(["response", name])["normalized_count"]
        .median()
        .reset_index()
    )
    ax = sns.stripplot(
        avgs,
        ax=ax,
        x=name,
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
for n, name in enumerate(["M", "N", "beta", "p", "tau", "eps", "init_w"]):
    for nn, (init_w, eps, mu) in enumerate(
        [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 1.0, 0.0)]
    ):
        ax = axs[n, nn]

        for coll in ax.collections:
            coll.set_clip_on(False)
        if n == 2:
            ax.set_ylabel("response frequency", fontsize=bigfs, va="center")
        else:
            ax.set_ylabel("")
        if n == 0:
            ax.set_title(
                names[(init_w, eps, mu, fixedBNatt100)],
                fontsize=bigfs,
                x=0.5,
                y=0.99,
                ha="center",
            )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=bigfs)
        if nn == 2:
            ax.text(
                1.02,
                0.1,
                experiments[name],
                va="center",
                ha="left",
                transform=ax.transAxes,
                fontdict={"weight": "bold"},
            )
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([-0.0, 0.5, 1.0])
        ax.set_yticklabels(rf"${int(x*100)}\,\%$" for x in [0, 0.5, 1.0])

axs[-1,0].axis("off")
axs[-1,1].axis("off")
axs[-2,0].axis("off")
axs[-2,1].axis("off")
for ax in axs[-2:, 2]:
    ax.set_yticklabels(rf"${int(x*100)}\,\%$" for x in [0, 0.5, 1.0])
    ax.tick_params(axis="y", which="both", left=True, labelleft=True)
import string
n=0
for ax in axs.flatten():
    if not ax in axs[-2:, :2]:
        ax.text(
            0.0 if n == 0 else 0,
            1.01,
            string.ascii_uppercase[n],
            fontsize=12,
            fontdict={"weight": "bold"},
            va="bottom",
            ha="left",
            transform=ax.transAxes,
        )
        n+=1

filename = "figs/AppendixFig_ofat"
print(filename)
fig.subplots_adjust(
    hspace=0.6, wspace=0.05, top=0.95, left=0.08, right=0.83, bottom=0.05
)
plt.savefig(filename+".png", dpi=600)
plt.savefig(filename+".pdf")

# %%
