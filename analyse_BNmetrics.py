# %%

from scipy import stats
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist
import os 
import matplotlib.path as mpath
import numpy as np

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
            (0.0000, 0.4470, 0.7410),
            (0.8500, 0.3250, 0.0980),
            (0.9290, 0.6940, 0.1250),
            (0.4940, 0.1840, 0.5560),
            (0.4660, 0.6740, 0.1880),
            (0.3010, 0.7450, 0.9330),
            (0.6350, 0.0780, 0.1840),
        ],
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
    # clust_foc=r"BN-clust$_{foc}$",
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
    # clust_foc=r"clustering focal BN",
    bc_foc=r"centrality focal BN",
    Hpersfoc=r"dissonance focal BN",
    Hpersnonfoc=r"dissonance non-focal BN",
    Hsoc=r"dissonance focal social",
)

response_cols = [
    "persistPos",
    "nonpersistPos",
    "compliant",
    "resilient",
    "resistant",
    "latecompliant",
]
negcols = ["compliant", "resilient", "resistant"]


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
    # "clust_foc",
    "bc_foc",
    "expI",
    "x_focal",
    "extr_nonfoc",
]
selected_metricsl_cols = [
    "n_nbs",
    "Hpersnonfoc",
    "Hpersfoc",
    "Hsoc",
    "tb_tot",
    "tb_foc",
    "absOm_tot",
    "absOm_foc",
    "clust",
    # "clust_foc",
    "bc_foc",
    "x_focal",
    "extr_nonfoc",
]

metrics_table_sort = [
        "x_focal",
        "extr_nonfoc",
        "n_nbs",
        "absOm_tot",
        "absOm_foc",
        "tb_tot",
        "tb_foc",
        "clust",
        # "clust_foc",
        "bc_foc",
        "Hpersfoc",
        "Hpersnonfoc",
        "Hsoc",
    ]



meta_cols = ["t", "id"]

cmap = dict(
    zip(
        response_cols + ["NA"],
        ["#4CAF50", "#AED581", "#2196F3", "#9C27B0", "#F44336", "#90CAF9", "#9E9E9E"],
    )
)
names = {
    (0.2, 0.0, 0.0, False): r"static ($\omega_0=0.2$)",
    (0.2, 1.0, 0.0, False): r"adaptive",
    (0.8, 0.0, 0.0, False): r"static ($\omega_0=0.8$)",
    (0.2, 1.0, 0.0, True): r"adaptive$\rightarrow$static",
}

# ----------------------------------------------
# -------    LOAD DATA
# ----------------------------------------------
res = []
mean_absedges = []
init_w, eps, mu, fixedBNat100 = (0.2, 1.0, 0.0, False)
res = pd.DataFrame()
all_pressures = [0, 1, 2, 4, 8, 16] if mu==0 else [4]
seeds = list(range(100))
for s in all_pressures:
    for seed in seeds:
        df = pd.read_csv(
            f"simOut/sim_link_prob0.10_init_w{init_w:.2f}_beta3.00_rho0.33_eps{eps:.2f}_mu{mu:.3f}{'_fixedBNat100' if fixedBNat100 else ''}_ext_strength{s}_seed{seed}.csv"
        )
        # df = df.rename(columns={"bc":"bc_foc"})
        # W = df.loc[df.t == 100, edges_columns].values
        # dists = pdist(W, metric="cityblock")
        # groupishness = 0 if eps == 0 else dists.std() / dists.mean()
        for t_eval in [95.5, 145.5, 195.5]:
            responses = (
                df.loc[df.t == t_eval][["id"] + response_cols]
                .melt(id_vars=["id"])
                .replace({False: np.nan})
                .dropna()
            )

            vals = df.loc[df.t == t_eval][meta_cols + metric_cols].reset_index()
            vals["response"] = None
            for col in response_cols:
                vals.loc[
                    (df.loc[df.t == 95.5, col] == 1).reset_index()[col], "response"
                ] = col

            for x in ["init_w", "eps", "mu", "s", "seed", "t_eval"]:
                vals[x] = eval(x)
            res = pd.concat([res, vals])


res["Hpersnonfoc"] = res.Hpers - res.Hpersfoc
metric_cols[1] = "Hpersnonfoc"
res = res.drop(columns=["Hpers"])


# %%

# ----------------------------------------------
# -------    PRINT TABLE
# ----------------------------------------------
t_eval = 95.5
pressures = [1,2,4,8,16] if mu==0 else [4]
for s_ext in pressures:
    print("% "
        +r"".join(["#"] * 50)
        + f"\n% time = {t_eval-4.5} - {t_eval+4.5}\n"
        + "% "
        +"".join(["#"] * 50)
        + f"\n% s_ext = {s_ext}\n"
        + "% "
        +"".join(["#"] * 50)
    )
    print(" " + " & ".join([""] + negcols) + " & \\\\ \\hline")
    for metric in metrics_table_sort:
        print(fr"{metric2titleVerb[metric]} &  " , end="") # + f"{metric2title[metric]} &  "
        for r in negcols:
            for t_eval in [95.5]:
                a = res.loc[(res.response == r) & (res.s == s_ext)& (res.t_eval == t_eval), metric]  # .mean()
                if len(a) > 0:
                    if abs(a.mean())>10:
                        print(f"${a.mean():.1f} \\pm {a.std():.1f}$", end=" & ")
                    else: 
                        print(f"${a.mean():.2f} \\pm {a.std():.2f}$", end=" & ")
                else:
                    print(" ", end=" & ")
        print("\\\\")

        if metric == metrics_table_sort[-1]:
            print(f"proportion &  ", end="")
            props = res.loc[res.s == s_ext, "response"].value_counts() / (
                len(res.id.unique()) * len(res.seed.unique())
            )
            summeNeg = (
                props[negcols].sum()
                if "resistant" in props
                else props[["compliant", "resilient"]].sum()
            )
            for r in negcols:
                if r in props:
                    print(
                        f"${100*props[r]/summeNeg:.1f}\,\%$",
                        end=" & ",
                    )
                else:
                    print(f"${0.0:.1f}\,\%$ & ")
            print("\\\\")


# %%


# ----------------------------------------------
# -------    PRINT CHANGE TABLE
# ----------------------------------------------

metric_change_table = []
columns = ["metric", "response", "time", "s", "stat", "value"] 
pressures = [4]
for s_ext in pressures:
    print(
        "% Change over time"
    )
    print("% " + " & ".join([""] + negcols) + " & \\\\ \\hline")
    for metric in metrics_table_sort:
        print(f"  " + f"{metric2titleVerb[metric]} &  ", end="")
        for r in negcols:
            for t_eval in [95.5,145.5, 195.5]:
                a = res.loc[(res.response.isin(negcols) if s_ext==0 else res.response==r) & (res.s == s_ext)& (res.t_eval == t_eval), metric]  # .mean()
                if len(a) > 0:
                    if abs(a.mean())>10:
                        print(f"${a.mean():.1f} \\pm {a.std():.1f}$", end=" & ")
                    else: 
                        print(f"${a.mean():.2f} \\pm {a.std():.2f}$", end=" & ")
                    metric_change_table.append([metric2title[metric], r, t_eval, s_ext, "mean", a.mean()])
                    metric_change_table.append([metric2title[metric], r, t_eval, s_ext, "std", a.std()])
                else:
                    print(" ", end=" & ")
                if r==negcols[-1]:
                    aNoPress = res.loc[(res.s == 0)& (res.t_eval == t_eval), metric]  # .mean()
                    if len(aNoPress)>0:
                        if abs(aNoPress.mean())>10:
                            print(f"${aNoPress.mean():.1f} \\pm {aNoPress.std():.1f}$", end=" & ")
                        else: 
                            print(f"${aNoPress.mean():.2f} \\pm {aNoPress.std():.2f}$", end=" & ")
                        metric_change_table.append([metric2title[metric], "RRC", t_eval, 0, "mean", aNoPress.mean()])
                        metric_change_table.append([metric2title[metric], "RRC", t_eval, 0, "std", aNoPress.std()])
                    else:
                        print(" ", end=" & ")
        print("\\\\")

        if metric == metrics_table_sort[-1]:
            print(f"proportion &   ", end="")
            props = res.loc[res.s == s_ext, "response"].value_counts() / (
                len(res.id.unique()) * len(res.seed.unique())
            )
            summeNeg = (
                props[negcols].sum()
                if "resistant" in props
                else props[["compliant", "resilient"]].sum()
            )
            for r in negcols:
                if r in props:
                    print(
                        f"${100*props[r]/summeNeg:.1f}\,\%$",
                        end=" & & &  ",
                    )
                else:
                    print(f"${0.0:.1f}\,\%$ & ")
            print("\\\\")


#%%
# metric_change_table = pd.DataFrame(metric_change_table, columns=columns)
cmap["RRC"] = "darkgrey"
######## PLOT   #####

s_ext = 4
# melt res from wide (one col per metric) to long format
plot_df = pd.concat([
    res.loc[(res.s == s_ext)][["response", "t_eval", "seed"] + metrics_table_sort],
    res.loc[(res.s == 0)][["response", "t_eval", "seed"] + metrics_table_sort].assign(response="RRC"),
]).melt(
    id_vars=["response", "t_eval", "seed"],
    var_name="metric", value_name="value"
)
plot_df["metric"] = plot_df["metric"].map(metric2titleVerb)

g = sns.FacetGrid(
    plot_df,
    col="metric",
    col_wrap=4,
    sharey=False,
    sharex=True,
    height=4/2.54,
    aspect=1.2,
)

g.map_dataframe(
    sns.lineplot,
    x="t_eval",
    y="value",
    hue="response",
    palette=cmap,
    hue_order=["RRC"]+negcols,
    errorbar="sd",        # seaborn computes mean ± sd directly from raw data
    marker="o",
    markersize=5,
    linewidth=2,
    err_kws={"alpha": 0.1},
)

g.set_titles(col_template="{col_name}", size=smallfs, pad=2)
g.set_axis_labels("", "", fontsize=smallfs)
g.tick_params(labelsize=smallfs)

for ax in g.axes.flat:
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.grid(True, linestyle=":", alpha=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([95.5, 145.5, 195.5])
    ax.set_xticklabels(["before", "during", "after"])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, min_n_ticks=3))  # max 3 yticks
# handles, labels = g.axes[0].get_legend_handles_labels()
# g.figure.legend(handles, labels, title="response", fontsize=smallfs,
#                 title_fontsize=smallfs, loc="lower center",
#                 ncol=len(cmap), bbox_to_anchor=(0.5, -0.05))
g.figure.set_size_inches(16/2.54, 10/2.54)
plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.06,right=0.99, top=0.95, bottom=0.06)

ax = g.axes.flat[0]
ax.text(153, 0.58, "compliant", color=cmap["compliant"], fontsize=smallfs, ha="left")
ax.text(150, -0.5, "resilient", color=cmap["resilient"], fontsize=smallfs, ha="left", rotation=-55)
ax.text(140, -0.7, "resistant", color=cmap["resistant"], fontsize=smallfs, ha="center", va="bottom" )
import string
for n, ax in enumerate(g.axes.flat):
    ax.text(
        0.03,
        0.975,
        string.ascii_uppercase[n],
        fontsize=12,
        fontdict={"weight": "bold"},
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
plt.savefig(f"figs/fig5_metricChange_mu{mu}.png", dpi=600)
plt.savefig(f"figs/fig5_metricChange_mu{mu}.pdf",)

#%%


# %%

# ----------------------------------------------
# -------    COHENS D
# ----------------------------------------------
resBefore = res.loc[res.t==95.5]

def cohens_d(a, b):
    n_a = a.shape[0]
    n_b = b.shape[0]
    var_a = a.var(ddof=1)  # unbiased (n-1) estimator
    var_b = b.var(ddof=1)
    
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    # Treat near-zero variance as zero
    pooled_var[pooled_var < 1e-10] = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (a.mean() - b.mean()) / pooled_var**0.5
    return d.where(pooled_var > 0, other=np.nan)


resBefore["noncompliant"] = np.nan
resBefore.loc[resBefore.response.isin(["resistant", "resilient"]), "noncompliant"] = 1
resBefore.loc[resBefore.response.isin(["compliant"]), "noncompliant"] = 0


pressures = [1,2,4,8,16] if mu==0 else [4]
cds_df = []
for s in pressures:
    res_s = resBefore.query(f"s=={s} and eps=={eps}")
    allagents = res_s.dropna(subset=["noncompliant"])
    for ty in ["rR-C", "R-r"]:
        if ty == "rR-C":
            group = allagents.loc[allagents["noncompliant"] == 1, selected_metricsl_cols]
            base = allagents.loc[allagents["noncompliant"] == 0, selected_metricsl_cols]
        elif ty == "R-r":
            group = allagents.loc[allagents.response == "resistant", selected_metricsl_cols]
            base = allagents.loc[allagents.response == "resilient", selected_metricsl_cols]
        cds = cohens_d(group, base)
        for metric in selected_metricsl_cols:
            cds_df.append(
                [s, metric2title[metric], metric2titleVerb[metric], ty, cds.loc[metric]]
            )
cohensd_df = pd.DataFrame(cds_df, columns=["s", "metricMath", "metric", "type", "d"])

# %%
# ----------------------------------------------
# -------    REGRESSION
# ----------------------------------------------
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

dependent_vars = [
    "n_nbs",
    # "tb_tot",
    "tb_foc",
    # "absOm_tot",
    "absOm_foc",
    "clust",
    # "clust_foc",
    'bc_foc',
    # 'extr_nonfoc'
]
control_vars = ["x_focal"]
coef_df = pd.DataFrame()

table_rows = []
for s in pressures:
    for ty in ["rR-C", "R-r"]:
        if not (ty=="R-r" and s>=8):
            df = resBefore.query(f"s=={s} and eps=={eps}")
            label_map = dict(zip(response_cols, [np.nan] * 6))
            if ty == "R-r":
                label_map["resistant"] = 1
                label_map["resilient"] = 0
            elif ty == "rR-C":
                label_map["resistant"] = 1
                label_map["resilient"] = 1
                label_map["compliant"] = 0
                label_map["latecompliant"] = 0

            df["response_group"] = df["response"].map(label_map)
            df = df.dropna(subset=["response_group"])
            # print(len(df))

            X = df[dependent_vars + control_vars]
            scaler = StandardScaler()
            X_scaled_df = pd.DataFrame(
                scaler.fit_transform(X), columns=dependent_vars + control_vars
            )
            y = df["response_group"].values
            X_scaled_df = X_scaled_df.loc[:, X.std() > 1e-6]

            model = sm.Logit(y, sm.add_constant(X_scaled_df)).fit()

            coefs = pd.DataFrame(
                {
                    "coefficient": model.params[1:],  # exclude intercept
                    "odds_ratio": np.exp(model.params[1:]),
                    "pvalue": model.pvalues[1:],
                    "ci_low": model.conf_int(alpha=0.05)[0][1:],
                    "ci_high": model.conf_int(alpha=0.05)[1][1:],
                },
                index=dependent_vars + control_vars,
            )
            coefs["type"] = ty
            coefs["s"] = s

            # --- Metrics ---
            pseudo_r2 = model.prsquared
            y_hat = model.predict()
            tjur_r2 = y_hat[y == 1].mean() - y_hat[y == 0].mean()

            model_null = sm.Logit(y, sm.add_constant(X_scaled_df[control_vars])).fit(disp=0)
            model_full = sm.Logit(y, sm.add_constant(X_scaled_df)).fit(disp=0)
            lr_stat = 2 * (model_full.llf - model_null.llf)
            p_val = stats.chi2.sf(lr_stat, df=len(dependent_vars))

            # --- Group proportions ---
            counts = df["response_group"].value_counts()
            total = counts.sum()
            prop_str = f"{counts.get(1, 0)/total:.2f} / {counts.get(0, 0)/total:.2f}"

            # --- Store row ---
            row = {
                "ty": ty,
                "s": s,
                "pseudo_r2": pseudo_r2,
                "tjur_r2": tjur_r2,
                "lr_stat": lr_stat,
                "lr_p": p_val,
                "prop_str": prop_str,
                "n": int(total),
            }
            table_rows.append(row)

            print("Coefficients")
            print(coefs[coefs["type"] == ty].sort_values("pvalue"))

            coef_df = pd.concat([coef_df, coefs])

coef_df["metric"] = coef_df.index.map(metric2titleVerb)


combined = (
    cohensd_df.set_index(["type", "metric", "s"])
    .join(coef_df.set_index(["type", "metric", "s"])[["coefficient", "ci_low", "ci_high"]])
    .reset_index()
)

#%%
# ----------------------------------------------
# -------    Table Performance
# ----------------------------------------------

# ── Build LaTeX tables ──────────────────────────────────────────────────────

# def p_stars(p):
#     if p < 0.001: return "***"
#     if p < 0.01:  return "**"
#     if p < 0.05:  return "*"
#     return ""

def make_latex_table(rows, ty_label, caption, label):
    header = r"""\begin{table}[ht]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\begin{tabular}{ccccccc}
\toprule
$s$ & $N$ & Proportions & McFadden $R^2$ & Tjur $R^2$ & LR $\chi^2$ & LR $p$ \\
     &     & (group 1 / group 0) &          &            &             &        \\
\midrule"""

    body_lines = []
    for r in rows:
        # stars = p_stars(r["lr_p"])
        p_fmt = f"{r['lr_p']:.2e}" if r['lr_p']>0.001 else "<0.001"
        line = (
            f"  {r['s']} & {r['n']} & {r['prop_str']} & "
            f"{r['pseudo_r2']:.3f} & {r['tjur_r2']:.3f} & "
            f"{r['lr_stat']:.2f} & {p_fmt} \\\\"
        )
        body_lines.append(line)

    footer = r"""\bottomrule
\end{tabular}
\end{table}"""

    return "\n".join([header] + body_lines + [footer])

# Split by type
rows_rRC = [r for r in table_rows if r["ty"] == "rR-C"]
rows_Rr  = [r for r in table_rows if r["ty"] == "R-r"]

print(make_latex_table(
    rows_rRC,
    ty_label="rR-C",
    caption=r"Model fit statistics across pressure levels --- \textit{Resistant/Resilient-Compliant} classification (Resistant+Resilient vs.\ Compliant)",
    label="tab:fit_rRC"
))

print()

print(make_latex_table(
    rows_Rr,
    ty_label="R-r",
    caption=r"Model fit statistics across pressure levels --- \textit{Resistant-Resilient} classification (Resistant vs.\ Resilient)",
    label="tab:fit_Rr"
))

# ----------------------------------------------
# -------    PLOTTING
# ----------------------------------------------

# %%
showRegression = True
diss = False
nmetric = len(combined.metric.unique())
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16 / 2.54, 8 / 2.54))
# axs[0].scatter([],[],marker="o", c="grey", s=5, edgecolor="grey", label="regression\ncoefficient")
if showRegression:
    dotpatch = mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="lightgrey",
        markeredgecolor="grey",
        markeredgewidth=0.7,
        markersize=4,
        label="regression\ncoefficient",
    )
patch = mpatches.Patch(color="grey", label=r"Cohen's $d$")
axs[0].legend(
    handles=[patch] if not showRegression else [patch, dotpatch],
    fontsize=smallfs,
    borderpad=0.25,
    handlelength=1,
    handleheight=0.5,
    handletextpad=0.7,
    facecolor="gainsboro",
    loc="lower right",
    edgecolor="none",
    framealpha=0.5,
)
for n, ty in enumerate(["rR-C", "R-r"]):
    ax = axs[n]
    ss = ([1, 2, 4] if ty == "R-r" else [1, 2, 4, 8, 16]) if mu==0 else [4]
    plot_df = combined.loc[(combined.type == ty) & (combined.s.isin(ss))].sort_values(
        "s", ascending=False
    )
    metric_order = list(metric2titleVerb.values())[::-1]
    sns.barplot(
        plot_df,
        x="d",
        y="metric",
        hue="s",
        hue_order=pressures[::-1],
        dodge=True,
        ax=ax,
        palette=cmapS,
        legend=False,
        zorder=10,
        order=metric_order,
        errorbar=None,
        alpha=0.8,
        lw=0,
    )
    if showRegression:
        ax2 = ax.twiny()
        if metric2titleVerb["x_focal"] in plot_df.metric.values:
            print("control for x_focal")
            plot_df = plot_df.loc[
                ~(plot_df.metric == metric2titleVerb["x_focal"])
            ].copy()

        plot_df_reg = plot_df.dropna(subset=["coefficient"]).copy()

        # Get y positions matching the stripplot dodge logic
        n_hues = len(ss)
        metric_positions = {m: i for i, m in enumerate(metric_order)}
        hue_offsets = (-0.4 + np.arange(pressures[::-1].index(ss[-1]), pressures[::-1].index(ss[0])+1) * 1/len(pressures)) if mu==0 else [0]  # same as seaborn's dodge spacing
        hue_order = ss[::-1]

        for hi, s_val in enumerate(hue_order):
            sub = plot_df_reg[plot_df_reg.s == s_val]
            color = cmapS[s_val]
            for _, row in sub.iterrows():
                y_pos = metric_positions.get(row["metric"])
                if y_pos is None:
                    continue
                y_dodge = y_pos + hue_offsets[hi]
                ax2.errorbar(
                    x=row["coefficient"],
                    y=y_dodge,
                    xerr=[[row["coefficient"] - row["ci_low"]],
                        [row["ci_high"] - row["coefficient"]]],
                    fmt="o",
                    color="k",
                    markersize=3,
                    markeredgecolor="k",
                    markeredgewidth=0.4,
                    markerfacecolor=color,
                    elinewidth=0.8,
                    capsize=2,
                    capthick=0.8,
                    zorder=20,
                )
        ax2.set_xlabel("Regression Coefficient (dots)", fontsize=smallfs)
    
        if diss:
            ax2.set_xlim(-1, 1)
        else:
            ax2.set_xlim(-3, 3)
    ax.set_xlabel(r"Cohen's $d$ (bars)", fontsize=smallfs)
    ax.set_title(
        (
            "\nResistant/Resilient vs. Compliant"
            if ty == "rR-C"
            else "\nResistant vs. Resilient"
        ),
        fontsize=bigfs + 1,
    )
    ax.set_ylabel("")
    maxy = (
        np.argsort(
            [p.get_y() for p in ax.patches],
        )[-10:-5]
        if n == 0
        else np.argsort(
            [p.get_y() for p in ax.patches],
        )[-6:-3]
    )
    for kk, (barind, text) in enumerate(zip(maxy, [rf"$s={s}$" for s in ss[::-1]])):
        bar = ax.patches[barind]
        h = bar.get_width()
        if np.isnan(h):
            continue

        y = bar.get_y() + bar.get_height() / 2
        x = bar.get_width()
        bar_fc = bar.get_facecolor()
        if mu == 0:
            ax.annotate(
                text,
                xy=(x + 0.1, y),
                xytext=(
                    1.8,
                    y
                    + (1.3 if n == 0 else 1)
                    - (0.56 if n == 0 else 0.46) * (len(maxy) - kk),
                ),
                ha="left",
                va="center",
                rotation=0,
                fontsize=smallfs - 1,
                arrowprops=dict(
                    arrowstyle="-",
                    lw=1,
                    connectionstyle="arc,rad=0.1",  # vertical then tilt
                    color=bar_fc,
                ),
                bbox={"pad": 0.02, "fc": "white", "ec": "none"},
                color=bar_fc,
                zorder=20,
            )
        if mu>0:
            ax.text(1.8,len(selected_metricsl_cols)-2.15, rf"$s={4}$", color= cmapS[4])

    ax.vlines([0], -0.5, nmetric + 0.5, lw=0.5, color="grey")
    xlim = 2.5
    ax.hlines([0.5 + k for k in range(nmetric)], -xlim, xlim, lw=0.5, color="grey")
    ax.hlines(
        [nmetric - 2.5, nmetric - 3.5, nmetric - 9.5], -xlim, xlim, lw=1.5, color="grey"
    )
    ax.set_ylim(-0.5, nmetric - 0.5)
    ax.set_xlim(-xlim, xlim)
    if showRegression:
        ax2.set_ylim(ax.get_ylim())

# draw once so tick labels exist
fig.canvas.draw()
if showRegression:
    for label in axs[0].get_yticklabels():
        if label.get_text() in list(coef_df.metric.unique()):
            label.set_fontweight("bold")


fig.subplots_adjust(
    left=0.23, top=0.82 if showRegression else 0.92, right=0.98, bottom=0.12
)
fname = (
    f"figs/fig4_mu{mu}{names[(init_w,eps,mu,fixedBNat100)] if eps!=1 else ''}"
)

if not os.path.isdir(fname.split("/")[0]):
    os.mkdir(fname.split("/")[0])
print(fname)
plt.savefig(fname + ".png", dpi=600)
plt.savefig(fname + ".pdf")
# baseline{'staticBNafterBurnIn' if staticBNafterBurnIn else ''}{'_withDiss' if showRegression and diss else ''}

# %%


# %%


fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=(12/2.54, 12/2.54))
axs=np.array([axs])
pressures = [0] if mu==0 else [4]

for s, ax in zip(pressures, axs.flatten()):
    df = res.query(f"s=={s} and eps=={eps}")
    a = df[selected_metricsl_cols].corr().loc[metrics_table_sort, metrics_table_sort]
    a = a.rename(columns=metric2titleVerb, index=metric2titleVerb)
    mask = np.triu(np.ones_like(a))
    sns.heatmap(a, ax=ax, cbar=False, cmap="coolwarm", vmin=-1,vmax=1, annot=True, fmt=".2f", annot_kws={'fontsize':6}, mask=mask, )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=smallfs-1)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=smallfs-1) 
    ax.set_aspect("equal")
    # ax.set_title(fr"$s={s}$", y=0.9, x=0.45, va="top", fontsize=12)
    ax.set_title("correlation of variables\nacross agents &\nsimulations\n"+f"at $t={int(df.t.unique()[0]-4.5)}-{int(df.t.unique()[0]+4.5)}$", y=0.9, x=0.8, va="top", fontsize=bigfs, ha="right")
    # Draw X on diagonal cells
    n = len(a)
    for i in range(n):
        ax.add_patch(plt.Circle((i+0.5,i+0.5), 0.1, color='gainsboro'))
# fig.autofmt_xdate(rotation=30, ha="right")
# fig.tight_layout()
fig.subplots_adjust(hspace=0.01, wspace=0.02, top=1, right=1, left=0.27, bottom=0.27 )
plt.savefig(f"figs/AppendixFig_metricCorrelations{f'_mu{mu}' if mu>0 else ''}.png", dpi=600)
plt.savefig(f"figs/AppendixFig_metricCorrelations{f'_mu{mu}' if mu>0 else ''}.pdf",)
# %%

def greedy_uncorrelated_subset(corr_matrix, n):
    abs_corr = corr_matrix.abs()
    # Start with the variable that has the lowest average correlation
    avg_corr = abs_corr.mean()
    selected = [avg_corr.idxmin()]
    
    while len(selected) < n:
        remaining = [c for c in corr_matrix.columns if c not in selected]
        # For each candidate, find its max correlation with already selected
        max_corr_with_selected = {
            c: abs_corr.loc[c, selected].max() for c in remaining
        }
        # Pick the one with the lowest max correlation to selected set
        next_var = min(max_corr_with_selected, key=max_corr_with_selected.get)
        selected.append(next_var)
    
    return selected

metrics_considered = [metric2titleVerb[m] for m in ["x_focal", "n_nbs", "tb_tot", "tb_foc", "absOm_tot", "absOm_foc", "clust",  "bc_foc", "extr_nonfoc"]]
chosen = greedy_uncorrelated_subset(a.loc[metrics_considered, metrics_considered], n=len(metrics_considered))

print("best metrics: ", chosen)
print("hand-chosen: ", [metric2title[m] for m in dependent_vars])
# %%
