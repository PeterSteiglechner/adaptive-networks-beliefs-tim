# Analysis Scripts

The five `analyse_*.py` scripts read the simulation CSVs in `simOut/` (relative to the repo root) and produce
the paper's figures and tables in `figs/`. `ANBmodel.py` writes into `simOut_old_model/`, so rename or link that
folder to `simOut/` first. Each script is organised into `# %%` cells, so it can be run as a whole
(`uv run analyse_X.py`) or cell by cell in VS Code or Jupyter.

For which simulations each script needs, see
[running-simulations.md](running-simulations.md#runs-needed-for-each-analysis-script). For the meaning of
the metric columns, see [model.md §8](model.md#8-metrics-get_energies-get_metrics-fill_metrics).

The images in this file are the copies in `figs/figsold/`. The scripts save to `figs/` directly, so a rerun
does not update them. `figs/` is git-ignored, which means the images only show up in a checkout that has the
folder.

## Shared conventions

All scripts redefine the same constants locally. None of them imports from `ANBmodel.py`:

- `belief_columns = ["0", …, "9"]`, `edges_columns = ["w01", …, "w89"]`, focal belief = `"0"`
  (hard-coded for M = 10).
- `response_cols`: `persistPos, nonpersistPos, compliant, resilient, resistant, latecompliant`.
  `negcols = ["compliant", "resilient", "resistant"]` are the misaligned agents (before < 0).
- A BN configuration is identified by the tuple `(init_w, eps, mu, fixedBNat100)`. The dicts `names` / `namesTex`
  map it to labels such as `static ($\omega_0=0.2$)` or `adaptive`.
- Response colours: compliant blue `#2196F3`, resilient purple `#9C27B0`, resistant red `#F44336`.
- In ensemble files, `t == 95.5` is the "before" average and the only row that carries response
  dummies. `145.5` is "during", `195.5` is "after", and `100` is the snapshot at the onset of the pressure.
  A window row holds each agent's mean of every metric, belief and edge over the 10 steps of the window, so
  a count such as `tb_tot` can be fractional there.
- **Relative response frequency** = count of a response ÷ (compliant + resilient + resistant [+ latecompliant]),
  computed per simulation.
- Figures are saved as both `.png` (600 dpi) and `.pdf`.

## Paper figure/table → script map

| Paper | Script | Output file |
|-------|--------|-------------|
| Fig. 2 (example dynamics + two agents' BNs) | `analyse_timeseries.py` | `figs/fig2_adaptive_t100` |
| Fig. A1, A2 (static examples) | presumably `analyse_timeseries.py` with `init_w`/`eps` changed (needs detailed static runs) | `figs/fig2_staticlow_t100`, `…statichigh…` |
| Fig. 3 (example run + response freq. vs. $s$) | `analyse_responseFrequencies.py` | `figs/fig3_responseFrequencies` |
| Fig. A3 (example run for each $s$) | `analyse_responseFrequencies.py` | `figs/AppendixFig_exampleSim_adaptive_seed1_overPressure` |
| Table 3, A1–A4 (metrics by response) | `analyse_BNmetrics.py` | LaTeX printed to stdout |
| Table A5 (metric change before/during/after) | `analyse_BNmetrics.py` | LaTeX printed to stdout |
| Fig. 4 (Cohen's d + logit coefficients) | `analyse_BNmetrics.py` | `figs/fig4_mu0.0` |
| Fig. 5 (metric change over time) | `analyse_BNmetrics.py` | `figs/fig5_metricChange_mu0.0` |
| Fig. A4 (metric correlation heatmap) | `analyse_BNmetrics.py` | `figs/AppendixFig_metricCorrelations` |
| Table A6, A7 (logit fit statistics) | `analyse_BNmetrics.py` | LaTeX printed to stdout |
| Fig. A5, A6 (Fig. 4 with $\mu>0$) | `analyse_BNmetrics.py` with `mu = 0.005` / `0.05` | `figs/fig4_mu0.005`, `figs/fig4_mu0.05` |
| Fig. 6 (social adaptation) | `analyse_socialAdaptation.py` | `figs/fig6_socialAdaptation` |
| Fig. A7 (edge-weight distributions) | `analyse_socialAdaptation.py` | `figs/AppendixFig_edgeweight_distributions` |
| Fig. A8 (OFAT sensitivity) | `analyse_sensitivity.py` | `figs/AppendixFig_ofat` |

`figs/figsold/` holds Fig. 2, 3, 4, 5, 6, A3, A4 and A7. Fig. A1, A2, A5, A6 and A8 are not in it.

Run the scripts in the order of the README. `analyse_timeseries.py`, `analyse_responseFrequencies.py` and
`analyse_BNmetrics.py` create `figs/` if it is missing. `analyse_socialAdaptation.py` saves Fig. A7 before it
creates the folder, and `analyse_sensitivity.py` never creates it, so both fail without an existing `figs/`.

---

## 0. Formulas the figures share

### Model equations behind the plotted quantities

The figures plot raw model state ($x$, $\omega$) and metrics derived from it. The state follows paper
eq. 1–3 (details in [model.md §5–6](model.md#5-dissonance-and-belief-updating-paper-eq-1-2-4)). With the
module constants $\tau = 1$ and $\lambda = 0$ they reduce to:

$$
D_i = -\sum_{m<n}\omega^{(i)}_{mn}x^{(i)}_m x^{(i)}_n \;-\; \rho\sum_{k\in\mathcal K_i}x^{(i)}_{foc}x^{(k)}_{foc} \;-\; s\,x^{(i)}_{foc}
$$

$$
p\big(x_d \to x'\big) \propto \frac{1}{1+e^{\beta\,\Delta D(x')}}, \qquad x' \in \{-1, -0.9, \dots, 1\}
$$

$$
\omega_{mn} \leftarrow \omega_{mn} + \epsilon\,\Delta x_m\,\Delta x_n + \mu\big(\bar\omega^{(\mathcal K)}_{mn} - \omega_{mn}\big)
$$

$\Delta x_m$ is the change of belief $m$ in the last step, and $\bar\omega^{(\mathcal K)}_{mn}$ is the mean
weight of edge $(m,n)$ over the agent's social contacts. Baseline: $N=100$ agents, $M=10$ beliefs, link probability
$p=0.1$, $\beta=3$, $\rho=1/3$, $\omega_0=0.2$, $\epsilon=1$ (adaptive) or $0$ (static), $\mu=0$. The pressure $s$
acts on $x_{foc}$ for $t = 101$–$150$.

The Hebbian term makes beliefs that change together get positive edges and beliefs that change in opposite
directions get negative edges. The $\mu$ term pulls each agent's edge towards its neighbours' mean.

### Time windows and response types

Each agent's focal belief is averaged over three windows of 10 steps:

$$
\bar x^{\,w}_{i} = \frac{1}{10}\sum_{t\in w} x^{(i)}_{foc}(t), \qquad
w \in \{\text{before} = 91\text{–}100,\ \text{during} = 141\text{–}150,\ \text{after} = 191\text{–}200\}
$$

| Response | before | during | after |
|----------|--------|--------|-------|
| `compliant` (lasting change) | < 0 | ≥ 0 | ≥ 0 |
| `resilient` (temporary change) | < 0 | ≥ 0 | < 0 |
| `resistant` (no change) | < 0 | < 0 | < 0 |
| `latecompliant` (negligible) | < 0 | < 0 | > 0 |

Agents with $\bar x^{\,\text{before}}_i < 0$ are *misaligned*: the pressure pushes them towards the opposite
sign. All response statistics use only them. See [model.md §9](model.md#9-response-classification-get_output)
for the two aligned types.

### Relative response frequency

For one simulation, with $n_r$ the number of agents with response $r$:

$$
f_r = \frac{n_r}{n_\text{compliant} + n_\text{resilient} + n_\text{resistant}\;[+\,n_\text{latecompliant}]}
$$

Fig. 3 includes `latecompliant` in the denominator, while Fig. 6 does not. Late compliance is almost
never observed, so the difference is negligible.

### Agent metrics

These are computed per agent at every tracked step (focal belief $foc = 0$, $M = 10$). In the window rows they are
averaged over the window like the beliefs above.

| Column | Label in figures | Formula |
|--------|------------------|---------|
| `x_focal` | focal belief | $x_{foc}$ |
| `extr_nonfoc` | extremity non-focal beliefs | $\frac{1}{M-1}\sum_{m\ne foc}\lvert x_m\rvert$ |
| `n_nbs` | nr social contacts | $\lvert\mathcal K\rvert$, degree in the fixed Erdős–Rényi social network |
| `absOm_tot` | connectedness BN | $\frac{1}{45}\sum_{m<n}\lvert\omega_{mn}\rvert$ |
| `absOm_foc` | connectedness focal BN | $\frac{1}{M-1}\sum_{m\ne foc}\lvert\omega_{foc,m}\rvert$ |
| `tb_tot` | balance BN | $\sum_{a<b<c}\mathbb 1[\omega_{ab}\,\omega_{ac}\,\omega_{bc} > 0]$, between 0 and $\binom{10}{3}=120$ |
| `tb_foc` | balance focal BN | the same sum over the triangles that contain $foc$, between 0 and $\binom{9}{2}=36$ |
| `clust` | clustering BN | $\frac{1}{M}\sum_m \frac{1}{k_m(k_m-1)}\sum_{j\ne l}\big(\hat\omega_{mj}\hat\omega_{jl}\hat\omega_{lm}\big)^{1/3}$ with $\hat\omega = \lvert\omega\rvert/\max\lvert\omega\rvert$ (Onnela et al. 2005) and $k_m$ the number of non-zero edges at $m$ |
| `bc_foc` | centrality focal BN | $\sum_{a<b,\ a,b\ne foc}\sigma_{ab}(foc)/\sigma_{ab}$, the unnormalised betweenness, where $\sigma_{ab}$ counts the shortest paths from $a$ to $b$ with edge length $1/(\lvert\omega\rvert + 10^{-6})$, and $\sigma_{ab}(foc)$ those through $foc$. Between 0 and 36 |
| `Hpersfoc` | dissonance focal BN | $-x_{foc}\sum_{m\ne foc}\omega_{foc,m}\,x_m$ |
| `Hpersnonfoc` | dissonance non-focal BN | $-\sum_{m<n,\ m,n\ne foc}\omega_{mn}x_mx_n$, derived in the scripts as `Hpers − Hpersfoc` |
| `Hsoc` | dissonance focal social | $-\rho\,x_{foc}\cdot\frac{1}{\lvert\mathcal K\rvert}\sum_{k}x^{(k)}_{foc}$. This uses the mean over contacts, while the dynamics use the sum (see [known-issues.md](known-issues.md)) |

A balanced triangle has an even number of negative edges, so its three beliefs can all be consonant at the same
time. Negative dissonance values mean consonance.

---

## 1. `analyse_timeseries.py`: Figure 2

**Input:** one detailed run: adaptive BN, $s=4$, `seed = 2` (`simOut/detailed/`).
You change the configuration through the variables `init_w, eps, mu, fixedBNat100, seed, s` at the top.

**Other output:** it prints `tb_tot`, `tb_foc` and `Hpers` of the two example agents at $t=100$, and the sign
counts of the focal beliefs.

### Figure 2: example dynamics and two agents' belief networks

![Figure 2: focal-belief trajectories of an adaptive run and the belief networks of agents 61 and 79 at t = 0 and t = 100](../figs/figsold/fig2_adaptive_t100.png)

`figs/figsold/fig2_adaptive_t100.{png,pdf}`

**What it shows.** How the population behaves before any pressure. Within a few steps the focal beliefs of
all 100 agents move to the extremes, near $\pm1$, and mostly stay there until $t = 100$ (panel A). The histogram
on the right of A is the bimodal distribution at $t = 100$. Panels B and C follow two agents, 61 (green) and
79 (purple), who both end with $x_{foc} \approx -1$. At $t=0$ every BN is the same weak complete graph
($\omega_{mn} = 0.2$). By $t=100$ each BN has strong signed edges that fit its beliefs. Amber (positive) edges
mostly join beliefs of the same sign, and teal (negative) edges mostly join beliefs of opposite sign. This is
the arrangement that minimises the BN term of $D$.

**How it's built.**

- **A (trajectories):** column `"0"` is pivoted to a time × agent table. Every agent is drawn for $t \le 100$
  as a thin line (width 0.4, alpha 0.2). Its grey level is $0.8 - 0.5\cdot\text{id}/100$, which is cosmetic
  only. Agents 61 and 79 are drawn again as thick coloured lines. The pressure band ($t = 101$–$150$) is
  outside the x-range, so the panel shows only the phase before the pressure.
- **Histogram:** the focal beliefs of all agents at the $t=100$ snapshot, in 20 bins on $[-1.001, 1.001]$.
  Each bar is coloured by its bin centre on the coolwarm map.
- **B, C (`plot_BN_ax`):** one agent's BN at $t=0$ and at $t=100$.
  - Node colour is $x_m$ on coolwarm over $[-1, 1]$ (blue −1, red +1). Labels are white when
    $\lvert x_m\rvert \ge 0.7$. The focal node is labelled `foc` and ringed in gold.
  - Edge colour is $\omega_{mn}$ on a diverging map: teal at −1, grey at 0, amber at +1. Values beyond
    $\pm1$ saturate.
  - Edge width is $1.4\,\omega_{mn}$. Matplotlib draws a negative width like its absolute value, so the width
    is effectively proportional to $\lvert\omega_{mn}\rvert$. Edge alpha is
    $\mathrm{clip}(0.5 + 0.45\,\lvert\omega_{mn}\rvert/5,\ 0,\ 1)$.
  - The layout is circular at $t=0$. At $t=100$ it is networkx's spring layout (Fruchterman–Reingold) with
    $\omega$ as weights, `k = 1.25` and a fixed seed per agent. Positive edges pull nodes together. Negative
    weights enter the attraction term with a negative sign and push nodes apart, so groups of same-sign beliefs
    show up as clusters.

**Formulas.** The figure has no statistics. It plots $x^{(i)}_{foc}(t)$ and $\omega_{mn}(t)$ straight from the
detailed CSV. The structure at $t=100$ comes from the Hebbian rule with $\mu = 0$:
$\omega_{mn}(t+1) = \omega_{mn}(t) + \epsilon\,\Delta x_m\Delta x_n$.

**Caveat.** The agent ids 61 and 79 are only meaningful for seed 2 of the baseline. If you change the seed or
the config, pick new agents.

---

## 2. `analyse_responseFrequencies.py`: Figure 3 and A3

**Input**
- Ensemble runs for all 4 BN configs × $s\in\{0,1,2,4,8,16\}$ × seeds 0–19 (`seeds = list(range(20))`).
- Detailed adaptive runs with seed 1 for each $s$.

**Non-figure cells**

1. **Load loop.** For every file, it sums the response dummies at `t == 95.5` and computes:
   - `groupishness` = std/mean of the pairwise Manhattan distances between agents' edge vectors at t=100.
     This measures BN heterogeneity and is 0 for static BNs.
   - `std_focal` = std of the focal beliefs at t=100. This measures polarisation; the paper reports
     $\sigma_{foc}$ for static vs. adaptive BNs.
   - `nr_negs` = number of agents with a before-mean < 0. Ensemble files have no rows for $t = 91$–$99$, so
     this actually uses only the $t=100$ snapshot.
   - The mean $|\omega|$ and mean $\omega$ at t=100 go into `mean_absedges`.
2. **Quick-look cells:** a bar plot of mean $|\omega|$ per config, and strip plots of absolute and relative counts
   for the adaptive config.
3. **Summary cells:** `std_focal` (mean ± std) and mean $|\omega|$ per config. These are the numbers in the
   paper's text, e.g. $\sigma_{foc}$ = 0.16 / 0.97 / 0.87.

### Figure 3: example run and response frequencies vs. pressure

![Figure 3: an adaptive example run at s = 4 and the relative response frequencies vs. s for the four BN configurations](../figs/figsold/fig3_responseFrequencies.png)

`figs/figsold/fig3_responseFrequencies.{png,pdf}`

**What it shows.** Panel A is one adaptive run at $s = 4$ with one agent of each response type. The
compliant agent (blue) flips to $+1$ during the pressure and stays there. The resilient agent (purple)
flips and returns to $-1$ after $t = 150$. The resistant agent (red) never leaves $-1$. Panels B–E show
the share of each response among the misaligned agents vs. $s$, one panel per BN configuration:

- **B, static ($\omega_0 = 0.2$):** all-or-nothing. Most agents resist at $s = 1$ (median ≈ 88 %), and
  everyone complies from $s = 2$ on.
- **C, static ($\omega_0 = 0.8$):** the strong BN holds the focal belief. Nearly everyone resists up to $s = 4$,
  and everyone is resilient (flips, then flips back) from $s = 8$ on.
- **D, adaptive → static** and **E, adaptive:** all three responses coexist at intermediate pressure. The
  heterogeneous BNs give different agents different thresholds. At strong pressure, resilience dominates and
  resistance disappears.

**How it's built.**

- **A:** the detailed run with seed 1 and $s = 4$. Every agent is a thin grey line (width 0.6, alpha 0.3). The
  example agents are hard-coded: 4 (resistant), 53 (resilient), 16 (compliant). The red band covers
  $t = 100$–$150$, and its alpha grows with $s$ (0.1, 0.175, 0.25, 0.325, 0.4 for $s = 1$ … 16). The black
  ellipse at $t \approx 95.5$ marks the "before" window that decides who is misaligned. The file path reuses the
  leftover `eps`, `init_w` and `fixedBNat100` from the B–E loop. These hold the adaptive values only because
  adaptive is the last panel.
- **B–E:** `res` has one row per simulation (4 configs × 6 values of $s$ × 20 seeds). It holds the response
  counts summed from the `t == 95.5` dummies, and from them $f_r$ is computed. The $s = 0$ runs are dropped.
  The x-axis is categorical at $\log_2 s$ and labelled with $s$. For every $(s, r)$:
  - the box is the interquartile range (`whis=0` and line width 0, so there are no whiskers and no median line),
  - the dots are the 20 simulations, jittered and dodged by response,
  - the square is the median over the 20 simulations.

  The alternating grey bands only separate the $s$ values.

**Formulas.** The relative response frequency $f_r$ (section 0), with `latecompliant` in the denominator, so
the three plotted shares of one simulation sum to $1 - f_\text{latecompliant}$. Box and square:
$Q_{25}, Q_{75}$ and $\mathrm{median}$ of $\{f_r^{(\text{seed})}\}_{\text{seed}=0}^{19}$.

### Figure A3: one run for each pressure strength

![Figure A3: the same adaptive run (seed 1) for s = 0, 1, 2, 4, 8, 16 with one highlighted agent per response type](../figs/figsold/AppendixFig_exampleSim_adaptive_seed1_overPressure.png)

`figs/figsold/AppendixFig_exampleSim_adaptive_seed1_overPressure.{png,pdf}`

**What it shows.** The same adaptive setup (seed 1) for every $s \in \{0, 1, 2, 4, 8, 16\}$, so you can see
how the response changes with the strength of the pressure. At $s = 0$ nothing happens: misaligned agents
stay negative, which counts as "resistant". As $s$ grows, more misaligned agents flip during the pressure. At
$s = 16$ all of them flip, and the only remaining question is whether they stay up (compliant) or fall back
(resilient) after $t = 150$.

**How it's built.** A 2 × 3 grid with shared axes, one detailed file per $s$. The grey lines are drawn as in
Fig. 3A, with alpha 0.2. For each response type that occurs in the run, one random agent is drawn in its colour (line width 1).
The agents are picked with `Series.sample()`, which draws from NumPy's global RNG, seeded once with
`np.random.seed(14)` before the loop. A panel's choice therefore depends on the panels drawn before it. A type
with no agents gets no line, which is why the $s = 0$ panel only has a red one. The red band and its alpha
are as in Fig. 3A. The "external pressure" label appears only at $s = 16$, and the ellipse only at $s = 4$.

**Formulas.** None beyond the response classification (section 0).

---

## 3. `analyse_BNmetrics.py`: Tables 3 and A1–A7, Figures 4, 5 and A4

This is the main statistical analysis: which pre-pressure properties distinguish compliant, resilient and
resistant agents?

**Configuration:** `init_w, eps, mu, fixedBNat100 = (0.2, 1.0, 0.0, False)` on line 161. With `mu == 0`
the script uses $s\in\{0,1,2,4,8,16\}$. With `mu > 0` it uses only $s=4$, which reproduces Appendix D
(Fig. A5/A6). It requires 100 seeds.

**Extra dependencies:** `scikit-learn` (StandardScaler) and `statsmodels` (Logit).

**Non-figure cells**

1. **Load.** It builds a long dataframe `res` with one row per agent × seed × $s$ × evaluation window
   (`t_eval` ∈ {95.5, 145.5, 195.5}). Each row gets the response label from the `t == 95.5` row.
   It derives `Hpersnonfoc = Hpers − Hpersfoc`.
2. **Table 3 / A1–A4** (LaTeX to stdout): mean ± std of each metric by response type in the "before"
   window, for each $s>0$, plus the proportions of the response types among the misaligned agents.
3. **Table A5** (LaTeX to stdout): the same for $s=4$ before, during and after. The last column is the
   no-pressure reference (`s == 0`, all agents).
4. **Table A6, A7** (LaTeX to stdout, `make_latex_table`): the fit statistics of the logistic regressions
   behind Fig. 4 (formulas there).
5. **`greedy_uncorrelated_subset`:** a helper that greedily picks weakly correlated metrics from the Fig. A4
   matrix. It starts with the metric that has the lowest mean $\lvert r\rvert$ to all others. Then it repeatedly
   adds the candidate whose largest $\lvert r\rvert$ to the already selected set is smallest. It prints this
   order next to the hand-chosen `dependent_vars`. This is how the paper justifies its choice of regression
   predictors ("we exclude highly correlated metrics").

The variable named `dependent_vars` actually holds the **predictors**. The dependent variable is
`response_group`.

### Figure 5: metric change before, during and after the pressure

![Figure 5: mean ± sd of the twelve metrics for compliant, resilient and resistant agents before, during and after the pressure at s = 4, with the no-pressure reference in grey](../figs/figsold/fig5_metricChange_mu0.0.png)

`figs/figsold/fig5_metricChange_mu0.0.{png,pdf}`

**What it shows.** For $s = 4$, how each of the 12 metrics develops from before to during to after the pressure,
separately for compliant, resilient and resistant agents. The grey line is the no-pressure run as a reference.
Some readings:

- Before the pressure, resistant agents have the most connected focal BN (E), the most balanced BN (F, G),
  the highest clustering (H) and the lowest focal BN dissonance (J). A coherent BN locks their focal belief in.
- Compliant agents' connectedness rises during the pressure and stays high afterwards (D, E). Their BN
  reorganises around the new belief.
- Resilient agents' focal BN dissonance jumps during the pressure (J). The pressure holds the focal belief
  against a BN that still favours the old sign, and that BN pulls it back once the pressure ends.
- The resistant agents' social dissonance turns positive during the pressure (L). Their contacts have
  flipped while they have not.
- The grey focal belief sits at ≈ 0 (A) because the reference averages over all agents, aligned and
  misaligned.

**How it's built.** The data are the rows of `res` for $s = 4$ and for $s = 0$ (the reference), for all 100
seeds and all three windows. Each agent contributes one value per window, its mean over the 10 steps. Its
response label comes from the `t == 95.5` row and is carried over to the other windows. A seaborn `FacetGrid`
draws one panel per metric (`col_wrap=4`, independent y-axes) with `sns.lineplot(hue="response")`. A point is
the mean over all agents of that type, pooled across seeds. The band is ±1 standard deviation
(`errorbar="sd"`), not a confidence interval. The grey line (`"RRC"`) contains every agent of the $s = 0$
runs.

**Formulas.** For response $r$ with agent set $A_r$ (pooled over seeds) and window $w$:

$$
\bar v_{r,w} = \frac{1}{\lvert A_r\rvert}\sum_{i\in A_r} v_{i,w}, \qquad
\text{band} = \bar v_{r,w} \pm \mathrm{sd}_{i\in A_r}(v_{i,w}), \qquad
v_{i,w} = \frac{1}{10}\sum_{t\in w} v_i(t)
$$

$v$ is any metric from the table in section 0.

### Figure 4: Cohen's d and logistic-regression coefficients

![Figure 4: Cohen's d per metric and pressure for resistant/resilient vs. compliant (left) and resistant vs. resilient (right), overlaid with logistic-regression coefficients and 95 % CIs](../figs/figsold/fig4_mu0.0.png)

`figs/figsold/fig4_mu0.0.{png,pdf}`

**What it shows.** Which properties measured *before* the pressure separate the response groups. The left
panel compares non-compliant (resistant + resilient) with compliant agents, and the right panel compares
resistant with resilient agents. Each bar is the standardised mean difference of one metric at one $s$
(colour). The dots are the logistic-regression coefficients of the five predictors (bold labels) with
95 % CIs, controlled for the focal belief. A positive value means the metric is higher in the first group
(non-compliant, or resistant). Some readings: non-compliant agents have a more negative focal belief and
more balanced focal BNs. What separates resistant from resilient agents is above all a more strongly connected
focal BN (coefficient ≈ 1.9 at $s = 4$) and a lower focal BN dissonance ($d \approx -2$ at $s = 4$).

**How it's built.**

- **Sample:** before-window rows (`t == 95.5`) of the adaptive runs, separately for each $s$. Only agents
  with a label in the contrast are used.
- **Bars:** Cohen's d of all 12 metrics, drawn with `sns.barplot(hue="s")` in the order of `metric2titleVerb`.
  The right panel shows only $s \in \{1, 2, 4\}$, because at $s \ge 8$ there are almost no resistant agents.
- **Dots:** one logit model per $(s, \text{contrast})$ with the predictors `n_nbs, tb_foc, absOm_foc, clust,
  bc_foc` and the control `x_focal`. The control's coefficient is fitted but not drawn. The resistant vs.
  resilient model is not fitted for $s \ge 8$. Each dot is $\hat\beta_k$ with its 95 % CI. The dots are offset
  vertically to sit on their bars.
- **Two x-axes:** the bottom one is Cohen's d on $[-2.5, 2.5]$, and the top one (`twiny`) is the coefficient on
  $[-3, 3]$. The zeros line up, but the scales do not, so don't compare a bar's length with a dot's position as
  numbers.
- **Layout:** thick horizontal lines separate the metric groups: belief content | social | BN structure |
  dissonance. Bold y-labels mark every regression variable, the control `x_focal` included.

**Formulas.**

Groups: for "Resistant/Resilient vs. Compliant" ($y=1$: resistant or resilient, $y=0$: compliant), for
"Resistant vs. Resilient" ($y=1$: resistant, $y=0$: resilient).

Cohen's d (pooled variance, sample variances with $n-1$):

$$
d = \frac{\bar v_1 - \bar v_0}{s_p}, \qquad
s_p^2 = \frac{(n_1-1)\,s_1^2 + (n_0-1)\,s_0^2}{n_1+n_0-2}
$$

$d$ is set to NaN when $s_p^2 < 10^{-10}$ (a metric that does not vary).

Logistic regression on standardised predictors (StandardScaler, i.e. population sd, fitted within each
$(s, \text{contrast})$ subset):

$$
\Pr(y_i = 1) = \frac{1}{1+\exp\!\big(-(\beta_0 + \sum_k \beta_k z_{ik} + \gamma\,z_{i,x_{foc}})\big)},
\qquad z_{ik} = \frac{v_{ik} - \bar v_k}{\sigma_k}
$$

$\beta_k$ is the change in the log-odds of being in group 1 per standard deviation of predictor $k$, and
$e^{\beta_k}$ is the odds ratio. CI: $\hat\beta_k \pm 1.96\,\mathrm{SE}(\hat\beta_k)$ (Wald, `conf_int`). In
the regression `latecompliant` agents are coded $y = 0$ together with the compliant ones, while Cohen's d
leaves them out. They are negligible.

Fit statistics of the same models (Tables A6, A7):

$$
R^2_\text{McFadden} = 1 - \frac{\ln L_\text{full}}{\ln L_\text{intercept}}, \qquad
R^2_\text{Tjur} = \overline{\hat p}_{\,y=1} - \overline{\hat p}_{\,y=0}, \qquad
\text{LR} = 2\,(\ln L_\text{full} - \ln L_\text{control}) \sim \chi^2_5
$$

$L_\text{control}$ is the model with only the intercept and `x_focal`.

### Figure A4: correlations between the metrics

![Figure A4: lower-triangle Pearson correlation matrix of the twelve metrics in the no-pressure runs](../figs/figsold/AppendixFig_metricCorrelations.png)

`figs/figsold/AppendixFig_metricCorrelations.{png,pdf}`

**What it shows.** How strongly the metrics co-vary across agents. This decides which ones can go into the
regression together. Strongly correlated pairs include balance BN ↔ balance focal BN ($r = 0.74$),
connectedness focal BN ↔ dissonance focal BN ($r = -0.78$), and extremity of the non-focal beliefs ↔
non-focal dissonance ($r = -0.72$). Of each such pair the regression keeps at most one variable.

**How it's built.** It uses the rows of `res` with $s = 0$ (adaptive, $\mu = 0$; with `mu > 0` it uses $s = 4$
and the file name gets the suffix `_mu{mu}`). That is 100 agents × 100 seeds × **3 windows** = 30,000 rows.
`DataFrame.corr()` gives the Pearson matrix of the 12 metrics, reordered like the tables. `sns.heatmap` draws
the lower triangle on coolwarm over $[-1, 1]$ with values to two decimals. The `np.triu` mask also hides the
diagonal, which is marked with grey dots instead.

**Formula.**

$$
r_{ab} = \frac{\sum_i (a_i-\bar a)(b_i-\bar b)}{\sqrt{\sum_i (a_i-\bar a)^2}\,\sqrt{\sum_i (b_i-\bar b)^2}}
$$

with $i$ running over all agent-window rows.

**Caveat.** The title says $t = 91$–$100$, but the rows of all three windows are pooled, and the title only reads
the first `t` value. At $s = 0$ the windows differ only by the slow drift of the BN (see the grey lines in
Fig. 5).

---

## 4. `analyse_socialAdaptation.py`: Figures 6 and A7

**Input:** adaptive BN ($\epsilon=1$), $s=4$, seeds 0–99, for
$\mu \in \{0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1\}$.

**Load loop.** For every file it computes, at t=100:

- pairwise **Euclidean (Frobenius) distances** between the agents' edge vectors. Their mean (`mean`) is
  the BN heterogeneity in Fig. 6C. It also stores the variance/mean, std and negative kurtosis.
- the number of negative edges and the mean $|\omega|$.
- the response counts at `t == 95.5`, and all per-agent metrics in the before window (into `metrics`).

A final quick-look cell plots the absolute response counts and `tb_foc` vs. $\mu$.

For Appendix D (Fig. A5/A6), run `analyse_BNmetrics.py` with `mu = 0.005` and `mu = 0.05`.

### Figure A7: edge-weight distributions

![Figure A7: kernel density estimates of all 45 edge weights at t = 100 for mu = 0, 0.005 and 0.2](../figs/figsold/AppendixFig_edgeweight_distributions.png)

`figs/figsold/AppendixFig_edgeweight_distributions.{png,pdf}`

**What it shows.** The distribution of each of the 45 edge weights across the 100 agents of one run at
$t = 100$, for $\mu = 0$, $0.005$ and $0.2$. Without or with weak social adaptation, the weights spread
over roughly $-2$ to $+2$ with long tails, so the agents' BNs differ a lot. At $\mu = 0.2$ each edge's
distribution collapses to a narrow peak, so all agents end up with nearly the same BN.

**How it's built.** It reads the file for seed 99 and $s = 4$. These are the **leftover loop variables** from
the load loop, not a deliberate choice. For each $\mu$, `sns.kdeplot` gets the 100 × 45 table of the
$t = 100$ edge columns. It draws one Gaussian KDE per column (colour = edge $(m,n)$, alpha 0.4). The dashed
line is $\omega_0 = 0.2$. The x-axis is cut to $[-6, 6]$ and the y-ticks are hidden.

**Formula.** For edge $(m,n)$ over the $N = 100$ agents:

$$
\hat f_{mn}(\omega) = \frac{1}{N h}\sum_{i=1}^{N}\varphi\!\left(\frac{\omega - \omega^{(i)}_{mn}}{h}\right),
\qquad h = \hat\sigma_{mn}\,N^{-1/5}\ \text{(Scott's rule)}
$$

$\varphi$ is the standard normal density. Seaborn's default `common_norm=True` scales each curve by its
share of all data points. All edges have $N = 100$ points, so each curve integrates to $1/45$ and the heights
stay comparable. The narrowing at large $\mu$ comes from the term $\mu(\bar\omega^{(\mathcal K)}_{mn} - \omega_{mn})$,
which pulls every agent's weight towards its neighbours' mean faster than the Hebbian term can spread them apart.

### Figure 6: social adaptation

![Figure 6: example belief networks at mu = 0.005 and 0.2, z-scored BN metrics vs. mu, and relative response frequencies vs. mu](../figs/figsold/fig6_socialAdaptation.png)

`figs/figsold/fig6_socialAdaptation.{png,pdf}`

**What it shows.** What happens when agents adapt their BN towards their neighbours' BNs ($\mu$).

- **A, B:** eight agents' BNs at $\mu = 0.005$ (diverse, many negative edges) and at $\mu = 0.2$ (nearly
  identical, almost all positive).
- **C:** BN heterogeneity and focal connectedness fall as $\mu$ grows. Focal balance and clustering stay low up
  to $\mu \approx 0.02$ and then rise.
- **D:** the shares of the responses among the misaligned agents. Compliance peaks around $\mu = 0.005$–$0.01$.
  Resistance shrinks steadily and is close to zero from $\mu \approx 0.005$. At strong $\mu$ almost every
  misaligned agent is resilient.
  The y-tick labels of this panel are wrong (see the caveat below).

**How it's built.**

- **A, B:** the ensemble file for seed 1 and $s = 4$ at $t = 100$. After `np.random.seed(42)`, eight agent ids
  are drawn with `np.random.randint(100)`. A and B show the same eight agents. The layout is circular, and the
  focal node is the rightmost one, ringed in black. Edge colour is as in Fig. 2. Edge width is
  $\min(5,\ 0.8\,\omega_{mn})$, drawn as its absolute value, so the cap at 5 applies only to positive weights.
  The nodes are small grey dots, because beliefs are not shown here.
- **C:** the x-axis is the index of $\mu$ (categorical and equally spaced). The axis label says "log-scale",
  but that is only approximate, and $\mu = 0$ is simply the first category.
  - *heterogeneity BNs* (pink): one value $H$ per simulation (below), z-scored over all 1,100 simulations.
    The line is the mean over the 100 seeds per $\mu$, and the band is ±1 sd.
  - *connectedness focal BN*, *balance focal BN*, *clustering BN*: per-agent values from the before window.
    Each metric is z-scored over all agents, seeds and $\mu$. The line is the mean per $\mu$, and the band is
    ±1 sd.
- **D:** $f_r$ per simulation, with the denominator compliant + resilient + resistant. For each $\mu$ and
  response there is a box (IQR, no whiskers) and a strip of the 100 simulations, as in Fig. 3.
- Dashed brackets connect the example rows in A and B to their $\mu$ in panel C.

**Formulas.** BN heterogeneity of one simulation, with $\boldsymbol\omega_i \in \mathbb R^{45}$ the edge
vector of agent $i$ at $t = 100$:

$$
H = \frac{2}{N(N-1)}\sum_{i<j}\lVert\boldsymbol\omega_i - \boldsymbol\omega_j\rVert_2,
\qquad
\lVert\boldsymbol\omega_i - \boldsymbol\omega_j\rVert_2 = \sqrt{\sum_{m<n}\big(\omega^{(i)}_{mn} - \omega^{(j)}_{mn}\big)^2}
$$

z-score: $z = (v - \bar v)/\mathrm{sd}(v)$ (pandas `std`, i.e. $n - 1$), with mean and sd taken over the
pooled set described above. Panel C therefore compares the *direction* of change of the metrics, not their
size. Panel D uses $f_r$ from section 0, without `latecompliant`.

**Caveat: y-tick labels in panel D.** The labels read 0 % to 125 %, but the tick marks sit at 0, 0.2, …, 1.0
(measured in the PNG). Each label is therefore 1.25× the true value: read "100 %" as 80 % and "125 %" as
100 %. The labels come from `ax_mu.set_yticklabels([... for x in ax_mu.get_yticks()])` without fixed tick
positions. Most likely the tick locator used steps of 0.25 when the labels were made, and switched to steps of
0.2 after `subplots_adjust` changed the axis height. The fix is to call `ax_mu.set_yticks([0, 0.2, 0.4, 0.6,
0.8, 1.0])` before setting the labels, as Fig. 3 does. The script still has the same code, so a rerun
reproduces the error.

---

## 5. `analyse_sensitivity.py`: Figure A8

Fig. A8 is not in `figs/figsold/`.

**Input:** one-factor-at-a-time variations around the baseline, $s=4$, seeds 0–99. See
[running-simulations.md → Sensitivity runs](running-simulations.md#sensitivity-runs) for the required
files, including the extra filename suffixes for `M`, `N` and `tau`.

**What it does**

1. `param_combis` lists the experiments as `[exp, M, N, init_w, eps, mu, beta, p, tau, s]`. Each one is
   run for static-low, static-high and adaptive, except `eps` and `init_w`, which are adaptive only.
2. It sums the response dummies at `t == 95.5` for each file and converts them to relative frequencies.
   `p` is re-expressed as the mean number of contacts, $p\cdot N$.
3. **Figure A8:** a 7×3 grid. Rows are `M, N, beta, p, tau, eps, init_w`, and columns are the BN
   configurations. Each panel shows strip plots of the response frequencies plus the medians, with the
   baseline included as a reference value. For `eps` and `init_w` only the adaptive column is filled.

**Output:** `figs/AppendixFig_ofat`.

The file is named `analyse_sensitivity.py`. The README calls it `analsyse_sensitivity.py`, which is a typo.
