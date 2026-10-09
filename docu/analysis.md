# Analysis Scripts

The five `analyse_*.py` scripts read the simulation CSVs in `simOut/` and produce the paper's figures
and tables in `figs/`. Each script is organised into `# %%` cells, so it can be run as a whole
(`uv run analyse_X.py`) or cell by cell in VS Code or Jupyter.

For which simulations each script needs, see
[running-simulations.md](running-simulations.md#runs-needed-for-each-analysis-script). For the meaning of
the metric columns, see [model.md §8](model.md#8-metrics-get_energies-get_metrics-fill_metrics).

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

Run the scripts in the order of the README. `analyse_BNmetrics.py` saves Fig. 5 before it creates
`figs/`, so it fails if the folder does not exist yet. Running `analyse_timeseries.py` first creates it.

---

## 1. `analyse_timeseries.py`: Figure 2

**Input:** one detailed run: adaptive BN, $s=4$, `seed = 2`
(`simOut/detailed/sim_link_prob0.10_init_w0.20_beta3.00_rho0.33_eps1.00_mu0.000_ext_strength4_seed2_detailed.csv`).
You change the configuration through the variables `init_w, eps, mu, fixedBNat100, seed, s` at the top.

**What it does**

- `plot_BN_ax(ag_b, ag_e, ax, …)` draws one agent's belief network with networkx:
  - Nodes are coloured by belief value (coolwarm, −1 blue → +1 red). Beliefs with $|x|\ge0.7$ get white labels.
  - Edge colour encodes the sign (teal < 0, amber > 0), and edge width and alpha encode $|\omega|$.
  - The focal node is ringed in gold.
  - The layout is circular for t = 0 or static BNs, and a weighted spring layout otherwise.
  - The function is reusable for any agent and time step.
- The main cell plots all focal-belief trajectories for $t \le 100$ (grey), plus a histogram of the focal
  beliefs at $t=100$.
- Two example agents (hard-coded ids **61** and **79**) are highlighted, and their BNs are drawn at $t=0$ and $t=100$.
- It prints the agents' `tb_tot`, `tb_foc` and `Hpers` at $t=100$, and the sign counts of the focal beliefs.

**Output:** `figs/fig2_<config>_t100.{png,pdf}`.

The agent ids are only meaningful for seed 2 of the baseline. If you change the seed or config, pick new agents.

---

## 2. `analyse_responseFrequencies.py`: Figure 3 and A3

**Input**
- Ensemble runs for all 4 BN configs × $s\in\{0,1,2,4,8,16\}$ × seeds 0–19 (`seeds = list(range(20))`).
- Detailed adaptive runs with seed 1 for each $s$.

**What it does**

1. **Load loop.** For every file, it sums the response dummies at `t == 95.5` and computes:
   - `groupishness` = std/mean of the pairwise Manhattan distances between agents' edge vectors at t=100.
     This measures BN heterogeneity and is 0 for static BNs.
   - `std_focal` = std of the focal beliefs at t=100. This measures polarisation; the paper reports
     $\sigma_{foc}$ for static vs. adaptive BNs.
   - `nr_negs` = number of agents with a before-mean < 0.
   - The mean $|\omega|$ and mean $\omega$ at t=100 go into `mean_absedges`.
2. **Quick-look cells:** a bar plot of mean $|\omega|$ per config, and strip plots of absolute and relative counts
   for the adaptive config.
3. **Figure 3:**
   - Panel A: focal-belief trajectories of the seed-1, $s=4$ detailed run. Hard-coded example agents:
     resistant = 4, resilient = 53, compliant = 16. The pressure window is shaded.
   - Panels B–E: box and strip plots of the relative response frequencies vs. $s$ (on a log₂ axis),
     one panel per BN config. Each dot is one simulation, and the squares are medians.
4. **Summary cells:** `std_focal` (mean ± std) and mean $|\omega|$ per config. These are the numbers in the
   paper's text, e.g. $\sigma_{foc}$ = 0.16 / 0.97 / 0.87.
5. **Figure A3:** a 2×3 grid of the same seed-1 adaptive run for each $s$. For each response type it
   highlights a randomly sampled agent (`np.random.seed(14)`).

**Output:** `figs/fig3_responseFrequencies`, `figs/AppendixFig_exampleSim_adaptive_seed1_overPressure`.

---

## 3. `analyse_BNmetrics.py`: Tables 3 and A1–A7, Figures 4, 5 and A4

This is the main statistical analysis: which pre-pressure properties distinguish compliant, resilient and
resistant agents?

**Configuration:** `init_w, eps, mu, fixedBNat100 = (0.2, 1.0, 0.0, False)` on line 159. With `mu == 0`
the script uses $s\in\{0,1,2,4,8,16\}$. With `mu > 0` it uses only $s=4$, which reproduces Appendix D
(Fig. A5/A6). It requires 100 seeds.

**Extra dependencies:** `scikit-learn` (StandardScaler) and `statsmodels` (Logit).

**What it does**

1. **Load.** It builds a long dataframe `res` with one row per agent × seed × $s$ × evaluation window
   (`t_eval` ∈ {95.5, 145.5, 195.5}). Each row gets the response label from the `t == 95.5` row.
   It derives `Hpersnonfoc = Hpers − Hpersfoc`.
2. **Table 3 / A1–A4** (LaTeX to stdout): mean ± std of each metric by response type in the "before"
   window, for each $s>0$, plus the proportions of the response types among the misaligned agents.
3. **Table A5** (LaTeX to stdout): the same for $s=4$ before, during and after. The last column is the
   no-pressure reference (`s == 0`, all agents).
4. **Figure 5:** a seaborn `FacetGrid` with one panel per metric. It shows the mean ± sd by response type
   across before, during and after, with the $s=0$ reference in grey (`"RRC"`).
5. **Cohen's d** (`cohens_d(a, b)`, pooled-variance d, `NaN` when the variance is ≈ 0), computed for two contrasts:
   - `rR-C`: non-compliant (resistant + resilient) vs. compliant
   - `R-r`: resistant vs. resilient
6. **Logistic regression** (statsmodels `Logit` on standardised predictors) for the same two contrasts:
   - predictors: `n_nbs, tb_foc, absOm_foc, clust, bc_foc`; control: `x_focal`
   - `R-r` is skipped for $s\ge8$, because almost no agents are resistant there
   - fit statistics: McFadden $R^2$, Tjur $R^2$ (difference in mean predicted probability between
     the two groups), and an LR test against a control-only model. These are printed as the LaTeX
     tables A6 and A7 (`make_latex_table`).
7. **Figure 4:** horizontal Cohen's-d bars per metric, coloured by $s$. The logit coefficients with 95 % CIs
   are overlaid as dots on a twin x-axis, and the regression predictors are shown in bold.
8. **Figure A4:** lower-triangle correlation heatmap of the metrics. It uses $s=0$ when `mu == 0`, and
   $s=4$ otherwise.
9. **`greedy_uncorrelated_subset`:** a helper that greedily picks weakly correlated metrics. It prints a
   suggested predictor set next to the hand-chosen `dependent_vars`. This is how the paper justifies its
   choice of regression predictors ("we exclude highly correlated metrics").

The variable named `dependent_vars` actually holds the **predictors**. The dependent variable is
`response_group`.

**Output:** `figs/fig4_mu{mu}`, `figs/fig5_metricChange_mu{mu}`,
`figs/AppendixFig_metricCorrelations[_mu{mu}]`, plus the LaTeX tables on stdout.

---

## 4. `analyse_socialAdaptation.py`: Figures 6 and A7

**Input:** adaptive BN ($\epsilon=1$), $s=4$, seeds 0–99, for
$\mu \in \{0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1\}$.

**What it does**

1. **Load loop.** For every file it computes, at t=100:
   - pairwise **Euclidean (Frobenius) distances** between the agents' edge vectors. Their mean (`mean`) is
     the BN heterogeneity in Fig. 6C. It also stores the variance/mean, std and negative kurtosis.
   - the number of negative edges and the mean $|\omega|$.
   - the response counts at `t == 95.5`, and all per-agent metrics in the before window (into `metrics`).
2. **Figure A7:** KDEs of all 45 edge-weight distributions at t=100 for $\mu \in \{0, 0.005, 0.2\}$.
   It uses the **leftover loop variables** `seed = 99` and `s = 4` from the load loop.
3. **Figure 6:**
   - Panels A/B: 8 random agents' BNs (`np.random.seed(42)`, simulation seed 1) at $\mu=0.005$ and $\mu=0.2$.
     They are drawn on a circular layout with sign-coloured edges, and the focal node is ringed.
   - Panel C: z-scored BN heterogeneity, `absOm_foc`, `tb_foc` and `clust` vs. $\mu$ (categorical axis).
   - Panel D: box and strip plots of the relative response frequencies vs. $\mu$.
   - Dashed brackets (`add_bracket_with_tick`) connect the example networks to their $\mu$ position.
4. A final quick-look cell plots the absolute response counts and `tb_foc` vs. $\mu$.

**Output:** `figs/fig6_socialAdaptation`, `figs/AppendixFig_edgeweight_distributions`.

For Appendix D (Fig. A5/A6), run `analyse_BNmetrics.py` with `mu = 0.005` and `mu = 0.05`.

---

## 5. `analyse_sensitivity.py`: Figure A8

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
