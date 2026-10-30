# Analysis Package (`anb.analysis`)

> **Status: in progress.** This describes the design agreed in `.scratch/analysis-refactor/spec.md`. Implemented
> so far: `index_folder`, the core (`Baseline`, selection, warnings, `Loader`, `FigureResult`, style) and `fig2`
> with `example_agent_stats` / `focal_sign_counts`, `fig3` / `figA3` / `polarisation_summary`, and `fig4` / `fig5` /
> `figA4` with their shared `metric_frame` (and `cohens_d` / `logit_fit` for Fig. 4), the tables `table3` /
> `tableA5` / `tableA6` / `tableA7` with `save_table` and `greedy_uncorrelated_subset`, `fig6` / `figA7`, and
> `figA8`. Update it as the tickets land. The old `analyse_*.py` scripts ([analysis.md](analysis.md)) stay
> unchanged as the reference.

`anb.analysis` makes the paper's figures and tables from **whatever runs a folder holds**. It finds runs by
parameter, not by a hard-coded filename. Each figure and table is its own function, and `make_all` runs all of
them. A figure whose runs are missing is skipped with a warning.

## Quick start

```python
from anb.analysis import make_all, fig3, Baseline, save_table, table3

report = make_all("simOut_old_model", "figs")                     # every applicable figure + table, PNG only
report = make_all("simOut_old_model", "figs", formats=("png", "pdf"), figs=["fig4", "fig6"])

res = fig3("simOut_old_model", seeds=range(20))                    # FigureResult, nothing saved yet
res.figure                                                         # matplotlib Figure
res.data                                                           # the plotted numbers
res.save("figs", formats=("png", "pdf"))

fig3("simOut_new_model", base=Baseline(link_prob=0.2))             # → fig3_responseFrequencies_link_prob0.20
save_table(table3("simOut_old_model", s=4), "figs")                # → figs/tables/table3_s4.tex
```

```bash
python -m anb.analysis simOut_old_model figs --figs fig3,fig4 --formats png,pdf [--strict]
```

## How runs are found

`index_folder(sim_dir)` parses every CSV name in `sim_dir/` and `sim_dir/detailed/` with the inverse of
`anb.io.legacy_filename` into one row of parameters:

- `link_prob, init_w, beta, rho, eps, mu, fixedBNat100, s, seed`;
- `lam, n_events, M, N, tau`;
- `detailed`.

When a suffix is absent, the default applies: M = 10, N = 100, τ = 1, λ = 0, one event. Files the parser
doesn't recognise are left out, with one warning.

A figure takes a **base config**, `Baseline(...)`, a frozen dataclass:

- The default is the paper's adaptive baseline: `link_prob 0.1, beta 3, rho 0.33, init_w 0.2, eps 1, mu 0,
  fixedBNat100 False, lam 0, n_events 1, M 10, N 100, tau 1`.
- The figure uses only runs that match the base, plus whatever the figure varies itself (pressure, seed, μ, BN
  config).
- Figures that compare BN configs derive the four paper configs from the base by replacing
  `(init_w, eps, mu, fixedBNat100)`:

| Config | `init_w` | `eps` | `mu` | `fixedBNat100` |
|--------|----------|-------|------|----------------|
| static ($\omega_0=0.2$) | 0.2 | 0 | 0 | False |
| static ($\omega_0=0.8$) | 0.8 | 0 | 0 | False |
| adaptive→static | 0.2 | 1 | 0 | True |
| adaptive | 0.2 | 1 | 0 | False |

M and N are read from each CSV (belief columns `"0"…"M-1"`, distinct `id`s), so every figure works for other
M, N and τ. The time structure is fixed: T = 200, pressure at t = 101–150, windows 95.5 / 145.5 / 195.5.

The package writes the `_M{M:.2f}`, `_N{N:.2f}` and `_tau{tau:.2f}` suffixes only when the value differs from
10 / 100 / 1. They go right after `seed…`. Baseline names are unchanged.

## Applicability and warnings

- **`FigureNotApplicable`** (a `UserWarning`): a structural requirement is missing. The function returns
  `None`, and the message names the missing cells.
- **`FewSeedsWarning`** (a `UserWarning`): a required cell has fewer seeds than the paper used. The figure is
  still made. With fewer than `min_seeds` (default 1), the figure is not applicable.
- If a figure needs a fixed grid (configs, pressure panels), every cell of the grid must be there. Fig. A8 is the
  exception: it needs its base cells, and a sensitivity row with a variant missing is left out of the grid with a
  **`MissingLevelsWarning`** naming the missing cells (not applicable if no row is complete).
- Pressure levels inside a panel (an x-axis, bars) use the paper's levels that exist. The missing ones get a
  **`MissingLevelsWarning`** (a `UserWarning`), and if none exist, the figure is not applicable. For Fig. 3 and
  the polarisation summary a level counts only if all four configs have it.

A model fit inside a figure that fails (an empty group, a perfect separation, no convergence) gives a **`FitWarning`**
(a `UserWarning`), and only that estimate is left out (Fig. 4's dots for one contrast × s).

Default example agents that are missing, or that don't have the intended response type, are replaced by a
seeded random agent of that type, with an `ExampleAgentFallback` warning (also a `UserWarning`).

| Item | Needs (on top of the base) | Paper seeds | Kwargs (defaults) |
|------|----------------------------|-------------|-------------------|
| `fig2` (Fig. 2, A1, A2) | detailed run, BN config, `s`, `seed` | 1 | `config="adaptive"`, `seed=2`, `s=4`, `agents=(61, 79)` |
| `fig3` | ensemble: all 4 configs × s > 0 levels, plus a detailed adaptive run for panel A | 100 (old script: 20) | `seeds=None`, `example_seed=1`, `example_s=4`, `example_agents={resistant: 4, resilient: 53, compliant: 16}`, `denominator` (the four types incl. `latecompliant`) |
| `polarisation_summary` | ensemble: all 4 configs × s ≥ 0 levels | 100 (old script: 20) | `seeds=None` |
| `figA3` | detailed adaptive run for each s ∈ {0, 1, 2, 4, 8, 16} | 1 | `seed=1`, `rng_seed=14` |
| `fig4` (Fig. 4, A5, A6) | ensemble at `mu` (replacing the base's μ): s ∈ {1, 2, 4, 8, 16} for μ = 0, s = 4 for μ > 0 | 100 | `mu=0.0`, `pressures=None` (paper rule), `contrast2_pressures=(1, 2, 4)`, `seeds=None` |
| `fig5` | ensemble at `mu` (replacing the base's μ): `s` and the s = 0 reference | 100 | `mu=0.0`, `s=4`, `seeds=None` |
| `figA4` | ensemble at `mu` (replacing the base's μ): s = 0 (μ = 0) or s = 4 (μ > 0) | 100 | `mu=0.0`, `seeds=None` |
| `fig6` | ensemble at each μ (replacing the base's μ), s = 4: at least two of `mus`, plus the run `example_seed` at both `example_mus` (which must be among `mus`) | 100 | `mus=None` (the paper's 11 that exist, missing ones warned about), `example_seed=1`, `example_mus=(0.005, 0.2)`, `agent_rng_seed=42`, `denominator` (compliant, resilient, resistant: no `latecompliant`), `seeds=None` |
| `figA7` | ensemble at each μ (replacing the base's μ): the run `seed` at `s` for every listed μ | 1 | `seed=99`, `s=4`, `mus=(0, 0.005, 0.2)` |
| `figA8` | ensemble s = 4: the base of staticlow, statichigh and adaptive (those the rows use), plus each complete OFAT row: M ∈ {5, 15}, N ∈ {50, 200} with p = 10/N, β ∈ {1.5, 6}, p ∈ {0.05, 0.2}, τ ∈ {2, 10} for the three configs, ε ∈ {0.5, 2} and ω₀ ∈ {0.1, 0.4} for adaptive | 100 | `rows=None` (all, in that order), `seeds=None` |
| `table3` (Table 3, A1–A4) | ensemble at `mu` (replacing the base's μ) and `s` | 100 | `s` (required, > 0), `mu=0.0`, `seeds=None` |
| `tableA5` | ensemble at `mu` (replacing the base's μ): s = 4 and the s = 0 reference | 100 | `mu=0.0`, `seeds=None` |
| `tableA6`, `tableA7` | as `fig4`; `tableA7` only at the pressures in `contrast2_pressures` | 100 | `mu=0.0`, `pressures=None` (paper rule), `seeds=None`; `tableA7` also `contrast2_pressures=(1, 2, 4)` |
| `greedy_uncorrelated_subset` | as `figA4` | 100 | `mu=0.0`, `metrics` (the old nine candidates), `n=None` (all), `seeds=None` |

In `make_all`, `fig2` is attempted for the adaptive, static-low and static-high configs (A1, A2). The `fig4` /
`figA4` / `table*` family is attempted for each μ in `mus=(0.0, 0.005, 0.05)`, and `table3` for each s > 0 that
exists.

## Output

A figure function returns a `FigureResult` with:

- `figure`: the matplotlib Figure;
- `data`: a tidy DataFrame of every plotted number;
- `name`: the file stem.

Figure functions never save or print anything. Pass one `Loader()` to several figure functions (`loader=`) to
read each request of a CSV only once, and a ready `index_folder(...)` (`index=`) to skip rescanning the folder.
`result.save(out_dir, formats=("png",))` creates `out_dir` and writes PNG at 600 dpi, plus a PDF if `"pdf"` is in
`formats`.

Names are the old scripts' names (`fig2_adaptive_t100`, `fig3_responseFrequencies`, `fig4_mu0.0`,
`fig5_metricChange_mu0.0`, `fig6_socialAdaptation`, `AppendixFig_…`, e.g. `AppendixFig_ofat` for Fig. A8). A base that differs from the paper's
baseline appends the differing fields as legacy tokens (`_link_prob0.20`, `_M15.00`), so variants don't
overwrite the paper figures. `fig5` at an `s` other than 4 appends `_s{s}`.

A table function returns its LaTeX as a `LatexTable`, a `str` that also carries the file stem in `.name`, and
prints nothing. `save_table(latex, out_dir, name=None)` writes it to `out_dir/tables/<name>.tex` (creating the
folder) and returns the path; `name` defaults to the table's own, and a plain string needs one. The names are
`table3_s{s}`, `tableA5_metricChange`, `tableA6_logit_rRC` and `tableA7_logit_Rr`, each plus `_mu{mu}` for μ > 0
and the base suffix (as for Fig. A4). The content and layout are the old script's stdout:

- `table3`: a comment block (window, `s_ext`), the column header, one row of mean ± sd per metric (one decimal for
  |mean| > 10, two otherwise) for compliant, resilient and resistant agents in the before window, and the
  proportions of the three types among them. A type without agents gets blank cells and 0.0 %.
- `tableA5`: per metric the three types at the three windows, then the s = 0 reference (every agent) at the three
  windows, and the proportions. The column header is a LaTeX comment, as before.
- `tableA6` / `tableA7`: a full `table` environment with one row per pressure: N, the proportions of group 1 /
  group 0, McFadden and Tjur R², and the LR χ² and p (`<0.001` below 0.001) of `logit_fit`. A fit that fails warns
  (`FitWarning`) and its row is left out.

Tables 3 and A5 are row fragments (no `tabular` environment), as the old output was.

`make_all` returns `{item: [paths] | "skipped: <reason>" | "error: <message>"}` and prints a summary. An
unexpected error in one item is warned about with its traceback and the run carries on. With `strict=True` it
is raised instead.

## Summaries

These functions return DataFrames:

- `polarisation_summary`: per config, the mean and std over runs (all paper pressures s ≥ 0 that all four configs
  have) of `std_focal` (σ_foc at t = 100), `abs_edge` / `mean_edge` (mean |ω| / mean ω at t = 100),
  `groupishness` and `nr_negs`. Columns are `(metric, "mean" | "std")` plus `("n_runs", "")`;
- `example_agent_stats`: the `tb_tot`, `tb_foc` and `Hpers` of the Fig. 2 agents at t = 100;
- `focal_sign_counts`;
- `cohens_d(before, contrast)` and `logit_fit(before, contrast)`: Fig. 4's statistics on one window and pressure
  of `metric_frame` rows, for the contrasts `"rR-C"` (resistant + resilient vs compliant) and `"R-r"` (resistant vs
  resilient). Cohen's d leaves `latecompliant` out, the logit counts it as compliant (as the old script). The logit
  standardises `n_nbs, tb_foc, absOm_foc, clust, bc_foc` and the control `x_focal` within the subset and returns a
  `LogitFit` (coefficients with 95 % CIs, McFadden and Tjur R², the LR test against the control model, the group
  sizes), or `None` with a `FitWarning`. Where the old script crashes (a near-constant column dropped), the LR test
  uses one degree of freedom per predictor kept, and without `x_focal` the control model is intercept-only;
- `metric_frame(runs, loader)`: the old `res` of `analyse_BNmetrics.py` for some index rows, one row per
  run × window × agent with the agent's response (from the 95.5 row) and `Hpersnonfoc`;
- `greedy_uncorrelated_subset`: the old script's greedy search for weakly correlated metrics on the Fig. A4
  correlations (same runs). It starts with the metric of the lowest mean |r| to all candidates and then adds the
  one whose largest |r| to those already picked is smallest. Columns are `rank`, `metric`, `mean_abs_r`,
  `max_abs_r` (to the earlier picks, NaN for the first) and `predictor` (one of the five hand-chosen predictors).

## Differences from the old scripts

These are the real bugs, and they are fixed:

- **Fig. 6D y-ticks:** the labels show the true values (the old ones were 1.25× too large).
- **Fig. A4 title:** it states that all three windows are pooled.
- **`fixedBNat100` filename typos:** gone, because runs are found through the index.
- **`figs/` creation order:** `save` creates `out_dir` first.
- **`nr_negs`:** it uses the before-window row (95.5), not the t = 100 snapshot.
- **Table A5 column order:** the s = 0 reference columns come after the three resistant ones, as in the paper. The
  old print cell interleaved them with the resistant columns (resistant before, reference before, resistant
  during, …). When a response type is absent, its proportion cell keeps its three columns (the old cell wrote one).

Everything else matches. The defaults reproduce the paper's choices: example agents, example seeds, the Fig. A3
draws (a local `RandomState(14)` instead of the global seed), the Fig. 6A/B agents (`RandomState(42).randint(N)`
instead of `randint(100)` on the global seed; the strip-plot jitter continues that RNG, as it did the global one) and
the response-frequency denominators (Fig. 3 counts `latecompliant`, Fig. 6 doesn't).

One deliberate change: figures use every seed they find, so Fig. 3 on a 100-seed folder uses 100 seeds instead
of the old script's 20. Pass `seeds=range(20)` to reproduce the old Fig. 3.

Fig. A8 finds its variants through the index, so the `_M`/`_N`/`_tau` runs the package writes are found (legacy runs
need the suffixes added by hand, known issue #4) and the old `_fixedBNatt100` typo doesn't apply. Two small
differences in the drawing: each row's middle value is the base's runs drawn once (the old script read the ε = 1 and
ω₀ = 0.2 runs twice, as a variant and as the base, so those dots were doubled; the medians are the same), and an
undrawn row leaves its panels empty, the panel letters running on over the drawn ones. The p row shows p·N with the N
of each run (from its distinct `id`s). The variant values are absolute, so for a non-paper base a variant equal to the
base is dropped.

### Checking against the old scripts

`pytest -m slow tests/test_old_vs_new_*.py` runs the unchanged `analyse_*.py` scripts on `simOut_old_model` and
compares their globals with the package's `FigureResult.data` and tables. It's skipped when that folder is
missing. The harness (`tests/old_scripts.py`, fixture `old_script`) runs a script cell by cell in a temporary
directory (`simOut` links to `simOut_old_model`, `figs/` exists, Agg backend). Each script runs once per session.
A cell that fails under the current dependencies is reported by number and line, and the later cells still run,
so a comparison can use the parts that ran. PNGs are compared with `png_difference`, the fraction of pixels whose
RGB values differ by more than 0.05.

| Figure | Old script | Compared | PNG threshold (measured) |
|---|---|---|---|
| Fig. 2 | `analyse_timeseries.py` | focal trajectories, example BNs, `example_agent_stats`, `focal_sign_counts`: exact | 0.1 % (0.004 %) |
| Fig. 3 | `analyse_responseFrequencies.py` | per-simulation frequencies (`seeds=range(20)`): exact | reported only (0.33 %, strip-plot jitter) |
| Fig. A3 | `analyse_responseFrequencies.py` | highlighted agents (from the script's printout): same | 0.1 % |
| `polarisation_summary` | `analyse_responseFrequencies.py` | σ_foc, groupishness (mean, std), mean \|ω\|, mean ω: rtol 1e-12; `nr_negs` differs on purpose | — |
| Fig. 5 | `analyse_BNmetrics.py` | mean, sd and n per metric × response × window (100 seeds): rtol 1e-12 | 0.1 % |
| Fig. A4 | `analyse_BNmetrics.py` | correlation matrix (s = 0, three windows pooled): rtol 1e-12 | reported only (0.47 %, the title) |
| Fig. 4 | `analyse_BNmetrics.py` | Cohen's d (rtol 1e-10), logit coefficients and CIs of the five predictors (rtol 1e-6), per contrast × s | 0.1 % (0 %) |
| Tables 3, A1–A7 | `analyse_BNmetrics.py` | the LaTeX against the script's captured stdout, whitespace ignored: identical (Table A5 after moving the old reference columns last) | — |
| `greedy_uncorrelated_subset` | `analyse_BNmetrics.py` | the order (`chosen`): same | — |
| Fig. 6 | `analyse_socialAdaptation.py` | H and its z-score, the z-scored metrics, the frequencies (11 μ × 100 seeds): rtol 1e-12 (z-scores 1e-10) | 0.1 % (0.05 %, the D tick labels) |
| Fig. A7 | `analyse_socialAdaptation.py` | the PNG | 0.1 % (0 %) |
| Fig. A8 | `analyse_sensitivity.py` | per-simulation frequencies per row × config × value (100 seeds): exact. `simOut_old_model` has no sensitivity runs, so that test skips, and a stand-in folder (every name the script reads linked to a baseline run of its config, each name at a different seed) is compared instead | reported only (1.2 %, jitter and the doubled dots) |

Fig. 3's strip plots jitter with the global NumPy RNG (seaborn has no RNG argument). The old script leaves it
unseeded, and the package seeds it per figure and restores it afterwards, so a figure is the same on every run.
