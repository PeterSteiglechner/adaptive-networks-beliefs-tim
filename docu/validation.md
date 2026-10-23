# Validation of the `anb` package against `ANBmodel.py`

**Date:** 2026-10-16 (the conditional and lean families added and rerun on 2026-10-23, from the cached runs)
**Status:** validation passed, all four families. The deliberate differences below still need the supervisor's sign-off before new
runs are used in results (ADR-001, ADR-002 action item 8).

This closes the validation and benchmark items of [ADR-001](adr-001-numba-over-gpu.md) (items 4, 5) and
[ADR-002](adr-002-testbed-architecture.md) (item 5), following seam 1 of the [spec](spec-anb-package.md).

## How to rerun

```bash
pytest -m slow -s tests/test_statistical_validation.py   # ~25 min on 12 cores, almost all of it legacy
pytest -m slow -s tests/test_benchmark.py                # ~3 min
```

The validation writes `report.md` and four tables (`pooled_responses.csv`, `conditional_responses.csv`,
`lean.csv`, `metrics.csv`) to `../validation_out/` (git-ignored). It caches one summary row per run in
`../validation_out/runs/`. A rerun only adds missing cells, so **delete that folder after changing the model**.
Older versions cached in `simOut_old_model/validation/`. That folder is no longer used and can be deleted. The
runs are made in `tests/validation.py`, but the paper configs, the `anb` testbed of a cell, the summaries, tests
and tables come from `anb.comparison`, the same code as the [folder comparison](#comparing-two-output-folders), so
the two can't disagree on method. The statistics are tested in `tests/test_comparison.py`, the folder comparison
in `tests/test_folder_comparison.py`, and the mechanism layer in `tests/test_ablation.py`,
`tests/test_shared_start.py` and `tests/test_shared_start_divergence.py`.

## Method

- **Cells:** the 4 paper configurations (static ω₀ = 0.2, static ω₀ = 0.8, adaptive→static, adaptive) × 6
  pressures (0, 1, 2, 4, 8, 16), with seeds 0–99 for each cell and implementation. p = 0.1, β = 3, ρ = 1/3,
  N = 100, M = 10, single event. Legacy runs through `run_simulation`, with a fresh module per run so the
  shuffle state can't leak between runs (known issue #2). `fixedBNat100` maps to `learning_until = 101`.
- **Unit of comparison: the run, not the agent.** Agents in one run share a network and pull each other to
  consensus, so agent-level tests are badly overdispersed. In the static ω₀ = 0.2 cells whole runs end on one
  side, and the pooled χ² is ~130 for two identical-looking seeds. Each run is reduced to its response counts
  and to the run mean of every metric in each window (95.5, 145.5, 195.5).
- **Response frequencies:** per cell, the χ² statistic of the pooled 2 × 7 table (6 responses + unlabelled). Its
  p-value comes from 10,000 permutations of whole runs between the two implementations.
- **Conditional response frequencies:** the same test on the agents negative before pressure only (compliant,
  resilient, resistant, latecompliant), which takes the lean out of the response shares.
- **Lean:** per cell, Welch on the run-level share positive before pressure and on the run mean of `x_focal` before
  pressure. There's no KS, because both are roughly bimodal (a run mostly ends on one side).
- **BN metric distributions:** per cell × window × metric (all 14 columns), a two-sample KS test on the 100 run
  means, plus the difference of means with a 95 % Welch CI to show the size of any deviation.
- **Multiple comparisons:** Benjamini–Hochberg within each family (24 pooled response tests, 24 conditional
  response tests, 48 lean tests, 1008 metric tests). A test fails at q < 0.05. A test with fewer than two runs on
  a side has no p-value; it gets q NaN, doesn't fail and doesn't enter the correction of the others.

## Results

| Family | Tests | Failures | Smallest p | Smallest q |
|--------|-------|----------|-----------|-----------|
| Response frequencies | 24 | **0** | 0.080 | 0.43 |
| Conditional response frequencies | 24 | **0** | 0.072 | 0.96 |
| Lean (share positive, `x_focal` before) | 48 | **0** | 0.073 | 0.32 |
| BN metrics | 1008 | **0** | 0.00074 | 0.37 |

Response shares (legacy → anb, share of all agents), for the configurations in the paper's Fig. 3 at s = 4:

| Configuration | compliant | resilient | resistant |
|---------------|-----------|-----------|-----------|
| static ω₀ = 0.2 | 0.539 → 0.459 | 0.001 → 0.000 | 0.000 → 0.000 |
| static ω₀ = 0.8 | 0.000 → 0.000 | 0.003 → 0.004 | 0.507 → 0.493 |
| adaptive→static | 0.101 → 0.095 | 0.300 → 0.284 | 0.107 → 0.104 |
| adaptive | 0.137 → 0.133 | 0.228 → 0.210 | 0.143 → 0.139 |

The full tables for every cell and every metric test are in `../validation_out/`. The lean is the same in the 6
pressures of a configuration (pattern 2 below), so its 48 tests are 8 distinct ones; the smallest p is
`x_focal` before pressure in static ω₀ = 0.8 (−0.022 → 0.006, 1.8 SE).

### Patterns looked at (not failures)

Both follow-ups can be rerun from `tests/` after `validate()`: `gate_check(pressure, learning_until)` and
`positive_share(impl, config)` in `validation.py`.

1. **`bc_foc` in adaptive→static at s = 8 and 16** (during and after windows, diff −0.19 and −0.23, raw
   p = 0.006 and 0.0007). The gap grows with pressure, which points to the `fixedBNat100` gate (difference 2
   below). Legacy still adapts the BN at t = 101, the first pressure step, and that step's large focal-belief
   change rewires the focal edges. Rerunning `anb` with `learning_until = 102` (legacy's effective gate) removes
   the gap: s = 8 gives diff +0.012 [−0.092, 0.115], p = 0.37, and s = 16 gives +0.013 [−0.093, 0.119],
   p = 0.70. So it is the deliberate difference, not a bug.
2. **The share of agents positive before the pressure is higher in `anb` in all 4 configurations** (e.g.
   persistPos 0.46 → 0.54 in static ω₀ = 0.2). The before window is identical across the 6 pressures of a
   configuration, so this is 4 correlated observations, not 24, and run-level signs are close to coin flips.
   With 1000 seeds, `anb` gives 0.50–0.51 in every configuration, and legacy gives 0.501 ± 0.002 in static
   ω₀ = 0.8, the configuration whose 100-seed value (0.489 ± 0.005) was furthest from 0.5. Neither
   implementation has a sign bias.

**Power.** With 100 runs a side, a single-cell difference of about 0.5 run-level SD (like pattern 1) reaches raw
p ≈ 10⁻³ but doesn't survive correction over 1008 tests. A bug that affects many cells, or one cell strongly,
would show up. A small effect confined to one cell needs a targeted rerun with more seeds, as above.

## Comparing two output folders

The validation above runs its own simulations of the 4 paper configs. The comparison framework instead compares
two folders of ensemble CSVs that already exist, for example the legacy sweep in `simOut_old_model` against the
`anb` sweep in `simOut_new_model`, over every cell on disk (the μ sweep included). It works the same for any two
`anb` sweeps, e.g. two network types. It uses the same statistics as the validation (see [Method](#method)), and
adds a mechanism layer that explains where a difference comes from: ablations, reruns and shared-start runs.

### Running it

```bash
python compare_old_new.py      # or cell by cell; it's a # %% script like the analyse_*.py scripts
```

The constants at the top of `compare_old_new.py` configure it. Most can be overridden by the environment
variables in the table (as the smoke tests do); the others (`PRESSURES`, `RERUN_MUS`, `RERUN_PRESSURES`,
`ABLATION_LEARNING_UNTIL`, `SHARED_START_CONFIG`, `SHARED_START_SEED`) are edited in the script.

| Environment variable | Default | Meaning |
|---|---|---|
| `COMPARE_OLD`, `COMPARE_NEW` | `simOut_old_model`, `simOut_new_model` | the two folders; "old" is reported as `legacy`, "new" as `anb` |
| `COMPARE_OUT` | `compare_out` | output folder (git-ignored) |
| `COMPARE_TOP_K` | 10 | rows in the summary's top table, and cells that get time-series plots |
| `COMPARE_N_PERMUTATIONS` | 10000 | permutations per χ² test |
| `COMPARE_N_JOBS` | -1 | `joblib` workers, for reading CSVs and for the optional cells' runs |
| `COMPARE_RUN_ABLATIONS`, `COMPARE_RUN_RERUN`, `COMPARE_RUN_SHARED_START` | 0 | set to 1 to run the [optional cells](#ablations-reruns-and-shared-start-runs) |

From Python, `anb.comparison.compare_folders(old, new, permutation_seed=0, n_permutations=10_000, cache_dir=None,
n_jobs=-1)` does the comparison and writes nothing but the summary cache (only when `cache_dir` is given). It
returns a `Comparison` with the `pairing` (`matched`, `only_old`, `only_new` and `unparsed` file stems), the run
summaries `old` and `new` (one row per run), and the tables `pooled_responses`, `conditional_responses`, `lean`,
`metrics` and `top_differences`.

**Pairing and cells.** Only `sim_*.csv` files directly in each folder count (subfolders such as `detailed/` are
ignored). Files are paired by name, seed included. Pairing only decides which runs enter a cell: within a cell the
two sets of runs are compared as samples, not seed by seed, because the RNG streams differ (deliberate difference
1). Files in one folder only, and paired files whose name isn't a legacy ensemble name (e.g. a run saved with
`legacy_names=False`), are listed in `summary.md` and left out; the latter are never read. Each file name is
parsed back into its parameters
(`parse_filename`, the inverse of `legacy_filename`), so any sweep is covered without a config list. A cell is
labelled with its paper config name (`static0.2`, `static0.8`, `adaptive-static`, `adaptive`) when it matches one,
otherwise with the parameters that differ from the adaptive config, e.g. `mu=0.1`.

**The summary cache.** Reading two sweeps of 8,400 CSVs each takes a while, so the run summaries are cached in
`compare_out/cache/`, one file per (folder, cell). The file holds every run of the cell in that folder, paired or
not, so comparing one folder with partners of different seed coverage (the main comparison and the ablation)
reuses it. A cell is reread when any of its files is added, removed or changed (name, modification time or size),
so the cache never needs deleting by hand. Uncached cells are read in parallel.

**What it writes to `compare_out/`:**

| File | Content |
|---|---|
| `summary.md` | the verdict in a minute: files matched, files not compared (in one folder only, or not legacy names), tests, untested tests (p NaN) and failing tests per family, the top-k effects |
| `pooled_responses.{csv,md}`, `conditional_responses.{csv,md}`, `lean.{csv,md}`, `metrics.{csv,md}` | the four test tables, described below |
| `top_differences.{csv,md}` | the four families stacked and ranked by effect |
| `response_shares_pooled.png` | pooled response shares over pressure, one panel per paper config |
| `response_shares_conditional_mu.png` | shares among agents negative before pressure over μ (the adaptive config plus the `mu=…` cells), one panel per pressure |
| `timeseries_<config>_pressure<s>.png` | mean of each metric at the tracked steps for the top-k cells of `top_differences`; characters of the config other than letters, digits and dots become `_` (`timeseries_adaptive_static_pressure4.png`, `timeseries_mu_0.1_pressure4.png`) |

In the plots old is solid and new dashed. The bands are 95 %: a bootstrap over whole runs for the response shares,
±1.96 SE over runs for the time series. Overlapping bands mean "no visible shift", not "proven equal".

### Reading the tables

Every table has one row per cell (the lean table: per cell × quantity; the metric table: per cell × window ×
metric), the cell's `config` and `pressure`, a raw `p`, the Benjamini–Hochberg `q` within its family and `fail`
(q < 0.05). Values are written as `<name>_legacy` / `<name>_anb` for old and new, and `diff` is always new − old. Each table is **ranked by effect
size, not by p**, so the biggest differences come first whatever their significance.

| Table | Test | Effect (ranking) | Other columns |
|---|---|---|---|
| `pooled_responses` | χ² on the pooled counts of the 6 responses + `unlabelled`, p from permuting whole runs | `cramers_v` | `chi2`, the share of each response on both sides |
| `conditional_responses` | the same, on the agents negative before pressure only (compliant, resilient, resistant, latecompliant) | `cramers_v` | shares within those agents |
| `lean` | Welch, per `quantity`: `share positive` (the run-level share positive before pressure) and `x_focal@before` (the run mean of the focal belief before pressure) | `smd` (by its absolute value) | `mean_legacy`, `mean_anb`, `diff`, `se`, 95 % CI `ci_low`…`ci_high`, `gap_se` (= diff / SE) |
| `metrics` | KS on the run means of each of the 14 metrics, per window (`before` 95.5, `during` 145.5, `after` 195.5) | `smd` (by its absolute value) | `mean_legacy`, `mean_anb`, `diff`, `se`, the Welch CI, `gap_se`, `ks` |
| `top_differences` | all four stacked | `effect` (Cramér's V or \|SMD\|) | `family`, `detail` (e.g. `bc_foc@after`, `share positive`), `q`, `fail` |

Cramér's V and the SMD aren't on the same scale, so compare effects within a family. A metric that is constant
and equal on both sides (e.g. the edge metrics of a static BN) gets p = 1. Two different constants still go to
the KS test (and the Welch CI is infinite). A test with fewer than two non-NaN runs on a side (e.g. a cell with
one seed, or a metric that is NaN in every run) has p and q NaN and doesn't fail; `summary.md` counts these as
untested.

**Separating the three sources of a difference.** Go through them in this order:

1. **Seed noise, mainly the lean.** A run ends mostly positive or mostly negative before pressure almost like one
   coin flip, so the share positive before pressure varies a lot between 100-run ensembles. The positive-before
   agents are persistPos/nonpersistPos by definition, so the lean moves the pooled shares directly. Look at the
   `lean` table first. A `gap_se` within about ±2 is seed noise. Then look at `conditional_responses`, which takes
   the lean out: if the pooled table differs but the conditional one doesn't, the gap is the lean, not a
   different response to pressure. Also keep in mind that the before window is the same in all pressures of one
   config (pattern 2 above), so a lean gap shows up in all 6 pressure
   cells at once. That's one observation, not six. A cell that passes BH but has a sizeable effect is
   underpowered rather than equal; settle it with a [rerun](#ablations-reruns-and-shared-start-runs).
2. **Deliberate differences** (listed [below](#deliberate-differences-from-legacy-for-sign-off)). The RNG stream
   only rules out seed-by-seed matching, it shouldn't move any table. The t = 101 gate is expected to show up only
   in the `adaptive-static` cells under pressure, in the focal edge metrics of the `during` and `after` windows
   (`bc_foc` in pattern 1 above; other focal edge metrics such as `absOm_foc` may move with it), not in `before`
   and not at pressure 0. Confirm it with the ablation: if
   `learning_until = 102` removes the gap, it's the deliberate difference.
3. **Unexplained differences.** What's left: a failing conditional response cell, a metric failure outside the
   gate's pattern, or a gap the ablation doesn't remove. First rerun the cell with more seeds. If it survives,
   run a shared start to find the first step and quantity where the two models diverge, then look there for a
   bug.

### Ablations, reruns and shared-start runs

These cells of `compare_old_new.py` are off by default, because they run simulations. Ablations and reruns save
normal output folders and go through `compare_folders`. Their tables and a `summary.md` like the main one go to
`compare_out/<name>/`, and the cell checks that every expected file was paired.

**Ablation** (`COMPARE_RUN_ABLATIONS=1`; seeds: `COMPARE_ABLATION_SEEDS`, default 100; folder: `COMPARE_ABLATION`,
default `simOut_ablation_learning102`). It runs `anb` for the adaptive→static config at all 6 pressures with the
t = 101 gate set back to legacy's (`learning_until = 102`), saves the runs under the `adaptive-static` file names
(`save_under_legacy_names(..., fixedBNat100=True)`) and compares them with the old folder. Tables:
`compare_out/ablation_learning_until/`. Only the `adaptive-static` cells pair; the rest of the old folder is
listed as old-only in that `summary.md`, which is expected. The gate is explained if the focal edge gaps of the
main report (`bc_foc` under strong pressure) are gone here. The ablation folder holds runs filed under another cell's name, so don't mix it with real
runs and don't point the analysis scripts at it. Another ablation (another deliberate difference reverted) is a
new cell of the same shape.

**Rerun** (`COMPARE_RUN_RERUN=1`; seeds: `COMPARE_RERUN_SEEDS`, default 1000; folders: `COMPARE_RERUN_OLD`,
`COMPARE_RERUN_NEW`, defaults `simOut_old_model_rerun`, `simOut_new_model_rerun`). It reruns the adaptive config
at the μ values in `RERUN_MUS` (default 0.1) at all 6 pressures with `anb`, and compares with the old rerun folder.
**Legacy is not run by the script.** Run it first over the same seeds, μ and pressures (edit the `__main__` block
of `ANBmodel.py`, see [running-simulations.md](running-simulations.md)) and copy its CSVs from
`simOut_old_model/` into the old rerun folder, since files pair by name, seed included. **Back up the main sweep's
files of those cells first.** `ANBmodel.py` always writes to `simOut_old_model/` and always overwrites (known
pitfall `if True:`), and its shuffle state carries over between runs (deliberate difference 1), so the rerun
replaces e.g. the μ = 0.1 seeds 0–99 of the main sweep with new realisations. Restore the backup afterwards, and
copy rather than move, or those cells drop out of the main comparison as new-only. The `anb` side uses
`skip_existing=True`, so an interrupted rerun resumes; delete the new rerun folder after changing the model.
Tables: `compare_out/rerun/`. With 10× the seeds, the SE is about a third, so a 100-seed gap that was noise
shrinks and a real one gets clearly significant.

**Shared start** (`COMPARE_RUN_SHARED_START=1`; pressures: `COMPARE_SHARED_START_PRESSURES`, default `4,16`;
replicas: `COMPARE_SHARED_START_K`, default 20; steps: `COMPARE_SHARED_START_WINDOW`, default `95,155`; the config
is `SHARED_START_CONFIG`, default `adaptive-static`, and the start seed `SHARED_START_SEED`, default 0). Legacy and
`anb` both start from one identical t = 0 state (beliefs, edge weights, adjacency, belief-change memory), drawn
from the start seed. Each runs K replicas, replica r with seed start seed + 1 + r (so no `anb` replica reuses the
stream that drew the start), and every step in the window is recorded. This rules out the initial conditions
as a source of a difference. `ANBmodel.py` isn't changed: each legacy replica gets a fresh module copy from the
legacy oracle, with its initialisation replaced by the shared state. From Python:
`run_shared_start(cell, pressure, start_seed=0, k=20, window=(95, 155), n_jobs=1, schedule=None,
load_legacy=None)` returns the trajectories in long format, and `compare_shared_start(trajectories)` compares
them. `cell` is a dict of `link_prob`, `init_w`, `beta`, `rho`, `eps`, `mu`, `fixedBNat100` and optionally `lam`,
as `parse_filename` returns it or `paper_cell(config)` builds it. `schedule` defaults to the paper's single event,
and `load_legacy` to the oracle's loader. Output in `compare_out/shared_start/`:

| File | Content |
|---|---|
| `divergence.csv/.md` | per pressure, step and quantity (the 14 metrics' agent means and `x_focal_sd`, the spread of the focal belief): `diff` (anb − legacy of the replica means) with 95 % CI, KS `p`, BH `q` over the steps × quantities of that pressure, `fail` |
| `first_divergence.csv/.md` | per pressure and quantity: `first_divergent_step` (the first step with q < 0.05, empty if none), `n_divergent_steps` of `n_steps` |
| `divergence_<config>_pressure<s>.png` | `diff` with its CI per quantity over the window; dots mark failing steps, a dashed line the first divergent step |

Reading it: the replicas are the unit, so with K = 20 only fairly large shifts are detected. A single marked step
among clean ones is likely a false discovery; a real divergence usually stays marked over a run of steps. For
`adaptive-static` the gate is expected to show up from t = 101/102 in the focal edge metrics. A divergence before
t = 101, or in the beliefs (`x_focal`, `x_focal_sd`) without one in the edges, points to something other than the
gate. The legacy replicas dominate the cost, so raise K with care.

## Benchmark

On 2026-10-16, on 12 cores (11 `joblib` workers, as legacy `__main__` uses). CPU seconds are summed over runs.

| | legacy | anb | speed-up |
|---|---|---|---|
| One run: update loop | 3.19 s | 0.052 s | 62× |
| One run: metrics | 0.32 s | 0.036 s | 9× |
| One run: total | 3.52 s | 0.100 s | 35× |
| 240-run ensemble: wall time | 151 s | 3.3 s | 46× |
| 240-run ensemble: update [CPU s] | 1506 | 20.3 | 74× |
| 240-run ensemble: metrics [CPU s] | 131 | 9.9 | 13× |
| 240-run ensemble: total [CPU s] | 1641 | 35.6 | 46× |

The update loop is inside ADR-001's 50–200× target. The first sweep compiles in about 2.8 s once per machine,
then loads from Numba's disk cache in 0.1 s. The paper's 100-seed sweep (2400 runs) drops from about 25 min to
under 1 min.

**`FocalBetweenness` (the Brandes swap).** After the sweep kernel, profiling one run (adaptive, s = 4) showed the
metrics at 56 ms, more than the 52 ms update loop, and `FocalBetweenness` at 40 ms, 72 % of the metrics. The
per-agent igraph builds were replaced with a private `@njit` Brandes helper over all agents, which ties path
lengths with igraph's relative tolerance (1e-10). It equals igraph on random edge matrices: continuous, on a
coarse grid (many tied shortest paths), sparse with zero edges, and all-equal. It also still equals legacy
`bc_foc` (`tests/test_focal_betweenness.py`). The `Metric` interface is unchanged. Per run it now takes 19 ms,
and the metrics come to 36 ms, 28 % of the run in the ensemble. The rest of the metric time is mostly
`WeightedClustering` (6 ms) and `FocalTriangleBalance` (5 ms), so further metric work isn't worth it for now.

## Deliberate differences from legacy, for sign-off

1. **The RNG stream (known issue #2).** `anb` creates one `np.random.default_rng(seed)` per run and passes it
   everywhere: initial beliefs, the Erdős–Rényi network, the agent and belief update orders, and Glauber sampling
   by inverse CDF instead of `np.random.choice`. Legacy uses the global `np.random`, reseeded twice per run, plus
   module-level lists that are shuffled in place and carry over between runs in one process. As a result:
   - A seed does **not** reproduce an old CSV, and `seed` 7 in `anb` is a different network and trajectory than
     `seed` 7 in legacy. Only the ensemble statistics match (shown above).
   - In exchange, a seed fully determines its run in `anb`, independent of `joblib` scheduling and of which runs
     came before. Legacy can't guarantee that.
   - Old and new CSVs can be pooled as samples from the same model, but not matched seed by seed.
2. **A single `fixedBNat100` meaning (known issue #8).** Legacy has two gates. The module-level flag (`t ≥ 101`,
   always off, so dead code) and the per-run flag (`t > 101`), which zeroes ε and μ but not λ. `anb` has one
   agent parameter, `learning_until`, and skips the whole edge update, λ included, once `t ≥ learning_until`.
   Paper runs use `learning_until = 101`, so the BN is static from the first pressure step on, as the paper
   describes ("adaptive until t = 100 and static thereafter"). Legacy adapts once more at t = 101. Effects:
   - Response frequencies show no detectable difference.
   - `bc_foc` of adaptive→static under strong pressure is lower by about 0.2 (pattern 1 above). Other metrics
     show no detectable difference.
   - λ = 0 in every paper run, so skipping the decay has no effect there.
   - To reproduce legacy's timing exactly, use `learning_until = 102`. Such runs get `fixedBNat100 = False`
     plus a `learning_until` column, and have no legacy filename (save them with `legacy_names=False`), so
     they can't overwrite or be mistaken for paper runs. The one exception is the
     [ablation](#ablations-reruns-and-shared-start-runs): `save_under_legacy_names(..., fixedBNat100=True)` files
     them under the `adaptive-static` names on purpose, in a folder of their own.

Unchanged on purpose, for comparability with the paper: `Hsoc` keeps the neighbours' mean (`SocialEnergy("sum")`
gives the eq. 1 term, known issue #1), the response rules keep their zero and boundary handling (#9), and file
names keep legacy's format without `_M`/`_N`/`_tau` suffixes (#4).

**Sign-off:** ☐ RNG stream (1) ☐ single `fixedBNat100` meaning (2), supervisor, date: ________
