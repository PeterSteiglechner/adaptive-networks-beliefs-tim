# Old-vs-new comparison framework

**Labels:** ready-for-agent
**Status:** open

## Problem Statement

I ran every simulation twice: with the frozen legacy model, written to `simOut_old_model`, and with the `anb`
package, written to `simOut_new_model`. Each folder holds 8400 ensemble CSVs: 14 configs × 6 pressures × 100
seeds. They cover the 4 paper configs and a μ sweep (eps = 1, init_w = 0.2, μ from 0.001 to 1.0). When I run
the analysis scripts on each folder, the figures differ, most visibly in how agents adapt under pressure: the
shares of compliant, resilient and resistant agents. There are differences in both the paper configs and the
μ sweep.

The existing statistical validation only covers the 4 paper configs at μ = 0. It runs its own simulations,
not the CSVs I made, and it reports only pooled response shares. That leaves me unable to tell which of the
following explains the gaps:

- (i) real model differences;
- (ii) the known, deliberate differences (the RNG stream and the t = 101 learning gate);
- (iii) seed noise, in particular the run-level "lean" (whether a run ends up mostly positive or negative
  before pressure), which acts almost like one coin flip per run.

An investigation by hand found three things:

- Most of the pooled gap comes from the lean (share positive before pressure: 0.492 in old, 0.517 in new, about
  1 SE apart). Conditioned on agents that are negative before pressure, the response shares agree.
- The t = 101 gate gives a real difference in `bc_foc` and `absOm_foc` for adaptive→static under pressure.
- At μ = 0.1 the conditional compliant share differs by 0.02–0.04, and it's unclear whether that is noise.

I want this analysis to be repeatable and to cover every config on disk, not done by hand once.

## Solution

There are two parts:

- **A comparison module in the `anb` package.** It takes two output folders, pairs their CSVs by filename,
  infers each cell (config × pressure) from the legacy filename, and reduces each run to a one-row summary. It
  then produces three ranked difference tables with multiple-testing correction:
  - **Response table:** pooled response shares, plus response shares conditional on being negative before
    pressure.
  - **Lean table:** the share positive before pressure, and `x_focal` before pressure.
  - **Metric table:** every BN metric per analysis window.
- **A mechanism layer that explains where any difference comes from.** It has two tools:
  - **Ablations:** anb runs with one known difference set back to legacy behaviour, e.g. `learning_until = 102`.
  - **Shared-start runs:** legacy and anb both start from the identical t = 0 state, K replicas each, so their
    trajectories around the pressure onset can be compared as ensembles.

A thin `# %%` report script, in the style of the `analyse_*.py` scripts, calls the module and writes these to
a git-ignored `compare_out/`:

- a ranked difference table;
- side-by-side old/new plots of the response shares (pooled and conditional) over μ and pressure;
- overlaid metric time series for the top-k differing cells.

The existing `tests/validation.py` is rewired to use the same summary and statistics code, so there is only
one implementation of the tests.

## User Stories

1. As a researcher, I want to point the framework at two output folders, so that I can compare any two sweeps
   without editing code.
2. As a researcher, I want CSVs paired by their legacy filename, so that each old run is compared with the
   new run of the same cell.
3. As a researcher, I want the framework to list files that exist in only one folder at the top of the
   report and then continue, so that a coverage gap is visible but doesn't block the analysis.
4. As a researcher, I want each cell's parameters (link_prob, init_w, beta, rho, eps, mu, fixedBNat100,
   ext_strength) parsed from the filename, so that the μ sweep and any future sweep are covered without a
   hard-coded config list.
5. As a researcher, I want each run reduced to one summary row (response counts, conditional response counts,
   share positive before pressure, per-window run means of every metric), so that the unit of comparison is
   the run and not the correlated agents within it.
6. As a researcher, I want the summaries cached per folder and cell, so that re-running the report after a
   change to the plotting doesn't re-read 16,800 CSVs.
7. As a researcher, I want the cache to be invalidated when the source CSVs change, so that I never compare
   stale summaries.
8. As a researcher, I want a χ² test with a run-level permutation p-value on the pooled response counts per
   cell, so that I see whether the response distribution differs.
9. As a researcher, I want the same χ² permutation test on response counts conditional on being negative
   before pressure (compliant, resilient, resistant, latecompliant), so that I can separate "agents respond
   differently to pressure" from "different agents were exposed to pressure".
10. As a researcher, I want a separate test on the run-level share positive before pressure per cell, so that
    I can see how much of a pooled gap is just the lean.
11. As a researcher, I want the lean reported with its standard error and the gap in units of SE, so that I
    can judge at a glance whether it's within seed noise.
12. As a researcher, I want KS tests plus a Welch difference-of-means CI on the run means of every metric per
    window (before, during, after), so that I can see BN-structure differences such as `bc_foc` and
    `absOm_foc`.
13. As a researcher, I want Benjamini–Hochberg correction at q = 0.05 within each test family (pooled
    responses, conditional responses, lean, metrics), so that 1000+ tests don't produce false alarms.
14. As a researcher, I want every table ranked by effect size (Cramér's V for the response tables, a
    standardised mean difference for the metrics), so that the biggest differences come first, whatever the
    p-value.
15. As a researcher, I want a combined "top differences" table across all families, so that I have one place
    to start.
16. As a researcher, I want old-vs-new plots of the pooled response shares over pressure for each paper config,
    so that I can see the figure from the analysis scripts side by side.
17. As a researcher, I want old-vs-new plots of the conditional response shares over μ for each pressure, so
    that I can check the μ sweep, which was never validated.
18. As a researcher, I want confidence bands (run-level bootstrap or SE) on those plots, so that I can tell a
    shift from noise by eye.
19. As a researcher, I want overlaid old/new time series of the mean metric values (at the tracked steps) for
    the top-k differing cells, so that I can see when they diverge.
20. As a researcher, I want everything written to a git-ignored `compare_out/`, so that reports don't clutter
    the repo.
21. As a researcher, I want a short markdown summary at the top of the report (the number of failing tests
    per family, the largest effects, the files missing on one side), so that I can read the verdict in a
    minute.
22. As a researcher, I want to run ablations: anb runs with one known deliberate difference reverted (e.g.
    `learning_until = 102` for the t = 101 gate), compared to the old folder with the same tables, so that I
    can confirm which known difference explains a gap.
23. As a researcher, I want ablation runs saved as a normal output folder, so that the folder comparison
    simply reuses them.
24. As a researcher, I want shared-start runs, where legacy and anb start from the identical t = 0 state
    (beliefs, edge weights, adjacency, belief-change memory) with K replicas each, so that the initial
    conditions are ruled out as a source of difference.
25. As a researcher, I want shared-start trajectories recorded at every step in a window around the pressure
    onset (t = 95–155 by default), so that I can see at which step and in which quantity the two ensembles
    diverge.
26. As a researcher, I want the shared-start ensembles compared step by step (mean and spread of focal
    belief, edge-weight metrics and energies), so that I get a "first divergent step" per quantity.
27. As a researcher, I want shared-start runs to need no change to `ANBmodel.py`, so that legacy stays the
    frozen reference.
28. As a researcher, I want the shared-start runs to use the legacy oracle's fresh module copy for each run,
    so that the module-level shuffled lists don't carry state between runs (known issue #2).
29. As a researcher, I want to be able to rerun selected cells with more seeds (e.g. 1000 at μ = 0.1) through
    the same comparison, so that an ambiguous cell can be settled.
30. As a researcher, I want the slow validation test to use the same summary and statistics code, so that the
    validation and my reports can never disagree on method.
31. As a future me, extending the network model, I want to compare any two `anb` sweeps (e.g. ER vs a new
    network type) with the same framework, so that it remains useful after the legacy comparison.
32. As a supervisor, I want a report that separates seed noise, deliberate differences and unexplained
    differences, so that I can sign off on the deliberate differences in `docu/validation.md`.

## Implementation Decisions

- **A new comparison module in the `anb` package.** It holds:
  - the folder pairing and filename parsing (the inverse of `legacy_filename`);
  - the run summariser;
  - the statistics: χ² permutation, KS and Welch, BH correction, effect sizes;
  - the table builders;
  - the shared-start runner.

  The statistics and summariser currently in `tests/validation.py` move here. They stay plain Python, with no
  Numba.
- **The main entry point takes two folders.** It is a function given an old and a new folder, plus options
  (cache directory, permutation seed, n_permutations). It returns a result object holding:
  - the pairing report (files matched, and files present on one side only);
  - the run-summary frames for both sides;
  - the four test tables (pooled responses, conditional responses, lean, metrics);
  - the combined top-differences table.

  It doesn't write files. Writing is the script's job.
- **Run summary.** This is one row per run.
  - **Response counts:** counts of the 6 response classes and `unlabelled`, from the t = 95.5 row.
  - **Conditional counts:** the same counts among agents negative before pressure. These use the 4 negative
    classes. persistPos/nonpersistPos are by definition the positive-before agents.
  - **Lean:** the share positive before pressure, (persistPos + nonpersistPos) / N.
  - **Metrics:** the run mean of each of the 14 metrics at t = 95.5, 145.5 and 195.5.
  - **Time series:** the mean of each metric at each tracked step, for the time-series plots.
- **The cell key is parsed from the filename.** It is the legacy filename tokens without the seed. The
  framework parses names only; it never builds legacy filenames differently from `legacy_filename`, since the
  analysis scripts depend on that format.
- **Labels on the cell.** A cell is labelled as a paper config (static0.2, static0.8, adaptive-static,
  adaptive) when its parameters match one. Otherwise its label comes from the parameters that differ, e.g.
  `mu=0.1`.
- **Families and correction.** Benjamini–Hochberg q < 0.05 within each family. Effect sizes:
  - response tables: Cramér's V from the pooled χ²;
  - lean and metrics: the standardised mean difference of the run-level values.

  Rankings use the effect size, and the q-value is shown next to it.
- **Lean test.** Welch on the run-level share positive, plus the gap in units of SE. A KS test isn't useful
  here, because the distribution is roughly bimodal (consensus on one side).
- **Summary cache.** The cache lives under the output directory, one file per (folder, cell). Its key is the
  sorted file list plus their modification times and sizes, so the cache is rebuilt when the folder changes.
- **Reading CSVs in parallel.** Summaries are read in parallel with joblib, as `tests/validation.py` already
  does.
- **The validation is rewired.** `tests/validation.py` keeps running the simulations itself (it creates its own
  runs with fresh legacy modules), but its summary, statistics and table code come from the new module. Its
  report keeps the same content. It gains the conditional and lean tables.
- **Ablations.** These need no new code path in the model. An ablation is an anb `Testbed` sweep with an
  agent parameter switched to the legacy behaviour (first case: `learning_until = 102` for the
  `fixedBNat100` configs). It's saved with `save_ensemble` into its own folder and compared with the main
  entry point. The report script has a cell that runs it for the adaptive→static configs.
- **Shared-start runner.** It is given a cell (parameters), a pressure, a start seed, K, a step window and an
  n_jobs value. It:
  - builds one t = 0 state from the start seed (beliefs, symmetric edge weights, adjacency, belief-change
    memory);
  - runs K legacy replicas: for each, it takes a fresh `load_legacy()` module copy and replaces that copy's
    `initialise_agents` and `initialise_network` with functions returning copies of the shared state in the
    legacy layout. `run_simulation` runs with replica seed r, and `track_times` is overridden in the copy, so
    every step in the window is recorded;
  - runs K anb replicas through `Testbed`, using a model spec whose build returns a copy of the shared state
    (an `ANBModel` built from the given `AgentState` and network) and a tracked window that covers the step
    range;
  - returns long-format trajectories (implementation, replica, t, agent, metric columns).

  `ANBModel.py` and `Testbed` aren't modified.
- **State conversion.** Legacy row layout ↔ `AgentState` reuses the mapping already in `tests/legacy_oracle.py`
  (`to_context`). It moves into the comparison module, or into a helper the oracle imports, so there is one
  conversion. The comparison module may depend on legacy only through that oracle loader, not by importing
  `ANBmodel.py` directly.
- **Step-wise comparison of shared-start runs.** At each step in the window and for each tracked quantity, it
  computes the difference of ensemble means across replicas, with a CI, and a KS test on the replica means.
  It reports the first step where the BH-corrected difference becomes significant.
- **The report script.** It is a `# %%` script at the repo root, configured by constants at the top: the two
  folders, the output folder `compare_out/`, top-k, and whether to run ablations or shared-start runs. Its
  cells:
  - load and compare;
  - write tables (CSV + markdown);
  - plots: pooled shares vs pressure per paper config; conditional shares vs μ per pressure; metric time
    series for the top-k cells; shared-start divergence plots.
- **`.gitignore`.** It gains `compare_out/`.
- **Documentation.** A short section in `docu/validation.md` on how to use the framework, plus a note in
  `docu/running-simulations.md`.

## Testing Decisions

- **Test only external behaviour:** what the main entry point and the shared-start runner return for given
  inputs. Don't test private helpers or the internal cache layout. A good test builds a small input with a
  known answer and checks the tables, e.g. two synthetic folders where one cell has a shifted response
  distribution, and checks that exactly that cell is flagged.
- **Seam A (folder comparison), fast tests.** These use synthetic ensemble-format CSVs written into tmp
  folders. The CSVs have the real legacy columns and the t = 95.5 response dummies, and a handful of seeds
  and cells. They check:
  - pairing, and the report of files present on one side only;
  - filename parsing for the paper configs, the μ sweep and the `fixedBNat100` token;
  - the conditional response counts;
  - the lean;
  - that an injected response shift is flagged in the pooled and conditional tables;
  - that an injected lean-only shift is flagged in the lean and pooled tables but not in the conditional
    table;
  - that BH correction and ranking by effect size hold;
  - cache reuse and cache invalidation.
- **Statistics.** The existing `tests/test_validation_stats.py` cases move along with the functions and keep
  passing (prior art for this kind of test).
- **Seam B (shared start).** These run a short run (small T, or a small window and K = 2) and are marked
  `slow` if needed. They check:
  - both implementations report identical t = 0 metrics from the shared state;
  - the replicas within an implementation differ after t = 0;
  - the trajectory frame has the expected shape and columns;
  - `ANBmodel.py` on disk is unchanged.

  The prior art is the state-equality tests (seam 2) built on `legacy_and_anb` in `tests/legacy_oracle.py`.
- **Ablation path.** This is covered by the existing `Testbed` tests (seam 1). There's one extra structural
  test: a `learning_until = 102` run keeps changing edge weights at t = 101 and stops afterwards.
- **Validation.** `pytest -m slow -s tests/test_statistical_validation.py` must still pass after the rewiring,
  and its report must match the previous one on the existing 4-config results.
- **Both `pytest` and `NUMBA_DISABLE_JIT=1 pytest` must pass.**
- **The report script** gets a smoke test only (it runs on the synthetic folders and produces files), like
  `tests/test_notebook.py`.

## Out of Scope

- Changing `ANBmodel.py` in any way, including fixing its known issues.
- Changing anb's dynamics or the deliberate differences themselves; whether to keep `learning_until = 101`
  is the supervisor's decision.
- Bit-for-bit or per-seed matching between legacy and anb. The RNG streams differ by design, so single-seed
  plots (`analyse_timeseries.py`, the example trajectories in `analyse_responseFrequencies.py`) can't match
  and aren't compared.
- Fixing the analysis scripts (hard-coded `simOut/`, the sensitivity filename typo and suffixes).
- Running the new 1000-seed sweeps themselves. The framework supports them; deciding which cells to rerun
  comes after the first report.
- Comparing detail-mode output (`detailed/`), beyond what the shared-start runner records itself.

## Further Notes

- Findings before this spec, which the first report should reproduce:
  - the lean is 0.492 in old vs 0.517 in new (about 1 SE). With 1000 seeds both are at about 0.50
    (`docu/validation.md`, pattern 2);
  - conditional response shares agree for the adaptive configs at s = 4 and 16;
  - legacy's extra edge update at t = 101 shifts adaptive→static `bc_foc` (after the pressure: 2.02 / 2.19 in
    old vs 1.96 in new at s = 4 / 16), and `learning_until = 102` removes the shift;
  - at μ = 0.1 the conditional compliant share differs by 0.02–0.04 (0.146 → 0.182 at s = 16), which is
    probably noise and the first candidate for a rerun with more seeds.
- The legacy `__main__` in the working copy has uncommitted edits (adaptive only, seeds [1, 2],
  `detail=True`). They don't touch the dynamics, but revert them before any new legacy sweep.
- `tests/validation.py` caches under `simOut_old_model/validation`, which is the same folder as the legacy
  output. When rewiring it, move that cache next to `compare_out/` or into its own folder, so it doesn't mix
  with the sweep CSVs that the folder comparison reads.
