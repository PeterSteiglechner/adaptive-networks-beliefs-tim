# Analysis package: figures and tables from whatever runs exist

**Labels:** ready-for-agent
**Status:** open

## Problem Statement

The five `analyse_*.py` scripts produce the paper's figures and tables, but only from one exact set of files.
Each script builds every filename from hard-coded lists of seeds, pressures and μ values. If one file is
missing, the script stops with `FileNotFoundError`. If I have more seeds than it expects, it ignores them. If I
change the social network, M, N or τ, nothing works: the scripts hard-code M = 10 and N = 100 (belief and edge
columns, agent-id colours, `randint(100)`, example agents 61/79 and 4/53/16), and they can't tell
`_M`/`_N`/`_tau` runs apart from the baseline.

For the MSc project I will run many variants of the model, especially of the social network. Each variant has
its own subset of seeds, pressures and parameters. To get any figure for them, I have to edit a script by hand.

There are other problems too:

- Each script recomputes and saves everything on every run, including the PDFs, and prints all LaTeX tables to
  stdout. I can't make just one figure, or just one table.
- The scripts have real bugs: the Fig. 6D y-tick labels are 1.25× too large, the A4 title claims one window
  but pools three, there are filename typos (`_fixedBNat1001.00`, `_fixedBNatt100`), and some scripts save into
  `figs/` before creating it.
- Some choices are implicit: Fig. A7 and Fig. 3A read leftover loop variables, and Fig. A3 seeds the global
  NumPy RNG.

## Solution

A subpackage `anb.analysis` replaces the scripts as the way to make figures. The old `analyse_*.py` scripts
stay untouched as a reference.

- It **indexes** a simulation folder once. Every CSV whose name follows the legacy naming scheme becomes one
  row of parameters.
- Every figure and every table is **its own function**. It takes the folder and a **base config** (defaults
  are the paper's baseline, all overridable, M/N/τ included), selects the runs it needs from the index, and
  uses every seed it finds.
- If the runs a figure *structurally* needs are missing, the function warns with `FigureNotApplicable` and
  returns nothing. If fewer seeds exist than the paper used, it still plots and warns with `FewSeedsWarning`.
- Figure functions only compute and draw. They return a `FigureResult` (the figure, the plotted numbers as
  a DataFrame, the output name). Saving is a separate, explicit step that writes the chosen formats. PNG is
  the default and PDF only when asked for. Tables are separate functions that return LaTeX and write `.tex`
  only on request.
- `make_all` runs every figure and table, skips the ones that don't apply with a warning, and returns a
  report of what was written and what was skipped and why.
- Real bugs are fixed. Everything else reproduces the paper's figures, with the implicit choices turned into
  keyword arguments whose defaults are the paper's.
- It runs from Python, from a CLI and from a notebook.

The user-facing reference (API, base config, the applicability rules for each figure, naming, CLI) is
`docu/analysis-package.md`. This spec doesn't repeat that table.

## User Stories

1. As a researcher, I want to point the analysis at any simulation folder, so that I can analyse
   `simOut_old_model`, `simOut_new_model` or a new variant folder without renaming anything.
2. As a researcher, I want the folder scanned and every legacy-named CSV parsed into parameters, so that
   figures find their runs by parameter rather than by a hard-coded filename.
3. As a researcher, I want files that don't follow the naming scheme to be reported once and ignored, so that
   stray files don't crash the analysis.
4. As a researcher, I want a figure to use every seed that exists for its configuration, so that adding seeds
   needs no code change.
5. As a researcher, I want to restrict a figure to a seed subset (`seeds=`), so that I can reproduce the old
   scripts' 20-seed Fig. 3 or run quick tests.
6. As a researcher, I want one function per paper figure (Fig. 2, 3, 4, 5, 6, A1–A8), so that I can make
   exactly the figure I need.
7. As a researcher, I want one function per paper table (Table 3 / A1–A4, A5, A6/A7), so that LaTeX is made
   only when I ask for it.
8. As a researcher, I want a single `make_all` call, so that I get every applicable figure and table in one go.
9. As a researcher, I want a figure that lacks structurally required runs to emit a `FigureNotApplicable`
   warning that names what's missing, so that I know what to simulate next.
10. As a researcher, I want a `FewSeedsWarning` when a figure uses fewer seeds than the paper, so that I don't
    mistake a 5-seed figure for the real one.
11. As a researcher, I want a `min_seeds` threshold per call, so that I can require a minimum sample size.
12. As a researcher, I want `make_all` to return a report mapping each figure/table to its saved paths or to
    the skip reason, and to print a short summary, so that I see at a glance what was produced.
13. As a researcher, I want `make_all` to carry on when one figure raises an unexpected error (warning with
    the traceback), so that one broken figure doesn't cost me the rest.
14. As a developer, I want `strict=True` on `make_all`, so that unexpected errors propagate while I debug.
15. As a researcher, I want the warnings to be custom `UserWarning` subclasses, so that I can filter them or
    turn them into errors with the standard `warnings` machinery.
16. As a researcher, I want figure functions to return the figure without saving it, so that I can look at it
    or tweak it in a notebook first.
17. As a researcher, I want an explicit save step with a `formats` choice (PNG by default, PDF opt-in), so
    that PDFs are only made when I need them for the paper.
18. As a researcher, I want the output directory created if it doesn't exist, so that a fresh checkout works.
19. As a researcher, I want tables written as `.tex` files into a `tables/` subfolder of the output directory,
    instead of printed to stdout, so that I can `\input` them.
20. As a researcher, I want each table function to also return the LaTeX string, so that I can inspect it in a
    notebook.
21. As a researcher, I want the figures to keep the old file names (`fig3_responseFrequencies`, `fig4_mu0.0`,
    `AppendixFig_ofat`, …), so that my LaTeX sources and `figs/figsold/` comparisons keep working.
22. As a researcher, I want a non-paper base config to append the differing parameters to the file name
    (e.g. `fig3_responseFrequencies_link_prob0.20`), so that variants don't overwrite the paper figures.
23. As a researcher, I want a base config dataclass (link_prob, beta, rho, init_w, eps, mu, fixedBNat100,
    lam, n_events, M, N, τ) with the paper's baseline as default, so that I can draw any figure for my network
    variants by overriding one field.
24. As a researcher, I want figures that compare BN configs (static low/high, adaptive→static, adaptive) to
    derive those configs from the base config, so that "the four configs" also works with `link_prob=0.2`.
25. As a researcher, I want every figure to work for M ≠ 10, N ≠ 100 and τ ≠ 1, with belief/edge/memory
    columns and N read from the data, so that I can analyse sensitivity runs and my own variants.
26. As a researcher, I want the anb package to write the `_M`, `_N`, `_tau` filename suffixes when those
    differ from 10/100/1, so that Fig. A8 and M/N/τ variants become findable (known issue #4).
27. As a researcher, I want baseline runs to keep their exact current filenames, so that existing folders
    stay valid.
28. As a researcher, I want the Fig. 4 / A4 family made for a default list of μ values (0, 0.005, 0.05) with
    an override, and a warning for each listed μ that's missing, so that Fig. A5/A6 come for free when the
    runs exist.
29. As a researcher, I want Fig. 2 made for the adaptive, static-low and static-high configs when their
    detailed runs exist, so that Fig. A1/A2 come for free.
30. As a researcher, I want example agents (Fig. 2: 61/79; Fig. 3A: 4/53/16) as keyword arguments with the
    paper's defaults, so that I can pick others.
31. As a researcher, I want a warning and a seeded random fallback when a default example agent doesn't exist
    (N smaller) or doesn't have the intended response type, so that example panels still render for other
    runs.
32. As a researcher, I want the example simulations (Fig. A7: seed 99, s = 4; Fig. 3A: seed 1, s = 4;
    Fig. 6A/B: seed 1, μ = 0.005 / 0.2, agent seed 42) as keyword arguments with the paper's defaults, so that
    the choice is explicit.
33. As a researcher, I want Fig. A3's agent sampling to use a local RNG seeded with 14, so that it draws the
    same agents as the paper without touching global state.
34. As a researcher, I want the Fig. 6D y-tick labels to show the true values, so that the figure isn't
    off by 1.25×.
35. As a researcher, I want the Fig. A4 title to describe the rows it actually pools, so that the figure
    doesn't misreport its sample.
36. As a researcher, I want the `fixedBNat100` runs found correctly in every figure, so that the old filename
    typos are gone.
37. As a researcher, I want the paper's summary numbers (σ_foc and mean |ω| per config, the example agents'
    `tb_tot`/`tb_foc`/`Hpers`, sign counts of focal beliefs, the `greedy_uncorrelated_subset` order) as
    functions that return DataFrames, so that I can quote them without the scripts' print cells.
38. As a researcher, I want each `FigureResult` to carry the plotted numbers as a DataFrame, so that I can
    check or reuse them without reading them off a plot.
39. As a researcher, I want runs shared between figures to be read from disk only once per `make_all` call, so
    that a full run doesn't parse the same 600 CSVs three times.
40. As a researcher, I want a CLI (`python -m anb.analysis SIM_DIR OUT_DIR [--figs …] [--formats …]
    [--strict]`), so that I can regenerate figures from the shell.
41. As a researcher, I want a notebook that runs the package figure by figure, so that I can look at the
    results interactively.
42. As a researcher, I want the old `analyse_*.py` scripts left byte-for-byte unchanged, so that they stay the
    reference for the paper's figures.
43. As a researcher, I want a one-off slow test that checks the package's numbers against the old scripts'
    numbers on `simOut_old_model`, so that I can trust "same figures" beyond eyeballing.
44. As a developer, I want the filename parser and `legacy_filename` to live side by side and be tested as a
    round-trip, so that the naming scheme is defined in one place.
45. As a developer, I want the comparison module to use the same parser, so that there's only one inverse of
    `legacy_filename`.

## Implementation Decisions

**Location and modules.** The new code is a subpackage `anb.analysis` of the existing package. Its modules:

- **index**: folder scan → index DataFrame.
- **select**: the base config, filtering, applicability and seed checks, and the warning classes.
- **load**: reading CSVs, deriving column layouts, the per-call cache.
- **output**: `FigureResult`, saving, naming.
- **style**: the shared constants (response colours, labels, metric titles, config names, BN-drawing helpers).
- One module per figure group: timeseries (Fig. 2, A1, A2), response frequencies (Fig. 3, A3, polarisation
  summaries), BN metrics (Fig. 4, 5, A4, A5, A6, Tables 3, A1–A7, greedy subset), social adaptation (Fig. 6,
  A7), sensitivity (Fig. A8).
- **runner**: `make_all` and the report.
- A `__main__` for the CLI.

The analysis modules don't import `ANBmodel.py`.

**Filename parsing moves to `anb.io`.** The existing inverse parser in the comparison module moves into
`anb.io`, next to `legacy_filename`. The comparison module then imports it. It's extended to:

- detailed files (the `detailed/` folder and the `_detailed` ending);
- the `_M`, `_N`, `_tau` suffixes;
- `_2events` and `_lambda…`;
- the bare `fixedBNat100` token.

Absent suffixes mean M = 10, N = 100, τ = 1, λ = 0 and one event.

**`legacy_filename` writes M/N/τ suffixes.** It gains optional M, N and τ. It appends `_M{M:.2f}`, `_N{N:.2f}`
and `_tau{tau:.2f}` only when the value differs from the default (10, 100, 1). The suffixes go right after the
`seed` token, in the order M, N, τ, and before `_2events`/`_lambda`. `analyse_sensitivity.py` expects exactly
this. Baseline names don't change. `SimulationResult.save`/`output_path` pass the spec's M, N and τ through.
Known issue #4 is updated: the package now writes the suffixes, legacy still doesn't. `ANBmodel.py` stays
frozen.

**Index.** `index_folder(sim_dir)` scans the folder and its `detailed/` subfolder. It returns one row per
parseable CSV:

- the `run_one` parameters (`link_prob`, `init_w`, `beta`, `rho`, `eps`, `mu`, `fixedBNat100`, `s` (from
  `ext_strength`), `seed`);
- `lam`, `n_events`, `M`, `N`, `tau`;
- `detailed`;
- the path.

Files it can't parse produce one summary warning and are left out. The index is computed once per call, or
once per `make_all`.

**Base config.** It's a frozen dataclass holding the non-swept parameters plus the BN-config fields.

- The defaults are the paper's adaptive baseline: `link_prob 0.1, beta 3, rho 0.33` (as rounded in the name),
  `init_w 0.2, eps 1, mu 0, fixedBNat100 False, lam 0, n_events 1, M 10, N 100, tau 1`.
- Figures that compare BN configs derive the four paper configs from the base by replacing `(init_w, eps, mu,
  fixedBNat100)`.
- Matching uses the filename-rounded values (`rho 0.33`) with a float tolerance.
- Runs that don't match the base exactly are ignored.
- The config labels (`static (ω₀=0.2)`, `adaptive`, …) come from one shared mapping.

**Applicability.** Each figure/table declares its structural requirements: which configs, which pressures,
detailed or ensemble, which example simulations. The base config and keyword arguments fill them in.

- **Missing structure** → `warnings.warn(FigureNotApplicable)` naming the missing cells, and the function
  returns `None`.
- **Fewer seeds than the paper's count** for any required cell → `FewSeedsWarning`, and the figure still plots.
- **Fewer than `min_seeds`** (default 1) → not applicable.
- **Figures with a fixed grid** (the Fig. 3 B–E configs, the Fig. A3 six pressures, the Fig. A8 rows) need their
  grid. Pressure lists inside a panel (Fig. 3 x-axis, Fig. 4 bars) use the paper's levels that exist and warn
  about the missing ones. If none exist, the figure isn't applicable.
- The rules for each figure are tabled in `docu/analysis-package.md`.

**Column layout from data.** M comes from the belief columns `"0"…"M-1"` in the CSV header, and N from the
number of distinct `id` values. τ comes from the filename. Edge columns (`w{m}{n}`, m < n) are derived from M,
and so are the memory columns. Anything that used to be a hard-coded 10 or 100 is computed:

- the grey levels in Fig. 2;
- the agent-id draws in Fig. 6A/B;
- the triangle-count ranges;
- the histogram.

The focal belief stays `"0"`. The time structure stays fixed: T = 200, pressure at t = 101–150, windows at
95.5 / 145.5 / 195.5, onset snapshot t = 100.

**FigureResult and saving.** A figure function returns a `FigureResult` with `figure` (matplotlib), `data` (a
tidy DataFrame of every plotted number) and `name` (the old file stem plus a suffix for a non-paper base).
Saving is explicit (`result.save(out_dir, formats=("png",))`):

- PNG at 600 dpi, as before; PDF only when listed in `formats`;
- `out_dir` is created if missing;
- the saved paths are returned.

Figure functions don't save anything, and neither do they print.

**Tables.** Table functions return the LaTeX string, with the same content and layout as the old scripts'
stdout. A separate save writes `out_dir/tables/<name>.tex`. Names: `table3_s{s}` for the one-per-pressure
Tables 3/A1–A4, `tableA5_metricChange`, `tableA6_logit_…`/`tableA7_logit_…`, all with the μ/base suffix rule.

**Naming suffix.** For a base config that differs from the paper baseline, the differing fields are appended
in base-config field order, using the legacy token format (`_link_prob0.20`, `_M15.00`). The paper baseline
gets exactly the old name.

**Implicit choices become keyword arguments** with the paper's values as defaults:

- Fig. 2: `seed=2, s=4, agents=(61, 79)`;
- Fig. 3A: `seed=1, s=4, agents={resistant: 4, resilient: 53, compliant: 16}`;
- Fig. 3: `seeds=None` (all available);
- Fig. A3: `seed=1`, a local `np.random.RandomState(14)` passed as `random_state` to the sampling, in the
  same panel order, so the draws match;
- Fig. 6A/B: `seed=1, mus=(0.005, 0.2), agent_rng_seed=42`;
- Fig. A7: `seed=99, s=4, mus=(0, 0.005, 0.2)`;
- the Fig. 4 family: `mus=(0.0, 0.005, 0.05)` (for `make_all`), the paper's pressure rule (all s > 0 for
  μ = 0, s = 4 for μ > 0), contrast-2 pressures `(1, 2, 4)`.

If a default example agent is missing or has the wrong response type, a warning is issued and an agent of the
right type is drawn with a seeded RNG. A type with no agents is left out, as in Fig. A3 today. The
response-frequency denominator keeps its paper difference (Fig. 3 counts `latecompliant`, Fig. 6 doesn't) as a
documented parameter.

**Bugs fixed** (the only intended visual differences from the old scripts):

- Fig. 6D: fix the y-tick positions before setting the labels.
- Fig. A4: the title states the pooled windows.
- Filename typos: gone, since files are found through the index.
- `figs/` creation order: gone, since `save` creates `out_dir`.
- `nr_negs` in the polarisation summary: computed from the 95.5 before-window row, not from the t = 100
  snapshot.

**Loading and caching.** Within one `make_all` call (or an explicitly shared loader passed to several figure
functions), each CSV is read once and kept as a reduced frame: the rows and columns the figures need. There's
no disk cache.

**make_all.** `make_all(sim_dir, out_dir, *, base=…, formats=("png",), figs=None, tables=True,
mus=(0.0, 0.005, 0.05), min_seeds=1, strict=False)`:

- runs the selected figures and tables;
- collects warnings into the report while still emitting them;
- catches unexpected exceptions per item (it warns with the traceback and records the error) unless `strict`;
- returns `{item_name: [paths] | "skipped: <reason>" | "error: <message>"}` and prints a short summary.

The CLI wraps it one to one.

**Summaries.** These are functions returning DataFrames: the polarisation/edge summary per config (σ_foc and
mean |ω| mean ± std), the example-agent BN stats for Fig. 2, the focal-belief sign counts and
`greedy_uncorrelated_subset`. The quick-look plots aren't carried over.

**Dependencies.** No new ones. matplotlib, seaborn, networkx, scikit-learn and statsmodels are already there.

## Testing Decisions

A good test calls the public API and checks what comes out: returned numbers, emitted warnings, the report and
the files written. It doesn't check private helpers or matplotlib internals beyond basic facts (number of
axes, tick labels where a bug was fixed).

- **Seam 1, the public API (main seam).** It runs against a tiny synthetic simulation folder built by a pytest
  fixture with `Testbed.save_ensemble`: a few seeds, a few pressures, a short horizon where possible, the four
  paper configs, one detailed run, one M/N/τ variant, one non-paper base (e.g. `link_prob 0.2`), cells left out
  on purpose and one stray file. Tests check:
  - each figure/table returns a `FigureResult`/string with the expected `data` shape and name;
  - a missing structural cell gives `FigureNotApplicable` and `None`;
  - too few seeds give `FewSeedsWarning` but still a result;
  - M/N/τ variants plot;
  - the name suffix rule holds;
  - `save` writes exactly the requested formats into a new `out_dir`;
  - tables go to `tables/*.tex`;
  - `make_all` returns a report covering every item, carries on after a forced error, and re-raises with
    `strict=True`.
- **Seam 2, the index.** The parser is tested as a round-trip against `legacy_filename` (baseline, the
  `fixedBNat100`, `_2events`, `_lambda`, `_M`/`_N`/`_tau` and detailed variants). Baseline names must stay
  unchanged, a stray file must be ignored with one warning, and the comparison module must still pass its
  tests.
- **Seam 3, old vs new (slow, one-off).** It runs the unchanged `analyse_*.py` scripts with `runpy` in a
  temporary working directory where `simOut` is linked to `simOut_old_model`, using a non-interactive
  matplotlib backend. Their module globals (response counts/frequencies, `res`, the Cohen's d and logit
  coefficient tables, the correlation matrix, the heterogeneity values, the LaTeX strings) are compared within
  a tolerance against the `FigureResult.data` and the table strings of the package on the same folder, with
  the old seed subsets (e.g. 20 seeds for Fig. 3). PNG pixel diffs are a sanity check. The expected
  differences are the fixed bugs (Fig. 6D ticks, A4 title). It's marked `slow`. It's skipped when
  `simOut_old_model` is missing, and it says so if an old script fails under the current dependencies.
- **Prior art:**
  - `tests/test_folder_comparison.py` and `tests/test_comparison.py`: synthetic folders and filename pairing;
  - `tests/test_legacy_filename.py`: naming round-trips;
  - `tests/test_testbed_ensemble.py`: building small ensembles in fixtures;
  - `tests/legacy_oracle.py`: loading frozen legacy code without modifying it;
  - `tests/test_notebook.py`: executing a notebook.

`pytest` and `NUMBA_DISABLE_JIT=1 pytest` must both pass.

## Out of Scope

- Modifying `ANBmodel.py` or the five `analyse_*.py` scripts.
- New figures or new statistics beyond the paper's, and the scripts' quick-look plots.
- A disk cache of parsed runs.
- Varying the time structure (T, pressure window, two events) inside the figures. `_2events`/`_lambda` runs
  are indexed but not plotted unless the base asks for them, and even then the windows stay the paper's.
- Picking the "best" example agents automatically beyond the fallback rule.
- Editing `docu/analysis.md` (the user has uncommitted edits there). A link to the new doc is added only after
  those are committed.

## Further Notes

- "Same figures" means visually identical to the old scripts' output on the same data and seeds, except for
  the fixed bugs. It doesn't mean pixel-identical.
- `make_all` on `simOut_old_model` with all 100 seeds gives Fig. 3 with 100 seeds where the old script used 20.
  That's intended. Seam 3 pins `seeds=range(20)` to compare.
- The deliberate legacy/anb differences in `docu/validation.md` still need supervisor sign-off before new runs
  go into results. This package only makes figures, from either folder.
