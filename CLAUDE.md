# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Simulation code for the Adaptive Networks of Beliefs (ANB) model (Steiglechner et al. 2026, full paper text in
`docu/ANB.md`). It's an MSc research project that extends the model, mainly by changing the social network
(see `projectIdea.md`). The detailed docs live in `docu/`. Start at `docu/README.md`, which has the model
summary and a paper ↔ code notation table.

## Commands

Uses `uv` with Python 3.12.

```bash
uv sync
uv add scikit-learn statsmodels   # analyse_BNmetrics.py needs these; they're missing from pyproject.toml
uv run ANBmodel.py                # runs the sweep configured in its __main__ block
uv run analyse_timeseries.py      # analysis scripts; each is a # %% cell script, also run interactively
```

There's no test suite, linter or build step. Debug a future Numba kernel with `NUMBA_DISABLE_JIT=1`.

## Architecture

**Two halves joined only by filenames.** `ANBmodel.py` writes one CSV per (param combo × pressure × seed) into
`simOut/` (ensemble mode) or `simOut/detailed/` (`detail=True`). Both are git-ignored. The five
`analyse_*.py` scripts import nothing from the model. They `pd.read_csv` exact filenames built the same way
as `run_one` builds them, so if you change how `run_one` formats a filename (`f"{k}{v:.2f}"`, `mu` with `.3f`,
`seed`/`ext_strength` unformatted, the bare `fixedBNat100` token) the analysis scripts break.
`docu/running-simulations.md` lists exactly which runs (seeds, pressures, μ values, sensitivity suffixes)
each script expects. The README table is incomplete.

**Configuring a run means editing source.** The swept parameters (`param_combis` =
`[link_prob, init_w, beta, rho, eps, mu, fixedBNat100]`, `pressures`, `seeds`, `detail`) are set in the
`if __name__ == "__main__":` block. `M`, `n_agents`, `tau`, `lam` and `two_external_events` are module
constants. `detail` and `track_times` are defined only under `__main__` but read by `run_simulation` and
`get_output`, so importing `ANBmodel` from another module and calling those functions fails.

**State layout in `ANBmodel.py`.** The whole population is one 2-D numpy array. Each row holds
meta (`t`, `id`), 14 metrics, 6 response dummies, M beliefs, 45 edge weights (`w{m}{n}`, m<n) and τ·M belief-change
memory. Columns are addressed through index lists (`beliefids`, `edgeids`, `delbeliefids`), not names. The
dynamics are random-sequential: agents update in shuffled order, each seeing neighbours already updated this
step, and beliefs within an agent are resampled one at a time (Glauber over 21 grid values). Ensemble
output keeps only the tracked steps plus averaged rows at t = 95.5/145.5/195.5. The response classification
(compliant/resilient/resistant…) is stored only on the 95.5 row.

**Planned `anb/` package (designed, not yet implemented).** `docu/adr-002-testbed-architecture.md` and
`docu/class-design.md` specify a refactor into `anb/` (`agents.py`, `networks.py`, `model.py`, `metrics.py`,
`testbed.py`, `io.py`) plus `notebooks/testbed.ipynb`, with Numba sweep kernels per `docu/adr-001-numba-over-gpu.md`.
These decisions are binding when you implement it:
- `ANBmodel.py` stays **frozen** as the reference model. Don't modify it to add features.
- `AgentState` dataclass holds arrays (edge weights as symmetric `(N, M, M)`). `Agent` ABC holds the rule and
  parameters. Per-agent params may be scalar or `(N,)`. One `Agent.sweep(...)` call per time step is the
  Numba boundary.
- No global RNG: one `np.random.default_rng(seed)` per run, passed explicitly everywhere.
- Output must be a drop-in replacement (same columns as legacy, plus a `legacy_filename` helper) so the
  analysis scripts keep working.
- Validate against legacy statistically (≥100 seeds per config), not bit for bit.

## Known pitfalls

`docu/known-issues.md` lists open bugs with locations. Check it before you "fix" an odd-looking behaviour,
because several quirks are needed to reproduce the paper:
- `Hsoc` uses the neighbours' mean focal belief, while the dynamics use the sum.
- There are two inconsistent `fixedBNat100` gates (the module-level flag and the per-run param).
- `run_one` never adds the `_M`/`_N`/`_tau` filename suffixes that `analyse_sensitivity.py` expects.
- `if True:` in `run_one` always overwrites existing CSVs.
- `tau` must be an int.
