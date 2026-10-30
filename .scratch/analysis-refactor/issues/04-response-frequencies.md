# 04: Fig. 3, Fig. A3 and the polarisation summary

**Labels:** ready-for-agent

**What to build:** The figures and numbers of `analyse_responseFrequencies.py` as package functions.

- `fig3(sim_dir, *, base, seeds=None, example_seed=1, example_s=4, example_agents=…, loader=None)`. Panels B–E need all four configs derived from the base. Pressure levels on the x-axis are the paper's s > 0 levels present for all four configs, and missing ones are warned about. Panel A needs the detailed adaptive run, and its config comes from the base, not from leftover loop variables. The denominator includes `latecompliant`, as in the paper. `data` holds the per-simulation relative frequencies.
- `figA3(sim_dir, *, base, seed=1, rng_seed=14, loader=None)`: needs the detailed adaptive runs for all six pressures. It samples agents with a local `RandomState(rng_seed)` passed as `random_state`, in the old panel order.
- `polarisation_summary(...)`: σ_foc, mean |ω| and mean ω per config, plus `groupishness` and `nr_negs` (from the 95.5 row, the bug fix).

**Blocked by:** #03

**Status:** open

- [ ] On the fixture: a missing config gives `FigureNotApplicable`, a missing pressure level gives a warning while the figure still renders, and few seeds give `FewSeedsWarning`
- [ ] Slow: with `seeds=range(20)` on `simOut_old_model`, the Fig. 3 frequencies match the old `res` exactly
- [ ] Slow: Fig. A3 picks the same agents as the old script (same highlighted ids)
- [ ] Slow: the summary numbers match the old summary cells (except `nr_negs`, which is documented)
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
