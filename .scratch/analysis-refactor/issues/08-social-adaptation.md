# 08: Fig. 6 and Fig. A7

**Labels:** ready-for-agent

**What to build:** The figures of `analyse_socialAdaptation.py`.

- `fig6(sim_dir, *, base, mus=None, example_seed=1, example_mus=(0.005, 0.2), agent_rng_seed=42, loader=None)`. It needs s = 4 for at least two μ values plus the example μ values for panels A/B. The x-axis uses the μ values present (the paper's 11 when they are all there), and missing paper μ values are warned about. The agent draws use a local RNG over N, not `randint(100)`. The response-frequency denominator excludes `latecompliant`, as in the paper. Panel D's y-ticks are set before their labels (the bug fix).
- `figA7(sim_dir, *, base, seed=99, s=4, mus=(0, 0.005, 0.2), loader=None)`. Its example simulation is explicit, not a set of leftover loop variables.
- `data` holds the heterogeneity H, the z-scored metrics and the frequencies per μ.

**Blocked by:** #03

**Status:** open

- [ ] On the fixture: a missing example μ gives `FigureNotApplicable`, and an M/N variant base renders
- [ ] Fig. 6D tick labels equal the tick positions as percentages
- [ ] Slow: H, the z-scores and the frequencies match the old script on `simOut_old_model`
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
