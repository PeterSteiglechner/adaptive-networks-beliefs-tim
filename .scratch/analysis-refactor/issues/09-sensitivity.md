# 09: Fig. A8 (OFAT sensitivity)

**Labels:** ready-for-agent

**What to build:** `figA8(sim_dir, *, base, rows=None, loader=None)`, which reproduces `analyse_sensitivity.py`.

- The OFAT experiments are as in the old script (`M, N, beta, p, tau, eps, init_w`, each with their values and configs). Variants are found through the index, so the `_M`/`_N`/`_tau` suffixes from #01 are matched, and the `_fixedBNatt100` typo doesn't apply.
- The 7×3 grid layout is kept. The base cell is required. A row whose variant runs are missing is not drawn, with a warning naming them. If no row is complete, the figure is not applicable.
- `p` is shown as the mean number of contacts p·N, using the N of each run.
- The fixture gets a few M/N/τ variant runs saved by the package (with suffixes).

**Blocked by:** #03, #01

**Status:** open

- [ ] On the fixture: the variant rows that are present render, and missing rows warn
- [ ] A missing base gives `FigureNotApplicable`
- [ ] Slow (if sensitivity runs exist in `simOut_old_model`): the frequencies match the old script, otherwise skipped with a reason
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
