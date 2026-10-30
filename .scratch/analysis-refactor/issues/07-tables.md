# 07: LaTeX tables and the greedy subset

**Labels:** ready-for-agent

**What to build:** The tables of `analyse_BNmetrics.py` as separate functions that return LaTeX and print nothing, plus a save helper.

- `table3(sim_dir, *, base, s, mu=0.0, loader=None)` covers Table 3 and A1–A4, one per pressure.
- `tableA5(...)` (s = 4 before/during/after plus the s = 0 column).
- `tableA6(...)`/`tableA7(...)` (the logit fit statistics: McFadden R², Tjur R², LR test against the control model), reusing #06's fits.
- `save_table(latex, out_dir, name)` writes `out_dir/tables/<name>.tex` and creates the folder. The names follow the spec, with the μ/base suffix rule.
- `greedy_uncorrelated_subset(...)` returns the selection order as a DataFrame.
- Each table follows the applicability rules (a missing s or s = 0 reference → `FigureNotApplicable`).

**Blocked by:** #06

**Status:** open

- [ ] Each function returns a string, and nothing is printed or written until `save_table`
- [ ] Slow: the LaTeX matches the old script's stdout for μ = 0 (whitespace-insensitive), and any intended difference is documented
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
