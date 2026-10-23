# 03: Conditional response, lean and metric tables, plus combined top differences

**Labels:** ready-for-agent

**What to build:** The comparison result gains the remaining three test families and a combined ranking.

The run summary gains:
- the conditional counts: response counts among agents negative before pressure (compliant, resilient, resistant, latecompliant);
- the lean: (persistPos + nonpersistPos) / N;
- the run mean of all 14 BN metrics at t = 95.5, 145.5 and 195.5;
- the mean of each metric at each tracked step, for the time-series plots later.

New tables:
- **Conditional response table:** χ² permutation, BH, Cramér's V.
- **Lean table:** Welch on the run-level share positive, the SMD, and the gap in units of SE (no KS, because the distribution is bimodal).
- **Metric table:** per cell, metric and window, KS plus a Welch difference-of-means CI, and the SMD.
- **Combined top-differences table:** across all four families, ranked by effect size.

BH runs within each family.

**Blocked by:** #02

**Status:** open

- [ ] Conditional counts and the lean are correct on hand-built synthetic runs
- [ ] An injected lean-only shift is flagged in the lean and pooled tables but not in the conditional table
- [ ] An injected metric shift in one cell and window is flagged in the metric table only for that cell and window
- [ ] BH correction applies within each family, and every table is ranked by effect size
- [ ] The combined table merges all four families
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
