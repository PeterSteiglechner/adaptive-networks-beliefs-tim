# 06: Fig. 4 / A5 / A6 (Cohen's d and logit)

**Labels:** ready-for-agent

**What to build:** `fig4(sim_dir, *, base, mu=0.0, pressures=None, contrast2_pressures=(1, 2, 4), loader=None)`, which reproduces Fig. 4 and, for μ > 0, Fig. A5/A6.

- The pressure rule defaults to the paper's: every s > 0 present for μ = 0, s = 4 for μ > 0. Missing paper levels are warned about.
- Cohen's d and the logit model (the same predictors and control as the old script, standardised within each subset) are computed in plain functions that the tables in #07 reuse. `data` holds d, β̂ and CI per metric × s × contrast.
- A logit fit that fails to converge, or a subset that is empty for a contrast, warns and leaves that dot out instead of crashing.
- The name is `fig4_mu{mu}`.

**Blocked by:** #05

**Status:** open

- [ ] On the fixture: renders with the available pressures, and a missing μ gives `FigureNotApplicable`
- [ ] Slow: Cohen's d and the logit coefficients/CIs match the old script for μ = 0 within a tolerance
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
