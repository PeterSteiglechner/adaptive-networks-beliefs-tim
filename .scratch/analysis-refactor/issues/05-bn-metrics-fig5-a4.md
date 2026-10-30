# 05: BN-metrics data, Fig. 5 and Fig. A4

**Labels:** ready-for-agent

**What to build:** The long agent × seed × s × window frame that `analyse_BNmetrics.py` builds, and the two figures that only need it.

- A shared builder for the long frame (response label from the 95.5 row, `Hpersnonfoc` derived) for a base and μ, through the loader.
- `fig5(sim_dir, *, base, mu=0.0, s=4, loader=None)`: needs s and the s = 0 reference. The name is `fig5_metricChange_mu{mu}`.
- `figA4(sim_dir, *, base, mu=0.0, loader=None)`: needs s = 0 for μ = 0 and s = 4 for μ > 0, as before. The title states the pooled windows (the bug fix). The name is as in the old script, with the `_mu{mu}` suffix for μ > 0.

**Blocked by:** #03

**Status:** open

- [ ] On the fixture: a missing s = 0 reference gives `FigureNotApplicable` for both figures
- [ ] Slow: the Fig. 5 means and sds and the Fig. A4 correlation matrix match the old script on `simOut_old_model`
- [ ] The Fig. A4 title names the pooled windows
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
