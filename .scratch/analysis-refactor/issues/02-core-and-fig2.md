# 02: Core (base config, selection, warnings, loading, FigureResult, saving) with Fig. 2 as the first slice

**Labels:** ready-for-agent

**What to build:** The shared machinery every figure uses, proven end to end by Fig. 2 / A1 / A2.

- The `Baseline` frozen dataclass holds the paper's defaults, including M/N/τ. The four paper BN configs are derived from a base. Matching uses the filename-rounded values with a float tolerance.
- Selection: given the index, a base and a figure's required cells, it returns the matching runs, or emits `FigureNotApplicable` (naming the missing cells) or `FewSeedsWarning` (with the paper's seed count per item), with a `min_seeds` threshold. Both are custom `UserWarning` subclasses.
- Loader: reads a CSV, derives M from the belief columns and N from the `id`s, derives the edge/memory columns from M, and keeps the needed rows and columns. A loader object can be shared to cache reads in memory. There is no disk cache.
- `FigureResult(figure, data, name)` with `.save(out_dir, formats=("png",))`: creates `out_dir`, writes PNG at 600 dpi, plus a PDF when asked, and returns the paths. A non-paper base appends its differing fields as legacy tokens to the name.
- A style module holds the shared colours, labels, config names and the BN-drawing helper.
- `fig2(sim_dir, *, base, config="adaptive", seed=2, s=4, agents=(61, 79), loader=None)` reproduces `analyse_timeseries.py`'s figure, and its name follows the old one (`fig2_adaptive_t100`, `fig2_staticlow_t100`, `fig2_statichigh_t100`). The hard-coded 100s are computed from N. A missing or out-of-range example agent warns and falls back to a seeded random agent.
- `example_agent_stats` and `focal_sign_counts` return DataFrames.
- A pytest fixture builds the synthetic sim folder used by all later tickets with `Testbed.save_ensemble`: the 4 configs, a few seeds and pressures, one detailed run, one N/M/τ variant, one `link_prob 0.2` base, deliberate gaps, one stray file.

**Blocked by:** #01

**Status:** open

- [ ] `fig2` on the fixture returns a `FigureResult` whose `data` holds the plotted trajectories and edges
- [ ] A missing detailed run gives `FigureNotApplicable` and `None`
- [ ] An N = 50 run plots, and agent 61/79 fall back with a warning
- [ ] `save` writes exactly the requested formats into a not-yet-existing directory
- [ ] The name suffix rule holds for a `link_prob 0.2` base, and the paper base gets the old name
- [ ] Figure functions write no files and print nothing
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
