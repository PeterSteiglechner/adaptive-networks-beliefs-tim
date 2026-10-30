# 10: `make_all`, report and CLI

**Labels:** ready-for-agent

**What to build:** One call that makes everything applicable.

- `make_all(sim_dir, out_dir, *, base=Baseline(), formats=("png",), figs=None, tables=True, mus=(0.0, 0.005, 0.05), min_seeds=1, strict=False)`:
  - indexes once and shares one loader;
  - attempts `fig2` for adaptive/static-low/static-high, `fig3`, `figA3`, and `fig4`/`fig5`/`figA4`/tables for each listed μ (`table3` for each s > 0 present), plus `fig6`, `figA7` and `figA8`;
  - saves the results and writes the tables.
- The report is `{item: [paths] | "skipped: <reason>" | "error: <message>"}` with a printed summary. Warnings are still emitted.
- An unexpected exception in one item is warned about with its traceback and the run continues. `strict=True` re-raises it.
- `figs=` takes the item names to restrict the run.
- `python -m anb.analysis SIM_DIR OUT_DIR [--figs …] [--formats …] [--no-tables] [--strict]` wraps `make_all` and prints the report.

**Blocked by:** #04, #07, #08, #09

**Status:** open

- [ ] On the fixture: the report has an entry for every item, and skipped items name their reason
- [ ] A forced error in one item doesn't stop the others, and `strict=True` raises
- [ ] Each CSV is read at most once per call (checked with the loader's counters)
- [ ] The CLI writes the same files as the API call
- [ ] Slow: `make_all(simOut_old_model, tmp)` produces every paper figure and table without errors
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
