# 03: Old-vs-new harness for the analysis scripts (slow)

**Labels:** ready-for-agent

**What to build:** A way to run the unchanged `analyse_*.py` scripts and read their results, so that each figure ticket can check its numbers against them.

- A test helper runs one old script with `runpy` in a temporary working directory, where `simOut` is a link to `simOut_old_model` and `figs/` is created first, so the known crash doesn't fire. It uses the Agg backend. It returns the script's module globals and the PNGs it wrote. Results are cached per session, because `analyse_BNmetrics.py` is slow.
- If an old script fails under the current dependencies (e.g. pandas 3), the helper reports which cell failed. That ticket's comparison then falls back to the parts that ran, plus a visual check against `figs/figsold/`.
- A slow test compares `fig2` (`data`) with `analyse_timeseries.py`'s globals, and does a PNG pixel-diff sanity check.
- Tests are marked `slow` and skipped when `simOut_old_model` is missing.

**Blocked by:** #02

**Status:** open

- [ ] The old scripts on disk are unchanged (checked with git), and nothing is written into the repo
- [ ] The Fig. 2 numbers match the old script exactly, and the PNG diff is small (the threshold is documented)
- [ ] It runs in `pytest -m slow`, and the default `pytest` doesn't touch it
