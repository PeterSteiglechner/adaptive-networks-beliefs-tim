# 05: Testbed notebook sweep and docs

**Labels:** ready-for-agent

**What to build:** Running the sweep in `notebooks/testbed.ipynb` is observable and resumable.
- The run-then-save double loop becomes one loop over (setting, pressure). Each iteration logs `ensemble i/n` and
  calls `save_ensemble` into the existing `../simOut` location.
- The notebook configures logging at INFO with timestamps. One handler writes to the notebook output, and a
  `FileHandler` writes `sweep.log` in the output directory, which is created first.
- `docu/running-simulations.md` explains `save_ensemble`, `skip_existing`, and how to turn logging on. If the
  package section of `docu/README.md` lists the testbed API, it gets the same. The docs warn that resuming with
  `skip_existing=True` after changing the model reuses stale CSVs, so delete or switch the output folder after
  model changes.

See `.scratch/testbed-logging/spec.md` (user stories 9, 22, 23).

**Blocked by:** 03 (Start, progress and finish logging for both ensemble methods), 04 (Resume with `skip_existing=True`)

**Status:** done

- [x] The notebook sweep is a single (setting, pressure) loop with an `ensemble i/n` log line, using `save_ensemble` into `../simOut`
- [x] Log lines are timestamped and go to both the notebook output and `sweep.log` next to the CSVs
- [x] The output CSVs have the same names and locations as the current notebook produces
- [x] The docs cover `save_ensemble`, `skip_existing`, turning logging on, and the stale-CSV warning
- [x] `demo.ipynb` and `ANBmodel.py` are unchanged, and `pytest` passes
