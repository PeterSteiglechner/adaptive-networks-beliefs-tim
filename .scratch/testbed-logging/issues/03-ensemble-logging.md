# 03: Start, progress and finish logging for both ensemble methods

**Labels:** ready-for-agent

**What to build:** Once a caller turns on logging with `logging.basicConfig(level=logging.INFO)`, both
`run_ensemble` and `save_ensemble` report what they're doing. Without that, the package stays silent. Both methods
go through one shared driver, so the logging logic lives in one place. All logging happens in the parent process,
from the records workers return, because log records from loky workers don't reach the parent.

- **Logger:** one hierarchy rooted at `anb`. The testbed module logs through its own module logger, a child of
  `anb`. The package root attaches a `NullHandler` and never installs real handlers or sets levels.
- **Start (INFO):** the parameter point, formatted compactly from `spec.params()` plus `ext_strength` and
  `n_events`. Floats get 2 decimals (`mu` gets 3), bools and ints are printed as they are, and `(N,)` arrays are
  shown as `array(N, mean=…)`. Then the seed count, `n_jobs` and, for `save_ensemble`, the output directory. The
  line must work for any spec, so it doesn't call `legacy_filename`, which raises for non-paper setups.
- **Progress (INFO):** results are collected with joblib's `return_as="generator_unordered"`. A line is logged
  every `max(1, ceil(n_to_run / 10))` completions and at the last one, with `done/total`, elapsed time, and an ETA
  from the mean time per finished seed.
- **Finish (INFO):** the elapsed wall time. For `save_ensemble`, also the number of files written and skipped and
  the output directory. For `run_ensemble`, the number of results.
- **DEBUG:** one line per written file path.
- `run_ensemble` keeps its signature and still returns results in seed order, otherwise unchanged.

See `.scratch/testbed-logging/spec.md` (user stories 1–8, 18–21, 24).

**Blocked by:** 02 (`Testbed.save_ensemble` writes each CSV from the worker as its seed finishes)

**Status:** done

- [x] With `caplog` at INFO, both methods emit a start record (with a parameter value and the seed count), at least one progress record (e.g. containing `"3/3"`), and a finish record. For `save_ensemble`, the finish record has the written and skipped counts and the output directory
- [x] Array parameters are summarised in the start record rather than printed in full
- [x] At DEBUG, `save_ensemble` emits one record per written path
- [x] With no logging configured, nothing reaches stderr (the `NullHandler` contract)
- [x] `run_ensemble` with `n_jobs=2` returns results in seed order, equal to sequential runs
- [x] Tests check key facts only, not exact wording, and pass with `NUMBA_DISABLE_JIT=1`
- [x] No new dependencies (no `tqdm`)
