# Spec: progress logging and incremental export for testbed sweeps

**Labels:** ready-for-agent

## Problem Statement

When I run the sweep in the testbed notebook (4 settings × 6 pressures, 100 seeds each = 2,400 runs) I get no
output at all for a long time. I can't tell which ensemble is running, how far along it is, how long the rest will
take, or what has been written to disk. Worse, nothing is written until every run has finished: all results sit in
memory, and a crash or kernel restart late in the sweep loses everything.

## Solution

The `anb` package logs what an ensemble is doing through the standard `logging` module: a line when an ensemble
starts (which parameter point, how many seeds, how many jobs, where the output goes), throttled progress lines with
elapsed time and ETA while seeds finish, and a summary line when it ends. A new `Testbed.save_ensemble` runs the
seeds and writes each CSV from the worker as soon as that seed finishes, so results never pile up in memory. It
returns the written paths and can skip seeds whose CSV already exists, which lets a crashed sweep resume. The
package stays silent unless the caller configures logging. The testbed notebook turns it on, adds an
"ensemble i/n" counter around its sweep loop, and also writes the log to a file next to the CSVs.

## User Stories

1. As a researcher running a sweep, I want a log line when each ensemble starts, so that I know which parameter point is being simulated.
2. As a researcher, I want the start line to show the parameters compactly (e.g. `init_w=0.20 eps=1.00 mu=0.000 fixedBNat100=True ext_strength=4`), so that I can read it at a glance.
3. As a researcher using per-agent parameters, I want array parameters summarised (e.g. `eps=array(100, mean=0.50)`), so that the line doesn't flood the output.
4. As a researcher, I want the start line to include the number of seeds, `n_jobs` and the output directory, so that I know the size of the job and where the files go.
5. As a researcher, I want progress lines while seeds finish (done/total, elapsed, ETA), so that I can see the ensemble is alive and estimate when it ends.
6. As a researcher, I want progress throttled to roughly every 10% of the seeds, so that a 100-seed ensemble gives about ten lines, not a hundred.
7. As a researcher, I want a finish line with the elapsed time, the number of files written and skipped and the output directory, so that I know what each ensemble exported.
8. As a researcher debugging output, I want every written file path logged at DEBUG, so that I can see each export when I need to without cluttering INFO.
9. As a researcher, I want an "ensemble i/n" counter in the notebook's sweep loop, so that I know where I am in the whole sweep.
10. As a researcher, I want each CSV written as soon as its seed finishes, so that a crash late in the sweep keeps everything finished so far.
11. As a researcher, I want finished results not kept in memory during a save-to-disk sweep, so that large sweeps don't exhaust RAM.
12. As a researcher, I want `save_ensemble` to return the paths it wrote, so that I can check or load them afterwards.
13. As a researcher, I want legacy filenames by default (`legacy_names=True`) and the `seed{seed}.csv` naming as an option, the same as `SimulationResult.save`, so that the analysis scripts keep finding the files.
14. As a researcher resuming a crashed sweep, I want `skip_existing=True` to skip seeds whose CSV already exists, so that I don't recompute finished work.
15. As a researcher, I want overwriting to stay the default, so that I never silently reuse stale CSVs after changing the model.
16. As a researcher, I want the number of skipped seeds logged, so that I know a resume actually reused files.
17. As a researcher, I want an ensemble where every seed is skipped to finish immediately, with a log line saying so, so that resuming a mostly finished sweep is fast.
18. As a user of `run_ensemble`, I want the same start/progress/finish logging there, so that in-memory ensembles are observable too.
19. As a user of `run_ensemble`, I want results still returned in seed order and otherwise unchanged, so that existing code and tests keep working.
20. As a test author or library user, I want the package to log nothing unless I configure logging, so that tests and imports stay quiet.
21. As a researcher, I want to turn logging on with a single `logging.basicConfig(level=logging.INFO)`, so that setup is trivial.
22. As a researcher, I want the notebook to also write the log to `sweep.log` next to the CSVs, so that I have a record of a long sweep after the kernel is gone.
23. As a researcher, I want log lines timestamped, so that I can see how long each part of the sweep took after the fact.
24. As a developer, I want the logging to work when Numba JIT is disabled and with `n_jobs > 1`, so that it behaves the same in every way the tests run.

## Implementation Decisions

- **Logger.** The package uses one logger hierarchy rooted at `anb`; the testbed module logs through its own module logger (a child of `anb`). The package root attaches a `NullHandler` and never installs real handlers or sets levels. Configuring output is the caller's job.
- **Levels.** INFO: ensemble start, throttled progress, ensemble finish, skip count. DEBUG: one line per written file.
- **New method `Testbed.save_ensemble(seeds, out_dir, n_jobs=1, legacy_names=True, skip_existing=False) -> list[Path]`.** It returns the paths of all seeds' CSVs (written or skipped), in seed order. `run_ensemble(seeds, n_jobs=1) -> list[SimulationResult]` keeps its signature and return value. Both methods share one internal driver that does the start/progress/finish logging, so the logic lives in one place.
- **Writing happens in the worker.** Each worker runs its seed, calls `SimulationResult.save(out_dir, legacy_names)` and returns only the seed, path and runtime to the parent, never the frame. The parent logs as these small records arrive.
- **Output path without running.** The CSV path for a seed has to be known before running (for `skip_existing`). It's derived from the spec's params, the schedule and the seed, using the same naming as `SimulationResult.save` (legacy name plus `_detailed` in detail mode, otherwise `seed{seed}`). That naming moves into one shared helper used by both, so they can't drift apart.
- **Skipping.** With `skip_existing=True`, the seeds whose path exists are filtered out in the parent before dispatch, and the skip count is logged. The default overwrites, as `save` does today.
- **Progress collection.** joblib `Parallel(..., return_as="generator_unordered")`. The parent consumes results as they finish and logs every `max(1, ceil(n_to_run / 10))` completions and at the last one: `done/total`, elapsed, ETA from the mean time per finished seed. `run_ensemble` sorts the collected results back into seed order.
- **Start line content.** Built from `spec.params()` plus `ext_strength` and `n_events`, formatted compactly: floats to 2 decimals (`mu` to 3), bools and ints as they are, `(N,)` arrays as `array(N, mean=…)`. It must work for any spec, not only legacy-nameable ones. It does not call `legacy_filename`, which raises for non-paper setups. Then the seed count, `n_jobs` and (for `save_ensemble`) the output directory.
- **Finish line content.** Elapsed wall time; for `save_ensemble` the number of files written and skipped and the output directory; for `run_ensemble` the number of results.
- **Testbed notebook.** Replaces its run-then-save double loop with one loop over `(setting, pressure)` that logs `ensemble i/n` and calls `save_ensemble` into the existing `../simOut` location. It configures logging at INFO with timestamps, with one handler to the notebook output and a `FileHandler` writing `sweep.log` in the output directory (created first).
- **Docs.** `docu/running-simulations.md` (and the package section of `docu/README.md` if it lists the testbed API) mention `save_ensemble`, `skip_existing` and how to turn logging on.
- **Binding constraints respected.** `ANBmodel.py` is untouched. There's no global RNG, since each worker still builds its own `default_rng(seed)`. Output files and their names are byte-for-byte the same as `SimulationResult.save` produces today.

## Testing Decisions

- Test only the external behaviour at the existing seam, the public `Testbed` API (`run_ensemble`, `save_ensemble`) plus captured log records, and not internal helpers or exact message wording beyond key facts (such as `"3/3"`, a parameter value, the output directory).
- Use a tiny spec and short schedule like the existing testbed tests (few agents, a pressure event of a few steps, short windows) so the tests are fast and pass under `NUMBA_DISABLE_JIT=1`.
- Cases:
  - `save_ensemble` writes one CSV per seed under the legacy names and returns those paths in seed order, and the written files equal `run(seed).save(...)` for the same seed.
  - `legacy_names=False` writes `seed{seed}.csv`.
  - `skip_existing=True` leaves existing files untouched (mtime/content), runs only the missing seeds, and logs the skip count. With all seeds present, nothing runs.
  - The default overwrites existing files.
  - With `caplog` at INFO, a start record (with a parameter value and seed count), at least one progress record, and a finish record (written/skipped counts) are emitted for both methods. At DEBUG, one record per written path.
  - Without logging configured, nothing reaches stderr (the `NullHandler` contract).
  - `run_ensemble` with `n_jobs=2` still returns results in seed order, equal to sequential runs.
- Prior art: the `Testbed.run` tests (small `make_spec`/`short_config` helpers), the testbed protocol tests, and the notebook test's pattern for executing notebook code cells.

## Out of Scope

- Any change to `ANBmodel.py` (frozen reference model) or its output.
- Progress bars (`tqdm`) or new dependencies.
- Logging inside a single run (per time step) or from the Numba kernel.
- Making the analysis scripts log.
- Changing `demo.ipynb` beyond what its existing test requires.
- A sweep/config abstraction above `Testbed` (the outer loop stays in the notebook).

## Further Notes

- Log records from worker processes don't reach the parent under joblib's loky backend. That's why all logging happens in the parent, from the records the workers return.
- Resuming with `skip_existing=True` after changing the model would reuse stale CSVs; the docs should say to delete or switch the output folder after model changes (same caution as the validation cache).
