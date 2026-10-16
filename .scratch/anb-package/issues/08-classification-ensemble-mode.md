# 08: Response classification, ensemble mode, `run_ensemble`

**What to build:** A researcher runs a testbed in ensemble mode and gets legacy-shaped rows: snapshots at 0/100/200 plus mean rows at 95.5/145.5/195.5, with the six response dummies (`persistPos`, `nonpersistPos`, `compliant`, `resilient`, `resistant`, `latecompliant`) only on the 95.5 row. Two-event runs add 245.5/295.5 and the 300 snapshot automatically. In detail mode the dummies go on the rows at t = T. `run_ensemble(seeds, n_jobs)` runs seeds in parallel with `joblib`, and each worker builds its own model from its own seed.

This adds:

- a `ResponseClassifier` built from `before`/`during`/`after` windows, using the legacy rules unchanged, including the zero and boundary handling from known issue #9
- a `WindowAggregator` built from `windows()` and `snapshot_times()`
- `Testbed` defaults that build both from `config.windows()`

See the spec `docu/spec-anb-package.md`, "`metrics` module" and "`testbed` module".

**Blocked by:** 04 (Tracer bullet: one detail-mode run end to end)

**Status:** ready-for-agent

- [x] `ResponseClassifier.classify` equals legacy classification on the same frames, including hand-built zero and boundary cases
- [x] Ensemble-mode row times are exactly {0, 95.5, 100, 145.5, 195.5, 200}, plus {245.5, 295.5, 300} for `two_events`
- [x] The dummies appear only on the 95.5 row in ensemble mode, and only on the t = T rows in detail mode
- [x] Changing `window_len`/`recovery` in `TestbedConfig` moves the classifier and aggregator windows consistently
- [x] There's one `MetricsRecorder` per run, with no metric state shared between runs
- [x] `run_ensemble(seeds, n_jobs > 1)` gives the same frames as serial `run` for each seed
