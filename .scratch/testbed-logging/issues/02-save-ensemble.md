# 02: `Testbed.save_ensemble` writes each CSV from the worker as its seed finishes

**Labels:** ready-for-agent

**What to build:** A researcher can call `Testbed.save_ensemble(seeds, out_dir, n_jobs=1, legacy_names=True)`.
Each seed's CSV goes to disk as soon as that seed finishes, so a crash late in a sweep keeps everything finished so
far, and finished frames don't pile up in memory. Each worker runs its seed, saves the CSV itself and sends back
only a small record (seed, path, runtime), never the frame. The method returns the paths in seed order. Existing
files are overwritten, the same as `SimulationResult.save`. Each worker still builds its own `default_rng(seed)`
(no global RNG). See `.scratch/testbed-logging/spec.md` (user stories 10–13, 15).

**Blocked by:** 01 (Pull the output path out into a shared helper)

**Status:** done

- [x] `save_ensemble` writes one CSV per seed under the legacy names and returns those paths in seed order
- [x] Each written file equals what `run(seed).save(...)` writes for the same seed
- [x] `legacy_names=False` writes `seed{seed}.csv`
- [x] Existing files are overwritten by default
- [x] Works with `n_jobs=2` (same files and paths as sequential)
- [x] Workers send back only seed, path and runtime, not frames
- [x] Tests use a tiny spec and short schedule (like the existing `Testbed.run` tests) and pass with `NUMBA_DISABLE_JIT=1`
