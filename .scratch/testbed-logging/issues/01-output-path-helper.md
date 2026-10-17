# 01: Pull the output path out into a shared helper

**Labels:** ready-for-agent

**What to build:** A prefactor with no behaviour change. The CSV path for one seed of a `Testbed` run can be worked
out before running, from the spec's params, the schedule (`ext_strength`, `n_events`), the seed, the detail flag,
the output directory and `legacy_names`. The naming is the same as `SimulationResult.save` uses today: the legacy
name, plus `_detailed` in detail mode, or `seed{seed}` without legacy names. `SimulationResult.save` uses this
helper too, so the two can't drift apart. Later tickets use it for writing from the worker and for `skip_existing`.
See `.scratch/testbed-logging/spec.md`, section "Output path without running".

**Blocked by:** None (can start immediately)

**Status:** done

- [x] One helper builds the per-seed CSV path, and `SimulationResult.save` uses it
- [x] A `Testbed` can produce a seed's path without running that seed
- [x] For the same seed, the path from the `Testbed` equals the path `run(seed).save(...)` returns, with legacy names and without, in ensemble and detail mode
- [x] `SimulationResult.save` output (names and file contents) is unchanged, and the existing tests pass, also with `NUMBA_DISABLE_JIT=1`
- [x] `ANBmodel.py` is untouched
