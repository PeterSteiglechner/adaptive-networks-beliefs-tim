# 01: Filename parsing in `anb.io`, M/N/τ suffixes, folder index

**Labels:** ready-for-agent

**What to build:** The naming scheme is defined once, in `anb.io`, for both directions, and a simulation folder can be turned into an index of runs.

- Move the inverse parser out of the comparison module into `anb.io`, next to `legacy_filename`. The comparison module imports it from there.
- The parser also handles detailed files (`detailed/` and `_detailed`), `_2events`, `_lambda…`, the bare `fixedBNat100` token, and the `_M{:.2f}`/`_N{:.2f}`/`_tau{:.2f}` suffixes. When a suffix is absent, the defaults apply (M 10, N 100, τ 1, λ 0, one event).
- `legacy_filename` takes optional M, N and τ. It writes a suffix only when the value differs from 10/100/1, right after `seed…` in the order M, N, τ, before `_2events`/`_lambda`. `output_path`/`SimulationResult.save` pass the spec's M, N and τ through.
- `anb.analysis.index_folder(sim_dir)` scans `sim_dir/` and `sim_dir/detailed/` and returns one row per parseable CSV (parameters, `s`, `seed`, `lam`, `n_events`, `M`, `N`, `tau`, `detailed`, path). It warns once, with a count and examples, about files it can't parse.
- Update known issue #4 in `docu/known-issues.md`: the package writes the suffixes, legacy still doesn't.

**Blocked by:** none

**Status:** open

- [ ] Round-trip tests `legacy_filename` → parser for the baseline, `fixedBNat100`, `_2events`, `_lambda`, each of M/N/τ, combinations, and detailed names
- [ ] Every existing baseline filename is unchanged (existing `test_legacy_filename.py` passes as is)
- [ ] The comparison module uses the moved parser, and its tests pass
- [ ] `index_folder` on a synthetic folder returns the expected rows, and a stray file gives exactly one warning
- [ ] `index_folder(simOut_old_model)` indexes all 8400 ensemble files plus the detailed ones (a manual check, noted in the ticket)
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
