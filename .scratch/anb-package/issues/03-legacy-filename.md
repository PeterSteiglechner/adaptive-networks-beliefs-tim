# 03: `legacy_filename`

**What to build:** An analysis-script user gets new CSVs under exactly the filenames the `analyse_*.py` scripts look for. `legacy_filename(params, detail)` reproduces `run_one`'s naming: `f"{k}{v:.2f}"`, μ with `.3f`, seed and pressure strength unformatted, the bare `fixedBNat100` token, and the `lam` and two-event variants. It reads `lam` and `n_events` from `params` and targets `simOut/` or `simOut/detailed/` according to `detail`. It does **not** add the `_M`/`_N`/`_tau` suffixes, because `run_one` doesn't either (known issue #4 stays open). See the spec `docu/spec-anb-package.md`, "`io` module".

**Blocked by:** 01 (Test scaffold and dependencies)

**Status:** ready-for-agent

- [ ] The result equals `run_one`'s string for every parameter combination in the paper sweeps and in `docu/running-simulations.md`
- [ ] Covered explicitly: `fixedBNat100` on/off, a non-default `lam`, two-event runs, and detail vs ensemble directory
- [ ] The legacy strings come from legacy code through the ticket-01 helper, not hand-typed expectations, wherever that's possible without editing `ANBmodel.py`
