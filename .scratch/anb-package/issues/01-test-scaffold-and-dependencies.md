# 01: Test scaffold and dependencies

**What to build:** A researcher can activate the `anb` env, import an (empty) `anb` package from any module or notebook, and run `pytest`, both with JIT on and with `NUMBA_DISABLE_JIT=1`. The suite already has a shared helper that loads the frozen legacy `ANBmodel.py` as an oracle: it injects the `detail` and `track_times` names that legacy defines only under `__main__` (known issue #7), without editing the file. Every later ticket's legacy comparisons build on this helper. See the spec `docu/spec-anb-package.md`, "Package and boundaries" and "Prior art".

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `numba` (≥ 0.56), `pytest`, `scikit-learn` and `statsmodels` are installed in the `anb` conda env and listed in `pyproject.toml`, so the `uv` route still works
- [x] An importable `anb` package exists next to `ANBmodel.py`, and `ANBmodel.py` and the `analyse_*.py` scripts are untouched
- [x] A pytest configuration exists. Slow tests (statistical validation, benchmarks) are behind a marker and excluded from the default run
- [x] A shared test helper loads legacy with `detail`/`track_times` injected into its namespace, and a smoke test calls one legacy function through it
- [x] `pytest` passes with JIT on and with `NUMBA_DISABLE_JIT=1`
