# 10: Numba sweep kernel

**What to build:** A researcher's runs get much faster with no change in behaviour. `ANBAgent.sweep` delegates to one `@njit(cache=True)` kernel that takes plain arrays and a `np.random.Generator`. It samples the Glauber step by inverse CDF instead of `np.random.choice`, and it's cached on disk so later sessions don't recompile. It stays the only Numba boundary: one call per time step, with the time loop and metrics still in Python. The plain-Python path stays usable for debugging with `NUMBA_DISABLE_JIT=1`. See the spec `docu/spec-anb-package.md`, "`agents` module", and `docu/adr-001-numba-over-gpu.md`.

**Blocked by:** 09 (Drop-in legacy output)

**Status:** done

- [x] The whole test suite passes with JIT on and with `NUMBA_DISABLE_JIT=1`
- [x] A second Python session reuses the on-disk cache instead of recompiling (compile time measured once, then cached)
- [x] On-demand benchmark: one full run (N = 100, M = 10, T = 200, all 14 metrics) new vs legacy, with the update loop timed separately from the metrics. The result is reported against the 50–200× target for the update loop
- [x] There's no global RNG in the kernel or its callers. The Generator is passed in
