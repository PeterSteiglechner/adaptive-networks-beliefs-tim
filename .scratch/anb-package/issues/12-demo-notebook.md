# 12: Demo notebook

**What to build:** A new user opens the testbed notebook and learns the API end to end:

1. Build a `ModelSpec` and `TestbedConfig`, run one seed, and plot the focal belief over time plus a few metrics.
2. Run a small ensemble with `run_ensemble` and show the response frequencies.
3. Define a toy `Network` subclass inline (only `build`) and run it through the same testbed, to show how little code an extension needs.

See the spec `docu/spec-anb-package.md`, "Notebook" (stories 49–51).

**Blocked by:** 09 (Drop-in legacy output)

**Status:** ready-for-agent

- [ ] The notebook runs top to bottom in a fresh `anb` kernel without errors
- [ ] It shows a single-run time series of `x_focal` and at least two other metrics
- [ ] It shows response frequencies from a small ensemble
- [ ] It defines a toy `Network` subclass inline, implementing only `build`, that runs through `Testbed` unchanged
- [ ] It has no source edits to the package or to `ANBmodel.py`
