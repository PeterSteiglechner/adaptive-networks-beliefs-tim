# 11: Analysis notebook and docs

**Labels:** ready-for-agent

**What to build:** A notebook `notebooks/analysis.ipynb` that runs the package figure by figure (and `make_all` at the end) on a configurable `sim_dir`, plus the final docs.

- Bring `docu/analysis-package.md` in line with what was built (drop the "planned" banner, fix any changed signatures).
- Add the package to `docu/README.md` and `CLAUDE.md` (commands, plus the note that the analysis no longer needs exact files).
- Add a link from `docu/analysis.md` and `docu/running-simulations.md` to the new doc, but only after the user's pending edits to `analysis.md` are committed (ask first).
- Extend `tests/test_notebook.py` (or similar) to execute the notebook on the synthetic fixture.

**Blocked by:** #10

**Status:** done

- [x] The notebook runs top to bottom on the fixture in the test
- [x] The docs match the implemented API
- [x] The old `analyse_*.py` scripts and `ANBmodel.py` are unchanged

**Done:**

- `notebooks/analysis.ipynb` (outputs cleared): a `parameters`-tagged cell (`SIM_DIR`, `OUT_DIR`, `BASE`, `FORMATS`),
  the index overview, one shared `Loader(whole=True)` and index, every figure, summary and table in paper order with
  a `show` helper (a `None` shows only the warning), `save` / `save_table` once each, then `make_all` with the shared
  loader. It also runs on `simOut_old_model` in a kernel (`make_all`: 38 written, 3 skipped: Fig. A1/A2, A8).
- `tests/test_notebook.py` runs it as a plain script on the synthetic fixture, with a cell injected after the
  `parameters` cell (default suite, a few seconds), and checks that it's committed without outputs.
- `docu/analysis-package.md`: banner dropped, notebook mentioned, `format_report(report, out_dir=None)` and the
  `fig3` `example_agents=None` default fixed, and a "Reference" section of stubs for everything exported, checked by
  `tests/test_analysis_docs.py` (parameter names, kinds and defaults, dataclass fields, warning bases).
- Links from `docu/analysis.md`, `docu/running-simulations.md` and `docu/README.md`; `CLAUDE.md` commands and a
  paragraph on the package. `ANBmodel.py` and `analyse_*.py` are unchanged since the feature's spec commit
  (`b1e43d1`). Their diff against `main` comes from the user's commits of Oct 16–23, before this feature.
