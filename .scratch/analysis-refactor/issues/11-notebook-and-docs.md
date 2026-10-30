# 11: Analysis notebook and docs

**Labels:** ready-for-agent

**What to build:** A notebook `notebooks/analysis.ipynb` that runs the package figure by figure (and `make_all` at the end) on a configurable `sim_dir`, plus the final docs.

- Bring `docu/analysis-package.md` in line with what was built (drop the "planned" banner, fix any changed signatures).
- Add the package to `docu/README.md` and `CLAUDE.md` (commands, plus the note that the analysis no longer needs exact files).
- Add a link from `docu/analysis.md` and `docu/running-simulations.md` to the new doc, but only after the user's pending edits to `analysis.md` are committed (ask first).
- Extend `tests/test_notebook.py` (or similar) to execute the notebook on the synthetic fixture.

**Blocked by:** #10

**Status:** open

- [ ] The notebook runs top to bottom on the fixture in the test
- [ ] The docs match the implemented API
- [ ] The old `analyse_*.py` scripts and `ANBmodel.py` are unchanged
