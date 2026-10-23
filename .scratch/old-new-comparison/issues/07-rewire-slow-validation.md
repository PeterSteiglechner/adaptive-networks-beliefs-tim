# 07: Rewire the slow validation onto the comparison module

**Labels:** ready-for-agent

**What to build:** The slow statistical validation keeps generating its own legacy and anb runs, with fresh legacy modules, but uses the comparison module's summary, statistics and table code throughout, so validation and reports can't disagree on method. Its report keeps the same content and gains the conditional and lean tables. Its run cache moves out of `simOut_old_model`, so it no longer mixes with the sweep CSVs that the folder comparison reads.

**Blocked by:** #03

**Status:** open

- [ ] The validation uses only the comparison module for summaries, statistics and tables (no duplicate implementation left in the tests)
- [ ] The report includes the conditional response and lean tables
- [ ] The validation cache no longer lives inside `simOut_old_model`
- [ ] `pytest -m slow -s tests/test_statistical_validation.py` passes, and the pooled response and metric results match the previous report on the 4 paper configs
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
