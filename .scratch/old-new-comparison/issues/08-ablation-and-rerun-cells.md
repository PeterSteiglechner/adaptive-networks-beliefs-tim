# 08: Ablation and more-seeds rerun cells

**Labels:** ready-for-agent

**What to build:** The report script gains cells that run with a known deliberate difference reverted, or with more seeds, saved as normal output folders and compared through the main entry point:
- an ablation: an anb `Testbed` sweep with `learning_until = 102` for the adaptive→static (`fixedBNat100`) configs, saved with `save_ensemble` into its own folder and compared against the old folder;
- a rerun: selected cells (e.g. μ = 0.1) with more seeds (e.g. 1000), saved and compared the same way.

Neither needs a new code path in the model.

**Blocked by:** #05

**Status:** open

- [ ] The ablation cell produces its own output folder and a comparison against old using the same tables
- [ ] The rerun cell is configured by constants (cells and seeds) and goes through the same comparison
- [ ] Structural test: a `learning_until = 102` run still changes edge weights at t = 101 and stops afterwards
- [ ] `ANBmodel.py` and `Testbed` are not modified
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
