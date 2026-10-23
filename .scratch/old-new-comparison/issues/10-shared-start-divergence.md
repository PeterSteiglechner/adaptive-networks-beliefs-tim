# 10: Step-wise shared-start comparison and divergence plots

**Labels:** ready-for-agent

**What to build:** Shared-start ensembles are compared step by step. At each step in the window, and for each tracked quantity (focal belief mean and spread, edge-weight metrics, energies), the comparison computes:
- the difference of ensemble means across replicas, with a CI;
- a KS test on the replica means;
- BH correction across steps and quantities.

It reports the first step at which each quantity diverges significantly. The report script runs this behind its shared-start toggle and writes a divergence table and divergence plots to `compare_out/`.

**Blocked by:** #09, #05

**Status:** done

- [x] Given trajectories, the comparison returns per-step differences with CIs, q-values, and a first-divergent-step per quantity
- [x] On synthetic trajectories with a divergence injected from a known step, that step is reported
- [x] With the toggle on, the report script writes the divergence table and plots
- [x] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
