# 09: Shared-start runner

**Labels:** ready-for-agent

**What to build:** Legacy and anb can be run from one identical t = 0 state, ruling out the initial conditions as a source of difference.

First, the legacy-row-layout ↔ `AgentState` conversion currently in the legacy oracle moves into one place (the comparison module, or a helper that the oracle imports), so there is a single conversion. The comparison module depends on legacy only through the oracle loader.

The runner takes a cell (parameters), a pressure, a start seed, K, a step window (default t = 95–155) and n_jobs. It:
- builds one t = 0 state (beliefs, symmetric edge weights, adjacency, belief-change memory);
- runs K legacy replicas, each on a fresh `load_legacy()` copy whose `initialise_agents`/`initialise_network` return copies of the shared state, with replica seed r and `track_times` overridden to record every step in the window;
- runs K anb replicas through `Testbed`, using a model spec built from the shared `AgentState` and network, with a tracked window covering the range;
- returns long-format trajectories (implementation, replica, t, agent, metric columns).

**Blocked by:** #01

**Status:** open

- [ ] There is a single legacy ↔ `AgentState` conversion, which the oracle imports
- [ ] Both implementations report identical t = 0 metrics from the shared state
- [ ] Replicas within one implementation differ after t = 0
- [ ] The trajectory frame has the expected columns and covers every step of the window for each implementation and replica
- [ ] `ANBmodel.py` on disk is unchanged, and `Testbed` is not modified
- [ ] Tests use a short run (small window, K = 2) and are marked `slow` if needed
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
