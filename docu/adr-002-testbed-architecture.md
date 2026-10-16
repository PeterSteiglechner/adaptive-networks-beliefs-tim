# ADR-002: Refactor into an `anb/` package with an injectable model and a testbed

**Status:** Accepted, amended 2026-10-14 (Q22–Q25). The RNG change (Q7, Q17) still needs the supervisor sign-off
named in [ADR-001](adr-001-numba-over-gpu.md)
**Date:** 2026-10-14
**Deciders:** Tim Krüger

## Context

All of the simulation lives in one script, `ANBmodel.py`. Module-level constants (`M`, `n_agents`, `tau`, `lam`,
`fixedBNat100`, `two_external_events`) configure it, the global `np.random` drives it, and one state array of
shape `100 × (16 + 6 + 10 + 45 + τ·10)` holds metadata, metrics, beliefs, edge weights and belief-change memory
side by side, addressed through column-index lists (`beliefids`, `edgeids`, `delbeliefids`). The model, the
experiment protocol (warm-up, pressure window, tracked time steps), the metrics and the output format are all
tangled together inside `run_simulation`, `fill_metrics` and `get_output`.

The planned extensions ([projectIdea.md](../projectIdea.md)) need these parts to change independently:

- other social networks: static topologies such as two clusters first, adaptive rewiring later
- other agent types: influencers with fixed beliefs, heterogeneous β, agents that pick their own focal belief
- other experiment protocols: two pressure events, ramps
- new metrics

[ADR-001](adr-001-numba-over-gpu.md) also plans to move the hot loop to Numba, which needs plain arrays and an
explicit RNG.

The goal is a clean **testbed**: a `Testbed` class that runs experiments on an **injected model**, separate
classes for **metrics**, abstract base classes for **agent** and **network**, and a notebook that drives the
testbed. This ADR records the design decisions from a structured design interview (Q1–Q21 below). The class
diagram doc (`docu/class-design.md`, still to be written) is built from the section
[Resulting class inventory](#resulting-class-inventory).

## Decision

1. Build a **new package `anb/`** next to `ANBmodel.py`. The legacy script stays frozen as the reference model.
2. Use **`abc.ABC`** abstract base classes for `Agent`, `Network` and `Metric`, with shared default behaviour
   in the base classes.
3. Store the population as **arrays** in an `AgentState` dataclass. `Agent` is an **agent type**: it holds the
   update rule and its parameters, not the data. Parameters can differ per agent (Q23).
4. A concrete **`ANBModel`** combines an `Agent` and a `Network`. Each `step()` hands one random-sequential sweep
   to the agent's Numba kernel (Q22).
5. The **`Testbed`** owns the time loop, the pressure schedule, seeding, metric tracking and result assembly. It
   receives the model through a **`ModelSpec`** factory.
6. **No global random state.** The testbed creates one `np.random.Generator` per seed and passes it to every
   component explicitly.
7. Output is a **drop-in replacement**: the same DataFrame columns as legacy, and an optional legacy filename
   helper, so the existing `analyse_*.py` scripts keep working.
8. The new package is checked against legacy **statistically**, not bit for bit.

## Decision log

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| Q1 | Scope of the design doc | Class diagram plus short responsibility notes for each class. No migration plan in it | Keeps the diagram doc focused. The plan lives in this ADR's action items |
| Q2 | Fate of `ANBmodel.py` | New package `anb/`. Legacy script stays untouched | Legacy remains the regression baseline and keeps producing paper-comparable CSVs |
| Q3 | Interface mechanism | `abc.ABC` with `@abstractmethod` | Base classes can share code (e.g. `Network.neighbours()` built on the adjacency), and "is-a" relationships show clearly in a class diagram |
| Q4 | Granularity of `Agent` | Population as arrays, not one object per agent | Matches today's layout and stays compatible with Numba (ADR-001). Subclasses express new update rules. Heterogeneous and mixed populations use per-agent parameters (Q23) |
| Q5 | Model vs. Testbed | Model = agent + network + ρ, exposes `step()`. Testbed owns the time loop, schedule, seeding, tracking and output. Model is injected as a factory | Each seed needs a fresh model. One place handles ensembles and parallelism |
| Q6 | Adaptive network hook | **Not now.** `Network` starts static. Adaptive networks come later as a new `Network` subclass | Keeps the first version small. *Consequence:* the rewiring ADR (ADR-001 item 7) decides whether rewiring runs between sweeps, as an extra call in `ANBModel.step()`, or inside the sweep kernel |
| Q7 | Equivalence with legacy | Statistical (response frequencies and metric distributions over many seeds) | ADR-001 already accepts a changed RNG stream. Bit-for-bit would force legacy RNG ordering into the design |
| Q8 | State vs. rule | `AgentState` dataclass holds arrays. `Agent` ABC holds rule and parameters | Metrics and networks read state without depending on the rule. Numba kernels get plain arrays |
| Q9 | Model class | Concrete `ANBModel`, not abstract. Each parameter lives with its owner: β, ε, μ, λ, τ, `init_w` on `Agent` (all but τ may be per agent, Q23); `link_prob` on `Network`; ρ on `ANBModel` | Only agent and network need to vary by subclass |
| Q10 | Metrics design | `Metric` ABC, one subclass per metric, collected by a `MetricsRecorder`. Response classification is a separate `ResponseClassifier` | Per-step metrics and post-hoc classification run at different times on different inputs |
| Q11 | Pressure configuration | Separate `PressureSchedule` class with `strength_at(t)` | Two-event runs, ramps and pulses become instances or subclasses instead of `if` branches. The analysis windows are derived from it (Q24) |
| Q12 | Output format | Same columns as legacy, so the DataFrame is a drop-in replacement | The five analysis scripts keep working on new runs |
| Q13 | Notebook purpose | Demo / smoke test, plus a playground for swapping in new `Agent` and `Network` subclasses. Large sweeps stay in a script | The notebook shows how to use the API. It is not the production runner |
| Q14 | Focal belief | Default focal dimension 0 as in legacy, but stored per agent: `AgentState.focal (N,)` | Costs nothing now and enables idea 3 (agents choose their focal belief) |
| Q15 | Legacy filenames | Include a `legacy_filename(params)` helper | Analysis scripts locate runs by exact filename |
| Q16 | Form of the factory | `ModelSpec` dataclass with `build(rng) -> ANBModel` and `params() -> dict` | Can be pickled for `joblib`, describes itself (parameters go into the output columns), and a sweep is a list of specs |
| Q17 | Randomness | `np.random.default_rng(seed)` per run, passed explicitly to `build`, network construction and `step` | Safe in parallel, reproducible per seed, and what Numba needs. Fixes known issue #2 (shared shuffle state) |
| Q18 | Edge weight storage | Symmetric matrix `(N, M, M)` in `AgentState`. Flattened to `w{m}{n}` columns only at output | Belief updates, energies, clustering and betweenness all work on the matrix directly |
| Q19 | `fixedBNat100` | Agent parameter `learning_until: int \| None`. The sweep receives `t` and skips the **whole** edge update (λ included) once `t ≥ learning_until` | Matches the module-level gate used for the paper's runs and removes the two inconsistent gates in legacy |
| Q20 | `Metric.compute` signature | `compute(ctx: StepContext) -> ndarray (N,)`, where `StepContext` holds `model`, `t` and `ext` | Energies need the network, ρ and the current pressure. New context fields don't break existing metrics |
| Q21 | Diagram layout | Mermaid `classDiagram` with one `namespace` block per module | The file mapping can be read straight from the diagram |
| Q22 | Numba boundary | `Agent` exposes one abstract `sweep(state, adjacency, t, ext, rng)`: shuffle the agent order, then belief and edge updates for every agent. `ANBAgent.sweep` is one `@njit` kernel. Replaces the per-agent `update_beliefs(state, i, …)` / `update_edges(…)` | Per-agent Python calls would leave T × N = 20,000 Python-level calls per run and cap ADR-001's speed-up. Subclass polymorphism now costs one call per step. The time loop stays in Python (200 calls per run) so metrics and the schedule stay outside Numba |
| Q23 | Heterogeneous and mixed populations | `beta`, `eps`, `mu`, `lam`, `init_w` accept a scalar or an `(N,)` array. A `fixed: ndarray[bool] (N,) \| None` mask marks agents that never update beliefs or edges. `Agent.param_arrays(n_agents)` broadcasts everything to `(N,)` arrays for the kernel | One `ModelSpec` holds one `Agent`, so a subclass would change the whole population. Influencers (idea 2) need a few fixed agents among normal ones, and heterogeneous β (idea 4) is a parameter, not a rule. Both become parameter settings, with no subclass needed. How influencers' fixed beliefs are initialised is decided with idea 2 |
| Q24 | Analysis windows | `PressureSchedule.events` is the single source for pressure timing. `TestbedConfig` holds the schedule plus `window_len = 10` and `recovery = 50`, and derives `T`, the analysis windows, the snapshot times and `track_times` | The first draft defined the pressure windows twice (schedule and config ranges) and brought back the two-event `if` through `Optional` `during2`/`after2` fields. The derived values reproduce legacy exactly |
| Q25 | Validation size | ≥ 100 seeds per configuration, shared with ADR-001 item 5 | The two ADRs asked for 50 and 100. One validation run serves both |

Q22–Q25 were added on 2026-10-14 after the design was checked against ADR-001.

## Resulting class inventory

This is the source for the class diagram. Signatures are indicative. Types use NumPy shapes: `N` agents, `M`
beliefs, `T` time steps.

### `anb/agents.py`

**`AgentState`** (dataclass): the population's data, with no behaviour beyond views.
- `beliefs: ndarray (N, M)`, values on the 21-point grid in [−1, 1]
- `edge_weights: ndarray (N, M, M)`, symmetric, zero diagonal
- `delta_memory: ndarray (N, τ, M)`, last τ belief changes
- `focal: ndarray[int] (N,)`, focal dimension per agent, default 0
- `n_agents`, `n_beliefs` (properties)
- `edge_vector() -> ndarray (N, M(M−1)/2)`, upper triangle in legacy column order
- `focal_beliefs() -> ndarray (N,)`

**`Agent`** (ABC): an agent type, meaning the update rule and its parameters.
- attributes: `n_beliefs (M)`, `belief_options`, `tau`, `learning_until`, and `beta`, `eps`, `mu`, `lam`,
  `init_w` as scalar or `(N,)` array, `fixed: ndarray[bool] (N,) | None` (Q23)
- `init_state(n_agents, rng) -> AgentState` (abstract)
- `param_arrays(n_agents) -> dict[str, ndarray]` (concrete): every per-agent parameter broadcast to `(N,)`
- `sweep(state, adjacency, t, ext, rng) -> None` (abstract, Q22): one random-sequential sweep. For each agent
  in shuffled order it computes the social input `ρ · Σ_k x_foc(k)` over the neighbours, resamples all beliefs in
  random order and writes `delta_memory`, then updates the edges with the neighbours' mean edge matrix (μ is 0
  without neighbours). Fixed agents are skipped, and the edge update is skipped once `t ≥ learning_until`

**`ANBAgent(Agent)`**: the paper's agent. Glauber belief updates (eq. 2) and Hebbian + social + decay edge
updates (eq. 3), in one `@njit` sweep kernel (ADR-001). Future subclasses are new rules only:
`CentralFocalAgent` (idea 3) and `BeliefObservingAgent` (idea 1). Influencers and heterogeneous β are
parameter settings of `ANBAgent` (Q23).

### `anb/networks.py`

**`Network`** (ABC): the social network, static in this version.
- `adjacency: ndarray (N, N)`, 0/1 or weighted
- `neighbours(i) -> ndarray[int]` (concrete default, built from `adjacency`)
- `degree() -> ndarray (N,)` (concrete default)
- `build(n_agents, rng) -> Network` (abstract classmethod or constructor contract)

**`ErdosRenyiNetwork(Network)`**: `link_prob`. Reproduces the legacy topology.
Future subclasses: `TwoClusterNetwork`, `AdaptiveNetwork` (needs a step hook, see Q6).

### `anb/model.py`

**`ModelSpec`** (dataclass): the injectable factory.
- `agent: Agent`, `network_cls: type[Network]`, `network_kwargs: dict`, `n_agents: int`, `rho: float`
- `build(rng) -> ANBModel`, which checks that every per-agent parameter has length `n_agents`
- `params() -> dict`, the parameters written to output columns (`link_prob`, `init_w`, `beta`, `rho`, `eps`,
  `mu`, `lam`, `fixedBNat100`, …). Uniform parameters are scalars. Per-agent ones are `(N,)` arrays and become
  per-agent columns of the same name, so homogeneous runs produce the legacy columns unchanged

**`ANBModel`** (concrete): agent + network + state.
- `agent: Agent`, `network: Network`, `state: AgentState`, `rho: float`
- `step(t, ext, rng) -> None`: calls `agent.sweep(state, network.adjacency, t, ext, rng)` (Q22). An adaptive
  network adds its update here or inside the kernel (Q6)

### `anb/metrics.py`

**`StepContext`** (dataclass): `model: ANBModel`, `t: int`, `ext: float`.

**`Metric`** (ABC): `name: str` (the output column), `compute(ctx) -> ndarray (N,)` (abstract).

Concrete metrics, one per legacy column: `Degree` (`n_nbs`), `PersonalEnergy` (`Hpers`), `FocalPersonalEnergy`
(`Hpersfoc`), `SocialEnergy` (`Hsoc`), `ExternalEnergy` (`Hext`), `TriangleBalance` (`tb_tot`),
`FocalTriangleBalance` (`tb_foc`), `MeanAbsEdge` (`absOm_tot`), `FocalMeanAbsEdge` (`absOm_foc`),
`WeightedClustering` (`clust`), `FocalBetweenness` (`bc_foc`), `ExpectedInfluence` (`expI`), `FocalBelief`
(`x_focal`), `NonFocalExtremity` (`extr_nonfoc`). Related metrics may share a private helper. For example,
the two balance metrics share one triangle pass.

**`MetricsRecorder`**: `metrics: list[Metric]`, `record(ctx) -> None` (stores a snapshot of all metrics plus
beliefs and the edge vector), `to_frame() -> DataFrame` (long format: `t`, `id`, metric columns, `0`…`9`,
`w01`…`w89`).

**`ResponseClassifier`**: before, during and after windows. `classify(frame) -> DataFrame` with the per-agent
dummies `persistPos`, `nonpersistPos`, `compliant`, `resilient`, `resistant`, `latecompliant`.

### `anb/testbed.py`

**`PressureSchedule`**: `strength`, `events: list[tuple[int, int]]` (inclusive pressure intervals),
`strength_at(t) -> float`, `n_events`. Factory helpers `single_event(s)` (101–150) and `two_events(s)` (101–150,
201–250) replace `two_external_events`.

**`TestbedConfig`** (dataclass, Q24): `schedule: PressureSchedule`, `window_len: int = 10`,
`recovery: int = 50`, `detail: bool`. Derived, for events `(s₁, e₁) … (sₖ, eₖ)`:
- `T = eₖ + recovery`
- `windows()`: `before` = the `window_len` steps before s₁, then per event `j`, `during{j}` = the last
  `window_len` steps of the event and `after{j}` = the last `window_len` steps of the `recovery` steps after it.
  Gives legacy's 91–100, 141–150, 191–200 (and 241–250, 291–300). The first event's windows keep the legacy
  names `during` and `after`
- `snapshot_times()`: 0, s₁ − 1, and eⱼ + `recovery` for each event `j` (0/100/200, plus 300)
- `track_times()`: every `t` if `detail`, otherwise {0, 1} ∪ windows ∪ snapshots, as in legacy

**`Testbed`**
- constructor: `spec: ModelSpec`, `config: TestbedConfig`, `metrics: list[Metric]`,
  `classifier: ResponseClassifier`, `aggregator: WindowAggregator` (both built from `config.windows()` by
  default)
- `run(seed) -> SimulationResult`: `rng = default_rng(seed)`, `model = spec.build(rng)`, record at t = 0,
  then for each `t`, `model.step(t, config.schedule.strength_at(t), rng)` and record if
  `t ∈ config.track_times()`
- `run_ensemble(seeds, n_jobs) -> list[SimulationResult]` (`joblib`)

**`WindowAggregator`**: ensemble-mode output. The snapshot rows plus window averages at 95.5/145.5/195.5 (and
245.5/295.5), with the response dummies on the 95.5 row (resolved open point 1, see
[class-design.md](class-design.md#resolved-open-points)).

**`SimulationResult`**: `frame: DataFrame`, `params: dict`, `seed: int`. `to_frame()` appends the parameter
columns (per-agent ones joined on `id`). `save(directory, legacy_names=True)`.

### `anb/io.py`

**`legacy_filename(params, detail) -> str`**: reproduces `run_one`'s naming exactly. `params` carries `lam` (from
`Agent`) and `n_events` (from the schedule), so they are not separate arguments.

### `notebooks/testbed.ipynb`

Not part of the class diagram. It (1) builds a `ModelSpec`, runs one seed and plots the focal belief over time
and a few metrics, (2) runs a small ensemble and shows response frequencies, (3) defines a toy `Network`
subclass inline and runs it through the same testbed.

## Consequences

**Easier**
- New agent types, networks, metrics and pressure protocols are new classes, not edits to the core loop.
- Testing pieces in isolation: a metric on a hand-made `AgentState`, a network's degree distribution, a schedule.
- ADR-001's Numba work: `ANBAgent.sweep` is the kernel, working on `AgentState` arrays and the adjacency, so the
  interface already matches it (Q22).
- Known issues #2 (shared shuffle state) and #7 (globals under `__main__`) go away by construction.

**Harder**
- Two code paths until the new package is validated against legacy.
- A new update rule means a new sweep kernel, written in Numba's subset of Python. Debug kernels with
  `NUMBA_DISABLE_JIT=1`.
- Adaptive networks (Q6) need either a network update call in `ANBModel.step` or rewiring inside the kernel.

**Behavioural differences from legacy (deliberate)**
- Different RNG stream, so seeds don't reproduce old CSVs.
- One `fixedBNat100` semantics (Q19) instead of two.

## Open points (not covered in the interview; resolved in [class-design.md](class-design.md#resolved-open-points))

- **Window aggregation owner.** The proposal above puts the before/during/after averaging and the ensemble-mode
  row selection in `SimulationResult.to_frame()`. It could instead be its own `WindowAggregator`.
- **`Hsoc` uses the neighbours' *mean* focal belief, but the dynamics use the *sum*** (legacy
  `get_energies` vs. `run_simulation`). The `SocialEnergy` metric should reproduce legacy for comparability, and
  the mismatch should be noted (see [known-issues.md](known-issues.md)).
- **Where `M` and the belief grid live.** The proposal puts them on `Agent`. `AgentState` takes its shape from
  them.

## Action Items

1. [x] Write `docu/class-design.md`: a Mermaid `classDiagram` with one namespace per module, built from the
   inventory above, and one short responsibility note per class.
2. [x] Resolve the three open points above in that doc.
3. [ ] Scaffold `anb/` with the ABCs and the `ANBAgent`, `ErdosRenyiNetwork`, `ANBModel`, `ModelSpec` and
   `Testbed` implementations. Write `ANBAgent.sweep` as plain Python first, then add `@njit` (ADR-001 item 2).
4. [ ] Port the 14 metrics and the response classifier, and check them against `fill_metrics` and `get_output`
   on a fixed legacy state.
5. [ ] Validate statistically against `ANBmodel.py` (the 4 paper configurations × 6 pressures, ≥ 100 seeds,
   Q25): response frequencies and BN metric distributions. This is also ADR-001 item 5.
6. [ ] Write `notebooks/testbed.ipynb` (Q13).
7. [x] Add the ADR and the class design doc to [docu/README.md](README.md).
8. [ ] Get the supervisor's sign-off on the RNG change (Q7, Q17, ADR-001).
