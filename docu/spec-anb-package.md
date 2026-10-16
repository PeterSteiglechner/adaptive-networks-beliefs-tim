# Spec: the `anb/` package (testbed refactor with a Numba sweep kernel)

**Label:** `ready-for-agent`
**Date:** 2026-10-16
**Sources:** [ADR-001](adr-001-numba-over-gpu.md) (with its 2026-10-14 and 2026-10-16 amendments),
[ADR-002](adr-002-testbed-architecture.md) (Q1–Q25) and [class-design.md](class-design.md) (including its resolved
open points). This spec merges them into one buildable unit of work. Where they disagree, the later amendment
wins. The decisions below are already made. Don't reopen them while implementing.

## Problem Statement

All of the ANB simulation lives in one script, `ANBmodel.py`. Module-level constants and source edits under
`__main__` configure it, the global `np.random` drives it, and one wide state array addressed through column-index
lists holds metadata, metrics, beliefs, edge weights and belief-change memory side by side. The model, the
experiment protocol (warm-up, pressure window, tracked steps), the metrics and the output format are all tangled
together inside `run_simulation`, `fill_metrics` and `get_output`.

This blocks the MSc project in two ways:

1. **It can't be extended.** The planned extensions in `projectIdea.md` are other social networks (two clusters,
   later adaptive rewiring), other agent types (influencers, heterogeneous β, agents that choose their focal
   belief), other pressure protocols (two events, ramps) and new metrics. Each one would mean editing the core
   loop and adding `if` branches to a script that has to stay the reference for the paper.
2. **It's too slow.** One run (N = 100, M = 10, T = 200) takes about 4.4 s. About 80 % of that is Python and
   NumPy call overhead in `glauber_fast`, which is called 200,000 times per run on arrays of 10 to 21 elements.
   The paper's 100 seeds per configuration take about 16 min, and sensitivity sweeps and extensions multiply
   that further.

There are also latent bugs that come from the structure: shuffle state shared between runs (known issue #2),
globals that exist only under `__main__` (#7), and two inconsistent `fixedBNat100` gates (#8).

## Solution

A new package `anb/` sits next to `ANBmodel.py`. The legacy script stays frozen, untouched, as the reference
model. The package separates:

- an **agent type** (update rule plus parameters) from the **agent state** (plain arrays),
- the **social network** (an adjacency matrix),
- a **model** that combines the two and advances one time step,
- a **testbed** that owns the time loop, the pressure schedule, seeding, metric tracking and assembly of the
  output,
- **metrics**, one class per output column, plus a separate response classifier and window aggregator.

The hot path, one random-sequential sweep over all agents per time step, is one Numba `@njit` kernel on the CPU.
It keeps the paper's dynamics exactly: shuffled agent order, beliefs resampled one at a time, and each agent
seeing neighbours already updated in that step. The time loop stays in Python (T calls per run), so metrics
and the schedule stay out of Numba. Parallelism across runs stays with `joblib`. There is no GPU.

The researcher drives everything from a `ModelSpec` (what to simulate) and a `TestbedConfig` (how to run the
experiment). One `np.random.default_rng(seed)` is created per run and passed explicitly everywhere. The output
DataFrame has the same columns as legacy, and a `legacy_filename` helper reproduces `run_one`'s naming, so the
five `analyse_*.py` scripts work on new runs unchanged. The new package is validated against legacy
statistically, over at least 100 seeds per configuration, not bit for bit. A notebook demonstrates the API and
how to plug in a new network.

## User Stories

### Running simulations

1. As a researcher, I want to describe a run as a `ModelSpec` (agent type, network type and its parameters,
   number of agents, ρ), so that the configuration is data instead of source edits under `__main__`.
2. As a researcher, I want to describe the experiment protocol as a `TestbedConfig` (pressure schedule, window
   length, recovery length, detail mode), so that the protocol is separate from the model.
3. As a researcher, I want `Testbed.run(seed)` to return a `SimulationResult`, so that I can work with one run
   in memory without going through the disk.
4. As a researcher, I want `Testbed.run_ensemble(seeds, n_jobs)` to run seeds in parallel with `joblib`, so that
   ensembles use all my cores.
5. As a researcher, I want a sweep to be just a list of `ModelSpec`s, so that parameter sweeps are plain Python
   loops or comprehensions.
6. As a researcher, I want `ModelSpec` to be picklable, so that it can be sent to `joblib` workers.
7. As a researcher, I want the same seed with the same spec and config to give an identical result, so that
   any run can be reproduced exactly.
8. As a researcher, I want parallel runs to share no random state, so that results don't depend on scheduling or
   on the order of runs (fixes known issue #2).
9. As a researcher, I want to import and call the package from any module or notebook, so that nothing depends
   on globals defined under `__main__` (fixes known issue #7).
10. As a researcher, I want one run at N = 100, M = 10, T = 200 to be far faster than 4.4 s (target 50–200× on
    the update loop), so that 100-seed ensembles and fine sweeps become cheap.
11. As a researcher, I want the Numba compile cost to be paid once and cached on disk, so that repeated sessions
    don't recompile each time.
12. As a researcher, I want to debug the sweep kernel as plain Python with `NUMBA_DISABLE_JIT=1`, so that I can
    step through the dynamics.

### Model fidelity

13. As a researcher, I want the sweep to keep random-sequential updating (shuffled agent order, each agent
    seeing neighbours already updated in the same step), so that the dynamics match the paper.
14. As a researcher, I want each agent's beliefs resampled one at a time in random order by Glauber dynamics over
    the 21-point grid (eq. 2), each seeing the dimensions updated before it, so that the belief update matches
    the paper.
15. As a researcher, I want the social input to be ρ times the **sum** of the neighbours' focal beliefs, so that
    the dynamics match legacy and eq. 1.
16. As a researcher, I want the edge update to be Hebbian plus social plus decay (eq. 3), with the social term
    using the neighbours' mean edge matrix and μ treated as 0 for an agent with no neighbours, so that the
    belief-network learning matches legacy.
17. As a researcher, I want the last τ belief changes per agent kept in `delta_memory`, so that the Hebbian term
    uses the same memory as legacy.
18. As a researcher, I want one `learning_until` agent parameter that skips the **whole** edge update (λ
    included) once `t ≥ learning_until`, so that `fixedBNat100` has one meaning, matching the module-level gate
    used for the paper's runs (resolves known issue #8).
19. As a supervisor, I want the new package's response frequencies and BN metric distributions to match
    `ANBmodel.py` over at least 100 seeds for each of the 4 paper configurations × 6 pressures, so that I can
    trust results produced with it.
20. As a supervisor, I want the deliberate differences from legacy (the RNG stream, a single `fixedBNat100`
    meaning) documented, so that I can sign off on them.

### Output compatibility

21. As an analysis-script user, I want the result frame to have exactly the legacy columns (`t`, `id`, the 14
    metric columns, the 6 response dummies, belief columns `0`…`9`, edge columns `w01`…`w89`, and the parameter
    columns), so that the `analyse_*.py` scripts read new CSVs unchanged.
22. As an analysis-script user, I want `SimulationResult.save(directory)` to write under the legacy filename by
    default, so that the scripts find new runs where they expect them.
23. As an analysis-script user, I want `legacy_filename(params, detail)` to reproduce `run_one`'s formatting
    exactly (`f"{k}{v:.2f}"`, μ with `.3f`, seed and pressure strength unformatted, the bare `fixedBNat100`
    token, the `lam` and two-event variants), so that no filename differs.
24. As an analysis-script user, I want ensemble mode to produce the snapshot rows at t = 0/100/200 plus averaged
    rows at 95.5/145.5/195.5, with the response dummies only on the 95.5 row, so that the output matches legacy
    row for row.
25. As an analysis-script user, I want detail mode to record every time step and attach the response dummies to
    the rows at t = T, so that the detailed time-series scripts work.
26. As an analysis-script user, I want two-event runs to add the averaged rows at 245.5/295.5 and the snapshot at
    300, so that two-event analyses work without special casing.
27. As a researcher, I want homogeneous runs to produce the scalar legacy parameter columns unchanged, so that
    old and new CSVs can be concatenated.

### Pressure protocols

28. As a researcher, I want a `PressureSchedule` with `strength_at(t)`, so that the pressure at any step comes
    from one place.
29. As a researcher, I want `PressureSchedule.single_event(s)` (t = 101–150) and `two_events(s)` (101–150 and
    201–250), so that the legacy `two_external_events` flag becomes a choice of schedule.
30. As a researcher, I want `T`, the analysis windows, the snapshot times and the tracked steps all derived from
    the schedule's events plus `window_len = 10` and `recovery = 50`, so that pressure timing is defined only
    once.
31. As a researcher, I want to add ramps or pulses as `PressureSchedule` subclasses, so that new protocols need
    no change to the testbed.

### Metrics and classification

32. As a researcher, I want each output metric to be its own `Metric` subclass with a column `name` and
    `compute(ctx) -> (N,)`, so that I can add a metric without touching the others.
33. As a researcher, I want the 14 legacy metrics ported one to one (`n_nbs`, `Hpers`, `Hpersfoc`, `Hsoc`,
    `Hext`, `tb_tot`, `tb_foc`, `absOm_tot`, `absOm_foc`, `clust`, `bc_foc`, `expI`, `x_focal`, `extr_nonfoc`),
    so that every legacy column is present.
34. As a researcher, I want metrics to receive a `StepContext` (model, t, current pressure), so that energies can
    read the network, ρ and the pressure, and new context fields don't break existing metrics.
35. As a researcher, I want `SocialEnergy` to default to the neighbours' **mean** focal belief, as legacy does,
    with an option for the sum, so that new runs stay comparable with the paper and the eq. 1 value is still
    available (known issue #1).
36. As a researcher, I want the energy metrics vectorised over agents (one `einsum` over the `(N, M, M)` edge
    matrix, one product with the adjacency), so that metrics don't become the new bottleneck after the kernel.
37. As a researcher, I want the response classifier to apply the legacy rules unchanged, including their zero
    and boundary handling (known issue #9), so that response frequencies are comparable.
38. As a researcher, I want the classifier and the aggregator to take their windows from
    `TestbedConfig.windows()`, so that changing the protocol changes both consistently.
39. As a researcher, I want one `MetricsRecorder` per run, so that runs share no metric state.

### Extensibility

40. As an extension developer, I want an `Agent` ABC whose subclasses implement `init_state` and `sweep`, so
    that a new update rule (`CentralFocalAgent`, `BeliefObservingAgent`) is a new class.
41. As an extension developer, I want a `Network` ABC whose subclasses implement only `build`, with
    `neighbours(i)` and `degree()` provided by the base class, so that a new topology (`TwoClusterNetwork`) is a
    few lines.
42. As an extension developer, I want the adjacency to be a dense `(N, N)` matrix that may be weighted, so that
    weighted and later adaptive networks fit the same interface.
43. As an extension developer, I want `beta`, `eps`, `mu`, `lam` and `init_w` to accept a scalar or an `(N,)`
    array, so that heterogeneous populations (idea 4) are a parameter setting, not a subclass.
44. As an extension developer, I want a `fixed` boolean mask that stops chosen agents from updating beliefs or
    edges, so that influencers (idea 2) are a parameter setting.
45. As an extension developer, I want per-agent parameters to appear as per-agent output columns of the same
    name, so that heterogeneous runs stay analysable.
46. As an extension developer, I want `ModelSpec.build` to reject per-agent parameters whose length isn't
    `n_agents`, and agent and network sizes that disagree, so that shape mistakes fail loudly at build time.
47. As an extension developer, I want the focal dimension stored per agent (default 0), so that agents choosing
    their own focal belief (idea 3) need no change to the state layout.
48. As an extension developer, I want `ANBModel.step` to be the one place where an adaptive network's update
    would be called, so that the rewiring ADR has a clear hook.

### Notebook

49. As a new user, I want a notebook that builds a `ModelSpec`, runs one seed and plots the focal belief over
    time and a few metrics, so that I can learn the API quickly.
50. As a new user, I want the notebook to run a small ensemble and show response frequencies, so that I can see
    the ensemble workflow end to end.
51. As an extension developer, I want the notebook to define a toy `Network` subclass inline and run it through
    the same testbed, so that I can see how much code an extension needs.

## Implementation Decisions

### Package and boundaries

- New package `anb/` with modules `agents`, `networks`, `model`, `metrics`, `testbed`, `io`, plus a notebook
  `notebooks/testbed.ipynb`. `ANBmodel.py` and the `analyse_*.py` scripts are **not modified**.
- Interfaces use `abc.ABC` with `@abstractmethod`. Base classes carry shared defaults.
- Install `numba` (≥ 0.56, for `np.random.Generator` arguments in `@njit`) and a test runner (`pytest`) into the
  `anb` conda env. Also install `scikit-learn` and `statsmodels`, which are needed to run the analysis on
  validation output (known issue #5). Add the same packages to `pyproject.toml` so the `uv` route still works.

### `agents` module

- **`AgentState`** (dataclass, data only):
  - `beliefs (N, M)` on the 21-point grid in [−1, 1]
  - `edge_weights (N, M, M)`, symmetric with a zero diagonal
  - `delta_memory (N, τ, M)`
  - `focal (N,)` as int, default 0
  - properties `n_agents` and `n_beliefs`, read from the array shapes (no separately stored M or N)
  - views `edge_vector() -> (N, M(M−1)/2)` (the upper triangle in legacy `w{m}{n}` order) and
    `focal_beliefs() -> (N,)`
  - metrics are **not** part of the state
- **`Agent`** (ABC) is an agent type, not data:
  - owns `n_beliefs` (M), `belief_options` (the grid), `tau` (an int, scalar only), `learning_until` (int or
    None), the per-agent-capable `beta`, `eps`, `mu`, `lam`, `init_w` (scalar or `(N,)`) and `fixed` (bool
    `(N,)` or None)
  - abstract `init_state(n_agents, rng) -> AgentState`
  - concrete `param_arrays(n_agents) -> dict[str, ndarray]` broadcasts every per-agent parameter to `(N,)` for
    the kernel
  - abstract `sweep(state, adjacency, t, ext, rng, rho) -> None` does one whole random-sequential step and mutates
    `state` in place. This is the **only** Numba boundary: one call per time step. There are no per-agent
    Python methods.
- **`ANBAgent(Agent)`**: the paper's rule. `sweep` delegates to one `@njit(cache=True)` kernel that takes plain
  arrays and a `np.random.Generator`. The kernel's semantics, per time step:
  1. Shuffle the agent order.
  2. For each agent in that order, skip it if it is `fixed`.
  3. Compute the social input ρ · Σ over neighbours k (from `adjacency`) of x_foc(k). ρ comes from the model
     and is passed into the kernel. Neighbours updated earlier in this step contribute their new values.
  4. Resample each belief dimension in a fresh random order by Glauber dynamics: probabilities over the grid
     with the agent's β, external pressure `ext` on the focal dimension, and sampling by inverse CDF instead of
     `np.random.choice`. Write the change into `delta_memory`.
  5. Unless `t ≥ learning_until`, update the edges by Hebbian + social + decay (eq. 3). The social term uses the
     neighbours' mean edge matrix, with μ = 0 when the agent has no neighbours. Keep the matrix symmetric.
- Write the kernel as plain Python first, then add `@njit`. The static network is passed as a 0/1 `adjacency`.

### `networks` module

- **`Network`** (ABC): `adjacency (N, N)` (0/1 or weighted), concrete `neighbours(i)` and `degree()` built from
  `adjacency`, abstract classmethod `build(n_agents, rng, **kwargs)`.
- **`ErdosRenyiNetwork`**: G(N, p) with `link_prob`, drawn from the injected `rng`. It matches the legacy
  topology statistically, not edge for edge.
- The network is static in this version. Adaptive networks are a later subclass.

### `model` module

- **`ModelSpec`** (picklable dataclass): `agent`, `network_cls`, `network_kwargs`, `n_agents`, `rho`.
  - `build(rng) -> ANBModel` calls `agent.init_state` and `network_cls.build` with the same `rng` and the same
    `n_agents`, and validates every per-agent parameter length.
  - `params() -> dict` gives the output-column parameters (`link_prob`, `init_w`, `beta`, `rho`, `eps`, `mu`,
    `lam`, `fixedBNat100`, …). Uniform parameters are scalars and per-agent ones are `(N,)` arrays.
    `fixedBNat100` is derived from `learning_until`.
- **`ANBModel`** (concrete, not abstract): `agent`, `network`, `state`, `rho`. `step(t, ext, rng)` calls
  `agent.sweep(state, network.adjacency, t, ext, rng, rho)`. It knows nothing about windows, protocols or metrics.
- Each parameter lives with its owner: β, ε, μ, λ, τ, `init_w` and `learning_until` on `Agent`; `link_prob` on
  `Network`; ρ on `ANBModel`/`ModelSpec`.

### `metrics` module

- **`StepContext`** (dataclass): `model`, `t`, `ext`.
- **`Metric`** (ABC): `name` (the output column) and abstract `compute(ctx) -> (N,)`.
- 14 concrete metrics, one per legacy column:

  | Column | Metric |
  |--------|--------|
  | `n_nbs` | `Degree` |
  | `Hpers` | `PersonalEnergy` |
  | `Hpersfoc` | `FocalPersonalEnergy` |
  | `Hsoc` | `SocialEnergy(aggregate="mean")`, with `"sum"` as the eq. 1 option |
  | `Hext` | `ExternalEnergy` |
  | `tb_tot` | `TriangleBalance` |
  | `tb_foc` | `FocalTriangleBalance` |
  | `absOm_tot` | `MeanAbsEdge` |
  | `absOm_foc` | `FocalMeanAbsEdge` |
  | `clust` | `WeightedClustering` |
  | `bc_foc` | `FocalBetweenness` |
  | `expI` | `ExpectedInfluence` |
  | `x_focal` | `FocalBelief` |
  | `extr_nonfoc` | `NonFocalExtremity` |

- Related metrics share private module-level helpers (for example one triangle pass for both balance metrics).
  The helpers are outside the class hierarchy.
- Metrics run in plain Python, only at tracked steps, and are vectorised over agents: the energies use `einsum`
  on the edge matrices and products with `adjacency`.
- `FocalBetweenness` starts on `igraph`. If profiling shows metrics dominate after the kernel lands, replace the
  N per-step graph builds with a private `@njit` Brandes helper over all agents. That swap is internal and
  doesn't change the `Metric` interface.
- **`MetricsRecorder`**: holds `metrics`. `record(ctx)` snapshots every metric plus beliefs and `edge_vector()`.
  `to_frame()` returns the long frame with `t`, `id`, the metric columns, `0`…`9` and `w01`…`w89`. There is one
  recorder per run.
- **`ResponseClassifier`**: built from `before`/`during`/`after` windows. `classify(frame)` returns one row per
  agent with `persistPos`, `nonpersistPos`, `compliant`, `resilient`, `resistant`, `latecompliant`, using the
  legacy rules unchanged.

### `testbed` module

- **`PressureSchedule`**: `strength`, `events` (inclusive `(start, end)` intervals, the single source of pressure
  timing), `n_events`, `strength_at(t)`. Factory classmethods `single_event(s)` → [(101, 150)] and
  `two_events(s)` → [(101, 150), (201, 250)]. Ramps and pulses are subclasses.
- **`TestbedConfig`** (dataclass): `schedule`, `window_len = 10`, `recovery = 50`, `detail`. For events
  (s₁, e₁) … (sₖ, eₖ) it derives:
  - `T` = eₖ + `recovery`
  - `windows()`: `before` = the `window_len` steps before s₁. For each event j, `during{j}` = the last
    `window_len` steps of the event and `after{j}` = the last `window_len` steps of the `recovery` after it. The
    first event keeps the bare names `during` and `after`. This gives 91–100, 141–150, 191–200, plus 241–250 and
    291–300 for two events.
  - `snapshot_times()`: 0, s₁ − 1, and eⱼ + `recovery` for each j (0, 100, 200, plus 300)
  - `track_times()`: every t if `detail`, otherwise {0, 1} ∪ all window steps ∪ snapshots
- **`WindowAggregator`** (built from `windows()` and `snapshot_times()`): `aggregate(frame, responses)` returns
  the snapshot rows plus one mean row per window, labelled with the window midpoint (95.5, 145.5, 195.5, plus
  245.5 and 295.5), with the response dummies on the `before` (95.5) row. It is its own class, not part of
  `SimulationResult`.
- **`Testbed`**: constructed with `spec`, `config`, `metrics`, and optionally `classifier` and `aggregator`
  (both default to being built from `config.windows()`). `run(seed)` does this:
  1. `rng = default_rng(seed)`
  2. `model = spec.build(rng)`
  3. record at t = 0
  4. for t = 1 … T, call `model.step(t, schedule.strength_at(t), rng)`, then record if t is tracked
  5. classify
  6. in detail mode, attach the dummies to the rows at t = T; in ensemble mode, aggregate
  7. return `SimulationResult(frame, spec.params() + schedule info, seed)`

  `run_ensemble(seeds, n_jobs)` uses `joblib`. Each worker builds its own model from its own seed.
- **`SimulationResult`** (plain container): `frame`, `params`, `seed`. `to_frame()` appends the parameter
  columns, joining per-agent ones on `id`. `save(directory, legacy_names=True)` writes a CSV.

### `io` module

- **`legacy_filename(params, detail) -> str`** reproduces `run_one`'s naming exactly. It reads `lam` and
  `n_events` from `params` rather than taking them as separate arguments. It doesn't add the `_M`/`_N`/`_tau`
  suffixes, because `run_one` doesn't either (known issue #4 stays open). It targets `../simOut_old_model` or
  `../simOut_old_model` according to `detail`.

### Randomness

- No global RNG anywhere in `anb/`. `default_rng(seed)` is created once per run in `Testbed.run` and passed to
  `ModelSpec.build`, `Agent.init_state`, `Network.build` and every `step`/`sweep`.
- The RNG stream deliberately differs from legacy, so a seed does not reproduce old CSVs. This needs supervisor
  sign-off, together with the single `fixedBNat100` meaning.

## Testing Decisions

**What makes a good test here.** Tests exercise behaviour through public interfaces: `Testbed.run`, a
`Metric.compute`, the `TestbedConfig` derived values. They don't assert on private helpers, kernel internals or
the RNG draw order. The model is stochastic, so model-level checks are statistical or structural, never
"this seed gives these numbers". The one exception is reproducibility (same seed, same output), which is
external behaviour. Tests must run with JIT on and with `NUMBA_DISABLE_JIT=1`.

**Seams (agreed):**

1. **Main seam: `Testbed.run(seed)` / `run_ensemble` → `SimulationResult`.**
   - Structural (fast, every run): the column set and order equal legacy's in both ensemble and detail mode;
     row times are {0, 95.5, 100, 145.5, 195.5, 200} in ensemble mode (plus 245.5, 295.5, 300 for two events);
     the dummies appear only on the 95.5 row (or at t = T in detail mode); beliefs stay on the grid; edge
     matrices stay symmetric with a zero diagonal; `fixed` agents' beliefs never change; edges stop changing at
     `learning_until`; same seed gives an identical frame and different seeds give different frames;
     `run_ensemble` matches serial `run`.
   - `legacy_filename` equals `run_one`'s string for every parameter combination used in the paper sweeps and
     in `docu/running-simulations.md`, including the `fixedBNat100`, `lam` and two-event variants.
   - Statistical validation (slow, run on demand, not in the default test run): the 4 paper configurations × 6
     pressures with at least 100 seeds each, new versus `ANBmodel.py`. Compare the response frequencies
     (compliant/resilient/resistant/…) with a two-proportion or χ² test per cell, and the BN metric
     distributions at each window with a two-sample KS test or a comparison of means with CIs. Report the
     failures after multiple-comparison correction. This one run serves both ADRs' validation items.
   - Benchmark (on demand): one run and the full 240-run ensemble, new versus legacy, with the metrics profiled
     separately.
2. **`Metric.compute(StepContext)` on a fixed legacy state.** Build a legacy state array, convert it into an
   `AgentState` plus adjacency, and check that every one of the 14 metrics equals `fill_metrics`/`get_output`
   to floating-point tolerance. That includes the `Hsoc` mean quirk, and `aggregate="sum"` should give the sum.
   Check `ResponseClassifier` against legacy classification on the same frame, including the zero and boundary
   cases. If the `@njit` Brandes helper is built, check it against `igraph` betweenness on random edge
   matrices under this seam.
3. **`TestbedConfig` derived values.** `T`, `windows()`, `snapshot_times()` and `track_times()` equal legacy's
   numbers for `single_event` and `two_events`, in both detail and ensemble mode. `strength_at` is 0 outside
   events and `strength` inside them, with inclusive bounds.

Not separate seams: the sweep kernel and `ErdosRenyiNetwork` are covered by seam 1. A light degree-distribution
check (mean degree ≈ p(N − 1)) can sit with the seam 1 structural tests.

**Prior art.** None. The repo has no test suite, so this adds the first `pytest` setup. Legacy `ANBmodel.py` is
the oracle for seams 2 and 3 and the baseline for the statistical validation. Because legacy defines `detail`
and `track_times` only under `__main__` (known issue #7), test helpers that call legacy functions must inject
those names into the legacy module's namespace without editing the file.

## Out of Scope

- Any change to `ANBmodel.py` or the `analyse_*.py` scripts. Their known issues (#3, #4, #6, #11, #12) stay open.
- Adaptive networks and rewiring rules (`AdaptiveNetwork`), including the decision whether rewiring runs between
  sweeps or inside the kernel. That needs its own ADR.
- `TwoClusterNetwork`, `CentralFocalAgent`, `BeliefObservingAgent`, `RampSchedule`, and how influencers' fixed
  beliefs are initialised. The interfaces allow them; implementing them is later work.
- GPU backends, synchronous updating, and batching runs across a tensor (ADR-001 options B and C).
- Reproducing legacy output bit for bit for a given seed.
- Fixing the `Hsoc` mean/sum mismatch in the default output (kept for comparability).
- A production sweep CLI. Large sweeps stay in a plain script that uses the package. The notebook is a demo.

## Further Notes

- The project works in the `anb` conda env (Python 3.12), as the ADRs and CLAUDE.md say. Install new dependencies
  there (`conda install -n anb numba pytest scikit-learn statsmodels`) and run tests and scripts with
  `conda activate anb`. `pyproject.toml` stays in sync so the `uv` route keeps working.
- Suggested build order, each step independently mergeable:
  1. `TestbedConfig` and `PressureSchedule` plus seam 3
  2. `legacy_filename`
  3. `AgentState`, `Agent`, `ANBAgent` with a plain-Python sweep, `ErdosRenyiNetwork`, `ModelSpec`, `ANBModel`
  4. the metrics, the classifier and seam 2
  5. `Testbed`, `WindowAggregator`, `SimulationResult` and seam 1 structural tests
  6. `@njit` on the sweep and the benchmark
  7. the statistical validation run
  8. the notebook
- The supervisor's sign-off on the RNG change and the single `fixedBNat100` meaning is still open. Collect it
  before using new runs in results, not before implementing.
- Once this lands, update `docu/README.md`, CLAUDE.md's "planned, not yet implemented" note and the ADR action
  item checkboxes.
