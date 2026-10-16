# ADR-001: Speed up the simulation with Numba, not a GPU

**Status:** Proposed, amended 2026-10-14 to fit [ADR-002](adr-002-testbed-architecture.md) (see [Amendments](#amendments))
**Date:** 2026-10-09
**Deciders:** Tim Krüger; supervisor sign-off needed because the RNG stream changes (see Consequences)

## Context

`ANBmodel.py` takes about **4.4 s per simulation** (N = 100 agents, M = 10 beliefs, T = 200, `anb` conda env,
12-core machine). The current ensemble of 240 runs (4 configurations × 6 pressures × 10 seeds) takes about 1.5 min
on 11 `joblib` workers. The paper's 100 seeds would take about 16 min. Sensitivity sweeps and the planned
extensions multiply that further.

A profile of one run shows where the time goes:

| Function | Calls per run | Share of runtime |
|----------|---------------|------------------|
| `update_belief` → `glauber_fast` | 200,000 (T × N × M) | ~80 % |
| `fill_metrics` (energies, balance, clustering, betweenness) | 32 | ~9 % |
| `update_edge_weights` | 20,000 | ~4 % |

Each `glauber_fast` call takes about 17 µs on arrays of 10 to 21 elements. That is almost all Python and NumPy
call overhead, not arithmetic. **The model is limited by interpreter overhead, not by compute.**

Two properties of the model constrain any speed-up:

1. **Random-sequential updating.** In each time step, agents update one after another in random order, and each
   agent sees neighbours already updated in the same step ([ANBmodel.py:439-447](../ANBmodel.py#L439-L447)).
   Within an agent, the M beliefs are resampled one by one, and each sees the dimensions updated before it
   ([ANBmodel.py:176-194](../ANBmodel.py#L176-L194)). The belief network is fully connected, so no two belief
   dimensions are independent.
2. **Small state.** One time step touches a 100 × 55 array (10 beliefs + 45 edge weights per agent).

The planned extensions ([projectIdea.md](../projectIdea.md)) add an **adaptive, weighted social network**:
agents rewire ties by comparing focal beliefs. That replaces the fixed neighbour lists `nb_list` with a weighted
N × N matrix `W` that changes over time, and adds branching rewiring rules (drop a tie if disagreement exceeds a
threshold, pick a new partner, and so on).

The question is how to make simulations fast enough for large ensembles and parameter sweeps, now and for the
dynamic-network extension.

## Decision

**JIT-compile one time step (the random-sequential agent sweep and the belief loop inside it) with Numba on the
CPU, keep the random-sequential semantics unchanged, and keep `joblib` for parallelism across runs.** The time loop
stays in Python, in ADR-002's `Testbed`. Do not move to a GPU now.

While rewriting, restructure the state so that a GPU port stays cheap if it is needed later (see Action Items).

## Options Considered

### Option A: Numba JIT on the CPU (chosen)

Compile the agent loop and belief loop of one time step into an `@njit` sweep kernel, called once per step from a
Python time loop. Sample from the Glauber probabilities by inverse CDF instead of `np.random.choice`.

The time loop stays outside the kernel for two reasons. It costs only T = 200 Python calls per run, which is
negligible next to the 200,000 `glauber_fast` calls removed. And metrics (`igraph` betweenness, item 3) and the
pressure schedule run in Python at the tracked time steps anyway.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low to medium. Same loop structure as today, written with explicit indices |
| Model fidelity | Exact. Same update order and equations |
| Expected speed-up | 50–200× on the update loop (typical for overhead-bound NumPy code; to be measured) |
| Scalability | Good up to N ≈ 1,000 and a few thousand runs; `joblib` scales across cores |
| Fit for dynamic networks | Good. Rewiring is branchy, per-agent logic, which Numba handles well |
| Hardware | Any CPU. Runs on laptops and clusters without CUDA |
| Familiarity | Plain Python and NumPy syntax |

**Pros**
- Keeps the paper's dynamics exactly, so results stay comparable with the published ones.
- Removes the bottleneck the profile actually shows.
- Simple to debug: the loop reads like the model description.
- No new hardware or driver dependency.

**Cons**
- Numba supports only a subset of Python and NumPy. Metrics that use `igraph` (betweenness) stay outside the
  JIT function.
- First call pays a compile cost of a few seconds (mitigated with `cache=True`).
- A different RNG stream, so old CSVs are not reproduced bit for bit for a given seed.

### Option B: GPU with agents as matrix columns (synchronous update)

Store each agent's beliefs and weights as a column and update all agents (and/or all belief dimensions) in one
matrix operation per step, in PyTorch, JAX or CuPy.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Model fidelity | **Changes the model.** Sequential becomes synchronous updating |
| Expected speed-up | Little or none at N = 100: a step is ~5,500 numbers, so kernel-launch overhead (~5–10 µs) dominates |
| Scalability | Good only for large N (thousands and more) |
| Fit for dynamic networks | Poor for event-based rewiring with branching and varying degrees |
| Hardware | Needs an NVIDIA GPU (available locally: RTX 4060 Ti; not guaranteed on a cluster) |

**Pros**
- Very fast if N grows to many thousands with dense weights.

**Cons**
- Synchronous Glauber dynamics behave differently. With β = 3 and strong weights, they can oscillate where
  sequential updating does not. Results would no longer be comparable with the paper.
- No speed-up at the current size.
- Adds a CUDA dependency.

### Option C: GPU batched across simulations (sequential semantics kept)

Hold all runs in a tensor of shape `(n_sims, N, M)`. Keep the sequential loops over agents and beliefs, and
vectorise each step over the independent runs (JAX `lax.scan` or PyTorch).

| Dimension | Assessment |
|-----------|------------|
| Complexity | High. Per-run random orders, varying degrees and rewiring must be expressed as masked tensor ops |
| Model fidelity | Exact |
| Expected speed-up | Worth it only for very large sweeps (thousands of runs at once) |
| Scalability | Excellent across runs; still sequential in N × M within a step |
| Fit for dynamic networks | Medium. Every run rewires differently, which forces masking and padding |
| Hardware | Needs an NVIDIA GPU |

**Pros**
- Correct semantics and real GPU use, because the batch dimension supplies the parallel work.

**Cons**
- Hardest to write and debug.
- Still runs N × M × T sequential steps per batch, so its advantage over Numba + `joblib` only shows at very large
  ensemble sizes.
- Not needed at the planned scale.

## Trade-off Analysis

The deciding factor is **model fidelity**. This project extends a published model, and its results must stay
comparable with the paper. Option B gives up the random-sequential updating, which is part of the model, for a
speed-up it would not deliver at N = 100. That rules it out.

Between A and C, both keep the semantics. Option C only pays off when thousands of runs execute at once. At the
planned scale, a few hundred to a few thousand runs with N between 100 and 1,000, Numba on 11 cores should bring
a full ensemble down from minutes to seconds. That makes the extra complexity of C unjustified.

The dynamic-network extension reinforces this. Rewiring is event-based and full of conditions, the kind of code
GPUs handle badly and Numba handles well. A dense `W` matrix adds O(N) work per agent update, which is small
at N ≤ 1,000.

## Consequences

**Easier**
- Large ensembles (100 seeds) and fine parameter sweeps become cheap.
- Adding the dynamic social network: plain loops over `W`, inside the sweep kernel or in a separate per-step
  kernel (decided in the rewiring ADR, item 7).
- Fixing [known issues](known-issues.md) #2 (shared shuffle state) and #7 (globals under `__main__`) as part of
  the rewrite, by passing the RNG and parameters explicitly.

**Harder**
- Two code paths during the transition (old reference model and new JIT model) until the new one is validated.
- Code inside the JIT function has to stay within Numba's supported subset.

**To revisit**
- Move to Option C if a study needs N ≳ 5,000 with dense weights, or tens of thousands of runs per sweep.
- Choose synchronous updating only if a model extension calls for it on modelling grounds (for example, "all
  agents revise their ties at the end of a step"). Then a GPU becomes natural, and that should be its own ADR.
- The RNG change means a given seed no longer reproduces old CSVs. Validation has to be statistical (see below).

## Action Items

1. [x] Restructure state into plain arrays: beliefs `(N, M)`, BN weights `(N, M, M)` symmetric (ADR-002 Q18),
   social network `W` `(N, N)` dense and weighted, past belief changes `(N, τ, M)`. Keep metrics out of the state
   array.
2. [x] Write the sweep kernel as an `@njit(cache=True)` function with explicit parameter arrays and a seeded
   `np.random.Generator`, keeping the random-sequential agent and belief order. It is `ANBAgent.sweep` in ADR-002
   (Q22). Represent the current static network as a fixed 0/1 `W`. Check that the installed Numba version
   accepts a `Generator` argument (supported since Numba 0.56).
3. [x] Keep `fill_metrics` and `get_output` in plain Python, called only at `track_times`.
4. [x] Benchmark against the current model (same parameters, one run, and the full 240-run ensemble), with
   `pytest -m slow -s tests/test_benchmark.py`. On 2026-10-16 the update loop took 3.2 s in legacy and
   0.05 s in `anb` (62×), and the 240-run ensemble took 151 s and 3.3 s of wall time (46×). The metrics
   dominated after the kernel, so `FocalBetweenness` got the `@njit` Brandes helper (amendment 2026-10-16).
   Numbers in [validation.md](validation.md#benchmark).
5. [x] Validate statistically: compare response frequencies (compliant / resilient / resistant) and BN metrics
   between old and new models over ≥ 100 seeds per configuration. This is the same validation as ADR-002 action
   item 5, run once. Passed on 2026-10-16, see [validation.md](validation.md).
6. [x] Add `numba` to the `anb` environment and to `pyproject.toml`.
7. [ ] Implement the adaptive social network on top of `W` (separate ADR for the rewiring rules). That ADR also
   decides whether ties are revised once per step, in a kernel between sweeps (`AdaptiveNetwork.update`), or
   during an agent's own update, inside the sweep kernel.

## Amendments

**2026-10-14, after ADR-002.** ADR-002 put the time loop in a Python `Testbed` and, in its first version, called
Python `Agent` methods once per agent. That kept T × N = 20,000 Python-level calls per run and would have capped
the speed-up this ADR is for. Both ADRs now agree on one sweep kernel per time step:

- Decision and Option A: one `@njit` kernel per time step instead of one function for the whole run.
- Action item 1: BN weights as a symmetric `(N, M, M)` matrix, following ADR-002 Q18.
- Action item 2: the kernel is `ANBAgent.sweep` and takes a `np.random.Generator`.
- Action item 7: the rewiring ADR decides where rewiring runs.

The validation threshold (≥ 100 seeds) is unchanged. ADR-002 now uses it too.

**2026-10-16, metrics after the sweep kernel.** Once the sweep kernel is in place, it removes most of the ~84 %
spent in `update_belief` and `update_edge_weights`. The metrics (~9 % now, about 0.4 s per run) are then likely
the largest cost. Action item 3 still applies, with two refinements:

- Action item 3: metrics stay outside the sweep kernel, but are vectorised over agents. `PersonalEnergy` and
  `FocalPersonalEnergy` become one `einsum` on the `(N, M, M)` edge matrix, and `SocialEnergy` one product with
  `W`, which replaces the per-agent list comprehensions in `get_energies`
  ([ANBmodel.py:199-238](../ANBmodel.py#L199-L238)). This needs no Numba.
- Action item 4: the benchmark also profiles the metrics separately. If they dominate, replace the per-agent
  `igraph` call in `FocalBetweenness` ([ANBmodel.py:280-286](../ANBmodel.py#L280-L286)), which builds N graphs
  per tracked step, with a private `@njit` Brandes helper over all agents. On M = 10 nodes it is short. This
  revises the Cons entry that betweenness stays outside the JIT. The helper is internal to the metric, so the
  `Metric` interface (ADR-002 Q10, Q20) does not change. Validate it against `igraph` on random edge matrices.
