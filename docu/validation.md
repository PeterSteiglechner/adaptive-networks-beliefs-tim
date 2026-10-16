# Validation of the `anb` package against `ANBmodel.py`

**Date:** 2026-10-16
**Status:** validation passed. The deliberate differences below still need the supervisor's sign-off before new
runs are used in results (ADR-001, ADR-002 action item 8).

This closes the validation and benchmark items of [ADR-001](adr-001-numba-over-gpu.md) (items 4, 5) and
[ADR-002](adr-002-testbed-architecture.md) (item 5), following seam 1 of the [spec](spec-anb-package.md).

## How to rerun

```bash
pytest -m slow -s tests/test_statistical_validation.py   # ~25 min on 12 cores, almost all of it legacy
pytest -m slow -s tests/test_benchmark.py                # ~3 min
```

The validation writes `report.md`, `responses.csv` and `metrics.csv` to `simOut/validation/` (git-ignored). It
caches one summary row per run in `simOut/validation/runs/`. A rerun only adds missing cells, so **delete that
folder after changing the model**. The code is in `tests/validation.py`, and its statistics are tested in
`tests/test_validation_stats.py`.

## Method

- **Cells:** the 4 paper configurations (static ω₀ = 0.2, static ω₀ = 0.8, adaptive→static, adaptive) × 6
  pressures (0, 1, 2, 4, 8, 16), with seeds 0–99 for each cell and implementation. p = 0.1, β = 3, ρ = 1/3,
  N = 100, M = 10, single event. Legacy runs through `run_simulation`, with a fresh module per run so the
  shuffle state can't leak between runs (known issue #2). `fixedBNat100` maps to `learning_until = 101`.
- **Unit of comparison: the run, not the agent.** Agents in one run share a network and pull each other to
  consensus, so agent-level tests are badly overdispersed. In the static ω₀ = 0.2 cells whole runs end on one
  side, and the pooled χ² is ~130 for two identical-looking seeds. Each run is reduced to its response counts
  and to the run mean of every metric in each window (95.5, 145.5, 195.5).
- **Response frequencies:** per cell, the χ² statistic of the pooled 2 × 7 table (6 responses + unlabelled). Its
  p-value comes from 10,000 permutations of whole runs between the two implementations.
- **BN metric distributions:** per cell × window × metric (all 14 columns), a two-sample KS test on the 100 run
  means, plus the difference of means with a 95 % Welch CI to show the size of any deviation.
- **Multiple comparisons:** Benjamini–Hochberg within each family (24 response tests, 1008 metric tests). A test
  fails at q < 0.05.

## Results

| Family | Tests | Failures | Smallest p | Smallest q |
|--------|-------|----------|-----------|-----------|
| Response frequencies | 24 | **0** | 0.080 | 0.43 |
| BN metrics | 1008 | **0** | 0.00074 | 0.37 |

Response shares (legacy → anb, share of all agents), for the configurations in the paper's Fig. 3 at s = 4:

| Configuration | compliant | resilient | resistant |
|---------------|-----------|-----------|-----------|
| static ω₀ = 0.2 | 0.539 → 0.459 | 0.001 → 0.000 | 0.000 → 0.000 |
| static ω₀ = 0.8 | 0.000 → 0.000 | 0.003 → 0.004 | 0.507 → 0.493 |
| adaptive→static | 0.101 → 0.095 | 0.300 → 0.284 | 0.107 → 0.104 |
| adaptive | 0.137 → 0.133 | 0.228 → 0.210 | 0.143 → 0.139 |

The full tables for every cell and every metric test are in `simOut/validation/report.md`.

### Patterns looked at (not failures)

Both follow-ups can be rerun from `tests/` after `validate()`: `gate_check(pressure, learning_until)` and
`positive_share(impl, config)` in `validation.py`.

1. **`bc_foc` in adaptive→static at s = 8 and 16** (during and after windows, diff −0.19 and −0.23, raw
   p = 0.006 and 0.0007). The gap grows with pressure, which points to the `fixedBNat100` gate (difference 2
   below). Legacy still adapts the BN at t = 101, the first pressure step, and that step's large focal-belief
   change rewires the focal edges. Rerunning `anb` with `learning_until = 102` (legacy's effective gate) removes
   the gap: s = 8 gives diff +0.012 [−0.092, 0.115], p = 0.37, and s = 16 gives +0.013 [−0.093, 0.119],
   p = 0.70. So it is the deliberate difference, not a bug.
2. **The share of agents positive before the pressure is higher in `anb` in all 4 configurations** (e.g.
   persistPos 0.46 → 0.54 in static ω₀ = 0.2). The before window is identical across the 6 pressures of a
   configuration, so this is 4 correlated observations, not 24, and run-level signs are close to coin flips.
   With 1000 seeds, `anb` gives 0.50–0.51 in every configuration, and legacy gives 0.501 ± 0.002 in static
   ω₀ = 0.8, the configuration whose 100-seed value (0.489 ± 0.005) was furthest from 0.5. Neither
   implementation has a sign bias.

**Power.** With 100 runs a side, a single-cell difference of about 0.5 run-level SD (like pattern 1) reaches raw
p ≈ 10⁻³ but doesn't survive correction over 1008 tests. A bug that affects many cells, or one cell strongly,
would show up. A small effect confined to one cell needs a targeted rerun with more seeds, as above.

## Benchmark

On 2026-10-16, on 12 cores (11 `joblib` workers, as legacy `__main__` uses). CPU seconds are summed over runs.

| | legacy | anb | speed-up |
|---|---|---|---|
| One run: update loop | 3.19 s | 0.052 s | 62× |
| One run: metrics | 0.32 s | 0.036 s | 9× |
| One run: total | 3.52 s | 0.100 s | 35× |
| 240-run ensemble: wall time | 151 s | 3.3 s | 46× |
| 240-run ensemble: update [CPU s] | 1506 | 20.3 | 74× |
| 240-run ensemble: metrics [CPU s] | 131 | 9.9 | 13× |
| 240-run ensemble: total [CPU s] | 1641 | 35.6 | 46× |

The update loop is inside ADR-001's 50–200× target. The first sweep compiles in about 2.8 s once per machine,
then loads from Numba's disk cache in 0.1 s. The paper's 100-seed sweep (2400 runs) drops from about 25 min to
under 1 min.

**`FocalBetweenness` (the Brandes swap).** After the sweep kernel, profiling one run (adaptive, s = 4) showed the
metrics at 56 ms, more than the 52 ms update loop, and `FocalBetweenness` at 40 ms, 72 % of the metrics. The
per-agent igraph builds were replaced with a private `@njit` Brandes helper over all agents, which ties path
lengths with igraph's relative tolerance (1e-10). It equals igraph on random edge matrices: continuous, on a
coarse grid (many tied shortest paths), sparse with zero edges, and all-equal. It also still equals legacy
`bc_foc` (`tests/test_focal_betweenness.py`). The `Metric` interface is unchanged. Per run it now takes 19 ms,
and the metrics come to 36 ms, 28 % of the run in the ensemble. The rest of the metric time is mostly
`WeightedClustering` (6 ms) and `FocalTriangleBalance` (5 ms), so further metric work isn't worth it for now.

## Deliberate differences from legacy, for sign-off

1. **The RNG stream (known issue #2).** `anb` creates one `np.random.default_rng(seed)` per run and passes it
   everywhere: initial beliefs, the Erdős–Rényi network, the agent and belief update orders, and Glauber sampling
   by inverse CDF instead of `np.random.choice`. Legacy uses the global `np.random`, reseeded twice per run, plus
   module-level lists that are shuffled in place and carry over between runs in one process. As a result:
   - A seed does **not** reproduce an old CSV, and `seed` 7 in `anb` is a different network and trajectory than
     `seed` 7 in legacy. Only the ensemble statistics match (shown above).
   - In exchange, a seed fully determines its run in `anb`, independent of `joblib` scheduling and of which runs
     came before. Legacy can't guarantee that.
   - Old and new CSVs can be pooled as samples from the same model, but not matched seed by seed.
2. **A single `fixedBNat100` meaning (known issue #8).** Legacy has two gates. The module-level flag (`t ≥ 101`,
   always off, so dead code) and the per-run flag (`t > 101`), which zeroes ε and μ but not λ. `anb` has one
   agent parameter, `learning_until`, and skips the whole edge update, λ included, once `t ≥ learning_until`.
   Paper runs use `learning_until = 101`, so the BN is static from the first pressure step on, as the paper
   describes ("adaptive until t = 100 and static thereafter"). Legacy adapts once more at t = 101. Effects:
   - Response frequencies show no detectable difference.
   - `bc_foc` of adaptive→static under strong pressure is lower by about 0.2 (pattern 1 above). Other metrics
     show no detectable difference.
   - λ = 0 in every paper run, so skipping the decay has no effect there.
   - To reproduce legacy's timing exactly, use `learning_until = 102`. Such runs get `fixedBNat100 = False`
     plus a `learning_until` column, and have no legacy filename (save them with `legacy_names=False`), so
     they can't overwrite or be mistaken for paper runs.

Unchanged on purpose, for comparability with the paper: `Hsoc` keeps the neighbours' mean (`SocialEnergy("sum")`
gives the eq. 1 term, known issue #1), the response rules keep their zero and boundary handling (#9), and file
names keep legacy's format without `_M`/`_N`/`_tau` suffixes (#4).

**Sign-off:** ☐ RNG stream (1) ☐ single `fixedBNat100` meaning (2), supervisor, date: ________
