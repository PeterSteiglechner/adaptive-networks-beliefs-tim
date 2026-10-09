# Running Simulations

How to produce the simulation output that the analysis scripts read. For what the model does, see
[model.md](model.md).

## Setup

```bash
uv sync
uv run ANBmodel.py
```

Simulations run in parallel on `cpu_count() - 1` cores via `joblib`. The output goes to `simOut/`
(ensemble mode) or `simOut/detailed/` (detailed mode). Both are git-ignored.

> **Missing dependencies for the analysis scripts.** `analyse_BNmetrics.py` imports `scikit-learn`
> and `statsmodels`, which are not in `pyproject.toml`. Add them before running that script:
> `uv add scikit-learn statsmodels`.

## What you configure in `ANBmodel.py`

Everything is configured by editing the `if __name__ == "__main__":` block (plus the module constants for
`M`, `N`, `tau`):

| Variable | Where | Meaning |
|----------|-------|---------|
| `param_combis` | main block | list of `[link_prob, init_w, beta, rho, eps, mu, fixedBNat100]` |
| `pressures` | main block | list of external pressure strengths $s$ |
| `seeds` | main block | list of random seeds |
| `detail` | main block | `True`: store every time step (slow, large files). `False`: store the compact ensemble output |
| `M`, `n_agents`, `tau`, `lam`, `two_external_events` | top of file | fixed constants |

Every combination of `param_combis × pressures × seeds` becomes one simulation and one CSV.
The default `param_combis` contains the four belief-network (BN) configurations from the paper:

| Name in the paper | `init_w` | `eps` | `mu` | `fixedBNat100` |
|-------------------|----------|-------|------|----------------|
| static ($\omega_0=0.2$) | 0.2 | 0.0 | 0.0 | False |
| static ($\omega_0=0.8$) | 0.8 | 0.0 | 0.0 | False |
| adaptive→static         | 0.2 | 1.0 | 0.0 | True  |
| adaptive (baseline)     | 0.2 | 1.0 | 0.0 | False |

> `detail=True` with more than 10 seeds aborts on purpose, because it would take very long.

## Runs needed for each analysis script

The analysis scripts load files by exact filename, so the runs below must exist. The seed counts are
the ones **hard-coded in the scripts**. The paper uses 100 seeds throughout.

| Script | Mode | BN configs | `mu` | pressures $s$ | seeds |
|--------|------|------------|------|---------------|-------|
| `analyse_timeseries.py` | detailed | adaptive | 0 | 4 | **2** |
| `analyse_responseFrequencies.py` | ensemble | all 4 | 0 | 0, 1, 2, 4, 8, 16 | 0–19 |
|  | detailed | adaptive | 0 | 0, 1, 2, 4, 8, 16 | **1** |
| `analyse_BNmetrics.py` (`mu = 0`) | ensemble | adaptive | 0 | 0, 1, 2, 4, 8, 16 | 0–99 |
| `analyse_BNmetrics.py` (`mu > 0`, App. D) | ensemble | adaptive | 0.005 or 0.05 | 4 | 0–99 |
| `analyse_socialAdaptation.py` | ensemble | adaptive | 0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 | 4 | 0–99 |
| `analyse_sensitivity.py` | ensemble | see below | 0 | 4 | 0–99 |

The README table ("single detailed: seeds `[1]`, pressures `[4]`") is incomplete.
`analyse_timeseries.py` reads seed **2**, and `analyse_responseFrequencies.py` reads seed **1**
for **all** six pressure levels. The ensemble runs must also include `s = 0`, because it is the
"no pressure" reference used in Fig. 5 and Fig. A4.

### Recipes

**1. Detailed example runs** (for the time-series and example figures):

```python
detail = True
seeds = [1, 2]
pressures = [0, 1, 2, 4, 8, 16]
param_combis = [[link_prob, 0.2, beta, rho, 1.0, 0.0, False]]   # adaptive only
```

**2. Main ensemble** (Fig. 3–5, Tables 3, A1–A7):

```python
detail = False
seeds = list(range(100))
pressures = [0, 1, 2, 4, 8, 16]
# default param_combis (4 BN configurations)
```

**3. Social adaptation** (Fig. 6, A5–A7). Use the commented-out block in the main section:

```python
detail = False
seeds = list(range(100))
pressures = [4]
param_combis = [[link_prob, 0.2, beta, rho, 1.0, mu, False]
                for mu in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]]
```

The `mu = 0` case is already covered by recipe 2.

### Sensitivity runs

`analyse_sensitivity.py` (Fig. A8) varies one parameter at a time around the baseline (`s = 4`,
100 seeds each). It uses the static-low, static-high and adaptive configs, except for the `eps` and
`init_w` rows, which only use adaptive:

| Experiment | Values | How to set it | Filename the script expects |
|------------|--------|---------------|-----------------------------|
| `base`   | –             | baseline (= recipe 2, `s=4`) | standard |
| `beta`   | 1.5, 6.0      | `beta` in main block | standard (`beta1.50`, `beta6.00`) |
| `p`      | 0.05, 0.2     | `link_prob` in main block | standard (`link_prob0.05`, …) |
| `eps`    | 0.5, 2.0      | `param_combis` | standard (`eps0.50`, `eps2.00`) |
| `init_w` | 0.1, 0.4      | `param_combis` | standard (`init_w0.10`, `init_w0.40`) |
| `M`      | 5, 15         | constant `M` | standard name **+ suffix `_M5.00` / `_M15.00`** |
| `N`      | 50, 200 (with `link_prob = 0.2`, `0.05`) | constant `n_agents` and `link_prob` | standard **+ `_N50.00` / `_N200.00`** |
| `tau`    | 2, 10         | constant `tau` (int) | standard **+ `_tau2.00` / `_tau10.00`** |

`run_one` does **not** add the `_M`, `_N` or `_tau` suffixes. For those experiments, add them to
`fname` in `run_one` yourself, e.g. `fname += f"_M{M:.2f}"`, or rename the files afterwards. The
suffix goes right after `seed{seed}` and before `.csv`.

## Rough cost

One ensemble simulation (N=100, M=10, T=200) is quick. Metrics are only computed at 34 tracked
time steps. Detailed mode computes metrics at all 201 steps and writes about 20k rows per run.
The full set of paper runs (recipes 1–3 + sensitivity) is several thousand simulations, so start with
a few seeds and reduce the `range(...)` in the analysis scripts to match while testing.
