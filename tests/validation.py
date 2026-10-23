"""Statistical validation of `anb` against legacy `ANBmodel.py` (spec, seam 1).

The paper's 4 BN configurations × 6 pressures are run with both implementations over the same seeds. Each run
is reduced to one row: its response counts and, per analysis window, the run mean of every metric. The unit
of comparison is the run, not the agent, because agents in one run share a network and influence each other.

- Responses, per cell: χ² statistic on the pooled 2 × K table (6 responses plus `unlabelled`). Its p-value
  comes from permuting the implementation labels of whole runs, so within-run correlation can't inflate it.
- Metrics, per cell × window × metric: two-sample KS test on the run means, plus the difference of means with
  a 95 % Welch CI for reading the size of any deviation.
- Failures: Benjamini–Hochberg q < `ALPHA`, within each family (responses, metrics).

Run summaries are cached per (implementation, cell) under `cache_dir`, so a rerun only adds missing cells.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from anb.agents import ANBAgent
from anb.comparison import ALPHA, metric_table, response_table, summarise_run
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig
from legacy_oracle import load_legacy

CONFIGS = {  # the paper's BN configurations: init_w, eps, mu, fixedBNat100 (legacy `__main__`)
    "static0.2": (0.2, 0.0, 0.0, False),
    "static0.8": (0.8, 0.0, 0.0, False),
    "adaptive-static": (0.2, 1.0, 0.0, True),
    "adaptive": (0.2, 1.0, 0.0, False),
}
PRESSURES = (0, 1, 2, 4, 8, 16)
FIXED = dict(link_prob=0.1, beta=3.0, rho=1 / 3)


# Running and summarising


def legacy_run_params(config: str, pressure: float, seed: int) -> dict:
    init_w, eps, mu, fixed = CONFIGS[config]
    return dict(FIXED, init_w=init_w, eps=eps, mu=mu, fixedBNat100=fixed, ext_strength=pressure, seed=seed)


def anb_testbed(config: str, pressure: float) -> Testbed:
    """The new-package counterpart of a legacy cell. `fixedBNat100` becomes `learning_until = 101`."""
    init_w, eps, mu, fixed = CONFIGS[config]
    agent = ANBAgent(beta=FIXED["beta"], eps=eps, mu=mu, init_w=init_w, learning_until=101 if fixed else None)
    spec = ModelSpec(agent, ErdosRenyiNetwork, {"link_prob": FIXED["link_prob"]}, n_agents=100, rho=FIXED["rho"])
    return Testbed(spec, TestbedConfig(PressureSchedule.single_event(pressure)))


def run_legacy(config: str, pressure: float, seed: int) -> dict:
    # a fresh legacy module per run, so the shuffled module-level lists don't carry over (known issue #2)
    return summarise_run(load_legacy().run_simulation(legacy_run_params(config, pressure, seed)))


def run_anb(config: str, pressure: float, seed: int) -> dict:
    return summarise_run(anb_testbed(config, pressure).run(seed).frame)


RUNNERS = {"legacy": run_legacy, "anb": run_anb}


def collect(impl: str, seeds, cache_dir: Path, n_jobs: int = -1) -> pd.DataFrame:
    """Run summaries for every cell and seed of `impl`, one row per run, reusing cached cells."""
    seeds = list(seeds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cells, missing = {}, []
    for config in CONFIGS:
        for pressure in PRESSURES:
            path = cache_dir / f"{impl}_{config}_s{pressure}.csv"
            cached = pd.read_csv(path) if path.exists() else None
            if cached is not None and cached.seed.tolist() == seeds:
                cells[config, pressure] = cached
            else:
                missing.append((config, pressure))
    jobs = [(config, pressure, seed) for config, pressure in missing for seed in seeds]
    rows = Parallel(n_jobs=n_jobs)(delayed(RUNNERS[impl])(*job) for job in jobs)
    for k, (config, pressure) in enumerate(missing):
        cell = pd.DataFrame(rows[k * len(seeds):(k + 1) * len(seeds)])
        cell.insert(0, "seed", seeds)
        path = cache_dir / f"{impl}_{config}_s{pressure}.csv"
        cell.to_csv(path, index=False)
        cells[config, pressure] = pd.read_csv(path)  # as cached cells are: CSV rounds floats by up to an ulp
    return pd.concat(
        [cell.assign(config=config, pressure=pressure) for (config, pressure), cell in cells.items()],
        ignore_index=True,
    )


# Report


def validate(seeds=range(100), out_dir=Path("simOut_old_model/validation"), n_jobs: int = -1, seed: int = 0):
    """Run (or load) both implementations, compare them, and write `responses.csv`, `metrics.csv` and
    `report.md` into `out_dir`. Returns `(responses, metrics)` tables with `q` and `fail` columns."""
    out_dir = Path(out_dir)
    legacy = collect("legacy", seeds, out_dir / "runs", n_jobs)
    new = collect("anb", seeds, out_dir / "runs", n_jobs)
    responses = response_table(legacy, new, np.random.default_rng(seed))
    metrics = metric_table(legacy, new)
    responses.to_csv(out_dir / "responses.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "report.md").write_text(report(responses, metrics, len(list(seeds))))
    return responses, metrics


def report(responses: pd.DataFrame, metrics: pd.DataFrame, n_seeds: int) -> str:
    lines = [
        "# Statistical validation: `anb` vs legacy `ANBmodel.py`",
        "",
        f"{len(CONFIGS)} configurations × {len(PRESSURES)} pressures, {n_seeds} seeds per cell and implementation.",
        f"Failures: Benjamini–Hochberg q < {ALPHA} within each family.",
        "",
        f"- Response frequencies (χ², run-level permutation): {int(responses.fail.sum())} of {len(responses)} "
        f"cells fail; smallest p = {responses.p.min():.3g}",
        f"- Metric distributions (KS on run means): {int(metrics.fail.sum())} of {len(metrics)} tests fail; "
        f"smallest p = {metrics.p.min():.3g}",
        "",
        "## Response frequencies (share of agents, legacy → anb)",
        "",
        "| config | s | " + " | ".join(RESPONSES) + " | p | q |",
        "|---|---|" + "---|" * (len(RESPONSES) + 2),
    ]
    for _, r in responses.iterrows():
        shares = " | ".join(f"{r[f'{n}_legacy']:.3f} → {r[f'{n}_anb']:.3f}" for n in RESPONSES)
        lines.append(f"| {r.config} | {r.pressure} | {shares} | {r.p:.3g} | {r.q:.3g} |")
    lines += ["", "## Metric failures", ""]
    failed = metrics[metrics.fail]
    if failed.empty:
        lines.append("None.")
    else:
        lines += ["| config | s | window | metric | legacy mean | anb mean | diff [95 % CI] | KS | p | q |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in failed.iterrows():
            lines.append(f"| {r.config} | {r.pressure} | {r.window} | {r.metric} | {r.mean_legacy:.4g} | "
                         f"{r.mean_anb:.4g} | {r['diff']:.3g} [{r.ci_low:.3g}, {r.ci_high:.3g}] | {r.ks:.3f} | "
                         f"{r.p:.3g} | {r.q:.3g} |")
    lines += ["", "## All metric tests, smallest p first", "",
              "| config | s | window | metric | diff [95 % CI] | KS | p | q |", "|---|---|---|---|---|---|---|---|"]
    for _, r in metrics.sort_values("p", kind="stable").iterrows():
        lines.append(f"| {r.config} | {r.pressure} | {r.window} | {r.metric} | {r['diff']:.3g} "
                     f"[{r.ci_low:.3g}, {r.ci_high:.3g}] | {r.ks:.3f} | {r.p:.3g} | {r.q:.3g} |")
    return "\n".join(lines) + "\n"


# Follow-ups to patterns in the report (docu/validation.md, "Patterns looked at")


def gate_check(pressure: float, learning_until: int, cache_dir=Path("simOut_old_model/validation/runs"), seeds=range(100),
               n_jobs: int = -1) -> dict:
    """`bc_foc` (during window) of adaptive→static: cached legacy runs versus `anb` with `learning_until`.

    101 is the package's gate (static from the first pressure step), 102 is legacy's effective one (t > 101).
    """
    def run(seed):
        testbed = anb_testbed("adaptive-static", pressure)
        testbed.spec.agent.learning_until = learning_until
        return summarise_run(testbed.run(seed).frame)["bc_foc@during"]

    legacy = pd.read_csv(Path(cache_dir) / f"legacy_adaptive-static_s{pressure}.csv")
    assert legacy.seed.tolist() == list(seeds), "run validate() first, with the same seeds"
    new = Parallel(n_jobs=n_jobs)(delayed(run)(seed) for seed in seeds)
    return compare_values(legacy["bc_foc@during"], new)


def positive_share(impl: str, config: str, seeds=range(1000), n_jobs: int = -1) -> tuple[float, float]:
    """Mean and standard error over runs of the share of agents whose focal belief is ≥ 0 before the pressure."""
    rows = Parallel(n_jobs=n_jobs)(delayed(RUNNERS[impl])(config, 0, seed) for seed in seeds)
    shares = np.array([(r["persistPos"] + r["nonpersistPos"]) / 100 for r in rows])
    return float(shares.mean()), float(shares.std(ddof=1) / np.sqrt(len(shares)))
