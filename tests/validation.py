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
from scipy import stats
from statsmodels.stats.multitest import multipletests

from anb.agents import ANBAgent
from anb.metrics import RESPONSE_COLUMNS
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig
from legacy_oracle import load_legacy

ALPHA = 0.05
N_PERMUTATIONS = 10_000
CONFIGS = {  # the paper's BN configurations: init_w, eps, mu, fixedBNat100 (legacy `__main__`)
    "static0.2": (0.2, 0.0, 0.0, False),
    "static0.8": (0.8, 0.0, 0.0, False),
    "adaptive-static": (0.2, 1.0, 0.0, True),
    "adaptive": (0.2, 1.0, 0.0, False),
}
PRESSURES = (0, 1, 2, 4, 8, 16)
FIXED = dict(link_prob=0.1, beta=3.0, rho=1 / 3)
WINDOWS = {"before": 95.5, "during": 145.5, "after": 195.5}
METRICS = ("n_nbs", "Hpers", "Hpersfoc", "Hsoc", "Hext", "tb_tot", "tb_foc", "absOm_tot", "absOm_foc", "clust",
           "bc_foc", "expI", "x_focal", "extr_nonfoc")
RESPONSES = (*RESPONSE_COLUMNS, "unlabelled")


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


def summarise_run(frame: pd.DataFrame) -> dict:
    """One ensemble-mode frame → response counts (from the 95.5 row) and per-window run means of each metric."""
    before = frame[frame.t == WINDOWS["before"]]
    labels = before[list(RESPONSE_COLUMNS)].astype(float).fillna(0).astype(bool)
    out = {name: int(labels[name].sum()) for name in RESPONSE_COLUMNS}
    out["unlabelled"] = int((~labels.any(axis=1)).sum())
    for window, t in WINDOWS.items():
        rows = frame[frame.t == t]
        for metric in METRICS:
            out[f"{metric}@{window}"] = float(np.nanmean(rows[metric].to_numpy(dtype=float)))
    return out


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
        cell.to_csv(cache_dir / f"{impl}_{config}_s{pressure}.csv", index=False)
        cells[config, pressure] = cell
    return pd.concat(
        [cell.assign(config=config, pressure=pressure) for (config, pressure), cell in cells.items()],
        ignore_index=True,
    )


# Statistics


def _chi2(counts_a: np.ndarray, counts_b: np.ndarray) -> np.ndarray:
    """χ² of the 2 × K tables [a; b]; `counts_*` are `(..., K)` with categories empty in both already dropped."""
    table = np.stack([counts_a, counts_b], axis=-2)  # (..., 2, K)
    expected = table.sum(axis=-1, keepdims=True) * table.sum(axis=-2, keepdims=True) / table.sum(
        axis=(-2, -1), keepdims=True)
    return ((table - expected) ** 2 / expected).sum(axis=(-2, -1))


def compare_responses(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                      n_permutations: int = N_PERMUTATIONS) -> dict:
    """χ² of pooled response counts with a run-level permutation p-value.

    `a (R_a, K)` and `b (R_b, K)` are per-run counts. The p-value is the share of label permutations (whole
    runs swapped between the groups) whose χ² is at least the observed one.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    used = (a.sum(axis=0) + b.sum(axis=0)) > 0
    a, b = a[:, used], b[:, used]
    if used.sum() <= 1:  # every agent in one category on both sides
        return {"chi2": 0.0, "p": 1.0}
    observed = float(_chi2(a.sum(axis=0), b.sum(axis=0)))
    runs = np.concatenate([a, b])
    in_a = np.zeros((n_permutations, len(runs)), dtype=bool)
    in_a[:, :len(a)] = True
    in_a = rng.permuted(in_a, axis=1)
    totals_a = in_a.astype(float) @ runs
    permuted = _chi2(totals_a, runs.sum(axis=0) - totals_a)
    p = (1 + np.count_nonzero(permuted >= observed - 1e-9)) / (n_permutations + 1)
    return {"chi2": observed, "p": float(p)}


def compare_values(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sample KS on run-level values, plus mean(b) − mean(a) with a 95 % Welch CI. NaNs are dropped."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    diff = b.mean() - a.mean()
    var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(var_a + var_b)
    if se == 0:
        half = 0.0
    else:
        dof = se**4 / (var_a**2 / (len(a) - 1) + var_b**2 / (len(b) - 1))
        half = stats.t.ppf(0.975, dof) * se
    if np.ptp(np.concatenate([a, b])) == 0:
        ks, p = 0.0, 1.0  # both constant and equal (e.g. edge metrics of a static BN)
    else:
        result = stats.ks_2samp(a, b)
        ks, p = float(result.statistic), float(result.pvalue)
    return {"mean_legacy": a.mean(), "mean_anb": b.mean(), "diff": diff, "ci_low": diff - half,
            "ci_high": diff + half, "ks": ks, "p": p}


def correct(table: pd.DataFrame) -> pd.DataFrame:
    """Add Benjamini–Hochberg `q` and `fail` (q < `ALPHA`) over all rows of one test family."""
    out = table.copy()
    out["q"] = multipletests(out.p, method="fdr_bh")[1] if len(out) else []
    out["fail"] = out.q < ALPHA
    return out


def response_table(legacy: pd.DataFrame, new: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (config, pressure), old in legacy.groupby(["config", "pressure"], sort=False):
        cell = new[(new.config == config) & (new.pressure == pressure)]
        a, b = old[list(RESPONSES)].to_numpy(), cell[list(RESPONSES)].to_numpy()
        row = {"config": config, "pressure": pressure, **compare_responses(a, b, rng)}
        for k, name in enumerate(RESPONSES):
            row[f"{name}_legacy"] = a[:, k].sum() / a.sum()
            row[f"{name}_anb"] = b[:, k].sum() / b.sum()
        rows.append(row)
    return correct(pd.DataFrame(rows))


def metric_table(legacy: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, pressure), old in legacy.groupby(["config", "pressure"], sort=False):
        cell = new[(new.config == config) & (new.pressure == pressure)]
        for window in WINDOWS:
            for metric in METRICS:
                column = f"{metric}@{window}"
                rows.append({"config": config, "pressure": pressure, "window": window, "metric": metric,
                             **compare_values(old[column], cell[column])})
    return correct(pd.DataFrame(rows))


# Report


def validate(seeds=range(100), out_dir=Path("simOut/validation"), n_jobs: int = -1, seed: int = 0):
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


def gate_check(pressure: float, learning_until: int, cache_dir=Path("simOut/validation/runs"), seeds=range(100),
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
