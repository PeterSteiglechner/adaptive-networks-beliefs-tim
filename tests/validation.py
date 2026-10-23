"""Statistical validation of `anb` against legacy `ANBmodel.py` (spec, seam 1).

The paper's 4 BN configurations × 6 pressures are run with both implementations over the same seeds. Each run
is reduced to one row: its response counts and, per analysis window, the run mean of every metric. The unit
of comparison is the run, not the agent, because agents in one run share a network and influence each other.

- Responses, per cell: χ² statistic on the pooled 2 × K table (6 responses plus `unlabelled`). Its p-value
  comes from permuting the implementation labels of whole runs, so within-run correlation can't inflate it.
  The same test on the agents negative before pressure gives the conditional table.
- Lean, per cell: Welch on the run-level share positive before pressure and on the run mean of `x_focal` before
  pressure (a run's lean is almost one coin flip).
- Metrics, per cell × window × metric: two-sample KS test on the run means, plus the difference of means with
  a 95 % Welch CI for reading the size of any deviation.
- Failures: Benjamini–Hochberg q < `ALPHA`, within each family (pooled, conditional, lean, metrics).

The paper configs, the anb testbed of a cell, run summaries, tests and tables all come from `anb.comparison`, as
in the folder comparison, so the two can't disagree on method. Run summaries are cached per (implementation, cell) under `cache_dir`, so a rerun only adds
missing cells. The default output folder `validation_out/` is separate from the sweep CSVs in `simOut_*`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from anb.comparison import (ALPHA, NEGATIVE_RESPONSES, PAPER_CONFIGS, RESPONSES, cell_testbed, compare_values,
                            lean_table, metric_table, paper_cell, response_table, summarise_run)
from anb.testbed import Testbed
from legacy_oracle import load_legacy

CONFIGS = PAPER_CONFIGS
PRESSURES = (0, 1, 2, 4, 8, 16)
OUT_DIR = Path(__file__).resolve().parent.parent / "validation_out"
LEGACY_PARAMS = ("link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100")  # what legacy's `params` holds


# Running and summarising


def legacy_run_params(config: str, pressure: float, seed: int) -> dict:
    cell = paper_cell(config)
    return {**{k: cell[k] for k in LEGACY_PARAMS}, "ext_strength": pressure, "seed": seed}


def anb_testbed(config: str, pressure: float, **agent_changes) -> Testbed:
    """The new-package counterpart of a legacy cell. `fixedBNat100` becomes `learning_until = 101`."""
    return cell_testbed(paper_cell(config), pressure, **agent_changes)


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
            if cached is not None and cached.seed.tolist() == seeds and "lean" in cached:
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


def validate(seeds=range(100), out_dir=OUT_DIR, n_jobs: int = -1, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Run (or load) both implementations, compare them, and write `pooled_responses.csv`,
    `conditional_responses.csv`, `lean.csv`, `metrics.csv` and `report.md` into `out_dir`.
    Returns the four tables (with `q` and `fail` columns) by name."""
    out_dir = Path(out_dir)
    legacy = collect("legacy", seeds, out_dir / "runs", n_jobs)
    new = collect("anb", seeds, out_dir / "runs", n_jobs)
    rng = np.random.default_rng(seed)
    tables = {
        "pooled_responses": response_table(legacy, new, rng),
        "conditional_responses": response_table(legacy, new, rng, categories=NEGATIVE_RESPONSES),
        "lean": lean_table(legacy, new),
        "metrics": metric_table(legacy, new),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "report.md").write_text(report(tables, len(list(seeds))))
    return tables


def _response_section(title: str, table: pd.DataFrame, categories: tuple[str, ...]) -> list[str]:
    lines = [f"## {title} (share of agents, legacy → anb)", "",
             "| config | s | " + " | ".join(categories) + " | p | q |",
             "|---|---|" + "---|" * (len(categories) + 2)]
    for _, r in table.iterrows():
        shares = " | ".join(f"{r[f'{n}_legacy']:.3f} → {r[f'{n}_anb']:.3f}" for n in categories)
        lines.append(f"| {r.config} | {r.pressure} | {shares} | {r.p:.3g} | {r.q:.3g} |")
    return lines + [""]


def report(tables: dict[str, pd.DataFrame], n_seeds: int) -> str:
    pooled, conditional = tables["pooled_responses"], tables["conditional_responses"]
    lean, metrics = tables["lean"], tables["metrics"]
    lines = [
        "# Statistical validation: `anb` vs legacy `ANBmodel.py`",
        "",
        f"{len(CONFIGS)} configurations × {len(PRESSURES)} pressures, {n_seeds} seeds per cell and implementation.",
        f"Failures: Benjamini–Hochberg q < {ALPHA} within each family.",
        "",
        f"- Response frequencies (χ², run-level permutation): {int(pooled.fail.sum())} of {len(pooled)} "
        f"cells fail; smallest p = {pooled.p.min():.3g}",
        f"- Response frequencies of agents negative before pressure: {int(conditional.fail.sum())} of "
        f"{len(conditional)} cells fail; smallest p = {conditional.p.min():.3g}",
        f"- Lean (share positive and `x_focal` before pressure, Welch on runs): {int(lean.fail.sum())} of "
        f"{len(lean)} tests fail; smallest p = {lean.p.min():.3g}",
        f"- Metric distributions (KS on run means): {int(metrics.fail.sum())} of {len(metrics)} tests fail; "
        f"smallest p = {metrics.p.min():.3g}",
        "",
    ]
    lines += _response_section("Response frequencies", pooled, RESPONSES)
    lines += _response_section("Response frequencies of agents negative before pressure", conditional,
                               NEGATIVE_RESPONSES)
    lines += ["## Lean: share positive and `x_focal` before pressure (legacy → anb)", "",
              "| config | s | quantity | legacy → anb | diff [95 % CI] | gap (SE) | SMD | p | q |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in lean.iterrows():
        lines.append(f"| {r.config} | {r.pressure} | {r.quantity} | {r.mean_legacy:.3f} → {r.mean_anb:.3f} | {r['diff']:.3g} "
                     f"[{r.ci_low:.3g}, {r.ci_high:.3g}] | {r.gap_se:.2f} | {r.smd:.2f} | {r.p:.3g} | {r.q:.3g} |")
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


def gate_check(pressure: float, learning_until: int, cache_dir=OUT_DIR / "runs", seeds=range(100),
               n_jobs: int = -1) -> dict:
    """`bc_foc` (during window) of adaptive→static: cached legacy runs versus `anb` with `learning_until`.

    101 is the package's gate (static from the first pressure step), 102 is legacy's effective one (t > 101).
    """
    def run(seed):
        testbed = anb_testbed("adaptive-static", pressure, learning_until=learning_until)
        return summarise_run(testbed.run(seed).frame)["bc_foc@during"]

    legacy = pd.read_csv(Path(cache_dir) / f"legacy_adaptive-static_s{pressure}.csv")
    assert legacy.seed.tolist() == list(seeds), "run validate() first, with the same seeds"
    new = Parallel(n_jobs=n_jobs)(delayed(run)(seed) for seed in seeds)
    return compare_values(legacy["bc_foc@during"], new)


def positive_share(impl: str, config: str, seeds=range(1000), n_jobs: int = -1) -> tuple[float, float]:
    """Mean and standard error over runs of the share of agents whose focal belief is ≥ 0 before the pressure."""
    rows = Parallel(n_jobs=n_jobs)(delayed(RUNNERS[impl])(config, 0, seed) for seed in seeds)
    shares = np.array([r["lean"] for r in rows])
    return float(shares.mean()), float(shares.std(ddof=1) / np.sqrt(len(shares)))
