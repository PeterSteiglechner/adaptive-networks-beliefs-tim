"""Comparison statistics for two sets of ensemble runs (legacy vs `anb`, and later old vs new network).

Each run is reduced to one row by `summarise_run`: its response counts and, per analysis window, the run mean of
every metric. The unit of comparison is the run, not the agent, because agents in one run share a network and
influence each other.

- Responses, per cell: χ² statistic on the pooled 2 × K table (6 responses plus `unlabelled`). Its p-value
  comes from permuting the implementation labels of whole runs, so within-run correlation can't inflate it.
- Metrics, per cell × window × metric: two-sample KS test on the run means, plus the difference of means with
  a 95 % Welch CI for reading the size of any deviation.
- Failures: Benjamini–Hochberg q < `ALPHA`, within each family (responses, metrics).

`compare_folders` pairs the CSVs of two output folders by legacy filename and runs the response test per cell.

Plain Python, no Numba.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from anb.metrics import RESPONSE_COLUMNS

ALPHA = 0.05
N_PERMUTATIONS = 10_000
WINDOWS = {"before": 95.5, "during": 145.5, "after": 195.5}
METRICS = ("n_nbs", "Hpers", "Hpersfoc", "Hsoc", "Hext", "tb_tot", "tb_foc", "absOm_tot", "absOm_foc", "clust",
           "bc_foc", "expI", "x_focal", "extr_nonfoc")
RESPONSES = (*RESPONSE_COLUMNS, "unlabelled")


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


def response_table(legacy: pd.DataFrame, new: pd.DataFrame, rng: np.random.Generator,
                   n_permutations: int = N_PERMUTATIONS) -> pd.DataFrame:
    rows = []
    for (config, pressure), old in legacy.groupby(["config", "pressure"], sort=False):
        cell = new[(new.config == config) & (new.pressure == pressure)]
        a, b = old[list(RESPONSES)].to_numpy(), cell[list(RESPONSES)].to_numpy()
        row = {"config": config, "pressure": pressure, **compare_responses(a, b, rng, n_permutations)}
        row["cramers_v"] = float(np.sqrt(row["chi2"] / (a.sum() + b.sum())))  # 2 × K table: min(2, K) − 1 = 1
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


# Folder comparison

_FILENAME = re.compile(
    r"sim_link_prob(?P<link_prob>[\d.]+)_init_w(?P<init_w>[\d.]+)_beta(?P<beta>[\d.]+)_rho(?P<rho>[\d.]+)"
    r"_eps(?P<eps>[\d.]+)_mu(?P<mu>[\d.]+)(?P<fixed>_fixedBNat100)?_ext_strength(?P<ext_strength>[\d.]+)"
    r"_seed(?P<seed>\d+)(?P<events>_2events)?(?:_lambda(?P<lam>[\d.]+))?")

# The paper's BN configurations (init_w, eps, mu, fixedBNat100) at the paper's fixed parameters (legacy `__main__`).
PAPER_CONFIGS = {
    "static0.2": (0.2, 0.0, 0.0, False),
    "static0.8": (0.8, 0.0, 0.0, False),
    "adaptive-static": (0.2, 1.0, 0.0, True),
    "adaptive": (0.2, 1.0, 0.0, False),
}
_BASELINE = dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=0.33, eps=1.0, mu=0.0, fixedBNat100=False, lam=0.0,
                 n_events=1)  # the adaptive paper config; a filename holds rho to two decimals


def parse_filename(name: str) -> tuple[dict, int]:
    """Inverse of `legacy_filename`: a file name (with or without `.csv`) → (cell parameters, seed).

    The cell holds every parameter except the seed, as `legacy_filename` takes them. Values are as rounded in
    the name (`rho = 0.33`).
    """
    match = _FILENAME.fullmatch(name.removesuffix(".csv"))
    if match is None:
        raise ValueError(f"not a legacy ensemble file name: {name}")
    g = match.groupdict()
    ext = float(g["ext_strength"])
    cell = dict(link_prob=float(g["link_prob"]), init_w=float(g["init_w"]), beta=float(g["beta"]),
                rho=float(g["rho"]), eps=float(g["eps"]), mu=float(g["mu"]), fixedBNat100=g["fixed"] is not None,
                ext_strength=int(ext) if ext.is_integer() else ext, lam=float(g["lam"] or 0.0),
                n_events=2 if g["events"] else 1)
    return cell, int(g["seed"])


def cell_label(cell: dict) -> str:
    """A paper config's name, else the parameters that differ from the adaptive config (the pressure doesn't count)."""
    others = ("link_prob", "beta", "rho", "lam", "n_events")
    if all(cell[k] == _BASELINE[k] for k in others):
        for name, values in PAPER_CONFIGS.items():
            if values == tuple(cell[k] for k in ("init_w", "eps", "mu", "fixedBNat100")):
                return name
    return ", ".join(f"{k}={cell[k]}" if k == "fixedBNat100" else f"{k}={cell[k]:g}"
                     for k in _BASELINE if cell[k] != _BASELINE[k])


@dataclass
class Pairing:
    """File stems (no `.csv`) found in both folders, and in one only."""
    matched: list[str]
    only_old: list[str]
    only_new: list[str]


@dataclass
class Comparison:
    """What `compare_folders` found: the pairing, one summary row per matched run and side, and the tables."""
    pairing: Pairing
    old: pd.DataFrame
    new: pd.DataFrame
    pooled_responses: pd.DataFrame = field(default_factory=pd.DataFrame)


def _stems(folder: Path) -> set[str]:
    return {path.stem for path in Path(folder).glob("sim_*.csv")}


def _summaries(folder: Path, stems: list[str]) -> pd.DataFrame:
    rows = []
    for stem in stems:
        cell, seed = parse_filename(stem)
        summary = summarise_run(pd.read_csv(Path(folder) / f"{stem}.csv"))
        rows.append({"config": cell_label(cell), "pressure": cell["ext_strength"], "seed": seed, **summary})
    return pd.DataFrame(rows)


def compare_folders(old: str | Path, new: str | Path, permutation_seed: int = 0,
                    n_permutations: int = N_PERMUTATIONS) -> Comparison:
    """Compare two ensemble output folders. Files are paired by name; those in one folder only are reported.

    The pooled response table has one row per cell (config × pressure), ranked by Cramér's V, with the
    Benjamini–Hochberg `q` of its permutation test alongside. Writes nothing.
    """
    in_old, in_new = _stems(old), _stems(new)
    pairing = Pairing(sorted(in_old & in_new), sorted(in_old - in_new), sorted(in_new - in_old))
    if not pairing.matched:
        raise ValueError(f"no CSV files with the same name in {old} and {new}")
    legacy, anb = _summaries(Path(old), pairing.matched), _summaries(Path(new), pairing.matched)
    rng = np.random.default_rng(permutation_seed)
    pooled = response_table(legacy, anb, rng, n_permutations)
    return Comparison(pairing, legacy, anb, pooled.sort_values("cramers_v", ascending=False, ignore_index=True))
