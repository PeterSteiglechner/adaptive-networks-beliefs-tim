"""Comparison statistics for two sets of ensemble runs (legacy vs `anb`, and later old vs new network).

Each run is reduced to one row by `summarise_run`: its response counts and, per analysis window, the run mean of
every metric. The unit of comparison is the run, not the agent, because agents in one run share a network and
influence each other.

- Responses, per cell: χ² statistic on the pooled 2 × K table (6 responses plus `unlabelled`). Its p-value
  comes from permuting the implementation labels of whole runs, so within-run correlation can't inflate it.
- Metrics, per cell × window × metric: two-sample KS test on the run means, plus the difference of means with
  a 95 % Welch CI for reading the size of any deviation.
- Failures: Benjamini–Hochberg q < `ALPHA`, within each family (responses, metrics).

Plain Python, no Numba.
"""

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
