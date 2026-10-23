"""Comparison statistics for two sets of ensemble runs (legacy vs `anb`, and later old vs new network).

Each run is reduced to one row by `summarise_run`: its response counts and, per analysis window, the run mean of
every metric. The unit of comparison is the run, not the agent, because agents in one run share a network and
influence each other.

- Responses, per cell: χ² statistic on the pooled 2 × K table (6 responses plus `unlabelled`). Its p-value
  comes from permuting the implementation labels of whole runs, so within-run correlation can't inflate it.
- Metrics, per cell × window × metric: two-sample KS test on the run means, plus the difference of means with
  a 95 % Welch CI for reading the size of any deviation.
- Conditional responses: the same χ² test on the four classes of agents negative before pressure.
- Lean, per cell: Welch on the run-level share positive before pressure (no KS, the distribution is bimodal).
- Failures: Benjamini–Hochberg q < `ALPHA`, within each family (pooled responses, conditional responses, lean,
  metrics). Tables are ranked by effect size: Cramér's V for responses, the standardised mean difference for the
  rest.

`compare_folders` pairs the CSVs of two output folders by legacy filename and runs these tests per cell.

Plain Python, no Numba.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

from anb.metrics import RESPONSE_COLUMNS

ALPHA = 0.05
N_PERMUTATIONS = 10_000
WINDOWS = {"before": 95.5, "during": 145.5, "after": 195.5}
METRICS = ("n_nbs", "Hpers", "Hpersfoc", "Hsoc", "Hext", "tb_tot", "tb_foc", "absOm_tot", "absOm_foc", "clust",
           "bc_foc", "expI", "x_focal", "extr_nonfoc")
RESPONSES = (*RESPONSE_COLUMNS, "unlabelled")
NEGATIVE_RESPONSES = RESPONSE_COLUMNS[2:]  # the agents negative before pressure; the first two are positive


def summarise_run(frame: pd.DataFrame) -> dict:
    """One ensemble-mode frame → response counts (from the 95.5 row), the lean, and run means of each metric.

    The counts of `NEGATIVE_RESPONSES` are the conditional counts. `lean` is the share positive before pressure.
    Metrics are averaged per window (`metric@window`) and per tracked step (`metric@t<step>`).
    """
    before = frame[frame.t == WINDOWS["before"]]
    labels = before[list(RESPONSE_COLUMNS)].astype(float).fillna(0).astype(bool)
    out = {name: int(labels[name].sum()) for name in RESPONSE_COLUMNS}
    out["unlabelled"] = int((~labels.any(axis=1)).sum())
    out["lean"] = (out["persistPos"] + out["nonpersistPos"]) / sum(out[name] for name in RESPONSES)
    for window, t in WINDOWS.items():
        rows = frame[frame.t == t]
        for metric in METRICS:
            out[f"{metric}@{window}"] = float(np.nanmean(rows[metric].to_numpy(dtype=float)))
    steps = frame[frame.t == frame.t.round()].groupby("t")  # the averaged rows sit at half steps
    for t, rows in steps:
        for metric in METRICS:
            out[f"{metric}@t{int(t)}"] = float(np.nanmean(rows[metric].to_numpy(dtype=float)))
    return out


def _chi2(counts_a: np.ndarray, counts_b: np.ndarray) -> np.ndarray:
    """χ² of the 2 × K tables [a; b]; `counts_*` are `(..., K)` with categories empty in both already dropped.
    A group with no agents at all (possible in a permutation of conditional counts) adds nothing."""
    table = np.stack([counts_a, counts_b], axis=-2)  # (..., 2, K)
    expected = table.sum(axis=-1, keepdims=True) * table.sum(axis=-2, keepdims=True) / table.sum(
        axis=(-2, -1), keepdims=True)
    deviation = (table - expected) ** 2
    return np.divide(deviation, expected, out=np.zeros_like(deviation), where=expected > 0).sum(axis=(-2, -1))


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


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """mean(b) − mean(a) with its SE, a 95 % Welch CI, the gap in SEs, the standardised mean difference and the
    Welch p-value. NaNs are dropped."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    diff = b.mean() - a.mean()
    var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(var_a + var_b)
    pooled_sd = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
    if se == 0:
        half, gap, p = 0.0, 0.0, 1.0
        if diff != 0:  # no spread at all but different levels
            gap, p = np.inf * np.sign(diff), 0.0
    else:
        dof = se**4 / (var_a**2 / (len(a) - 1) + var_b**2 / (len(b) - 1))
        half = stats.t.ppf(0.975, dof) * se
        gap = diff / se
        p = float(2 * stats.t.sf(abs(gap), dof))
    smd = 0.0 if diff == 0 else diff / pooled_sd if pooled_sd > 0 else np.inf * np.sign(diff)
    return {"mean_legacy": a.mean(), "mean_anb": b.mean(), "diff": diff, "se": se, "ci_low": diff - half,
            "ci_high": diff + half, "gap_se": gap, "smd": smd, "p": p}


def compare_values(a: np.ndarray, b: np.ndarray) -> dict:
    """`welch` plus a two-sample KS on run-level values (its p-value replaces the Welch one). NaNs are dropped."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = welch(a, b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if np.ptp(np.concatenate([a, b])) == 0:
        out["ks"], out["p"] = 0.0, 1.0  # both constant and equal (e.g. edge metrics of a static BN)
    else:
        result = stats.ks_2samp(a, b)
        out["ks"], out["p"] = float(result.statistic), float(result.pvalue)
    return out


def correct(table: pd.DataFrame) -> pd.DataFrame:
    """Add Benjamini–Hochberg `q` and `fail` (q < `ALPHA`) over all rows of one test family."""
    out = table.copy()
    out["q"] = multipletests(out.p, method="fdr_bh")[1] if len(out) else []
    out["fail"] = out.q < ALPHA
    return out


def response_table(legacy: pd.DataFrame, new: pd.DataFrame, rng: np.random.Generator,
                   n_permutations: int = N_PERMUTATIONS, categories: tuple[str, ...] = RESPONSES) -> pd.DataFrame:
    """One row per cell: χ² permutation test on the run counts of `categories`, Cramér's V, and the shares."""
    rows = []
    for (config, pressure), old in legacy.groupby(["config", "pressure"], sort=False):
        cell = new[(new.config == config) & (new.pressure == pressure)]
        a, b = old[list(categories)].to_numpy(), cell[list(categories)].to_numpy()
        row = {"config": config, "pressure": pressure, **compare_responses(a, b, rng, n_permutations)}
        total = a.sum() + b.sum()
        row["cramers_v"] = float(np.sqrt(row["chi2"] / total)) if total else 0.0  # 2 × K table: min(2, K) − 1 = 1
        for k, name in enumerate(categories):
            row[f"{name}_legacy"] = a[:, k].sum() / a.sum() if a.sum() else np.nan
            row[f"{name}_anb"] = b[:, k].sum() / b.sum() if b.sum() else np.nan
        rows.append(row)
    return correct(pd.DataFrame(rows))


def lean_table(legacy: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: Welch on the run-level share positive before pressure, the gap in SEs and the SMD."""
    rows = []
    for (config, pressure), old in legacy.groupby(["config", "pressure"], sort=False):
        cell = new[(new.config == config) & (new.pressure == pressure)]
        w = welch(old.lean, cell.lean)
        rows.append({"config": config, "pressure": pressure, "lean_legacy": w.pop("mean_legacy"),
                     "lean_anb": w.pop("mean_anb"), **w})
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
    """What `compare_folders` found: the pairing, one summary row per matched run and side, the four test tables
    and `top_differences`, which stacks them ranked by `effect`."""
    pairing: Pairing
    old: pd.DataFrame
    new: pd.DataFrame
    pooled_responses: pd.DataFrame = field(default_factory=pd.DataFrame)
    conditional_responses: pd.DataFrame = field(default_factory=pd.DataFrame)
    lean: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_differences: pd.DataFrame = field(default_factory=pd.DataFrame)


def _stems(folder: Path) -> set[str]:
    return {path.stem for path in Path(folder).glob("sim_*.csv")}


def _summarise_file(path: Path) -> dict:
    return summarise_run(pd.read_csv(path))


def _cell_key(folder: Path, stems: list[str]) -> str:
    """Identifies a cell's files and their state (names, modification times, sizes) and what a summary holds, so
    a change to the folder or to the summarised columns misses the cache."""
    parts = [str(folder.resolve()), repr((METRICS, WINDOWS, RESPONSES))]
    for stem in stems:
        stat = (folder / f"{stem}.csv").stat()
        parts.append(f"{stem}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _summaries(folder: Path, stems: list[str], cache_dir: Path | None = None, n_jobs: int = -1) -> pd.DataFrame:
    """One summary row per run. With `cache_dir`, each cell's rows are stored under (folder, cell) and reused
    while the cell's files are unchanged; the other cells' CSVs are read in parallel."""
    folder = Path(folder)
    cells: dict[str, list[str]] = {}
    for stem in stems:
        cells.setdefault(re.sub(r"_seed\d+", "", stem), []).append(stem)
    frames, todo = {}, {}
    for cell_name, members in cells.items():
        entry = None
        if cache_dir is not None:
            token = hashlib.sha256(f"{folder.resolve()}|{cell_name}".encode()).hexdigest()[:24]
            entry = Path(cache_dir) / f"{token}.pkl", _cell_key(folder, members)
            try:
                cached = pd.read_pickle(entry[0])
                if cached["key"] == entry[1]:
                    frames[cell_name] = cached["rows"]
                    continue
            except Exception:  # missing or unreadable cache file: rebuild
                pass
        todo[cell_name] = entry
    jobs = [(name, stem) for name in todo for stem in cells[name]]
    summaries = Parallel(n_jobs=n_jobs)(delayed(_summarise_file)(folder / f"{stem}.csv") for _, stem in jobs)
    rows: dict[str, list[dict]] = {name: [] for name in todo}
    for (name, stem), summary in zip(jobs, summaries):
        cell, seed = parse_filename(stem)
        rows[name].append({"config": cell_label(cell), "pressure": cell["ext_strength"], "seed": seed, **summary})
    for name, entry in todo.items():
        frames[name] = pd.DataFrame(rows[name])
        if entry is not None:
            entry[0].parent.mkdir(parents=True, exist_ok=True)
            partial = entry[0].with_suffix(".tmp")
            pd.to_pickle({"key": entry[1], "rows": frames[name]}, partial)
            partial.replace(entry[0])  # atomic, so an interrupted run never leaves a truncated cache file
    return pd.concat([frames[name] for name in cells], ignore_index=True)


def compare_folders(old: str | Path, new: str | Path, permutation_seed: int = 0,
                    n_permutations: int = N_PERMUTATIONS, cache_dir: str | Path | None = None,
                    n_jobs: int = -1) -> Comparison:
    """Compare two ensemble output folders. Files are paired by name; those in one folder only are reported.

    Four tables, one row per cell (metrics: per cell × window × metric), each corrected with Benjamini–Hochberg
    within itself and ranked by effect size, with `q` alongside: pooled responses and conditional responses
    (Cramér's V), lean and metrics (|SMD|). `top_differences` merges them. Run summaries are read with `n_jobs` workers; with `cache_dir` they
    are cached per (folder, cell) and rebuilt for any cell whose files changed. Writes nothing else.
    """
    in_old, in_new = _stems(old), _stems(new)
    pairing = Pairing(sorted(in_old & in_new), sorted(in_old - in_new), sorted(in_new - in_old))
    if not pairing.matched:
        raise ValueError(f"no CSV files with the same name in {old} and {new}")
    legacy = _summaries(Path(old), pairing.matched, cache_dir, n_jobs)
    anb = _summaries(Path(new), pairing.matched, cache_dir, n_jobs)
    rng = np.random.default_rng(permutation_seed)
    pooled = response_table(legacy, anb, rng, n_permutations)
    conditional = response_table(legacy, anb, rng, n_permutations, NEGATIVE_RESPONSES)
    lean = lean_table(legacy, anb)
    metrics = metric_table(legacy, anb)
    pooled = pooled.sort_values("cramers_v", ascending=False, ignore_index=True)
    conditional = conditional.sort_values("cramers_v", ascending=False, ignore_index=True)
    lean = lean.iloc[(-lean.smd.abs()).argsort(kind="stable")].reset_index(drop=True)
    metrics = metrics.iloc[(-metrics.smd.abs()).argsort(kind="stable")].reset_index(drop=True)
    return Comparison(pairing, legacy, anb, pooled, conditional, lean, metrics,
                      _top_differences(pooled, conditional, lean, metrics))


def _top_differences(pooled, conditional, lean, metrics) -> pd.DataFrame:
    """The four families stacked: family, cell, what was tested (`detail`), `effect` (≥ 0), `q` and `fail`."""
    parts = []
    for family, table in (("pooled responses", pooled), ("conditional responses", conditional)):
        parts.append(table.assign(family=family, detail="", effect=table.cramers_v))
    parts.append(lean.assign(family="lean", detail="share positive", effect=lean.smd.abs()))
    parts.append(metrics.assign(family="metrics", detail=metrics.metric + "@" + metrics.window,
                                effect=metrics.smd.abs()))
    columns = ["family", "config", "pressure", "detail", "effect", "q", "fail"]
    top = pd.concat([part[columns] for part in parts], ignore_index=True)
    return top.iloc[(-top.effect).argsort(kind="stable")].reset_index(drop=True)
