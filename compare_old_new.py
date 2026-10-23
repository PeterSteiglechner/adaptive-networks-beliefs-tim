# %%
"""Compare two ensemble output folders (legacy vs `anb`, or any two sweeps) and write the tables to `compare_out/`.

Run it as a plain script or cell by cell. Most constants below can be overridden by a `COMPARE_*` environment
variable (`OLD_FOLDER` by `COMPARE_OLD`, ...), which the smoke tests use; the others are edited here.
"""
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anb.comparison import (METRICS, NEGATIVE_RESPONSES, PAPER_CONFIGS, RESPONSES, SHARED_START_QUANTITIES,
                            Comparison, cell_testbed, compare_folders, compare_shared_start, metric_series,
                            paper_cell, response_shares, run_shared_start, save_under_legacy_names, top_cells)

# %%
OLD_FOLDER = Path(os.environ.get("COMPARE_OLD", "simOut_old_model"))
NEW_FOLDER = Path(os.environ.get("COMPARE_NEW", "simOut_new_model"))
OUT_FOLDER = Path(os.environ.get("COMPARE_OUT", "compare_out"))
TOP_K = int(os.environ.get("COMPARE_TOP_K", 10))
N_PERMUTATIONS = int(os.environ.get("COMPARE_N_PERMUTATIONS", 10_000))
PRESSURES = (0, 1, 2, 4, 8, 16)
N_JOBS = int(os.environ.get("COMPARE_N_JOBS", -1))

# Ablation: anb with the t = 101 gate set back to legacy's (`learning_until = 102`) for the adaptive→static config,
# saved under that config's file names and compared with the old folder.
RUN_ABLATIONS = bool(int(os.environ.get("COMPARE_RUN_ABLATIONS", 0)))
ABLATION_FOLDER = Path(os.environ.get("COMPARE_ABLATION", "simOut_ablation_learning102"))
ABLATION_SEEDS = range(int(os.environ.get("COMPARE_ABLATION_SEEDS", 100)))
ABLATION_LEARNING_UNTIL = 102

# Rerun: selected μ cells (the adaptive config at these μ, eps = 1, init_w = 0.2) with more seeds. Legacy has to be
# rerun over the same seeds into RERUN_OLD_FOLDER first (`ANBmodel.py`, `__main__` block), as files are paired by
# name, seed included.
RUN_RERUN = bool(int(os.environ.get("COMPARE_RUN_RERUN", 0)))
RERUN_MUS = (0.1,)
RERUN_PRESSURES = PRESSURES
RERUN_SEEDS = range(int(os.environ.get("COMPARE_RERUN_SEEDS", 1000)))
RERUN_OLD_FOLDER = Path(os.environ.get("COMPARE_RERUN_OLD", "simOut_old_model_rerun"))
RERUN_NEW_FOLDER = Path(os.environ.get("COMPARE_RERUN_NEW", "simOut_new_model_rerun"))

# Shared start: legacy and anb from one t = 0 state, K replicas each, compared step by step over the window.
# Tables and plots go to compare_out/shared_start/.
RUN_SHARED_START = bool(int(os.environ.get("COMPARE_RUN_SHARED_START", 0)))
SHARED_START_CONFIG = "adaptive-static"
SHARED_START_PRESSURES = tuple(float(p) for p in os.environ.get("COMPARE_SHARED_START_PRESSURES", "4,16").split(","))
SHARED_START_K = int(os.environ.get("COMPARE_SHARED_START_K", 20))
SHARED_START_WINDOW = tuple(int(t) for t in os.environ.get("COMPARE_SHARED_START_WINDOW", "95,155").split(","))
SHARED_START_SEED = 0

# %%
result = compare_folders(OLD_FOLDER, NEW_FOLDER, n_permutations=N_PERMUTATIONS, cache_dir=OUT_FOLDER / "cache",
                         n_jobs=N_JOBS)


# %%
def tables(res: Comparison) -> dict[str, pd.DataFrame]:
    return {"pooled_responses": res.pooled_responses, "conditional_responses": res.conditional_responses,
            "lean": res.lean, "metrics": res.metrics, "top_differences": res.top_differences}


def markdown_table(table: pd.DataFrame) -> str:
    """A GitHub markdown table; floats to 4 significant digits."""
    cells = table.map(lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "\\|"))
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def file_list(stems: list[str], limit: int = 20) -> str:
    shown = "\n".join(f"- `{stem}`" for stem in stems[:limit])
    return shown + (f"\n- … and {len(stems) - limit} more" if len(stems) > limit else "") if stems else "none"


def summary_markdown(res: Comparison, old: Path, new: Path) -> str:
    pairing = res.pairing
    families = pd.DataFrame(
        [{"family": name, "tests": len(table), "untested (p NaN)": int(table.p.isna().sum()),
          "failing (q < 0.05)": int(table.fail.sum())}
         for name, table in tables(res).items() if name != "top_differences"])
    return "\n".join([
        "# Old vs new comparison", "",
        f"Old: `{old}`, new: `{new}`. {len(pairing.matched)} files matched.", "",
        "## Files not compared", "",
        f"Only in old ({len(pairing.only_old)}):", "", file_list(pairing.only_old), "",
        f"Only in new ({len(pairing.only_new)}):", "", file_list(pairing.only_new), "",
        f"In both, but not legacy ensemble names ({len(pairing.unparsed)}):", "", file_list(pairing.unparsed), "",
        "## Failing tests per family", "", markdown_table(families), "",
        f"## Top {TOP_K} effects", "", markdown_table(res.top_differences.head(TOP_K)), ""])


def write_report(res: Comparison, old: Path, new: Path, folder: Path) -> None:
    """Every table as CSV and markdown, plus `summary.md`, into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables(res).items():
        table.to_csv(folder / f"{name}.csv", index=False)
        (folder / f"{name}.md").write_text(markdown_table(table) + "\n")
    (folder / "summary.md").write_text(summary_markdown(res, old, new))


# %%
write_report(result, OLD_FOLDER, NEW_FOLDER, OUT_FOLDER)
print((OUT_FOLDER / "summary.md").read_text())

# %%
# Plots. Old is solid and new dashed; every line has a 95 % band (response shares: bootstrap over runs; metric
# time series: ±1.96 SE over runs). The colours are the Okabe-Ito palette, so they hold up for colour-blind readers.
COLORS = dict(zip(RESPONSES, ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#7F7F7F"]))
SIDES = {"old": ("-", "o"), "new": ("--", "s")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def draw_band(ax, x, mean, low, high, side, color, label=None):
    style, marker = SIDES[side]
    ax.plot(x, mean, style, marker=marker, markersize=3, color=color, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)


def share_panel(ax, shares: dict[str, pd.DataFrame], x_column: str, categories) -> list:
    """Draw old and new shares of `categories` against `x_column` (evenly spaced, the values as tick labels)."""
    positions = sorted(set(np.concatenate([table[x_column].unique() for table in shares.values()])))
    for side, table in shares.items():
        for name in categories:
            rows = table[table.response == name].sort_values(x_column)
            x = [positions.index(v) for v in rows[x_column]]
            draw_band(ax, x, rows.share, rows.low, rows.high, side, COLORS[name], name if side == "old" else None)
    ax.set_xticks(range(len(positions)), [f"{v:g}" for v in positions], rotation=90 if len(positions) > 8 else 0)
    ax.set_ylim(0, 1)
    return positions


def save(fig, name: str) -> None:
    fig.savefig(OUT_FOLDER / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def side_by_side(summary_old: pd.DataFrame, summary_new: pd.DataFrame, categories) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    return {"old": response_shares(summary_old, categories, rng=rng), "new": response_shares(summary_new, categories, rng=rng)}


# %%
# Pooled response shares over pressure, one panel per paper config
pooled = side_by_side(result.old, result.new, RESPONSES)
configs = [name for name in PAPER_CONFIGS if name in set(pooled["old"].config)]
if configs:
    fig, axs = plt.subplots(1, len(configs), figsize=(3.4 * len(configs), 3.2), sharey=True, squeeze=False)
    for ax, config in zip(axs[0], configs):
        share_panel(ax, {side: table[table.config == config] for side, table in pooled.items()}, "pressure", RESPONSES)
        ax.set(title=config, xlabel="pressure")
    axs[0, 0].set_ylabel("share of agents")
    axs[0, -1].legend(title="solid: old, dashed: new", fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "response_shares_pooled.png")

# %%
# Response shares among agents negative before pressure, over μ, one panel per pressure.
# The μ sweep is the adaptive config (μ = 0) plus the cells labelled by μ alone.
conditional = side_by_side(result.old, result.new, NEGATIVE_RESPONSES)
sweep = {side: table[(table.config == "adaptive") | table.config.str.fullmatch(r"mu=[\d.]+")]
         for side, table in conditional.items()}
pressures = sorted(sweep["old"].pressure.unique())
if pressures:
    fig, axs = plt.subplots(1, len(pressures), figsize=(3.4 * len(pressures), 3.2), sharey=True, squeeze=False)
    for ax, pressure in zip(axs[0], pressures):
        share_panel(ax, {side: table[table.pressure == pressure] for side, table in sweep.items()}, "mu",
                    NEGATIVE_RESPONSES)
        ax.set(title=f"pressure {pressure:g}", xlabel="μ")
    axs[0, 0].set_ylabel("share among agents negative before pressure")
    axs[0, -1].legend(title="solid: old, dashed: new", fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "response_shares_conditional_mu.png")

# %%
# Mean metric time series (tracked steps) for the cells with the largest differences in the combined table
series = {metric: {"old": metric_series(result.old, metric), "new": metric_series(result.new, metric)}
          for metric in METRICS}
for config, pressure in top_cells(result.top_differences, TOP_K):
    fig, axs = plt.subplots(3, 5, figsize=(14, 7), sharex=True)
    for ax, metric in zip(axs.flat, METRICS):
        for side, table in series[metric].items():
            rows = table[(table.config == config) & (table.pressure == pressure)]
            draw_band(ax, rows.t, rows["mean"], rows["mean"] - 1.96 * rows.se, rows["mean"] + 1.96 * rows.se, side,
                      "#0072B2" if side == "old" else "#D55E00", side)
        ax.set_title(metric, fontsize=8)
    for ax in axs.flat[len(METRICS):]:
        ax.set_visible(False)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"{config}, pressure {pressure:g}")
    fig.supxlabel("t")
    save(fig, f"timeseries_{re.sub(r'[^A-Za-z0-9.]+', '_', config)}_pressure{pressure:g}.png")

# %%
# Ablation and rerun cells. Both save a normal output folder and go through `compare_folders`; their tables are
# written to `compare_out/<name>/`, with a `summary.md` like the main one.
def compare_and_write(old: Path, new: Path, name: str, expected_files: int) -> Comparison:
    res = compare_folders(old, new, n_permutations=N_PERMUTATIONS, cache_dir=OUT_FOLDER / "cache", n_jobs=N_JOBS)
    write_report(res, old, new, OUT_FOLDER / name)
    assert len(res.pairing.matched) == expected_files and not res.pairing.only_new, (
        f"{name}: {len(res.pairing.matched)} of {expected_files} files paired, {len(res.pairing.only_new)} only in new")
    print(f"{name}: {len(res.pairing.matched)} files matched\n" + markdown_table(res.top_differences.head(TOP_K)))
    return res


# %%
if RUN_ABLATIONS:
    for pressure in PRESSURES:
        bed = cell_testbed(paper_cell("adaptive-static"), pressure, learning_until=ABLATION_LEARNING_UNTIL)
        save_under_legacy_names(bed, ABLATION_SEEDS, ABLATION_FOLDER, n_jobs=N_JOBS, fixedBNat100=True)
    ablation = compare_and_write(OLD_FOLDER, ABLATION_FOLDER, "ablation_learning_until",
                                len(PRESSURES) * len(ABLATION_SEEDS))

# %%
# `skip_existing` resumes an interrupted rerun; delete RERUN_NEW_FOLDER after changing the model.
if RUN_RERUN:
    for mu in RERUN_MUS:
        for pressure in RERUN_PRESSURES:
            cell_testbed(paper_cell("adaptive", mu=mu), pressure).save_ensemble(RERUN_SEEDS, RERUN_NEW_FOLDER,
                                                                                n_jobs=N_JOBS, skip_existing=True)
    rerun = compare_and_write(RERUN_OLD_FOLDER, RERUN_NEW_FOLDER, "rerun",
                        len(RERUN_MUS) * len(RERUN_PRESSURES) * len(RERUN_SEEDS))

# %%
# Shared-start divergence. Per pressure: the step-wise table (difference anb − legacy of the ensemble means with a
# 95 % CI, KS p, BH q over steps × quantities), the first divergent step per quantity, and one plot of the
# differences. A step with q < 0.05 is marked; a lone marked step among clean ones is likely a false discovery.
if RUN_SHARED_START:
    shared_cell = paper_cell(SHARED_START_CONFIG)
    shared_folder = OUT_FOLDER / "shared_start"
    shared_folder.mkdir(parents=True, exist_ok=True)
    divergence_steps, first_divergence = [], []
    for pressure in SHARED_START_PRESSURES:
        trajectories = run_shared_start(shared_cell, pressure, SHARED_START_SEED, SHARED_START_K, SHARED_START_WINDOW,
                                        n_jobs=N_JOBS)
        divergence = compare_shared_start(trajectories)
        divergence_steps.append(divergence.steps.assign(config=SHARED_START_CONFIG, pressure=pressure))
        first_divergence.append(divergence.first_divergence.assign(config=SHARED_START_CONFIG, pressure=pressure))

        fig, axs = plt.subplots(3, 5, figsize=(14, 7), sharex=True)
        for ax, quantity in zip(axs.flat, SHARED_START_QUANTITIES):
            rows = divergence.steps[divergence.steps.quantity == quantity]
            ax.axhline(0, color="#7F7F7F", linewidth=0.8)
            ax.plot(rows.t, rows["diff"], color="#0072B2")
            ax.fill_between(rows.t, rows.ci_low, rows.ci_high, color="#0072B2", alpha=0.15, linewidth=0)
            ax.plot(rows.t[rows.fail], rows["diff"][rows.fail], "o", markersize=3, color="#D55E00")
            first = divergence.first_divergence.set_index("quantity").first_divergent_step[quantity]
            if not np.isnan(first):
                ax.axvline(first, color="#D55E00", linestyle="--", linewidth=0.8)
            ax.set_title(quantity, fontsize=8)
        for ax in axs.flat[len(SHARED_START_QUANTITIES):]:
            ax.set_visible(False)
        fig.suptitle(f"{SHARED_START_CONFIG}, pressure {pressure:g}: anb − legacy, K = {SHARED_START_K} "
                     "(band 95 % CI, dots q < 0.05, dashed first divergent step)")
        fig.supxlabel("t")
        fig.savefig(shared_folder / f"divergence_{SHARED_START_CONFIG}_pressure{pressure:g}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    for name, parts in {"divergence": divergence_steps, "first_divergence": first_divergence}.items():
        table = pd.concat(parts, ignore_index=True)
        table = table[["config", "pressure", *[c for c in table if c not in ("config", "pressure")]]]
        table.to_csv(shared_folder / f"{name}.csv", index=False)
        (shared_folder / f"{name}.md").write_text(markdown_table(table) + "\n")
    print("Shared start, first divergent step:\n" + (shared_folder / "first_divergence.md").read_text())
