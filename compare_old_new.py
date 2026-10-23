# %%
"""Compare two ensemble output folders (legacy vs `anb`, or any two sweeps) and write the tables to `compare_out/`.

Run it as a plain script or cell by cell. The constants below can be overridden by environment variables of the
same name (`COMPARE_OLD`, ...), which the smoke test uses.
"""
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anb.comparison import (METRICS, NEGATIVE_RESPONSES, PAPER_CONFIGS, RESPONSES, compare_folders, metric_series,
                            response_shares, top_cells)

# %%
OLD_FOLDER = Path(os.environ.get("COMPARE_OLD", "simOut_old_model"))
NEW_FOLDER = Path(os.environ.get("COMPARE_NEW", "simOut_new_model"))
OUT_FOLDER = Path(os.environ.get("COMPARE_OUT", "compare_out"))
TOP_K = int(os.environ.get("COMPARE_TOP_K", 10))
N_PERMUTATIONS = int(os.environ.get("COMPARE_N_PERMUTATIONS", 10_000))
RUN_ABLATIONS = False  # the ablation cell arrives with its own ticket
RUN_SHARED_START = False  # the shared-start cell arrives with its own ticket

# %%
result = compare_folders(OLD_FOLDER, NEW_FOLDER, n_permutations=N_PERMUTATIONS, cache_dir=OUT_FOLDER / "cache")

# %%
TABLES = {
    "pooled_responses": result.pooled_responses,
    "conditional_responses": result.conditional_responses,
    "lean": result.lean,
    "metrics": result.metrics,
    "top_differences": result.top_differences,
}


def markdown_table(table: pd.DataFrame) -> str:
    """A GitHub markdown table; floats to 4 significant digits."""
    cells = table.map(lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "\\|"))
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def file_list(stems: list[str], limit: int = 20) -> str:
    shown = "\n".join(f"- `{stem}`" for stem in stems[:limit])
    return shown + (f"\n- … and {len(stems) - limit} more" if len(stems) > limit else "") if stems else "none"


def summary_markdown() -> str:
    pairing = result.pairing
    families = pd.DataFrame(
        [{"family": name, "tests": len(table), "failing (q < 0.05)": int(table.fail.sum()) if "fail" in table else 0}
         for name, table in TABLES.items() if name != "top_differences"])
    top = result.top_differences.head(TOP_K)
    return "\n".join([
        "# Old vs new comparison", "",
        f"Old: `{OLD_FOLDER}`, new: `{NEW_FOLDER}`. {len(pairing.matched)} files matched.", "",
        "## Files in one folder only", "",
        f"Only in old ({len(pairing.only_old)}):", "", file_list(pairing.only_old), "",
        f"Only in new ({len(pairing.only_new)}):", "", file_list(pairing.only_new), "",
        "## Failing tests per family", "", markdown_table(families), "",
        f"## Top {TOP_K} effects", "", markdown_table(top), ""])


# %%
OUT_FOLDER.mkdir(parents=True, exist_ok=True)
for name, table in TABLES.items():
    table.to_csv(OUT_FOLDER / f"{name}.csv", index=False)
    (OUT_FOLDER / f"{name}.md").write_text(markdown_table(table) + "\n")
(OUT_FOLDER / "summary.md").write_text(summary_markdown())
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
