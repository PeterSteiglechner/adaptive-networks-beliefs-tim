"""Fig. A8: the one-factor-at-a-time (OFAT) sensitivity of the response frequencies to M, N, β, p, τ, ε and ω₀.

Reproduces `analyse_sensitivity.py`. The variant runs are found through the index (the `_M`/`_N`/`_tau` suffixes the
package writes), so the old script's `_fixedBNatt100` typo doesn't apply. A row whose variant runs are missing is
left out of the 7 × 3 grid. Each row's middle value is the base's own runs, drawn once (the old script read the ε = 1
and ω₀ = 0.2 runs twice, as a variant and as the base).
"""

import string
import warnings
from dataclasses import replace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .index import index_folder
from .load import Loader
from .output import FigureResult
from .responses import JITTER_SEED, PLOTTED, RESPONSES
from .select import CONFIG_FIELDS, PAPER, Baseline, Cell, FigureNotApplicable, MissingLevelsWarning, select
from .style import CONFIG_LABELS, RESPONSE_COLORS, SCRIPT_RC, figure_style, seeded_jitter

S = 4  # the pressure of every run
PAPER_SEEDS = 100
BEFORE = 95.5  # the row that holds the response classification
COLUMNS = ("staticlow", "statichigh", "adaptive")  # the grid's columns
# The OFAT rows in figure order: the configs (columns) each is drawn for, and its variants as the fields they replace
# in the config. The values are absolute, so for a non-paper base a variant equal to the base is dropped, and one
# shown at the base's value in its row (N = 50, p = 0.2 for a base of N = 50, p = 0.1) is dropped with a warning.
ROWS = {
    "M": (COLUMNS, ({"M": 5}, {"M": 15})),
    "N": (COLUMNS, ({"N": 50, "link_prob": 0.2}, {"N": 200, "link_prob": 0.05})),  # p = 10/N: as many contacts
    "beta": (COLUMNS, ({"beta": 1.5}, {"beta": 6.0})),
    "p": (COLUMNS, ({"link_prob": 0.05}, {"link_prob": 0.2})),
    "tau": (COLUMNS, ({"tau": 2}, {"tau": 10})),
    "eps": (("adaptive",), ({"eps": 0.5}, {"eps": 2.0})),
    "init_w": (("adaptive",), ({"init_w": 0.1}, {"init_w": 0.4})),
}
ROW_LABELS = {  # the old `experiments`, right of each row
    "M": r"nr of beliefs $\boldsymbol{M}$",
    "N": r"nr of agents $\boldsymbol{N}$" + "\n" + r"(with $\boldsymbol{p}=10/N$)",
    "beta": r"attention to" + "\n" + r"dissonance $\boldsymbol{\beta}$",
    "p": r"average nr of " + "\n" + r"social contacts" + "\n" + r"$\boldsymbol{p} \cdot N$",
    "tau": r"activation" + "\n" + r"memory $\boldsymbol{\tau}$",
    "eps": r"internal" + "\n" + r"adaptation" + "\n" + r"rate $\boldsymbol{\epsilon}$",
    "init_w": r"initial edge" + "\n" + r"weight $\boldsymbol{\omega_0}$",
}
_INTEGER_ROWS = ("M", "N", "tau")  # tick labels without decimals (p·N: without them where it's whole)
_LABEL_ROW = 2  # the row carrying the y label (β), or the drawn row nearest to it
_FS = SCRIPT_RC["font.size"]


def _variants(row: str, base: Baseline, seeds) -> list[tuple[str, Cell]]:
    """The variant cells of `row` as (config, cell), leaving out a variant equal to the base config.

    A variant at the base's value in `row` would be pooled with the base's runs under one tick, so it is left out
    with a `MissingLevelsWarning` naming it.
    """
    configs, changes = ROWS[row]
    cells, coinciding = [], []
    for config in configs:
        own = base.config(config)
        for fields in changes:
            variant = replace(own, **fields)
            if variant.rounded() != own.rounded():
                label = f"{config} " + ", ".join(f"{k}={v}" for k, v in fields.items())
                if _value(row, variant, variant.N) == _value(row, own, own.N):
                    coinciding.append(label)
                else:
                    cells.append((config, Cell(label, variant, S, seeds=seeds)))
    if coinciding:
        warnings.warn(f"figA8: leaving out {'; '.join(coinciding)}, at the base's value in row {row} (they'd be "
                      f"pooled with its runs)", MissingLevelsWarning, stacklevel=3)
    return cells


def _value(row: str, config: Baseline, N: int):
    """The x value of a run in `row`: the varied field, or for `p` the mean number of contacts p·N (N of the run)."""
    return round(config.link_prob * N, 2) if row == "p" else getattr(config, row)  # p has 2 decimals


def figA8(sim_dir, *, base: Baseline = PAPER, rows=None, seeds=None, min_seeds=1, loader=None,
          index=None) -> FigureResult | None:
    """Fig. A8: the relative response frequencies at s = 4 when one parameter at a time is changed from the base.

    Rows (in order, `rows` picks some): M ∈ {5, 15}, N ∈ {50, 200} with p = 10/N, β ∈ {1.5, 6}, p ∈ {0.05, 0.2} shown
    as p·N (N of each run), τ ∈ {2, 10}, each for the staticlow, statichigh and adaptive configs, and ε ∈ {0.5, 2},
    ω₀ ∈ {0.1, 0.4} for the adaptive config only. Every row also shows the base's runs. The base runs of the configs
    the rows use are required. A row with a variant missing is not drawn (a `MissingLevelsWarning` names the missing
    cells), and with no row left the figure isn't applicable. The 7 × 3 grid stays, its undrawn panels empty. For a
    non-paper base a variant equal to it is dropped, and one at its value in the row (N = 50, p = 0.2 for a base of
    N = 50, p = 0.1) is dropped with a `MissingLevelsWarning`.

    `data`: row, config, value (the x value), base (whether the run is the base's), seed, response (compliant,
    resilient, resistant, latecompliant: the old denominator), frequency.
    """
    rows = list(ROWS) if rows is None else list(rows)
    unknown = [row for row in rows if row not in ROWS]
    if unknown:
        raise ValueError(f"unknown OFAT row(s) {', '.join(unknown)}; choose from {', '.join(ROWS)}")
    index = index_folder(sim_dir) if index is None else index
    loader = loader or Loader()

    def short(cell):
        return len(cell.runs(index)) < min_seeds

    bases = {config: Cell(f"base {config}", base.config(config), S, seeds=seeds)
             for config in COLUMNS if any(config in ROWS[row][0] for row in rows)}
    missing = [cell.label for cell in bases.values() if short(cell)]
    if missing:
        warnings.warn(f"figA8 not applicable: needs at least {min_seeds} run(s) at s={S} of {', '.join(missing)}",
                      FigureNotApplicable, stacklevel=2)
        return None
    variants = {row: _variants(row, base, seeds) for row in rows}
    lacking = {row: [cell.label for _, cell in cells if short(cell)] for row, cells in variants.items()}
    complete = [row for row in rows if not lacking[row]]
    gaps = "; ".join(f"{row}: no {', '.join(labels)}" for row, labels in lacking.items() if labels)
    if not complete:
        warnings.warn(f"figA8 not applicable: no OFAT row has at least {min_seeds} run(s) of each variant ({gaps})",
                      FigureNotApplicable, stacklevel=2)
        return None
    if gaps:
        left_out = ", ".join(row for row in rows if lacking[row])
        warnings.warn(f"figA8: leaving out row(s) {left_out}, which lack variant runs ({gaps})", MissingLevelsWarning,
                      stacklevel=2)
    used = [bases[config] for config in bases if any(config in ROWS[row][0] for row in complete)]
    used += [cell for row in complete for _, cell in variants[row]]
    selected = select(index, used, item="figA8", paper_seeds=PAPER_SEEDS, min_seeds=min_seeds)
    data = _frequencies(complete, bases, variants, selected, loader)
    with figure_style(SCRIPT_RC), seeded_jitter(JITTER_SEED):
        fig = _draw(data)
    return FigureResult(fig, data, "AppendixFig_ofat" + base.suffix(ignore=CONFIG_FIELDS))


def _frequencies(rows, bases, variants, selected, loader) -> pd.DataFrame:
    """Tidy relative response frequencies (95.5 row, the four `RESPONSES` as denominator) per row × run."""
    records = []
    for row in rows:
        cells = [(config, bases[config], True) for config in ROWS[row][0]]
        cells += [(config, cell, False) for config, cell in variants[row]]
        for config, cell, is_base in cells:
            frame = selected[cell.label]
            for seed, path in zip(frame.seed, frame.path):
                run = loader.load(path, t=(BEFORE,), columns=RESPONSES)
                counts = run.frame[list(RESPONSES)].sum().astype(float)
                freq = counts / counts.sum()  # NaN (silently, as the old script) for a run without these types
                value = _value(row, cell.base, run.layout.N)
                records += [{"row": row, "config": config, "value": value, "base": is_base, "seed": int(seed),
                             "response": name, "frequency": freq[name]} for name in RESPONSES]
    data = pd.DataFrame(records)
    order = {"row": {row: k for k, row in enumerate(ROWS)}, "config": {config: k for k, config in enumerate(COLUMNS)}}
    return data.sort_values(["row", "config", "value", "seed"], key=lambda c: c.map(order[c.name]) if c.name in order
                            else c, kind="stable", ignore_index=True)


def _label(row: str, value) -> str:
    if row == "p":
        return f"{float(value):g}"
    return f"{int(value)}" if row in _INTEGER_ROWS else f"{float(value)}"


def _strips(ax, panel: pd.DataFrame, row: str):
    """One dot per run and response (jittered, dodged) and the medians as squares, over the row's values."""
    labels = {value: _label(row, value) for value in sorted(panel.value.unique())}
    plotted = panel.assign(x=panel.value.map(labels)).rename(columns={"frequency": "normalized_count"})
    style = dict(x="x", order=list(labels.values()), hue="response", y="normalized_count", palette=RESPONSE_COLORS,
                 hue_order=list(PLOTTED), legend=False, dodge=True, jitter=True)
    sns.stripplot(plotted, ax=ax, **style, size=2, alpha=0.4)
    medians = plotted.groupby(["response", "x"])["normalized_count"].median().reset_index()
    sns.stripplot(medians, ax=ax, **style, size=4, alpha=0.8, marker="s")


def _draw(data: pd.DataFrame):
    fig, axs = plt.subplots(len(ROWS), len(COLUMNS), sharex=False, sharey=True, figsize=(18 / 2.54, 16 / 2.54))
    plotted = data[data.response.isin(PLOTTED)]
    drawn = {}
    for n, row in enumerate(ROWS):
        for config in ROWS[row][0]:
            panel = plotted[(plotted.row == row) & (plotted.config == config)]
            if not panel.empty:
                drawn[(n, COLUMNS.index(config))] = row
                _strips(axs[n, COLUMNS.index(config)], panel, row)
    label_row = min({n for n, _ in drawn}, key=lambda n: abs(n - _LABEL_ROW))
    titled = set()
    for (n, nn), row in sorted(drawn.items()):
        ax = axs[n, nn]
        for coll in ax.collections:
            coll.set_clip_on(False)
        ax.set_ylabel("response frequency" if n == label_row else "", fontsize=_FS, va="center")
        if nn not in titled:  # the top drawn panel of each column
            titled.add(nn)
            ax.set_title(CONFIG_LABELS[COLUMNS[nn]], fontsize=_FS, x=0.5, y=0.99, ha="center")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=_FS)
        if nn == len(COLUMNS) - 1:
            ax.text(1.02, 0.1, ROW_LABELS[row], va="center", ha="left", transform=ax.transAxes,
                    fontdict={"weight": "bold"})
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(rf"${int(x * 100)}\,\%$" for x in [0, 0.5, 1.0])
        if nn > 0 and (n, nn - 1) not in drawn:  # sharey hides these, but nothing is left of them (ε, ω₀)
            ax.tick_params(axis="y", which="both", left=True, labelleft=True)
    letters = iter(string.ascii_uppercase)
    for n in range(len(ROWS)):
        for nn in range(len(COLUMNS)):
            if (n, nn) in drawn:
                axs[n, nn].text(0, 1.01, next(letters), fontsize=12, fontdict={"weight": "bold"}, va="bottom",
                                ha="left", transform=axs[n, nn].transAxes)
            else:
                axs[n, nn].axis("off")
    fig.subplots_adjust(hspace=0.6, wspace=0.05, top=0.95, left=0.08, right=0.83, bottom=0.05)
    return fig
