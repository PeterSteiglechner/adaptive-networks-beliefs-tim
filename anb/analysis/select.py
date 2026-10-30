"""The base config, finding a figure's runs in the index, and the applicability warnings."""

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd


class FigureNotApplicable(UserWarning):
    """A figure or table can't be made: runs it structurally needs are missing. The function returns `None`."""


class FewSeedsWarning(UserWarning):
    """A figure is made from fewer seeds than the paper used for at least one of its cells."""


class MissingLevelsWarning(UserWarning):
    """Some of the paper's levels (pressures, μ values) inside a panel have no runs; the figure uses the others."""


class ExampleAgentFallback(UserWarning):
    """A default example agent doesn't exist in the run (or has the wrong type) and a seeded random one is used."""


# The BN config fields and the paper's four configs (docu/analysis-package.md), keyed by their file-name token.
CONFIG_FIELDS = ("init_w", "eps", "mu", "fixedBNat100")
CONFIGS = {
    "staticlow": (0.2, 0.0, 0.0, False),
    "statichigh": (0.8, 0.0, 0.0, False),
    "adaptive2static": (0.2, 1.0, 0.0, True),
    "adaptive": (0.2, 1.0, 0.0, False),
}

# Decimals each field has in legacy file names. Fields not listed (bools, ints) are compared exactly.
_DECIMALS = {"link_prob": 2, "beta": 2, "rho": 2, "init_w": 2, "eps": 2, "mu": 3, "lam": 4}
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Baseline:
    """The non-swept parameters a figure's runs must have. The default is the paper's adaptive baseline.

    Values are matched against file names as rounded there (`rho 0.33` matches `1/3`). Figures comparing BN
    configs replace `CONFIG_FIELDS` via `config`.
    """

    link_prob: float = 0.1
    beta: float = 3.0
    rho: float = 0.33
    init_w: float = 0.2
    eps: float = 1.0
    mu: float = 0.0
    fixedBNat100: bool = False
    lam: float = 0.0
    n_events: int = 1
    M: int = 10
    N: int = 100
    tau: int = 1

    def config(self, name: str) -> "Baseline":
        """This base with the BN fields of one of the paper's four `CONFIGS`."""
        if name not in CONFIGS:
            raise ValueError(f"unknown BN config {name!r}; one of {', '.join(CONFIGS)}")
        return replace(self, **dict(zip(CONFIG_FIELDS, CONFIGS[name])))

    def rounded(self) -> dict:
        """The fields as legacy file names write them."""
        return {f.name: _round(f.name, getattr(self, f.name)) for f in fields(self)}

    def suffix(self, ignore: Iterable[str] = ()) -> str:
        """The legacy tokens of the fields that differ from `PAPER`, in field order: `""` for the paper's base.

        `ignore` leaves out fields the figure sets itself (e.g. `CONFIG_FIELDS` for a config comparison).
        """
        ignore = set(ignore)
        own, paper = self.rounded(), PAPER.rounded()
        return "".join(_token(k, own[k]) for k in own if k not in ignore and not _same(own[k], paper[k]))


PAPER = Baseline()


def _round(name, value):
    return float(f"{value:.{_DECIMALS[name]}f}") if name in _DECIMALS else value  # as the name writes it


def _same(a, b) -> bool:
    return bool(np.isclose(a, b, rtol=0, atol=_TOLERANCE))


def _token(name, value) -> str:
    """`name=value` as a legacy file-name token (see `anb.io.legacy_filename`)."""
    if name == "fixedBNat100":
        return "_fixedBNat100" if value else ""
    if name == "n_events":
        return "_2events"  # the only other count legacy names
    if name == "lam":
        return f"_lambda{value:.4f}"
    if name == "mu":
        return f"_mu{value:.3f}"
    return f"_{name}{value:.2f}"


def runs(index: pd.DataFrame, base: Baseline, *, s=None, detailed: bool | None = False, seeds=None) -> pd.DataFrame:
    """The rows of `index` that match `base` (as rounded in file names), sorted by seed.

    `s`, `detailed` and `seeds` narrow them further; `None` doesn't filter. Ensemble runs (`detailed=False`) are
    the default.
    """
    mask = np.ones(len(index), dtype=bool)
    for name, value in base.rounded().items():
        mask &= np.isclose(index[name].to_numpy(float), float(value), rtol=0, atol=_TOLERANCE)
    if s is not None:
        mask &= np.isclose(index.s.to_numpy(float), float(s), rtol=0, atol=_TOLERANCE)
    if detailed is not None:
        mask &= index.detailed.to_numpy(bool) == detailed
    if seeds is not None:
        mask &= index.seed.isin(list(seeds)).to_numpy()
    return index[mask].sort_values("seed", kind="stable")


@dataclass(frozen=True)
class Cell:
    """One structural requirement of a figure: the runs of `base` at pressure `s`, ensemble or detailed."""

    label: str
    base: Baseline
    s: float
    detailed: bool = False
    seeds: Sequence[int] | None = None

    def runs(self, index: pd.DataFrame) -> pd.DataFrame:
        return runs(index, self.base, s=self.s, detailed=self.detailed, seeds=self.seeds)


def select(index: pd.DataFrame, cells: Iterable[Cell], *, item: str, paper_seeds: int,
           min_seeds: int = 1) -> dict[str, pd.DataFrame] | None:
    """Each cell's runs, by label, or `None` if any cell has fewer than `min_seeds` runs.

    In that case a `FigureNotApplicable` names the short cells. Otherwise cells with fewer than `paper_seeds` runs
    (the paper's count for `item`) get one `FewSeedsWarning` listing them.
    """
    selected = {cell.label: cell.runs(index) for cell in cells}
    short = {label: len(frame) for label, frame in selected.items() if len(frame) < min_seeds}
    if short:
        missing = ", ".join(f"{label} ({n} run(s))" for label, n in short.items())
        warnings.warn(f"{item} not applicable: needs at least {min_seeds} run(s) of {missing}",
                      FigureNotApplicable, stacklevel=3)
        return None
    few = {label: len(frame) for label, frame in selected.items() if len(frame) < paper_seeds}
    if few:
        counts = ", ".join(f"{label}: {n}" for label, n in few.items())
        warnings.warn(f"{item}: fewer seeds than the paper's {paper_seeds} for {counts}", FewSeedsWarning,
                      stacklevel=3)
    return selected
