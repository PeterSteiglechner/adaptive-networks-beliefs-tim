"""Reading simulation CSVs: the column layout from the data, and an in-memory cache of the reduced frames."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from anb.agents import edge_columns
from anb.io import parse_filename


@dataclass(frozen=True)
class Layout:
    """A run's sizes: M from the belief columns `"0"…"M-1"`, N from the distinct `id`s, τ from the file name."""

    M: int
    N: int
    tau: int

    @property
    def beliefs(self) -> list[str]:
        return [str(m) for m in range(self.M)]

    @property
    def edges(self) -> list[str]:
        """`w{m}{n}` for m < n, in the CSV's order."""
        return edge_columns(self.M)


@dataclass(frozen=True)
class Run:
    """One CSV, reduced to the requested rows and columns (`t` and `id` first, `id` as int)."""

    path: Path
    frame: pd.DataFrame
    layout: Layout


def _n_beliefs(header: Iterable[str]) -> int:
    header = set(header)
    M = 0
    while str(M) in header:
        M += 1
    return M


@dataclass
class Loader:
    """Reads runs and keeps each reduced frame in memory, so figures sharing a loader read a request once.

    There is no disk cache. Treat the returned frames as read-only: they are shared between callers.
    """

    _runs: dict = field(default_factory=dict, repr=False)

    @property
    def cached(self) -> int:
        """The number of reduced frames held."""
        return len(self._runs)

    def load(self, path, *, t=None, columns: Iterable[str] = (), beliefs: bool = False, edges: bool = False) -> Run:
        """The run at `path` with `t`, `id`, then `columns`, the belief columns and the edge columns (as asked).

        `t` keeps only those time steps (any iterable); `None` keeps all.
        """
        path = Path(path)
        t_key = None if t is None else tuple(sorted(float(x) for x in t))
        key = (path, t_key, tuple(columns), beliefs, edges)
        if key not in self._runs:
            self._runs[key] = self._read(path, t_key, list(columns), beliefs, edges)
        return self._runs[key]

    @staticmethod
    def _read(path: Path, t, columns: list[str], beliefs: bool, edges: bool) -> Run:
        header = pd.read_csv(path, nrows=0).columns
        M = _n_beliefs(header)
        wanted = ["t", "id", *columns, *(str(m) for m in range(M) if beliefs), *(edge_columns(M) if edges else [])]
        frame = pd.read_csv(path, usecols=wanted)[wanted]
        frame["id"] = frame["id"].astype(int)
        layout = Layout(M=M, N=frame.id.nunique(), tau=parse_filename(path)[0]["tau"])
        if t is not None:
            frame = frame[frame.t.isin(t)].reset_index(drop=True)
        return Run(path, frame, layout)
