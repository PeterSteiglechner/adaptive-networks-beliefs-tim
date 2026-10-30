"""Reading simulation CSVs: the column layout from the data, and an in-memory cache of the reduced frames."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from anb.agents import edge_columns
from anb.io import is_detailed, parse_filename


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


# The ensemble rows the figures read: the before / during / after window averages and the pressure-onset snapshot.
ANALYSIS_TIMES = (95.5, 100.0, 145.5, 195.5)


@dataclass
class Loader:
    """Reads runs and keeps one reduced frame per CSV in memory, so figures sharing a loader read a CSV once.

    A request is cut from the frame held for its CSV if that has the request's rows and columns. Otherwise the CSV is
    read again, for both. With `whole=True` the first read takes every column (ensemble runs at `ANALYSIS_TIMES`,
    detailed runs at every t), so the figures' later requests are cut from it: `make_all` uses one. `reads` counts
    the reads from disk per path. There is no disk cache. Treat the returned frames as read-only: they are shared
    between callers.
    """

    whole: bool = False
    reads: Counter = field(default_factory=Counter)
    _runs: dict = field(default_factory=dict, repr=False)  # path → (the t held, None for all; the Run)

    @property
    def cached(self) -> int:
        """The number of CSVs held."""
        return len(self._runs)

    def load(self, path, *, t=None, columns: Iterable[str] = (), beliefs: bool = False, edges: bool = False) -> Run:
        """The run at `path` with `t`, `id`, then `columns`, the belief columns and the edge columns (as asked).

        `t` keeps only those time steps (any iterable); `None` keeps all.
        """
        path = Path(path)
        t = _times(t)
        held = self._runs.get(path)
        if held is not None:
            held_t, run = held
            wanted = _wanted(run.layout.M, columns, beliefs, edges)
            if _covers(held_t, t) and set(wanted) <= set(run.frame.columns):
                return _cut(run, held_t, t, wanted)
            read_t, extra = _union(held_t, t), [*run.frame.columns, *wanted]
        elif self.whole:
            read_t, extra = (None if t is None or is_detailed(path) else _union(t, ANALYSIS_TIMES)), None
        else:
            read_t, extra = t, ()
        run = self._read(path, read_t, extra, columns, beliefs, edges)
        self._runs[path] = (read_t, run)
        return _cut(run, read_t, t, _wanted(run.layout.M, columns, beliefs, edges))

    def _read(self, path: Path, t, extra, columns, beliefs: bool, edges: bool) -> Run:
        """Rows `t` with the request's columns plus `extra`, or with every named column if `extra` is `None`."""
        self.reads[path] += 1
        if extra is None:
            frame = pd.read_csv(path, usecols=lambda c: not c.startswith("Unnamed"))
            M = _n_beliefs(frame.columns)
            frame = frame[list(dict.fromkeys(["t", "id", *frame.columns]))]
        else:
            M = _n_beliefs(pd.read_csv(path, nrows=0).columns)
            wanted = list(dict.fromkeys([*_wanted(M, columns, beliefs, edges), *extra]))
            frame = pd.read_csv(path, usecols=wanted)[wanted]
        frame["id"] = frame["id"].astype(int)
        layout = Layout(M=M, N=frame.id.nunique(), tau=parse_filename(path)[0]["tau"])
        if t is not None:
            frame = frame[frame.t.isin(t)]
        # One block per dtype: thousands of held frames of ~90 one-column blocks each fragment the heap (measured:
        # `make_all` on simOut_old_model peaks at a median 2.7 instead of 3.6 GB, and runs faster).
        return Run(path, frame.reset_index(drop=True).copy(), layout)


def _times(t) -> tuple | None:
    return None if t is None else tuple(sorted({float(x) for x in t}))


def _wanted(M: int, columns: Iterable[str], beliefs: bool, edges: bool) -> list[str]:
    return list(dict.fromkeys(["t", "id", *columns, *(str(m) for m in range(M) if beliefs),
                               *(edge_columns(M) if edges else [])]))


def _covers(held_t, t) -> bool:
    return held_t is None or (t is not None and set(t) <= set(held_t))


def _union(a, b) -> tuple | None:
    return None if a is None or b is None else _times([*a, *b])


def _cut(run: Run, held_t, t, wanted: list[str]) -> Run:
    """The rows `t` and columns `wanted` of a held run (the held `Run` itself if that is all of it)."""
    if t == held_t and wanted == list(run.frame.columns):
        return run
    frame = run.frame if t == held_t else run.frame[run.frame.t.isin(t)]
    return Run(run.path, frame[wanted].reset_index(drop=True), run.layout)
