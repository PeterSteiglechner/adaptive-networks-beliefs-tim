"""Output metrics (one `Metric` per column) and the per-run recorder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .agents import edge_columns

if TYPE_CHECKING:
    from .model import ANBModel


@dataclass
class StepContext:
    """Everything a metric may read at time `t`."""

    model: ANBModel
    t: int
    ext: float


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(self, ctx: StepContext) -> np.ndarray:
        """One value per agent, `(N,)`."""


class Degree(Metric):
    name = "n_nbs"

    def compute(self, ctx):
        return ctx.model.network.degree()


class FocalBelief(Metric):
    name = "x_focal"

    def compute(self, ctx):
        return ctx.model.state.focal_beliefs()


class MetricsRecorder:
    """Snapshots every metric plus beliefs and edges at each recorded step. One recorder per run."""

    def __init__(self, metrics: list[Metric]):
        self.metrics = list(metrics)
        self._blocks: list[np.ndarray] = []
        self._columns: list[str] | None = None

    def record(self, ctx: StepContext) -> None:
        state = ctx.model.state
        n, m = state.n_agents, state.n_beliefs
        if self._columns is None:
            metric_cols = [metric.name for metric in self.metrics]
            self._columns = ["t", "id"] + metric_cols + [str(d) for d in range(m)] + edge_columns(m)
        self._blocks.append(np.column_stack(
            [np.full(n, ctx.t), np.arange(n)]
            + [metric.compute(ctx) for metric in self.metrics]
            + [state.beliefs, state.edge_vector()]
        ))

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(np.concatenate(self._blocks), columns=self._columns)
        return frame.astype({"t": int, "id": int})
