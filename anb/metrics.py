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


def _focal_rows(state) -> np.ndarray:
    """`(N, M)`: each agent's edge-matrix row for its own focal dimension."""
    return state.edge_weights[np.arange(state.n_agents), state.focal]


class PersonalEnergy(Metric):
    """Hpers = −½ Σ_ij w_ij x_i x_j (BN dissonance over all edges)."""

    name = "Hpers"

    def compute(self, ctx):
        x = ctx.model.state.beliefs
        return -0.5 * np.einsum("ni,nij,nj->n", x, ctx.model.state.edge_weights, x)


class FocalPersonalEnergy(Metric):
    """Hpersfoc = −x_foc Σ_j w_foc,j x_j (BN dissonance on the focal belief's edges)."""

    name = "Hpersfoc"

    def compute(self, ctx):
        state = ctx.model.state
        return -state.focal_beliefs() * np.einsum("nj,nj->n", _focal_rows(state), state.beliefs)


class SocialEnergy(Metric):
    """Hsoc = −ρ x_foc · (neighbours' focal beliefs, A-weighted), 0 without neighbours.

    `aggregate="mean"` (default) divides by the summed link weights, as legacy does. `"sum"` gives the eq. 1
    term −ρ x_foc Σ_k A_ik x_foc(k) that the dynamics use (known issue #1).
    """

    name = "Hsoc"

    def __init__(self, aggregate: str = "mean"):
        if aggregate not in ("mean", "sum"):
            raise ValueError(f"aggregate must be 'mean' or 'sum', got {aggregate!r}")
        self.aggregate = aggregate

    def compute(self, ctx):
        x = ctx.model.state.focal_beliefs()
        adjacency = ctx.model.network.adjacency
        social = adjacency @ x
        if self.aggregate == "mean":
            weight = adjacency.sum(axis=1)
            social = np.divide(social, weight, out=np.zeros_like(social), where=weight != 0)
        return -ctx.model.rho * x * social


class ExternalEnergy(Metric):
    """Hext = −ext · x_foc, where `ext` is the pressure at this step (0 outside events)."""

    name = "Hext"

    def compute(self, ctx):
        return -ctx.ext * ctx.model.state.focal_beliefs()


class NonFocalExtremity(Metric):
    """extr_nonfoc: mean |x| over the non-focal beliefs."""

    name = "extr_nonfoc"

    def compute(self, ctx):
        state = ctx.model.state
        total = np.abs(state.beliefs).sum(axis=1)
        return (total - np.abs(state.focal_beliefs())) / (state.n_beliefs - 1)


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
