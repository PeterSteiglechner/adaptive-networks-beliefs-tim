"""Output metrics (one `Metric` per column) and the per-run recorder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit

from .agents import edge_columns

if TYPE_CHECKING:
    from .model import ANBModel


@dataclass
class StepContext:
    """Everything a metric may read at time `t`. Metrics cache shared work in it, so build a new one after
    changing the state."""

    model: ANBModel
    t: int
    ext: float
    cache: dict = field(default_factory=dict, repr=False, compare=False)  # work shared by metrics at this step


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


def _focal_field(state) -> np.ndarray:
    """`(N,)`: Σ_j w_foc,j x_j, the BN's pull on each agent's focal belief (shared by Hpersfoc and expI)."""
    return np.einsum("nj,nj->n", _focal_rows(state), state.beliefs)


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
        return -state.focal_beliefs() * _focal_field(state)


class SocialEnergy(Metric):
    """Hsoc = −ρ x_foc · (neighbours' focal beliefs, A-weighted), 0 without neighbours.

    `aggregate="mean"` (default) divides by the summed |link weights| (the degree, as in legacy, for 0/1 links). `"sum"` gives the eq. 1
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
            weight = np.abs(adjacency).sum(axis=1)
            social = np.divide(social, weight, out=np.zeros_like(social), where=weight != 0)
        return -ctx.model.rho * x * social


class ExternalEnergy(Metric):
    """Hext = −ext · x_foc, where `ext` is the pressure at this step (0 outside events)."""

    name = "Hext"

    def compute(self, ctx):
        return -ctx.ext * ctx.model.state.focal_beliefs()


class FocalBetweenness(Metric):
    """bc_foc: betweenness of the focal belief in the agent's own (complete) belief network, as igraph gives it.

    Edge lengths follow legacy: 1 / (|w| + 1e-6), and 1e9 for an edge with w = 0 (kept in the graph, so all
    M(M−1)/2 edges always exist). Unnormalised, undirected. Computed by a Numba Brandes pass over all agents
    instead of one igraph graph per agent; path lengths tie within igraph's relative tolerance.
    """

    name = "bc_foc"
    LENGTH_EPS = 1e-6
    ZERO_EDGE_LENGTH = 1e9

    def compute(self, ctx):
        state = ctx.model.state
        abs_w = np.abs(state.edge_weights)
        lengths = np.where(abs_w > 0, 1.0 / (abs_w + self.LENGTH_EPS), self.ZERO_EDGE_LENGTH)
        return _focal_betweenness(lengths, state.focal)


_PATH_TIE_EPS = 1e-10  # igraph's IGRAPH_SHORTEST_PATH_EPSILON: path lengths this close (relatively) are equal


@njit(cache=True)
def _compare_lengths(a, b):
    """−1, 0 or 1 as `a` is shorter than, tied with or longer than `b` (igraph's `igraph_cmp_epsilon`)."""
    if a == b:
        return 0
    if abs(a - b) < _PATH_TIE_EPS * (abs(a) + abs(b)):
        return 0
    return -1 if a < b else 1


@njit(cache=True)
def _focal_betweenness(lengths, focal):
    """Brandes betweenness of node `focal[n]` in each agent's complete weighted graph `lengths[n] (M, M)`.

    Dense Dijkstra from every source; undirected, so each pair is counted from both ends and halved.
    """
    n_agents, m, _ = lengths.shape
    out = np.zeros(n_agents)
    dist = np.empty(m)
    sigma = np.empty(m)
    delta = np.empty(m)
    done = np.empty(m, dtype=np.bool_)
    order = np.empty(m, dtype=np.int64)
    pred = np.empty((m, m), dtype=np.bool_)
    for a in range(n_agents):
        f = focal[a]
        total = 0.0
        for s in range(m):
            dist[:] = np.inf
            sigma[:] = 0.0
            done[:] = False
            pred[:, :] = False
            dist[s] = 0.0
            sigma[s] = 1.0
            for k in range(m):
                u = -1
                for v in range(m):
                    if not done[v] and (u < 0 or dist[v] < dist[u]):
                        u = v
                done[u] = True
                order[k] = u
                for v in range(m):
                    if done[v] or v == u:
                        continue
                    alt = dist[u] + lengths[a, u, v]
                    cmp = _compare_lengths(alt, dist[v])
                    if cmp < 0:
                        dist[v] = alt
                        sigma[v] = sigma[u]
                        pred[v, :] = False
                        pred[v, u] = True
                    elif cmp == 0:
                        sigma[v] += sigma[u]
                        pred[v, u] = True
            delta[:] = 0.0
            for k in range(m - 1, -1, -1):
                w = order[k]
                for v in range(m):
                    if pred[w, v]:
                        delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if s != f:
                total += delta[f]
        out[a] = total / 2.0
    return out


class NonFocalExtremity(Metric):
    """extr_nonfoc: mean |x| over the non-focal beliefs."""

    name = "extr_nonfoc"

    def compute(self, ctx):
        state = ctx.model.state
        total = np.abs(state.beliefs).sum(axis=1)
        return (total - np.abs(state.focal_beliefs())) / (state.n_beliefs - 1)


@cache
def _triangles(n_beliefs: int) -> np.ndarray:
    """`(T, 3)`: every node triple a < b < c of an `n_beliefs`-node BN."""
    return np.array(list(combinations(range(n_beliefs), 3)), dtype=np.int64).reshape(-1, 3)


def _balanced_triangles(ctx: StepContext) -> tuple[np.ndarray, np.ndarray]:
    """The one triangle pass both balance metrics share, computed once per `StepContext`."""
    if "balanced_triangles" not in ctx.cache:
        ctx.cache["balanced_triangles"] = _triangle_pass(ctx.model.state.edge_weights)
    return ctx.cache["balanced_triangles"]


def _triangle_pass(edge_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns `triangles (T, 3)`, every node triple a < b < c, and `balanced (N, T)`, true where
    w_ab w_ac w_bc > 0. A triangle with a zero edge counts as unbalanced, as in legacy.
    """
    triangles = _triangles(edge_weights.shape[1])
    a, b, c = triangles.T
    balanced = edge_weights[:, a, b] * edge_weights[:, a, c] * edge_weights[:, b, c] > 0
    return triangles, balanced


class TriangleBalance(Metric):
    """tb_tot: number of balanced triangles in the BN."""

    name = "tb_tot"

    def compute(self, ctx):
        _, balanced = _balanced_triangles(ctx)
        return balanced.sum(axis=1).astype(float)


class FocalTriangleBalance(Metric):
    """tb_foc: number of balanced triangles that contain the focal belief."""

    name = "tb_foc"

    def compute(self, ctx):
        state = ctx.model.state
        triangles, balanced = _balanced_triangles(ctx)
        through_focal = (triangles[None, :, :] == state.focal[:, None, None]).any(axis=2)
        return (balanced & through_focal).sum(axis=1).astype(float)


class MeanAbsEdge(Metric):
    """absOm_tot: mean |w| over all M(M−1)/2 BN edges."""

    name = "absOm_tot"

    def compute(self, ctx):
        return np.abs(ctx.model.state.edge_vector()).mean(axis=1)


class FocalMeanAbsEdge(Metric):
    """absOm_foc: mean |w_foc,j| over the M − 1 edges of the focal belief."""

    name = "absOm_foc"

    def compute(self, ctx):
        state = ctx.model.state
        return np.abs(_focal_rows(state)).sum(axis=1) / (state.n_beliefs - 1)


class WeightedClustering(Metric):
    """clust: mean over BN nodes of the weighted clustering coefficient, as legacy computes it.

    Node i gets Σ_jk (ŵ_ij ŵ_jk ŵ_ki)^(1/3) / (k_i (k_i − 1)) with ŵ = |w| / max |w| and k_i the number of
    nonzero edges of i. Nodes with k_i < 2 are skipped. An agent with no node left (e.g. an all-zero BN) gets
    NaN, as in legacy.
    """

    name = "clust"

    def compute(self, ctx):
        abs_w = np.abs(ctx.model.state.edge_weights)
        w_max = abs_w.max(axis=(1, 2), keepdims=True)
        w_cbrt = np.divide(abs_w, w_max, out=np.zeros_like(abs_w), where=w_max > 0) ** (1 / 3)
        cycles = np.einsum("aij,ajk,aki->ai", w_cbrt, w_cbrt, w_cbrt)
        degree = (abs_w > 0).sum(axis=2)
        pairs = degree * (degree - 1)
        valid = pairs > 0
        node_clust = np.divide(cycles, pairs, out=np.zeros_like(cycles), where=valid)
        n_valid = valid.sum(axis=1)
        return np.divide(
            node_clust.sum(axis=1), n_valid, out=np.full(len(node_clust), np.nan), where=n_valid > 0
        )


class ExpectedInfluence(Metric):
    """expI: Σ_j w_foc,j x_j, the focal belief's edge-weighted pull from the other beliefs."""

    name = "expI"

    def compute(self, ctx):
        return _focal_field(ctx.model.state)


def default_metrics() -> list[Metric]:
    """Fresh instances of the 14 legacy metrics, in legacy's column order."""
    return [
        Degree(), PersonalEnergy(), FocalPersonalEnergy(), SocialEnergy(), ExternalEnergy(), TriangleBalance(),
        FocalTriangleBalance(), MeanAbsEdge(), FocalMeanAbsEdge(), WeightedClustering(), FocalBetweenness(),
        ExpectedInfluence(), FocalBelief(), NonFocalExtremity(),
    ]


RESPONSE_COLUMNS = ("persistPos", "nonpersistPos", "compliant", "resilient", "resistant", "latecompliant")


class ResponseClassifier:
    """Labels each agent's response to the pressure from its focal-belief means over three windows.

    The rules are legacy's, unchanged (known issue #9): a mean of exactly 0 counts as positive, except in
    `latecompliant`, which needs `after > 0`, so before < 0, during < 0, after = 0 gets no label.
    """

    def __init__(self, before: range, during: range, after: range, column: str = "x_focal"):
        self.before, self.during, self.after = before, during, after
        self.column = column

    @classmethod
    def from_windows(cls, windows: dict[str, range], column: str = "x_focal") -> ResponseClassifier:
        return cls(windows["before"], windows["during"], windows["after"], column)

    def classify(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One row per agent (indexed by `id`), one bool column per response."""
        if self.column not in frame:
            raise KeyError(f"classification needs the {self.column!r} column; add the FocalBelief metric")
        pivot = frame.pivot_table(index="t", columns="id", values=self.column)
        before, during, after = (pivot.loc[pivot.index.isin(w)].mean(axis=0)
                                 for w in (self.before, self.during, self.after))
        out = pd.DataFrame({
            "persistPos": (before >= 0) & (during >= 0) & (after >= 0),
            "nonpersistPos": (before >= 0) & (after < 0),
            "compliant": (before < 0) & (during >= 0) & (after >= 0),
            "resilient": (before < 0) & (during >= 0) & (after < 0),
            "resistant": (before < 0) & (during < 0) & (after < 0),
            "latecompliant": (before < 0) & (during < 0) & (after > 0),
        })
        out.index = out.index.astype(int)
        return out


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
