"""Agent types (update rule plus parameters) and the population's state as plain arrays."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

PER_AGENT_PARAMS = ("beta", "eps", "mu", "lam", "init_w")


def edge_columns(n_beliefs: int) -> list[str]:
    """Legacy edge column names, in the order of `AgentState.edge_vector()`."""
    return [f"w{m}{n}" for m, n in zip(*np.triu_indices(n_beliefs, k=1))]


@dataclass
class AgentState:
    """The population's data. Shapes: `beliefs (N, M)`, `edge_weights (N, M, M)`, `delta_memory (N, tau, M)`.

    `edge_weights` is symmetric with a zero diagonal. `delta_memory[:, 0]` is the latest belief change.
    """

    beliefs: np.ndarray
    edge_weights: np.ndarray
    delta_memory: np.ndarray
    focal: np.ndarray

    @property
    def n_agents(self) -> int:
        return self.beliefs.shape[0]

    @property
    def n_beliefs(self) -> int:
        return self.beliefs.shape[1]

    def edge_vector(self) -> np.ndarray:
        """`(N, M(M-1)/2)`: the upper triangle in legacy `w{m}{n}` order (`w01`, `w02`, …, `w89`)."""
        m, n = np.triu_indices(self.n_beliefs, k=1)
        return self.edge_weights[:, m, n]

    def focal_beliefs(self) -> np.ndarray:
        return self.beliefs[np.arange(self.n_agents), self.focal]


class Agent(ABC):
    """An agent type: the update rule and its parameters.

    `beta`, `eps`, `mu`, `lam` and `init_w` are scalars or `(N,)` arrays. `fixed` is a bool `(N,)` mask of
    agents that never update. The edge update is skipped entirely once `t >= learning_until`.
    """

    def __init__(
        self,
        *,
        beta=3.0,
        eps=0.0,
        mu=0.0,
        lam=0.0,
        init_w=0.2,
        fixed=None,
        learning_until: int | None = None,
        tau: int = 1,
        n_beliefs: int = 10,
        belief_options=None,
    ):
        if not isinstance(tau, (int, np.integer)) or tau < 1:
            raise ValueError(f"tau must be a positive int, got {tau!r}")
        self.beta = beta
        self.eps = eps
        self.mu = mu
        self.lam = lam
        self.init_w = init_w
        self.fixed = None if fixed is None else np.asarray(fixed, dtype=bool)
        self.learning_until = learning_until
        self.tau = int(tau)
        self.n_beliefs = n_beliefs
        self.belief_options = np.linspace(-1, 1, 21) if belief_options is None else np.asarray(belief_options)

    def param_arrays(self, n_agents: int) -> dict[str, np.ndarray]:
        """Every per-agent parameter broadcast to `(N,)`. Raises if an array's length isn't `n_agents`."""
        out = {}
        for name in PER_AGENT_PARAMS:
            out[name] = self._broadcast(name, getattr(self, name), n_agents, float)
        fixed = np.zeros(n_agents, dtype=bool) if self.fixed is None else self.fixed
        out["fixed"] = self._broadcast("fixed", fixed, n_agents, bool)
        return out

    @staticmethod
    def _broadcast(name, value, n_agents, dtype):
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            return np.full(n_agents, arr, dtype=dtype)
        if arr.shape != (n_agents,):
            raise ValueError(f"per-agent parameter {name!r} has shape {arr.shape}, expected ({n_agents},)")
        return arr

    @abstractmethod
    def init_state(self, n_agents: int, rng: np.random.Generator) -> AgentState: ...

    @abstractmethod
    def sweep(self, state: AgentState, adjacency: np.ndarray, t: int, ext: float, rng: np.random.Generator,
              rho: float) -> None:
        """One whole random-sequential time step, mutating `state` in place."""


class ANBAgent(Agent):
    """The paper's agent: Glauber belief updates (eq. 2) and Hebbian + social + decay edge updates (eq. 3)."""

    def init_state(self, n_agents, rng):
        m = self.n_beliefs
        init_w = self.param_arrays(n_agents)["init_w"]
        edges = np.broadcast_to(init_w[:, None, None], (n_agents, m, m)).copy()
        edges[:, np.arange(m), np.arange(m)] = 0.0
        return AgentState(
            beliefs=self.belief_options[rng.integers(len(self.belief_options), size=(n_agents, m))],
            edge_weights=edges,
            delta_memory=np.zeros((n_agents, self.tau, m)),
            focal=np.zeros(n_agents, dtype=np.int64),
        )

    def sweep(self, state, adjacency, t, ext, rng, rho):
        p = self.param_arrays(state.n_agents)
        learning = self.learning_until is None or t < self.learning_until
        _sweep(
            state.beliefs, state.edge_weights, state.delta_memory, state.focal, adjacency,
            self.belief_options, p["beta"], p["eps"], p["mu"], p["lam"], p["fixed"],
            float(rho), float(ext), learning, rng,
        )


def _sweep(beliefs, edges, memory, focal, adjacency, options, beta, eps, mu, lam, fixed, rho, ext, learning, rng):
    """The sweep kernel on plain arrays (to become `@njit` in ticket 10).

    Agents update in shuffled order and see neighbours already updated in this step. The social input is
    ρ · Σ_k A_ik x_foc(k); the social edge term uses the A-weighted mean of the neighbours' edge matrices.
    """
    n_agents, m = beliefs.shape
    for i in rng.permutation(n_agents):
        if fixed[i]:
            continue
        nbs = np.flatnonzero(adjacency[i])
        f = focal[i]
        social = 0.0
        for k in nbs:
            social += adjacency[i, k] * beliefs[k, focal[k]]

        prior = beliefs[i].copy()
        for d in rng.permutation(m):
            field = edges[i, d] @ beliefs[i]  # zero diagonal: only the other dimensions contribute
            if d == f:
                field += ext + rho * social
            dH = -(options - beliefs[i, d]) * field
            prob = 0.5 * (1.0 + np.tanh(-0.5 * beta[i] * dH))  # = 1 / (1 + exp(beta * dH)), overflow-free
            cdf = np.cumsum(prob)
            idx = min(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"), len(options) - 1)
            beliefs[i, d] = options[idx]
        memory[i, 1:] = memory[i, :-1]
        memory[i, 0] = beliefs[i] - prior

        if learning:
            delta = memory[i].mean(axis=0)
            hebb = np.outer(delta, delta)
            np.fill_diagonal(hebb, 0.0)
            if len(nbs) > 0:
                w = adjacency[i, nbs]
                social_edges = np.tensordot(w, edges[nbs], axes=1) / w.sum()
                mu_i = mu[i]
            else:
                social_edges = edges[i]
                mu_i = 0.0
            edges[i] += eps[i] * hebb + mu_i * (social_edges - edges[i]) - lam[i] * edges[i]
