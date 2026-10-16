"""Social networks as `(N, N)` adjacency matrices (0/1 or weighted)."""

from abc import ABC, abstractmethod

import numpy as np


class Network(ABC):
    """A static social network. Subclasses only implement `build`."""

    def __init__(self, adjacency: np.ndarray):
        self.adjacency = adjacency

    @property
    def n_agents(self) -> int:
        return self.adjacency.shape[0]

    def neighbours(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.adjacency[i])

    def degree(self) -> np.ndarray:
        """Number of neighbours per agent (unweighted)."""
        return np.count_nonzero(self.adjacency, axis=1)

    @classmethod
    @abstractmethod
    def build(cls, n_agents: int, rng: np.random.Generator, **kwargs) -> "Network": ...


class ErdosRenyiNetwork(Network):
    """G(N, p): each pair is linked independently with probability `link_prob`."""

    def __init__(self, adjacency, link_prob: float):
        super().__init__(adjacency)
        self.link_prob = link_prob

    @classmethod
    def build(cls, n_agents, rng, link_prob):
        upper = np.triu(rng.random((n_agents, n_agents)) <= link_prob, k=1)
        return cls((upper | upper.T).astype(float), link_prob)
