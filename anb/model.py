"""What to simulate: `ModelSpec` (a picklable factory) and `ANBModel` (agents on a network)."""

from dataclasses import dataclass, field

import numpy as np

from .agents import PER_AGENT_PARAMS, Agent, AgentState
from .io import LEGACY_LEARNING_UNTIL
from .networks import Network


class ANBModel:
    """Agents on a network. Knows nothing about windows, protocols or metrics."""

    def __init__(self, agent: Agent, network: Network, state: AgentState, rho: float):
        self.agent = agent
        self.network = network
        self.state = state
        self.rho = rho

    def step(self, t: int, ext: float, rng: np.random.Generator) -> None:
        self.agent.sweep(self.state, self.network.adjacency, t, ext, rng, self.rho)


@dataclass
class ModelSpec:
    agent: Agent
    network_cls: type[Network]
    network_kwargs: dict = field(default_factory=dict)
    n_agents: int = 100
    rho: float = 1 / 3

    def build(self, rng: np.random.Generator) -> ANBModel:
        n = self.n_agents
        self.agent.param_arrays(n)  # raises on per-agent parameters of the wrong length
        state = self.agent.init_state(n, rng)
        network = self.network_cls.build(n, rng, **self.network_kwargs)
        return self.checked_model(state, network)

    def checked_model(self, state: AgentState, network: Network) -> ANBModel:
        """`ANBModel` of `state` on `network`, after checking both against `n_agents`."""
        n = self.n_agents
        if state.n_agents != n:
            raise ValueError(f"agent state has {state.n_agents} agents, spec has n_agents={n}")
        if network.adjacency.shape != (n, n):
            raise ValueError(f"network adjacency has shape {network.adjacency.shape}, expected ({n}, {n})")
        return ANBModel(self.agent, network, state, self.rho)

    def params(self) -> dict:
        """Output-column parameters: scalars for uniform parameters, `(N,)` arrays for per-agent ones.

        `fixedBNat100` is true only for `learning_until = 101`, the gate legacy's flag means. Any other gate
        gets its own `learning_until` column, and a `fixed` mask a per-agent `fixed` column.
        """
        out = dict(self.network_kwargs)
        for name in PER_AGENT_PARAMS:
            value = getattr(self.agent, name)
            out[name] = value if np.ndim(value) == 0 else np.asarray(value)
        out["rho"] = self.rho
        learning_until = self.agent.learning_until
        out["fixedBNat100"] = learning_until == LEGACY_LEARNING_UNTIL
        if learning_until not in (None, LEGACY_LEARNING_UNTIL):
            out["learning_until"] = learning_until
        if self.agent.fixed is not None:
            out["fixed"] = np.asarray(self.agent.fixed)
        return out
