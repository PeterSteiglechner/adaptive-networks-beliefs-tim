"""Load the frozen legacy `ANBmodel.py` as a test oracle.

Legacy defines `detail` and `track_times` only under `if __name__ == "__main__":` but reads them in
`run_simulation` and `get_output` (known issue #7). `load_legacy` injects both into a freshly executed copy
of the module, so the file itself stays untouched.
"""

from pathlib import Path
from types import ModuleType

import numpy as np

LEGACY_PATH = Path(__file__).resolve().parent.parent / "ANBmodel.py"


def legacy_track_times(legacy: ModuleType, detail: bool):
    """`track_times` exactly as legacy's `__main__` block builds it."""
    if detail:
        return np.arange(legacy.T + 1)
    return (
        [0, 1]
        + legacy.beforeRange
        + [100]
        + legacy.duringRange
        + legacy.afterRange
        + [200]
        + ((legacy.during2Range + legacy.after2Range + [300]) if legacy.two_external_events else [])
    )


def load_legacy(detail: bool = False, two_events: bool = False) -> ModuleType:
    """Return a fresh, independent copy of legacy with `detail` and `track_times` set.

    Each call re-executes the file, so module state that legacy mutates (e.g. the shuffled `agentids`)
    doesn't leak between tests. `two_events` flips legacy's module constant `two_external_events` in the
    executed source, so the file itself stays untouched.
    """
    source = LEGACY_PATH.read_text()
    flag = "two_external_events = False"
    assert source.count(flag) == 1, "legacy no longer defines the two_external_events flag as expected"
    if two_events:
        source = source.replace(flag, "two_external_events = True")
    legacy = ModuleType("ANBmodel_legacy")
    legacy.__file__ = str(LEGACY_PATH)
    exec(compile(source, str(LEGACY_PATH), "exec"), legacy.__dict__)
    legacy.detail = detail
    legacy.track_times = legacy_track_times(legacy, detail)
    return legacy


# Seam 2: the same population state on both sides, for comparing `Metric.compute` with `fill_metrics`.


def random_legacy_state(legacy: ModuleType, seed: int, link_prob: float = 0.1, isolate=()):
    """A legacy `(agents, nb_list)` pair with random grid beliefs and random edges in [-1, 1].

    Agents in `isolate` lose all their links (on both ends), so every state has agents without neighbours.
    The `agents` metric columns are still NaN; `legacy_metrics` fills them.
    """
    np.random.seed(seed)
    agents = legacy.initialise_agents(0.0)
    agents[:, legacy.edgeids] = np.random.uniform(-1, 1, size=(legacy.n_agents, len(legacy.edgeids)))
    nb_list = legacy.initialise_network(seed, link_prob)
    cut = set(isolate)
    nb_list = [
        np.array([], dtype=int) if n in cut else nbs[~np.isin(nbs, list(cut))] for n, nbs in enumerate(nb_list)
    ]
    return agents, nb_list


def legacy_params(rho: float = 1 / 3, ext_strength: float = 0.0) -> dict:
    """The `params` entries legacy `fill_metrics` reads."""
    return {"rho": rho, "ext_strength": ext_strength}


def legacy_metrics(legacy: ModuleType, agents, nb_list, t: int, params: dict) -> dict[str, np.ndarray]:
    """Legacy `fill_metrics` on a copy of `agents`, as `{metric column: (N,) values}`."""
    filled = legacy.fill_metrics(t, agents.copy(), nb_list, params)
    return {name: filled[:, 2 + k] for k, name in enumerate(legacy.metric_cols)}


def to_context(legacy: ModuleType, agents, nb_list, t: int, params: dict):
    """The same state as an `anb` `StepContext`: `AgentState`, 0/1 adjacency, ρ, and the pressure at `t`."""
    from anb.agents import ANBAgent, AgentState
    from anb.metrics import StepContext
    from anb.model import ANBModel
    from anb.networks import ErdosRenyiNetwork

    n, m = legacy.n_agents, legacy.M
    edges = np.zeros((n, m, m))
    rows, cols = legacy.edge_arr.T
    edges[:, rows, cols] = agents[:, legacy.edgeids]
    edges[:, cols, rows] = agents[:, legacy.edgeids]
    state = AgentState(
        beliefs=agents[:, legacy.beliefids].copy(),
        edge_weights=edges,
        delta_memory=agents[:, list(legacy.delbeliefids)].reshape(n, legacy.tau, m).copy(),
        focal=np.full(n, legacy.focal, dtype=np.int64),
    )
    adjacency = np.zeros((n, n))
    for i, nbs in enumerate(nb_list):
        adjacency[i, nbs] = 1.0
    model = ANBModel(ANBAgent(), ErdosRenyiNetwork(adjacency, link_prob=np.nan), state, params["rho"])
    ext = params["ext_strength"] if t in legacy.ext_time else 0.0
    return StepContext(model, t, ext)


def legacy_and_anb(
    legacy: ModuleType, seed: int, t: int, ext_strength: float, rho: float = 1 / 3, isolate=(), edit=None
):
    """One random state on both sides: `(legacy metric columns, anb StepContext)`.

    `edit(agents)` may change the legacy state in place before either side sees it.
    """
    agents, nb_list = random_legacy_state(legacy, seed, isolate=isolate)
    if edit is not None:
        edit(agents)
    params = legacy_params(rho=rho, ext_strength=ext_strength)
    return legacy_metrics(legacy, agents, nb_list, t, params), to_context(legacy, agents, nb_list, t, params)


def refocus_randomly(state, rng) -> None:
    """Give each agent a random focal k, swapping belief dimensions 0 and k so the BN itself is unchanged.

    Every metric should then give the same values as before, if it follows each agent's `focal`.
    """
    focal = rng.integers(state.n_beliefs, size=state.n_agents)
    for i, k in enumerate(focal):
        perm = np.arange(state.n_beliefs)
        perm[[0, k]] = perm[[k, 0]]
        state.beliefs[i] = state.beliefs[i, perm]
        state.edge_weights[i] = state.edge_weights[i][np.ix_(perm, perm)]
    state.focal[:] = focal
