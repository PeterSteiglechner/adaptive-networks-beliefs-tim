"""Seam 1 (detail mode): `Testbed.run(seed)` drives every layer and returns a `SimulationResult`."""

import pickle
import subprocess
import sys

import numpy as np
import pytest

from anb.agents import ANBAgent
from anb.metrics import default_metrics
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig

N = 12
GRID = np.linspace(-1, 1, 21)
EDGE_COLS = [f"w{m}{n}" for m in range(10) for n in range(m + 1, 10)]


def make_spec(**agent_kwargs):
    agent_kwargs = {"beta": 3.0, "eps": 1.0, "init_w": 0.2, **agent_kwargs}
    return ModelSpec(
        agent=ANBAgent(**agent_kwargs),
        network_cls=ErdosRenyiNetwork,
        network_kwargs={"link_prob": 0.3},
        n_agents=N,
        rho=1 / 3,
    )


def short_config(strength=2.0):
    """A short detail-mode protocol: pressure on t = 4…6, T = 9."""
    return TestbedConfig(PressureSchedule(strength, [(4, 6)]), window_len=2, recovery=3, detail=True)


def run(spec=None, config=None, seed=0):
    return Testbed(spec or make_spec(), config or short_config()).run(seed)


def test_detail_frame_has_every_step_and_the_expected_columns():
    result = run(seed=3)
    frame = result.frame

    dummies = ["persistPos", "nonpersistPos", "compliant", "resilient", "resistant", "latecompliant"]
    metrics = [m.name for m in default_metrics()]
    assert list(frame.columns) == ["t", "id"] + metrics + dummies + [str(m) for m in range(10)] + EDGE_COLS
    assert sorted(frame.t.unique()) == list(range(10))
    assert all(sorted(g.id) == list(range(N)) for _, g in frame.groupby("t"))
    assert result.seed == 3


def test_beliefs_stay_on_the_grid_and_beliefs_actually_move():
    frame = run(make_spec(eps=1.0, lam=0.05), seed=1).frame
    beliefs = frame[[str(m) for m in range(10)]].to_numpy()

    assert np.isin(beliefs, GRID).all()
    first, last = (frame[frame.t == t][[str(m) for m in range(10)]].to_numpy() for t in (0, 9))
    assert not np.array_equal(first, last)


def test_edge_matrices_stay_symmetric_with_zero_diagonal():
    spec = make_spec(eps=1.0, mu=0.3, lam=0.05)
    model = spec.build(np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for t in range(1, 6):
        model.step(t, 2.0, rng)
    w = model.state.edge_weights

    assert not np.allclose(w, 0.2 * (1 - np.eye(10)))  # the edges did learn
    np.testing.assert_array_equal(w, w.transpose(0, 2, 1))
    assert (np.diagonal(w, axis1=1, axis2=2) == 0).all()


def test_fixed_agents_never_change_beliefs_or_edges():
    fixed = np.zeros(N, dtype=bool)
    fixed[[0, 5, 7]] = True
    frame = run(make_spec(eps=1.0, mu=0.3, lam=0.05, fixed=fixed), seed=2).frame
    cols = [str(m) for m in range(10)] + EDGE_COLS

    for i in range(N):
        rows = frame[frame.id == i][cols].to_numpy()
        changed = not (rows == rows[0]).all()
        assert changed != fixed[i], f"agent {i} (fixed={fixed[i]})"


def test_edges_stop_changing_from_learning_until_on_including_decay():
    frame = run(make_spec(eps=1.0, mu=0.3, lam=0.1, learning_until=5), seed=4).frame
    edges = {t: g.sort_values("id")[EDGE_COLS].to_numpy() for t, g in frame.groupby("t")}

    assert not np.array_equal(edges[3], edges[4])  # still learning at t = 4
    for t in range(5, 10):
        np.testing.assert_array_equal(edges[t], edges[4])


def test_same_seed_gives_identical_frame_and_different_seeds_differ():
    spec, config = make_spec(mu=0.2, lam=0.05), short_config()

    a, b = run(spec, config, seed=7).frame, run(spec, config, seed=7).frame
    c = run(spec, config, seed=8).frame

    assert a.equals(b)
    assert not a.equals(c)


def test_positive_pressure_pushes_focal_beliefs_up():
    frames = [run(make_spec(), short_config(strength=8.0), seed=s).frame for s in range(5)]
    mean_focal = np.mean([f.groupby("t").x_focal.mean() for f in frames], axis=0)

    assert mean_focal[6] > 0.5
    assert mean_focal[6] - mean_focal[3] > 0.3  # t = 3 is before the event (t = 4…6)


def test_mean_degree_matches_link_probability():
    n, p = 100, 0.1
    spec = ModelSpec(ANBAgent(), ErdosRenyiNetwork, {"link_prob": p}, n_agents=n)
    degrees = [spec.build(np.random.default_rng(s)).network.degree() for s in range(10)]

    assert np.mean(degrees) == pytest.approx(p * (n - 1), rel=0.05)


def test_n_nbs_column_is_the_network_degree():
    frame = run(seed=6).frame

    assert (frame.groupby("id").n_nbs.nunique() == 1).all()
    assert frame.n_nbs.sum() > 0


@pytest.mark.parametrize("param", ["beta", "eps", "mu", "lam", "init_w", "fixed"])
def test_build_rejects_per_agent_parameters_of_the_wrong_length(param):
    value = np.zeros(N - 1, dtype=bool) if param == "fixed" else np.full(N - 1, 0.5)
    spec = make_spec(**{param: value})

    with pytest.raises(ValueError, match=param):
        spec.build(np.random.default_rng(0))


def test_build_accepts_per_agent_parameters_of_the_right_length():
    spec = make_spec(beta=np.linspace(1, 4, N), fixed=np.arange(N) < 2)

    assert spec.build(np.random.default_rng(0)).state.n_agents == N


class WrongSizeNetwork(ErdosRenyiNetwork):
    @classmethod
    def build(cls, n_agents, rng, link_prob):
        return super().build(n_agents + 1, rng, link_prob)


class WrongSizeAgent(ANBAgent):
    def init_state(self, n_agents, rng):
        return super().init_state(n_agents + 1, rng)


@pytest.mark.parametrize("agent_cls, network_cls", [(ANBAgent, WrongSizeNetwork), (WrongSizeAgent, ErdosRenyiNetwork)])
def test_build_rejects_agent_and_network_sizes_that_disagree(agent_cls, network_cls):
    spec = ModelSpec(agent_cls(), network_cls, {"link_prob": 0.3}, n_agents=N)

    with pytest.raises(ValueError, match="expected|n_agents"):
        spec.build(np.random.default_rng(0))


def test_model_spec_round_trips_through_pickle():
    spec = make_spec(beta=np.linspace(1, 4, N), learning_until=5)

    clone = pickle.loads(pickle.dumps(spec))

    assert run(clone, seed=9).frame.equals(run(spec, seed=9).frame)


def test_package_runs_from_a_plain_module(tmp_path):
    """Nothing in `anb` reads names from `__main__` (legacy known issue #7)."""
    (tmp_path / "experiment.py").write_text(
        "from anb.agents import ANBAgent\n"
        "from anb.model import ModelSpec\n"
        "from anb.networks import ErdosRenyiNetwork\n"
        "from anb.testbed import PressureSchedule, Testbed, TestbedConfig\n"
        "def main():\n"
        "    spec = ModelSpec(ANBAgent(eps=1.0), ErdosRenyiNetwork, {'link_prob': 0.3}, n_agents=5)\n"
        "    config = TestbedConfig(PressureSchedule(1.0, [(2, 3)]), window_len=1, recovery=1, detail=True)\n"
        "    return len(Testbed(spec, config).run(0).frame)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", "import experiment; print(experiment.main())"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )

    assert out.stdout.strip() == str(5 * 5)
