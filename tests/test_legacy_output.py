"""Drop-in legacy output: `SimulationResult.to_frame()` and `save()` against CSVs written by legacy `run_one`.

The legacy side is real `run_one` + `get_output`; only `run_simulation`'s dynamics are stubbed, with zero
snapshots at legacy's tracked steps, so the reference CSV is legacy-produced (path, index column and all).
"""

import numpy as np
import pandas as pd
import pytest

from anb.agents import ANBAgent
from anb.metrics import RESPONSE_COLUMNS, default_metrics
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig
from legacy_oracle import load_legacy

N = 5
LEGACY_PARAMS = ["link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100", "ext_strength", "seed"]
COMBOS = [  # link_prob, init_w, beta, eps, mu, fixedBNat100 (rho = 1/3)
    (0.1, 0.2, 3.0, 1.0, 0.0, False),
    (0.1, 0.2, 3.0, 1.0, 0.025, True),
]


def legacy_csv(tmp_path, monkeypatch, combo, pressure, seed, detail, two_events=False):
    """`(relative path, frame)` of the CSV legacy `run_one` writes, with the dynamics stubbed out."""
    legacy = load_legacy(detail=detail, two_events=two_events)

    def run_simulation(params):
        # legacy records each t once if it is in track_times (which lists t = 100 twice) or is the last step
        times = sorted({*legacy.track_times, legacy.T})
        snapshots = np.zeros((len(times) * legacy.n_agents, legacy.columnIndex + legacy.M + len(legacy.edge_list)))
        snapshots[:, 0] = np.repeat(times, legacy.n_agents)
        snapshots[:, 1] = np.tile(np.arange(legacy.n_agents), len(times))
        return legacy.get_output(snapshots)

    legacy.run_simulation = run_simulation
    link_prob, init_w, beta, eps, mu, fixed = combo
    out = tmp_path / "legacy"
    out.mkdir()
    monkeypatch.chdir(out)
    legacy.run_one(seed, link_prob, init_w, beta, 1 / 3, eps, mu, fixed, pressure)
    (path,) = [p.relative_to(out) for p in out.rglob("*.csv")]
    return path, pd.read_csv(out / path)


def make_spec(combo, n_agents=N, **overrides):
    link_prob, init_w, beta, eps, mu, fixed = combo
    agent = {"beta": beta, "eps": eps, "mu": mu, "init_w": init_w, "learning_until": 101 if fixed else None}
    return ModelSpec(ANBAgent(**(agent | overrides)), ErdosRenyiNetwork, {"link_prob": link_prob},
                     n_agents=n_agents, rho=1 / 3)


def anb_result(combo, pressure, seed, detail, two_events=False, **overrides):
    schedule = (PressureSchedule.two_events if two_events else PressureSchedule.single_event)(pressure)
    return Testbed(make_spec(combo, **overrides), TestbedConfig(schedule, detail=detail)).run(seed)


def test_default_metrics_are_the_14_legacy_columns_in_legacy_order():
    legacy = load_legacy()

    assert [m.name for m in default_metrics()] == legacy.metric_cols
    assert [m.name for m in Testbed(make_spec(COMBOS[0]), TestbedConfig(PressureSchedule.single_event(1))).metrics] \
        == legacy.metric_cols


@pytest.mark.parametrize("detail, two_events", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("combo", COMBOS)
def test_saved_csv_matches_legacy_path_columns_and_order(combo, detail, two_events, tmp_path, monkeypatch):
    expected_path, expected = legacy_csv(tmp_path, monkeypatch, combo, 4, 7, detail, two_events)
    result = anb_result(combo, 4, 7, detail, two_events)

    path = result.save(tmp_path / "anb")

    assert path == tmp_path / "anb" / expected_path
    got = pd.read_csv(path)
    assert list(got.columns) == list(expected.columns)
    assert list(result.to_frame().columns) == list(expected.columns)[1:]  # legacy's first column is the index
    legacy_n = len(expected) // expected.t.nunique()
    assert list(got.t[::N]) == list(expected.t[::legacy_n])  # the same t blocks, in the same order
    assert (got.groupby("t").size() == N).all()


@pytest.mark.parametrize("detail", [False, True])
def test_homogeneous_parameter_columns_equal_legacy_and_concatenate_cleanly(detail, tmp_path, monkeypatch):
    combo = COMBOS[1]
    _, expected = legacy_csv(tmp_path, monkeypatch, combo, 4, 7, detail)
    got = pd.read_csv(anb_result(combo, 4, 7, detail).save(tmp_path / "anb"))

    for name in LEGACY_PARAMS:
        assert got[name].dtype == expected[name].dtype, name
        assert got[name].unique().tolist() == expected[name].unique().tolist(), name
    both = pd.concat([expected, got], ignore_index=True)
    assert list(both.columns) == list(expected.columns)
    assert both[LEGACY_PARAMS].notna().all().all()


@pytest.mark.parametrize("detail", [False, True])
def test_per_agent_beta_becomes_a_per_agent_column(detail):
    betas = np.linspace(1.0, 4.0, N)
    frame = anb_result(COMBOS[0], 2, 3, detail, beta=betas).to_frame()

    np.testing.assert_array_equal(frame.beta.to_numpy(), betas[frame.id.to_numpy()])
    assert list(frame.columns)[-len(LEGACY_PARAMS):] == LEGACY_PARAMS


def test_scalar_lam_stays_in_the_filename_and_per_agent_lam_becomes_a_column(tmp_path):
    scalar = anb_result(COMBOS[0], 2, 3, False, lam=0.05)
    per_agent = anb_result(COMBOS[0], 2, 3, False, lam=np.linspace(0, 0.1, N))

    assert "lam" not in scalar.to_frame()
    assert scalar.save(tmp_path).name.endswith("_lambda0.0500.csv")
    frame = per_agent.to_frame()
    np.testing.assert_array_equal(frame.lam.to_numpy(), np.linspace(0, 0.1, N)[frame.id.to_numpy()])


def test_fixedBNat100_column_follows_learning_until():
    assert anb_result(COMBOS[0], 2, 3, False).to_frame().fixedBNat100.eq(False).all()
    assert anb_result(COMBOS[0], 2, 3, False, learning_until=101).to_frame().fixedBNat100.eq(True).all()


def test_legacy_names_need_uniform_parameters(tmp_path):
    result = anb_result(COMBOS[0], 2, 3, False, beta=np.linspace(1.0, 4.0, N))

    with pytest.raises(ValueError, match="beta"):
        result.save(tmp_path)
    path = result.save(tmp_path, legacy_names=False)
    pd.testing.assert_frame_equal(pd.read_csv(path, index_col=0), result.to_frame())  # tolerant: CSV floats round-trip to the last digit only


def test_to_frame_leaves_the_result_frame_untouched():
    result = anb_result(COMBOS[0], 2, 3, False)
    before = result.frame.copy()

    result.to_frame()

    assert result.frame.equals(before)
    assert not set(LEGACY_PARAMS) & set(result.frame.columns)
    assert list(result.frame.columns)[-len(RESPONSE_COLUMNS):] == list(RESPONSE_COLUMNS)
