"""Seam B (shared start): legacy and anb from one identical t = 0 state, K replicas each, every step of a window."""

import numpy as np
import pytest

import legacy_oracle
from anb.comparison import (METRICS, build_shared_start, legacy_rows_from_state, run_shared_start,
                            state_from_legacy_rows)
from anb.testbed import PressureSchedule

CELL = dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=0.33, eps=1.0, mu=0.0, fixedBNat100=False, lam=0.0)
SCHEDULE = PressureSchedule(2.0, [(2, 3)])  # a short protocol: pressure on t = 2…3
WINDOW = (1, 5)


@pytest.fixture(scope="module")
def trajectories():
    return run_shared_start(CELL, 2.0, start_seed=5, k=2, window=WINDOW, schedule=SCHEDULE,
                            load_legacy=legacy_oracle.load_legacy)


def test_the_legacy_row_conversion_round_trips_and_the_oracle_uses_it(legacy):
    agents, nb_list = legacy_oracle.random_legacy_state(legacy, seed=3)
    agents[:, list(legacy.delbeliefids)] = np.random.default_rng(0).normal(size=(legacy.n_agents, legacy.M))

    state = state_from_legacy_rows(legacy, agents)
    again = legacy_rows_from_state(legacy, state)

    columns = [*legacy.beliefids, *legacy.edgeids, *legacy.delbeliefids]
    np.testing.assert_array_equal(again[:, columns], agents[:, columns])
    assert legacy_oracle.state_from_legacy_rows is state_from_legacy_rows


def test_both_implementations_report_identical_t0_metrics_from_the_shared_state(trajectories):
    t0 = trajectories[trajectories.t == 0]

    legacy_rows, anb_rows = (t0[t0.implementation == name].sort_values(["replica", "agent"]) for name in ("legacy", "anb"))
    assert len(legacy_rows) == len(anb_rows) == 2 * 100
    for metric in METRICS:
        np.testing.assert_allclose(legacy_rows[metric].to_numpy(), anb_rows[metric].to_numpy(), atol=1e-9,
                                   err_msg=metric)
    first, second = (legacy_rows[legacy_rows.replica == r].x_focal.to_numpy() for r in (0, 1))
    np.testing.assert_array_equal(first, second)  # replicas share the start


@pytest.mark.parametrize("implementation", ["legacy", "anb"])
def test_replicas_within_an_implementation_differ_after_t0(trajectories, implementation):
    last = trajectories[(trajectories.implementation == implementation) & (trajectories.t == WINDOW[1])]

    first, second = (last[last.replica == r].sort_values("agent").x_focal.to_numpy() for r in (0, 1))
    assert not np.array_equal(first, second)


def test_the_trajectory_frame_covers_t0_and_every_step_of_the_window(trajectories):
    assert list(trajectories.columns) == ["implementation", "replica", "t", "agent", *METRICS]
    for (implementation, replica), run in trajectories.groupby(["implementation", "replica"]):
        assert sorted(run.t.unique()) == [0, *range(WINDOW[0], WINDOW[1] + 1)]
        assert all(sorted(g.agent) == list(range(100)) for _, g in run.groupby("t"))
    assert set(trajectories.implementation) == {"legacy", "anb"}
    assert set(trajectories.replica) == {0, 1}


def test_the_start_state_depends_on_the_start_seed_only():
    a, b, c = (build_shared_start(CELL, seed) for seed in (1, 1, 2))

    np.testing.assert_array_equal(a[0].beliefs, b[0].beliefs)
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0].beliefs, c[0].beliefs)


def test_running_leaves_legacy_on_disk_unchanged():
    before = legacy_oracle.LEGACY_PATH.read_bytes()
    run_shared_start(CELL, 0.0, k=1, window=(1, 2), schedule=SCHEDULE, load_legacy=legacy_oracle.load_legacy)
    assert legacy_oracle.LEGACY_PATH.read_bytes() == before
