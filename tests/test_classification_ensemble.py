"""Response classification (seam 2), window aggregation and ensemble mode (seam 1)."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from anb.agents import ANBAgent
from anb.metrics import RESPONSE_COLUMNS, ResponseClassifier
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig, WindowAggregator
from legacy_oracle import load_legacy

N = 6
BELIEF_COLS = [str(m) for m in range(10)]


def make_spec(n_agents=N, **agent_kwargs):
    agent_kwargs = {"beta": 3.0, "eps": 1.0, "mu": 0.2, "init_w": 0.2, **agent_kwargs}
    return ModelSpec(ANBAgent(**agent_kwargs), ErdosRenyiNetwork, {"link_prob": 0.4}, n_agents=n_agents)


def short_config(detail=False, **kwargs):
    """Pressure on t = 4…6; with window_len 2 and recovery 3: before 2–3, during 5–6, after 8–9, T = 9."""
    kwargs = {"window_len": 2, "recovery": 3, **kwargs}
    return TestbedConfig(PressureSchedule(4.0, [(4, 6)]), detail=detail, **kwargs)


# ResponseClassifier against legacy, on hand-built frames.

LEGACY_TIMES = list(range(201))
WINDOW_LEVELS = (-0.1, 0.0, 0.1)


def hand_built_focal(seed: int) -> np.ndarray:
    """`(T + 1, 100)` focal beliefs on the grid. The first 27 agents hold one constant value per window,
    covering every sign combination including exact zeros; the rest are random grid values."""
    rng = np.random.default_rng(seed)
    x = rng.choice(np.linspace(-1, 1, 21), size=(len(LEGACY_TIMES), 100))
    windows = (range(91, 101), range(141, 151), range(191, 201))
    for agent, levels in enumerate(product(WINDOW_LEVELS, repeat=3)):
        for window, level in zip(windows, levels):
            x[list(window), agent] = level
    return x


def legacy_dummies(focal: np.ndarray) -> pd.DataFrame:
    """Legacy detail-mode `get_output` on a snapshot array holding `focal` as belief 0."""
    legacy = load_legacy(detail=True)
    n = focal.shape[1]
    snapshots = np.zeros((focal.size, legacy.columnIndex + legacy.M + len(legacy.edge_list)))
    snapshots[:, 0] = np.repeat(LEGACY_TIMES, n)
    snapshots[:, 1] = np.tile(np.arange(n), len(LEGACY_TIMES))
    snapshots[:, legacy.beliefids[legacy.focal]] = focal.ravel()
    out = legacy.get_output(snapshots)
    at_end = out[out.t == LEGACY_TIMES[-1]].astype({"id": int})
    return at_end.set_index("id")[list(RESPONSE_COLUMNS)]


def anb_frame(focal: np.ndarray) -> pd.DataFrame:
    n = focal.shape[1]
    return pd.DataFrame({
        "t": np.repeat(LEGACY_TIMES, n),
        "id": np.tile(np.arange(n), len(LEGACY_TIMES)),
        "x_focal": focal.ravel(),
    })


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classifier_equals_legacy_including_zero_and_boundary_cases(seed):
    focal = hand_built_focal(seed)
    windows = TestbedConfig(PressureSchedule.single_event(1.0)).windows()

    ours = ResponseClassifier.from_windows(windows).classify(anb_frame(focal))

    expected = legacy_dummies(focal)
    assert list(ours.columns) == list(RESPONSE_COLUMNS)
    pd.testing.assert_frame_equal(ours.astype(int), expected.astype(int), check_names=False)


def test_classifier_keeps_the_legacy_zero_quirks():
    """Known issue #9: a zero before-mean is aligned; before < 0, during < 0, after = 0 gets no label."""
    frame = pd.DataFrame({"t": [0, 1, 2, 0, 1, 2], "id": [0, 0, 0, 1, 1, 1],
                          "x_focal": [0.0, -0.1, -0.1, -0.1, -0.1, 0.0]})
    out = ResponseClassifier(before=range(0, 1), during=range(1, 2), after=range(2, 3)).classify(frame)

    assert out.loc[0, "nonpersistPos"] and out.loc[0].sum() == 1
    assert out.loc[1].sum() == 0  # unlabelled


# Ensemble mode through Testbed.run.


@pytest.mark.parametrize("schedule, times", [
    (PressureSchedule.single_event(2.0), [0, 95.5, 100, 145.5, 195.5, 200]),
    (PressureSchedule.two_events(2.0), [0, 95.5, 100, 145.5, 195.5, 200, 245.5, 295.5, 300]),
])
def test_ensemble_row_times_are_snapshots_plus_window_midpoints(schedule, times):
    frame = Testbed(make_spec(), TestbedConfig(schedule)).run(0).frame

    assert sorted(frame.t.unique()) == times
    assert list(frame.t) == sorted(frame.t)
    assert all(sorted(g.id) == list(range(N)) for _, g in frame.groupby("t"))


def test_ensemble_dummies_appear_only_on_the_before_row():
    frame = Testbed(make_spec(), TestbedConfig(PressureSchedule.single_event(2.0))).run(1).frame
    dummies = frame[list(RESPONSE_COLUMNS)]
    on_before = frame.t == 95.5

    assert dummies[on_before].notna().all().all()
    assert dummies[~on_before].isna().all().all()
    assert (dummies[on_before].astype(int).sum(axis=1) <= 1).all()  # the labels are exclusive


def test_detail_dummies_appear_only_on_the_last_step():
    frame = Testbed(make_spec(), short_config(detail=True)).run(1).frame
    dummies = frame[list(RESPONSE_COLUMNS)]
    at_end = frame.t == 9

    assert dummies[at_end].notna().all().all()
    assert dummies[~at_end].isna().all().all()


@pytest.mark.parametrize("window_len, recovery", [(2, 3), (3, 5)])
def test_window_settings_move_classifier_and_aggregator_together(window_len, recovery):
    """Metrics draw no random numbers, so the same seed gives the same trajectory in both modes."""
    spec = make_spec()
    config = short_config(window_len=window_len, recovery=recovery)
    detail = Testbed(spec, short_config(detail=True, window_len=window_len, recovery=recovery)).run(5).frame
    ensemble = Testbed(spec, config).run(5).frame

    windows = config.windows()
    midpoints = {name: float(np.mean(w)) for name, w in windows.items()}
    assert sorted(ensemble.t.unique()) == sorted(set(config.snapshot_times()) | set(midpoints.values()))
    for name, window in windows.items():
        expected = detail[detail.t.isin(window)].groupby("id")[["x_focal", *BELIEF_COLS]].mean()
        got = ensemble[ensemble.t == midpoints[name]].set_index("id")[["x_focal", *BELIEF_COLS]]
        pd.testing.assert_frame_equal(got, expected, check_names=False)

    expected = ResponseClassifier.from_windows(windows).classify(detail)
    got = ensemble[ensemble.t == midpoints["before"]].set_index("id")[list(RESPONSE_COLUMNS)]
    pd.testing.assert_frame_equal(got.astype(int), expected.astype(int), check_names=False)
    at_end = detail[detail.t == config.T].set_index("id")[list(RESPONSE_COLUMNS)]
    pd.testing.assert_frame_equal(at_end.astype(int), expected.astype(int), check_names=False)


def test_aggregator_keeps_snapshot_rows_unchanged():
    detail = Testbed(make_spec(), short_config(detail=True)).run(2).frame.drop(columns=list(RESPONSE_COLUMNS))
    config = short_config()
    responses = ResponseClassifier.from_windows(config.windows()).classify(detail)

    out = WindowAggregator(config.windows(), config.snapshot_times()).aggregate(detail, responses)

    for t in config.snapshot_times():
        got = out[out.t == t].drop(columns=list(RESPONSE_COLUMNS)).reset_index(drop=True)
        expected = detail[detail.t == t].reset_index(drop=True).astype({"t": got.t.dtype})
        pd.testing.assert_frame_equal(got, expected)


def test_repeated_runs_share_no_metric_state():
    spec, config = make_spec(), short_config()
    testbed = Testbed(spec, config)

    first = testbed.run(3).frame
    testbed.run(4)
    again = testbed.run(3).frame

    assert again.equals(first)
    assert again.equals(Testbed(spec, config).run(3).frame)


@pytest.mark.parametrize("detail", [False, True])
def test_run_ensemble_in_parallel_matches_serial_runs(detail):
    testbed = Testbed(make_spec(), short_config(detail=detail))
    seeds = [0, 1, 2, 3]

    results = testbed.run_ensemble(seeds, n_jobs=2)

    assert [r.seed for r in results] == seeds
    for result, seed in zip(results, seeds):
        assert result.frame.equals(testbed.run(seed).frame)
    assert not results[0].frame.equals(results[1].frame)


def test_testbed_rejects_a_metric_list_without_the_classified_column():
    from anb.metrics import Degree

    with pytest.raises(ValueError, match="x_focal"):
        Testbed(make_spec(), short_config(), metrics=[Degree()])
