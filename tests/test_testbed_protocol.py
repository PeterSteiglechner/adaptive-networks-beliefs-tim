"""Seam 3: `PressureSchedule` and the values `TestbedConfig` derives from it match legacy."""

import numpy as np
import pytest

from anb.testbed import PressureSchedule, TestbedConfig
from legacy_oracle import load_legacy

SCHEDULES = {
    "single": (PressureSchedule.single_event, False),
    "two": (PressureSchedule.two_events, True),
}


@pytest.fixture(params=SCHEDULES)
def schedule_and_flag(request):
    make, two = SCHEDULES[request.param]
    return make(0.3), two


def test_strength_is_zero_outside_and_full_inside_inclusive():
    s = PressureSchedule.two_events(0.4)

    assert [s.strength_at(t) for t in (0, 100, 101, 150, 151, 200, 201, 250, 251, 300)] == [
        0, 0, 0.4, 0.4, 0, 0, 0.4, 0.4, 0, 0,
    ]


def test_factories_define_the_events():
    assert PressureSchedule.single_event(1).events == [(101, 150)]
    assert PressureSchedule.two_events(1).events == [(101, 150), (201, 250)]
    assert PressureSchedule.two_events(1).n_events == 2


def test_single_event_literal_values():
    cfg = TestbedConfig(PressureSchedule.single_event(0.3))

    assert cfg.T == 200
    assert cfg.windows() == {
        "before": range(91, 101),
        "during": range(141, 151),
        "after": range(191, 201),
    }
    assert cfg.snapshot_times() == [0, 100, 200]


def test_two_event_literal_values():
    cfg = TestbedConfig(PressureSchedule.two_events(0.3))

    assert cfg.T == 300
    assert cfg.windows()["during2"] == range(241, 251)
    assert cfg.windows()["after2"] == range(291, 301)
    assert cfg.snapshot_times() == [0, 100, 200, 300]


@pytest.mark.parametrize("detail", [False, True])
def test_derived_values_equal_legacy(schedule_and_flag, detail):
    schedule, two = schedule_and_flag
    legacy = load_legacy(detail=detail, two_events=two)
    cfg = TestbedConfig(schedule, detail=detail)
    w = cfg.windows()

    assert cfg.T == legacy.T
    assert list(w["before"]) == legacy.beforeRange
    assert list(w["during"]) == legacy.duringRange
    assert list(w["after"]) == legacy.afterRange
    if two:
        assert list(w["during2"]) == legacy.during2Range
        assert list(w["after2"]) == legacy.after2Range
    else:
        assert set(w) == {"before", "during", "after"}
    assert set(cfg.track_times()) == set(legacy.track_times)
    assert [schedule.strength_at(t) > 0 for t in range(legacy.T + 1)] == [
        t in legacy.ext_time for t in range(legacy.T + 1)
    ]


def test_detail_tracks_every_step_and_ensemble_tracks_a_subset():
    schedule = PressureSchedule.single_event(0.3)

    assert list(TestbedConfig(schedule, detail=True).track_times()) == list(range(201))
    expected = {0, 1, 100, 200} | set(range(91, 101)) | set(range(141, 151)) | set(range(191, 201))
    assert set(TestbedConfig(schedule).track_times()) == expected
