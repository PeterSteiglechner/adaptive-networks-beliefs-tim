"""Experiment protocol: when pressure is applied and which steps are analysed."""

from dataclasses import dataclass


class PressureSchedule:
    """External pressure as data: a `strength` applied during inclusive `(start, end)` events."""

    def __init__(self, strength: float, events: list[tuple[int, int]]):
        self.strength = strength
        self.events = [tuple(e) for e in events]

    @property
    def n_events(self) -> int:
        return len(self.events)

    def strength_at(self, t: int) -> float:
        return self.strength if any(s <= t <= e for s, e in self.events) else 0

    @classmethod
    def single_event(cls, strength: float) -> "PressureSchedule":
        return cls(strength, [(101, 150)])

    @classmethod
    def two_events(cls, strength: float) -> "PressureSchedule":
        return cls(strength, [(101, 150), (201, 250)])


@dataclass
class TestbedConfig:
    """The experiment protocol. Everything time-related is derived from the schedule's events."""

    __test__ = False  # not a pytest test class

    schedule: PressureSchedule
    window_len: int = 10
    recovery: int = 50
    detail: bool = False

    @property
    def T(self) -> int:
        return self.schedule.events[-1][1] + self.recovery

    def windows(self) -> dict[str, range]:
        """`before`, then `during{j}`/`after{j}` per event; the first event has the bare names."""
        n = self.window_len
        start = self.schedule.events[0][0]
        out = {"before": range(start - n, start)}
        for j, (_, end) in enumerate(self.schedule.events, start=1):
            suffix = "" if j == 1 else str(j)
            out[f"during{suffix}"] = range(end - n + 1, end + 1)
            out[f"after{suffix}"] = range(end + self.recovery - n + 1, end + self.recovery + 1)
        return out

    def snapshot_times(self) -> list[int]:
        return [0, self.schedule.events[0][0] - 1] + [e + self.recovery for _, e in self.schedule.events]

    def track_times(self) -> list[int]:
        if self.detail:
            return list(range(self.T + 1))
        steps = {0, 1, *self.snapshot_times()}
        for w in self.windows().values():
            steps.update(w)
        return sorted(steps)
