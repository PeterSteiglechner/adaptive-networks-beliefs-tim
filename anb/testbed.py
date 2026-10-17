"""Experiment protocol: when pressure is applied and which steps are analysed, and the run loop."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .io import legacy_filename, parameter_columns
from .metrics import RESPONSE_COLUMNS, Metric, MetricsRecorder, ResponseClassifier, StepContext, default_metrics
from .model import ModelSpec


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


class WindowAggregator:
    """Ensemble-mode output: the snapshot rows plus one per-agent mean row per window.

    Each mean row is labelled with its window's midpoint (95.5, 145.5, …). The response dummies go on the
    `before` row only, as the last columns, and are NaN everywhere else (legacy's column order).
    """

    def __init__(self, windows: dict[str, range], snapshot_times: list[int]):
        self.windows = dict(windows)
        self.snapshot_times = list(snapshot_times)

    def aggregate(self, frame: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
        values = [c for c in frame.columns if c not in ("t", "id")]
        blocks = [frame[frame.t.isin(self.snapshot_times)]]
        for name, window in self.windows.items():
            means = frame[frame.t.isin(window)].groupby("id")[values].mean().reset_index()
            means.insert(0, "t", np.mean(window))
            if name == "before":
                means[list(RESPONSE_COLUMNS)] = responses.reindex(means.id).to_numpy()
            blocks.append(means)
        out = pd.concat(blocks, ignore_index=True)
        return out.sort_values(["t", "id"], kind="stable", ignore_index=True)


def output_path(params: dict, seed: int, detail: bool, directory="simOut", legacy_names: bool = True) -> Path:
    """The CSV path for one seed's run, as `SimulationResult.save` writes it. `params` are the spec's plus the schedule's.

    With `legacy_names`, the name is legacy `run_one`'s (`sim_…`, under `detailed/` and ending in `_detailed` in
    detail mode). Otherwise it is `seed{seed}.csv`, so use one directory per configuration.
    """
    if legacy_names:
        name = legacy_filename({**params, "seed": seed}, detail)
        name += "_detailed" if detail else ""
    else:
        name = f"seed{seed}"
    return Path(directory) / f"{name}.csv"


@dataclass
class SimulationResult:
    """One run: the finished frame, the parameters (spec and schedule), the seed and the output mode."""

    frame: pd.DataFrame
    params: dict
    seed: int
    detail: bool = False

    def output_params(self) -> dict:
        """`params` plus the seed: what the output columns and the legacy filename are built from."""
        return {**self.params, "seed": self.seed}

    def to_frame(self) -> pd.DataFrame:
        """`frame` plus the parameter columns, as legacy `run_one` writes them.

        Uniform parameters are scalar columns. Per-agent ones are `(N,)` arrays, joined on `id`. `t` and `id`
        are floats, as in legacy's CSVs (its state array is all float), so old and new frames merge on `id`.
        """
        params = self.output_params()
        frame = self.frame.copy()
        ids = frame.id.to_numpy()
        for name in parameter_columns(params):
            value = params[name]
            frame[name] = value if np.ndim(value) == 0 else np.asarray(value)[ids]
        return frame.astype({"t": float, "id": float})

    def save(self, directory="simOut", legacy_names: bool = True) -> Path:
        """Write `to_frame()` as CSV under `directory` and return the path.

        With `legacy_names`, the name is legacy `run_one`'s (`sim_…`, under `detailed/` and ending in `_detailed` in
        detail mode), so `save("simOut")` from the repo root writes where the `analyse_*.py` scripts look. Legacy
        itself writes to `simOut_old_model/`, which keeps the two models' runs apart. Otherwise the name is
        `seed{seed}.csv`, so use one directory per configuration.
        """
        path = output_path(self.params, self.seed, self.detail, directory, legacy_names)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path)
        return path


@dataclass(frozen=True)
class SavedRun:
    """What a `save_ensemble` worker sends back instead of the frame: the seed, where its CSV went and how long it took."""

    seed: int
    path: Path
    runtime: float


class Testbed:
    """Owns the experiment: seeding, the time loop, the pressure schedule and metric recording."""

    __test__ = False  # not a pytest test class

    def __init__(
        self,
        spec: ModelSpec,
        config: TestbedConfig,
        metrics: list[Metric] | None = None,
        classifier: ResponseClassifier | None = None,
        aggregator: WindowAggregator | None = None,
    ):
        self.spec = spec
        self.config = config
        self.metrics = default_metrics() if metrics is None else list(metrics)
        self.classifier = classifier or ResponseClassifier.from_windows(config.windows())
        self.aggregator = aggregator or WindowAggregator(config.windows(), config.snapshot_times())
        if self.classifier.column not in {metric.name for metric in self.metrics}:
            raise ValueError(f"the response classifier reads {self.classifier.column!r}; add its metric")

    def run(self, seed: int) -> SimulationResult:
        schedule = self.config.schedule
        rng = np.random.default_rng(seed)
        model = self.spec.build(rng)
        recorder = MetricsRecorder(self.metrics)
        tracked = set(self.config.track_times())

        recorder.record(StepContext(model, 0, schedule.strength_at(0)))
        for t in range(1, self.config.T + 1):
            ext = schedule.strength_at(t)
            model.step(t, ext, rng)
            if t in tracked:
                recorder.record(StepContext(model, t, ext))

        frame = recorder.to_frame()
        responses = self.classifier.classify(frame)
        if self.config.detail:
            frame = self._attach_at_end(frame, responses)
        else:
            frame = self.aggregator.aggregate(frame, responses)

        return SimulationResult(frame, self.params(), seed, self.config.detail)

    def params(self) -> dict:
        """The run's parameters, without the seed: the spec's plus the schedule's `ext_strength` and `n_events`."""
        schedule = self.config.schedule
        return {**self.spec.params(), "ext_strength": schedule.strength, "n_events": schedule.n_events}

    def output_path(self, seed: int, directory="simOut", legacy_names: bool = True) -> Path:
        """Where `run(seed).save(directory, legacy_names)` writes, worked out without running the seed."""
        return output_path(self.params(), seed, self.config.detail, directory, legacy_names)

    def run_ensemble(self, seeds, n_jobs: int = 1) -> list[SimulationResult]:
        """`run(seed)` for each seed, in parallel with joblib. Each worker builds its own model."""
        return Parallel(n_jobs=n_jobs)(delayed(self.run)(seed) for seed in seeds)

    def save_ensemble(self, seeds, out_dir, n_jobs: int = 1, legacy_names: bool = True) -> list[Path]:
        """`run(seed).save(out_dir, legacy_names)` for each seed, and the paths in seed order.

        Each worker writes its own CSV as soon as its seed finishes and returns only a `SavedRun`, so finished
        frames never pile up in memory and a crash keeps the files written so far. Existing files are overwritten.
        """
        runs = Parallel(n_jobs=n_jobs)(delayed(self._run_and_save)(seed, out_dir, legacy_names) for seed in seeds)
        return [r.path for r in runs]

    def _run_and_save(self, seed: int, out_dir, legacy_names: bool) -> SavedRun:
        """One `save_ensemble` worker: run the seed, write its CSV and return only the small record."""
        start = time.perf_counter()
        path = self.run(seed).save(out_dir, legacy_names)
        return SavedRun(seed, path, time.perf_counter() - start)

    def _attach_at_end(self, frame: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
        """Detail mode: the dummies as 0/1 on the rows at t = T, NaN elsewhere, after the metric columns."""
        at_end = frame.t == self.config.T
        position = 2 + len(self.metrics)
        for k, name in enumerate(RESPONSE_COLUMNS):
            column = np.full(len(frame), np.nan)
            column[at_end.to_numpy()] = responses[name].reindex(frame.id[at_end]).astype(int).to_numpy()
            frame.insert(position + k, name, column)
        return frame
