"""Step-wise comparison of shared-start trajectories: per-step differences, q-values, first divergent step."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from anb.comparison import METRICS, SHARED_START_QUANTITIES, compare_shared_start

K, N_AGENTS, STEPS = 30, 10, list(range(95, 111))


def synthetic_trajectories(shift_from: int | None = None, shift: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """`run_shared_start`-shaped trajectories: noise around a common level, with the anb `x_focal` of every
    agent shifted by `shift` from step `shift_from` on. t = 0 is identical for all, as in a shared start."""
    rng = np.random.default_rng(seed)
    parts = []
    for implementation in ("legacy", "anb"):
        for replica in range(K):
            for t in [0, *STEPS]:
                values = rng.normal(size=(N_AGENTS, len(METRICS))) if t else np.zeros((N_AGENTS, len(METRICS)))
                frame = pd.DataFrame(values, columns=list(METRICS))
                if implementation == "anb" and shift_from is not None and t >= shift_from:
                    frame["x_focal"] += shift
                parts.append(pd.concat([pd.DataFrame({"implementation": implementation, "replica": replica, "t": t,
                                                      "agent": range(N_AGENTS)}), frame], axis=1))
    return pd.concat(parts, ignore_index=True)


def test_each_step_and_quantity_gets_a_difference_with_a_ci_and_a_q_value():
    steps = compare_shared_start(synthetic_trajectories()).steps

    assert len(steps) == len(STEPS) * len(SHARED_START_QUANTITIES)
    assert set(steps.t) == set(STEPS)  # t = 0 is the shared state, not a test
    assert set(steps.quantity) == set(SHARED_START_QUANTITIES)
    for column in ("mean_legacy", "mean_anb", "diff", "ci_low", "ci_high", "ks", "p", "q", "fail"):
        assert column in steps
    assert ((steps.ci_low <= steps["diff"]) & (steps["diff"] <= steps.ci_high)).all()
    assert (steps.q >= steps.p).all()


def test_the_quantities_are_ensemble_means_of_replica_means_and_spreads():
    trajectories = synthetic_trajectories()
    steps = compare_shared_start(trajectories).steps

    at = trajectories[trajectories.t == 100]
    replica_sd = at.groupby(["implementation", "replica"]).x_focal.std(ddof=0)
    row = steps[(steps.t == 100) & (steps.quantity == "x_focal_sd")].iloc[0]
    assert row.mean_legacy == np.float64(replica_sd["legacy"].mean())
    assert row.mean_anb == np.float64(replica_sd["anb"].mean())
    row = steps[(steps.t == 100) & (steps.quantity == "Hsoc")].iloc[0]
    assert np.isclose(row.mean_anb, at[at.implementation == "anb"].Hsoc.mean())


def test_a_divergence_injected_from_a_known_step_is_reported_at_that_step():
    first = compare_shared_start(synthetic_trajectories(shift_from=103)).first_divergence.set_index("quantity")

    assert first.loc["x_focal", "first_divergent_step"] == 103
    others = first.drop(index="x_focal")
    assert others.first_divergent_step.isna().all()
    assert set(first.index) == set(SHARED_START_QUANTITIES)


def test_without_a_divergence_no_quantity_diverges():
    first = compare_shared_start(synthetic_trajectories()).first_divergence

    assert first.first_divergent_step.isna().all()
    assert (first.n_divergent_steps == 0).all()


def test_the_report_script_writes_the_divergence_table_and_plots(tmp_path):
    from anb.agents import ANBAgent
    from anb.model import ModelSpec
    from anb.networks import ErdosRenyiNetwork
    from anb.testbed import PressureSchedule, Testbed, TestbedConfig

    old, new, out = tmp_path / "old", tmp_path / "new", tmp_path / "out"
    spec = ModelSpec(ANBAgent(beta=3.0, eps=1.0, mu=0.0, init_w=0.2), ErdosRenyiNetwork, {"link_prob": 0.1},
                     n_agents=12, rho=1 / 3)
    for folder in (old, new):  # the main comparison needs a cell to pair; it isn't what's tested
        Testbed(spec, TestbedConfig(PressureSchedule.single_event(4.0))).save_ensemble(range(2), folder)
    env = {**os.environ, "COMPARE_OLD": str(old), "COMPARE_NEW": str(new), "COMPARE_OUT": str(out),
           "COMPARE_N_PERMUTATIONS": "20", "COMPARE_N_JOBS": "1", "COMPARE_RUN_SHARED_START": "1",
           "COMPARE_SHARED_START_K": "2", "COMPARE_SHARED_START_WINDOW": "100,102",
           "COMPARE_SHARED_START_PRESSURES": "4"}

    proc = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "compare_old_new.py")], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=900)

    assert proc.returncode == 0, proc.stderr
    folder = out / "shared_start"
    for name in ("divergence", "first_divergence"):
        assert (folder / f"{name}.csv").exists() and (folder / f"{name}.md").exists()
    steps = pd.read_csv(folder / "divergence.csv")
    assert set(steps.t) == {100, 101, 102} and set(steps.pressure) == {4}
    plots = list(folder.glob("divergence_*.png"))
    assert plots and all(p.stat().st_size > 0 for p in plots)
