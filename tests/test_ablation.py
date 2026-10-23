"""The ablation and rerun path: a `Testbed` with `learning_until = 102`, saved under legacy names and compared."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from anb.agents import ANBAgent
from anb.comparison import compare_folders, save_under_legacy_names
from anb.io import legacy_filename
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig

N = 12
EDGE_COLS = [f"w{m}{n}" for m in range(10) for n in range(m + 1, 10)]


def make_testbed(learning_until, detail=False, strength=4.0, mu=0.3) -> Testbed:
    agent = ANBAgent(beta=3.0, eps=1.0, mu=mu, init_w=0.2, learning_until=learning_until)
    spec = ModelSpec(agent, ErdosRenyiNetwork, {"link_prob": 0.1}, n_agents=N, rho=1 / 3)
    return Testbed(spec, TestbedConfig(PressureSchedule.single_event(strength), detail=detail))


def edges_at(frame: pd.DataFrame, t: int) -> np.ndarray:
    return frame[frame.t == t].sort_values("id")[EDGE_COLS].to_numpy()


def test_a_gate_of_102_still_changes_edges_at_t101_and_stops_afterwards():
    frame = make_testbed(102, detail=True).run(seed=1).frame

    assert not np.array_equal(edges_at(frame, 100), edges_at(frame, 101))
    for t in (102, 103, 150):
        np.testing.assert_array_equal(edges_at(frame, t), edges_at(frame, 101))


def test_a_gate_of_101_stops_one_step_earlier():
    frame = make_testbed(101, detail=True).run(seed=1).frame

    np.testing.assert_array_equal(edges_at(frame, 101), edges_at(frame, 100))


def test_the_ablation_is_saved_under_the_adaptive_static_names_and_pairs_with_it(tmp_path):
    ablation, reference = tmp_path / "ablation", tmp_path / "reference"
    paths = save_under_legacy_names(make_testbed(102), range(2), ablation, fixedBNat100=True)
    make_testbed(101).save_ensemble(range(2), reference)

    expected = {legacy_filename({**make_testbed(101).params(), "seed": s}, detail=False) + ".csv" for s in range(2)}
    assert {p.name for p in paths} == expected == {p.name for p in ablation.glob("*.csv")}
    assert all("_fixedBNat100_" in name for name in expected)
    result = compare_folders(reference, ablation, n_permutations=20, n_jobs=1)
    assert len(result.pairing.matched) == 2 and not result.pairing.only_old and not result.pairing.only_new


def test_the_report_script_runs_the_ablation_and_rerun_cells_into_their_own_folders(tmp_path):
    old, new, out, rerun_old = (tmp_path / name for name in ("old", "new", "out", "rerun_old"))
    for pressure in (0, 1, 2, 4, 8, 16):  # stand-in legacy folders: cheap runs under the names the cells pair with
        make_testbed(101, strength=pressure, mu=0.0).save_ensemble(range(2), old)
        make_testbed(None, strength=pressure, mu=0.1).save_ensemble(range(2), rerun_old)
    make_testbed(None, strength=4, mu=0.0).save_ensemble(range(2), old)  # the main comparison's cell
    make_testbed(None, strength=4, mu=0.0).save_ensemble(range(2), new)
    env = {**os.environ, "COMPARE_OLD": str(old), "COMPARE_NEW": str(new), "COMPARE_OUT": str(out),
           "COMPARE_N_PERMUTATIONS": "50", "COMPARE_N_JOBS": "1",
           "COMPARE_RUN_ABLATIONS": "1", "COMPARE_ABLATION": str(tmp_path / "ablation"), "COMPARE_ABLATION_SEEDS": "2",
           "COMPARE_RUN_RERUN": "1", "COMPARE_RERUN_SEEDS": "2", "COMPARE_RERUN_OLD": str(rerun_old),
           "COMPARE_RERUN_NEW": str(tmp_path / "rerun_new")}

    proc = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "compare_old_new.py")], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=900)

    assert proc.returncode == 0, proc.stderr
    assert len(list((tmp_path / "ablation").glob("*_fixedBNat100_*.csv"))) == 12
    assert len(list((tmp_path / "rerun_new").glob("*.csv"))) == 12
    for name in ("ablation_learning_until", "rerun"):
        assert (out / name / "pooled_responses.csv").exists() and (out / name / "top_differences.md").exists()
