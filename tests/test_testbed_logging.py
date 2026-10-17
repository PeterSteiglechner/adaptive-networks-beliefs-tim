"""Ensemble logging: start, progress and finish records from `run_ensemble` and `save_ensemble`, silent by default."""

import logging
import subprocess
import sys

import numpy as np
import pytest

from anb.agents import ANBAgent
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig

SEEDS = [5, 2, 9]


def make_testbed(eps=1.0):
    """6 agents, pressure on t = 4…6, ensemble mode, T = 9."""
    spec = ModelSpec(ANBAgent(beta=3.0, eps=eps, init_w=0.2), ErdosRenyiNetwork, {"link_prob": 0.3},
                     n_agents=6, rho=1 / 3)
    return Testbed(spec, TestbedConfig(PressureSchedule(4, [(4, 6)]), window_len=2, recovery=3))


def messages(caplog, level=logging.INFO):
    """The text of the `anb` records at exactly `level`."""
    return [r.getMessage() for r in caplog.records if r.name.startswith("anb") and r.levelno == level]


def run_method(method, testbed, tmp_path, **kwargs):
    """Call `run_ensemble` or `save_ensemble` (into `tmp_path`) on `SEEDS`."""
    if method == "save_ensemble":
        return testbed.save_ensemble(SEEDS, tmp_path, **kwargs)
    return testbed.run_ensemble(SEEDS, **kwargs)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("method", ["run_ensemble", "save_ensemble"])
def test_both_methods_log_start_progress_and_finish(method, n_jobs, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="anb")
    run_method(method, make_testbed(), tmp_path, n_jobs=n_jobs)

    lines = messages(caplog)
    start, finish = lines[0], lines[-1]
    assert "beta=3.00" in start and "init_w=0.20" in start and "mu=0.000" in start and "ext_strength=4" in start
    assert "3 seeds" in start and "n_jobs=" + str(n_jobs) in start
    assert any("3/3" in line for line in lines[1:-1])
    if method == "save_ensemble":
        assert str(tmp_path) in start
        assert "3 written" in finish and "0 skipped" in finish and str(tmp_path) in finish
    else:
        assert "3 results" in finish


def test_progress_is_throttled_to_about_every_tenth_of_the_seeds(caplog):
    caplog.set_level(logging.INFO, logger="anb")
    make_testbed().run_ensemble(range(25))

    progress = [line for line in messages(caplog) if "/25" in line]
    assert [line.split("/25")[0].split()[-1] for line in progress] == ["3", "6", "9", "12", "15", "18", "21", "24", "25"]


def test_array_parameters_are_summarised_in_the_start_record(caplog):
    caplog.set_level(logging.INFO, logger="anb")
    make_testbed(eps=np.linspace(0.0, 1.0, 6)).run_ensemble(SEEDS[:1])

    start = messages(caplog)[0]
    assert "eps=array(6, mean=0.50)" in start
    assert "[" not in start


def test_save_ensemble_logs_each_written_path_at_debug(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="anb")
    paths = make_testbed().save_ensemble(SEEDS, tmp_path)

    debug = messages(caplog, logging.DEBUG)
    assert len(debug) == len(SEEDS)
    assert all(any(str(p) in line for line in debug) for p in paths)


def test_package_logs_nothing_unless_logging_is_configured():
    code = (
        "import logging, anb.testbed\n"
        "assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger('anb').handlers)\n"
        "logging.getLogger('anb.testbed').warning('should not be seen')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert done.stderr == ""


def test_run_ensemble_in_parallel_returns_results_in_seed_order():
    testbed = make_testbed()
    results = testbed.run_ensemble(SEEDS, n_jobs=2)

    assert [r.seed for r in results] == SEEDS
    for result, seed in zip(results, SEEDS):
        assert result.frame.equals(testbed.run(seed).frame)
