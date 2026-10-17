"""`Testbed.save_ensemble`: each seed's CSV is written by its worker, and the paths come back in seed order."""

import logging
from dataclasses import fields

import pytest

from anb.agents import ANBAgent
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, SavedRun, Testbed, TestbedConfig

SEEDS = [5, 2, 9]


def make_testbed():
    """A tiny, legacy-nameable setup: 6 agents, pressure on t = 4…6, ensemble mode, T = 9."""
    spec = ModelSpec(ANBAgent(beta=3.0, eps=1.0, init_w=0.2), ErdosRenyiNetwork, {"link_prob": 0.3},
                     n_agents=6, rho=1 / 3)
    return Testbed(spec, TestbedConfig(PressureSchedule(4, [(4, 6)]), window_len=2, recovery=3))


def write_stale(testbed, seeds, directory):
    """Put a placeholder CSV where each seed's output goes, and return the paths."""
    paths = [testbed.output_path(seed, directory) for seed in seeds]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale")
    return paths



def anb_messages(caplog, level):
    """The text of the `anb` records at exactly `level`."""
    return [r.getMessage() for r in caplog.records if r.name.startswith("anb") and r.levelno == level]

@pytest.mark.parametrize("legacy_names", [True, False])
def test_save_ensemble_writes_what_run_then_save_writes_and_returns_paths_in_seed_order(legacy_names, tmp_path):
    testbed = make_testbed()
    paths = testbed.save_ensemble(SEEDS, tmp_path / "ensemble", legacy_names=legacy_names)

    expected = [testbed.run(seed).save(tmp_path / "single", legacy_names=legacy_names) for seed in SEEDS]
    assert [p.relative_to(tmp_path / "ensemble") for p in paths] == \
        [p.relative_to(tmp_path / "single") for p in expected]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(paths, expected))
    assert sorted(p for p in (tmp_path / "ensemble").rglob("*.csv")) == sorted(paths)


def test_save_ensemble_names_files_by_seed_without_legacy_names(tmp_path):
    paths = make_testbed().save_ensemble(SEEDS, tmp_path, legacy_names=False)

    assert paths == [tmp_path / f"seed{seed}.csv" for seed in SEEDS]


def test_save_ensemble_writes_legacy_names_by_default(tmp_path):
    testbed = make_testbed()
    paths = testbed.save_ensemble(SEEDS, tmp_path)

    assert paths == [testbed.output_path(seed, tmp_path) for seed in SEEDS]
    assert all(p.name.startswith("sim_") for p in paths)


def test_save_ensemble_overwrites_existing_files(tmp_path):
    testbed = make_testbed()
    (stale,) = write_stale(testbed, SEEDS[:1], tmp_path)

    testbed.save_ensemble(SEEDS, tmp_path)

    assert stale.read_bytes() == testbed.run(SEEDS[0]).save(tmp_path / "fresh").read_bytes()


def test_save_ensemble_in_parallel_writes_the_same_files_as_sequentially(tmp_path):
    testbed = make_testbed()
    sequential = testbed.save_ensemble(SEEDS, tmp_path / "seq")
    parallel = testbed.save_ensemble(SEEDS, tmp_path / "par", n_jobs=2)

    assert [p.relative_to(tmp_path / "par") for p in parallel] == [p.relative_to(tmp_path / "seq") for p in sequential]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(parallel, sequential))


def test_workers_send_back_only_seed_path_and_runtime(tmp_path):
    record = make_testbed()._run_and_save(SEEDS[0], tmp_path, True)

    assert [f.name for f in fields(SavedRun)] == ["seed", "path", "runtime"]
    assert record.seed == SEEDS[0] and record.path.exists() and record.runtime > 0


def test_skip_existing_leaves_existing_files_untouched_and_runs_only_missing_seeds(tmp_path, caplog):
    testbed = make_testbed()
    (stale,) = write_stale(testbed, SEEDS[1:2], tmp_path)
    mtime = stale.stat().st_mtime_ns
    caplog.set_level(logging.DEBUG, logger="anb")

    paths = testbed.save_ensemble(SEEDS, tmp_path, skip_existing=True)

    assert paths == [testbed.output_path(seed, tmp_path) for seed in SEEDS]
    assert stale.read_text() == "stale" and stale.stat().st_mtime_ns == mtime
    for seed in (SEEDS[0], SEEDS[2]):
        assert testbed.output_path(seed, tmp_path).read_bytes() == \
            testbed.run(seed).save(tmp_path / "fresh").read_bytes()
    written = anb_messages(caplog, logging.DEBUG)
    assert len(written) == 2 and not any(str(stale) in line for line in written)
    info = anb_messages(caplog, logging.INFO)
    assert any("skipping 1" in line for line in info)
    assert any("2/2" in line for line in info)
    assert "2 written" in info[-1] and "1 skipped" in info[-1]


def test_skip_existing_with_every_seed_present_runs_nothing(tmp_path, caplog, monkeypatch):
    testbed = make_testbed()
    stale = write_stale(testbed, SEEDS, tmp_path)
    monkeypatch.setattr(Testbed, "run", lambda self, seed: pytest.fail("no seed should run"))
    caplog.set_level(logging.INFO, logger="anb")

    paths = testbed.save_ensemble(SEEDS, tmp_path, skip_existing=True)

    assert paths == stale
    assert all(p.read_text() == "stale" for p in paths)
    info = anb_messages(caplog, logging.INFO)
    assert any("all 3 seeds skipped" in line for line in info)
    assert "0 written" in info[-1] and "3 skipped" in info[-1]
