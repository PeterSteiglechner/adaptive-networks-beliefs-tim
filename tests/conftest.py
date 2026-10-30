import os
import subprocess
import sys
from pathlib import Path

import pytest

import old_scripts
import sim_folder
from legacy_oracle import load_legacy


@pytest.fixture
def legacy():
    """A fresh legacy `ANBmodel` in ensemble mode (`detail=False`)."""
    return load_legacy()


@pytest.fixture(scope="session")
def sim_dir(tmp_path_factory) -> Path:
    """The synthetic simulation folder of `sim_folder.py`, built once per session. Treat it as read-only.

    Under `NUMBA_DISABLE_JIT=1` it is built in a fresh interpreter with the JIT on: the analysis tests don't test
    the kernel, and without the JIT the runs would take minutes.
    """
    out_dir = tmp_path_factory.mktemp("sim")
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_JIT"}
        subprocess.run([sys.executable, sim_folder.__file__, str(out_dir)], env=env, check=True)
    else:
        sim_folder.build(out_dir)
    return out_dir


@pytest.fixture(scope="session")
def old_script(tmp_path_factory):
    """`old_script("analyse_….py")` runs that unchanged script on `simOut_old_model` once per session (cached).

    Skips when `simOut_old_model` is missing. The result's `repo_changes` lists what `git status` gained during the
    run (should be empty: the scripts work in a temporary directory).
    """
    if not old_scripts.OLD_SIM_DIR.is_dir():
        pytest.skip(f"{old_scripts.OLD_SIM_DIR.name} is missing")
    cache = {}

    def run(name):
        if name not in cache:
            before = _git_status()
            result = old_scripts.run_old_script(name, tmp_path_factory.mktemp(Path(name).stem))
            result.repo_changes = sorted(set(_git_status()) - set(before))
            cache[name] = result
        return cache[name]

    return run


def _git_status() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=old_scripts.REPO,
                         capture_output=True, text=True, check=True)
    return out.stdout.splitlines()
