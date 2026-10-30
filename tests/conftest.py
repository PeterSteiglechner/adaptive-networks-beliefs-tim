import os
import subprocess
import sys
from pathlib import Path

import pytest

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
