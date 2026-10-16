"""The compiled sweep kernel is cached on disk, so a second Python session doesn't recompile it."""

import os

import pytest

from numba_session import run_session

pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="nothing is compiled or cached without the JIT"
)


def test_a_second_session_loads_the_kernel_from_the_disk_cache(tmp_path):
    first = run_session(tmp_path)
    second = run_session(tmp_path)

    assert first["misses"] >= 1 and first["hits"] == 0
    assert second["hits"] >= 1 and second["misses"] == 0
