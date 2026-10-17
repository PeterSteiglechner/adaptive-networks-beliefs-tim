"""The demo notebook runs top to bottom (stories 49–51). Slow: it runs one detail seed plus two small ensembles.

The testbed notebook's full sweep is far too slow for a test, so it's only checked to parse."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "demo.ipynb"
TESTBED_NOTEBOOK = NOTEBOOK.with_name("testbed.ipynb")


def code_cells(notebook=NOTEBOOK):
    cells = json.loads(notebook.read_text())["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def test_notebook_has_no_magics_so_it_runs_as_a_plain_script():
    assert all(not line.lstrip().startswith(("%", "!")) for cell in code_cells() for line in cell.splitlines())


def test_testbed_notebook_compiles_as_a_plain_script():
    compile("\n\n".join(code_cells(TESTBED_NOTEBOOK)), str(TESTBED_NOTEBOOK), "exec")


@pytest.mark.slow
def test_notebook_runs_top_to_bottom_in_a_fresh_interpreter(tmp_path):
    script = tmp_path / "testbed.py"
    script.write_text("\n\n".join(code_cells()))
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr
