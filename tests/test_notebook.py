"""The notebooks run top to bottom (stories 49–51 of the anb package, story 41 of the analysis package).

The demo notebook's run is slow: it runs one detail seed plus two small ensembles. The analysis notebook runs on the
synthetic folder of `sim_folder.py` in a few seconds, so its run is part of the default suite. The testbed
notebook's full sweep is far too slow for a test, so it's only checked to parse.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "demo.ipynb"
TESTBED_NOTEBOOK = NOTEBOOK.with_name("testbed.ipynb")
ANALYSIS_NOTEBOOK = NOTEBOOK.with_name("analysis.ipynb")


def code_cells(notebook=NOTEBOOK):
    return [cell for cell in json.loads(notebook.read_text())["cells"] if cell["cell_type"] == "code"]


def code_sources(notebook=NOTEBOOK):
    return ["".join(cell["source"]) for cell in code_cells(notebook)]


def run_as_script(sources, cwd: Path) -> subprocess.CompletedProcess:
    """The cells in a fresh interpreter (Agg backend), as one plain script in `cwd`."""
    script = cwd / "notebook.py"
    script.write_text("\n\n".join(sources))
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run([sys.executable, str(script)], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=600)


@pytest.mark.parametrize("notebook", [NOTEBOOK, ANALYSIS_NOTEBOOK], ids=lambda path: path.stem)
def test_notebook_has_no_magics_so_it_runs_as_a_plain_script(notebook):
    assert all(not line.lstrip().startswith(("%", "!"))
               for source in code_sources(notebook) for line in source.splitlines())


def test_testbed_notebook_compiles_as_a_plain_script():
    compile("\n\n".join(code_sources(TESTBED_NOTEBOOK)), str(TESTBED_NOTEBOOK), "exec")


@pytest.mark.slow
def test_notebook_runs_top_to_bottom_in_a_fresh_interpreter(tmp_path):
    proc = run_as_script(code_sources(), tmp_path)
    assert proc.returncode == 0, proc.stderr


def test_analysis_notebook_is_committed_with_outputs_cleared():
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells(ANALYSIS_NOTEBOOK))


def test_analysis_notebook_runs_top_to_bottom_on_the_synthetic_folder(sim_dir, tmp_path):
    # The cell tagged `parameters` sets the folders; an injected cell right after it points them at the fixture
    # (as papermill would), so the notebook's own defaults still run.
    tagged = [k for k, cell in enumerate(code_cells(ANALYSIS_NOTEBOOK))
              if "parameters" in cell["metadata"].get("tags", [])]
    assert len(tagged) == 1
    out_dir = tmp_path / "figs"
    sources = code_sources(ANALYSIS_NOTEBOOK)
    sources.insert(tagged[0] + 1, f"SIM_DIR = Path({str(sim_dir)!r})\nOUT_DIR = Path({str(out_dir)!r})")

    proc = run_as_script(sources, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "make_all: " in proc.stdout  # its printed report
    assert (out_dir / "fig2_adaptive_t100.png").is_file()
    assert (out_dir / "tables" / "table3_s4.tex").is_file()
    assert not (out_dir / "fig2_adaptive_t100.pdf").exists()  # PNG only by default
