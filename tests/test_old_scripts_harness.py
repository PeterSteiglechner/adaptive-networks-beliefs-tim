"""The old-script harness (`old_scripts.py`) on a stand-in script: cells, failures, PNGs, restored global state."""

import matplotlib
import numpy as np

from old_scripts import cells, png_difference, run_old_script

SCRIPT = '''# %%
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rc("font", size=3)
np.random.seed(5)
n_files = len(os.listdir("simOut"))

# %%
raise ValueError("pandas 3 broke this cell")

# %%
fig = plt.figure(figsize=(1, 1))
plt.savefig("figs/one.png", dpi=20)
print("saved")
'''


def test_cells_keep_line_numbers():
    [(l0, c0), (l1, c1), (l2, _)] = cells(SCRIPT)
    assert (l0, l1, l2) == (1, 9, 12)
    assert c1.splitlines()[l1] == 'raise ValueError("pandas 3 broke this cell")'
    assert c0.startswith("# %%")


def test_runs_every_cell_reports_the_failing_one_and_restores_global_state(tmp_path):
    script, sim_dir = tmp_path / "analyse_fake.py", tmp_path / "sim"
    script.write_text(SCRIPT)
    sim_dir.mkdir()
    (sim_dir / "a.csv").write_text("")
    font_size, rng = matplotlib.rcParams["font.size"], np.random.get_state()[1].copy()

    result = run_old_script(script, tmp_path / "work", sim_dir)

    assert [(f.cell, f.line) for f in result.failures] == [(1, 9)]
    assert "pandas 3 broke this cell" in str(result.failures[0]) and str(script) in result.failures[0].error
    assert "line 10" in result.failures[0].error  # tracebacks point at the script's own line
    assert result.globals["n_files"] == 1 and result.stdout == "saved\n"
    assert list(result.pngs) == ["one.png"] and not result.ok
    assert png_difference(result.pngs["one.png"], result.pngs["one.png"]) == 0
    assert matplotlib.rcParams["font.size"] == font_size
    np.testing.assert_array_equal(np.random.get_state()[1], rng)
