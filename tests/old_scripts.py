"""Run an unchanged `analyse_*.py` script and collect what it computed (seam 3, old vs new).

The script runs cell by cell (`# %%`) in one shared `__main__` namespace, in a temporary working directory where
`simOut` links to `simOut_old_model` and `figs/` already exists (some scripts save into it before creating it). The
backend is Agg, stdout is captured, and the global rcParams and NumPy RNG are restored afterwards, so the scripts'
`plt.rc`/`sns.set_style`/`np.random.seed` don't leak into other tests. A failing cell is recorded and the next
cells still run, so a comparison can fall back to the parts that ran (cells that depend on a failed one fail too).

Running cell by cell instead of `runpy.run_path` is what keeps the globals of a script that fails part-way;
otherwise it's the same: the code is compiled with the script's own path and line numbers.
"""

import contextlib
import io
import os
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OLD_SIM_DIR = REPO / "simOut_old_model"
OLD_FIGS = REPO / "figs" / "figsold"
CELL = re.compile(r"^# %%", re.MULTILINE)


@dataclass
class CellFailure:
    cell: int  # 0-based, in `# %%` order
    line: int  # 1-based first line of the cell in the script
    error: str  # the formatted traceback

    def __str__(self):
        return f"cell {self.cell} (line {self.line}): {self.error.strip().splitlines()[-1]}"


@dataclass
class OldScriptResult:
    script: str
    globals: dict
    pngs: dict[str, Path]  # file name (without `figs/`) → path, for every PNG the script wrote
    stdout: str
    failures: list[CellFailure] = field(default_factory=list)
    repo_changes: list[str] = field(default_factory=list)  # filled by the `old_script` fixture

    @property
    def ok(self) -> bool:
        return not self.failures

    def report(self) -> str:
        return f"{self.script}: " + ("ran" if self.ok else "failed in " + "; ".join(map(str, self.failures)))


def cells(source: str) -> list[tuple[int, str]]:
    """The script's `# %%` cells as (first line, code), the code padded with blank lines to keep line numbers."""
    starts = [0] + [m.start() for m in CELL.finditer(source) if m.start() > 0] + [len(source)]
    out = []
    for a, b in zip(starts, starts[1:]):
        line = source.count("\n", 0, a) + 1
        out.append((line, "\n" * (line - 1) + source[a:b]))
    return out


def run_old_script(name, work_dir: Path, sim_dir: Path = OLD_SIM_DIR) -> OldScriptResult:
    """Run `REPO/name` in `work_dir` (made if needed, `simOut` → `sim_dir`, `figs/` created) and collect its results.

    `name` may also be an absolute path (the harness's own tests run a stand-in script).
    """
    script = REPO / name
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "simOut").symlink_to(sim_dir.resolve(), target_is_directory=True)
    (work_dir / "figs").mkdir()
    namespace = {"__name__": "__main__", "__file__": str(script), "__builtins__": __builtins__}
    failures, stdout = [], io.StringIO()
    previous_backend, previous_cwd, rng_state = matplotlib.get_backend(), os.getcwd(), np.random.get_state()
    plt.switch_backend("Agg")
    try:
        os.chdir(work_dir)
        with matplotlib.rc_context(), contextlib.redirect_stdout(stdout):
            for k, (line, code) in enumerate(cells(script.read_text())):
                try:
                    exec(compile(code, str(script), "exec"), namespace)
                except Exception:
                    failures.append(CellFailure(k, line, traceback.format_exc()))
    finally:
        plt.close("all")  # a cell may draw on a figure an earlier cell made, so close only at the end
        os.chdir(previous_cwd)
        np.random.set_state(rng_state)
        plt.switch_backend(previous_backend)
    pngs = {p.name: p for p in sorted((work_dir / "figs").rglob("*.png"))}
    return OldScriptResult(str(name), namespace, pngs, stdout.getvalue(), failures)


def png_difference(a: Path, b: Path, *, tol=0.05) -> float:
    """The fraction of pixels whose RGB channels differ by more than `tol` (on 0–1) between two same-size PNGs."""
    x, y = plt.imread(a)[..., :3], plt.imread(b)[..., :3]
    if x.shape != y.shape:
        raise AssertionError(f"{a.name} is {x.shape[:2]} pixels, {b.name} is {y.shape[:2]}")
    return float((np.abs(x - y) > tol).any(axis=-1).mean())
