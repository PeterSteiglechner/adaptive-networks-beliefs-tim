"""`anb.analysis.make_all`, its report and the CLI (`python -m anb.analysis`), on the folder of `sim_folder.py`.

The fixture's paper base has the four BN configs at s ∈ {0, 1, 4} (ensemble), one detailed adaptive run (s = 4,
seed 2) and μ ∈ {0.005, 0.2} at s = 4 for seed 1 only, so most of the paper's items are skipped on it: the report
then names why.
"""

import contextlib
import io
import subprocess
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import sim_folder  # noqa: E402
from anb.analysis import FigureFailed, FigureNotApplicable, Loader, fig2, make_all  # noqa: E402
from anb.analysis.__main__ import main  # noqa: E402
from anb.io import legacy_filename  # noqa: E402

FIG2_KEYS = ["fig2 config=adaptive", "fig2 config=staticlow", "fig2 config=statichigh"]
MUS = (0.0, 0.005, 0.05)
EXPECTED_KEYS = [
    *FIG2_KEYS, "fig3", "figA3",
    *(f"{item} mu={mu}" for mu in MUS for item in ("fig4", "fig5", "figA4")),
    "fig6", "figA7", "figA8",
    "table3 s=1 mu=0.0", "table3 s=4 mu=0.0", "tableA5 mu=0.0", "tableA6 mu=0.0", "tableA7 mu=0.0",
    "table3 s=4 mu=0.005", "tableA5 mu=0.005", "tableA6 mu=0.005", "tableA7 mu=0.005",
    "table3 mu=0.05", "tableA5 mu=0.05", "tableA6 mu=0.05", "tableA7 mu=0.05",
]
DETAILED_RUN = legacy_filename(dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=1 / 3, eps=1.0, mu=0.0, fixedBNat100=False,
                                    ext_strength=sim_folder.DETAILED["s"], seed=sim_folder.DETAILED["seed"], lam=0.0,
                                    n_events=1), True) + "_detailed.csv"


def run(*args, **kwargs):
    """`make_all(...)`, what it printed and the warnings it emitted."""
    out = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out):
        warnings.simplefilter("always")
        report = make_all(*args, **kwargs)
    return report, out.getvalue(), caught


def written(out_dir: Path) -> set[Path]:
    return {path.relative_to(out_dir) for path in out_dir.rglob("*") if path.is_file()}


class Full(NamedTuple):
    """`make_all` on the whole fixture: its report, what it printed, its warnings, the output folder and the loader."""

    report: dict
    printed: str
    caught: list
    out_dir: Path
    loader: Loader


@pytest.fixture(scope="module")
def full(sim_dir, tmp_path_factory) -> Full:
    """`make_all` on the whole fixture with a shared whole-reading loader."""
    out_dir = tmp_path_factory.mktemp("make_all")
    loader = Loader(whole=True)
    return Full(*run(sim_dir, out_dir, loader=loader), out_dir, loader)


@pytest.fixture
def broken_dir(sim_dir, tmp_path):
    """The fixture folder (links to its files) with Fig. 2's detailed run replaced by a file that can't be read."""
    for path in sim_dir.rglob("*"):
        if path.is_file():
            link = tmp_path / path.relative_to(sim_dir)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path)
    broken = tmp_path / DETAILED_RUN
    broken.unlink()
    broken.write_text("t,id\nnot,a number\n")
    return tmp_path


# The report

def test_the_report_has_an_entry_for_every_item(full):
    assert list(full.report) == EXPECTED_KEYS


def test_items_are_written_or_skipped_with_their_reason(full):
    report, out_dir = full.report, full.out_dir

    for key, value in report.items():
        if isinstance(value, str):
            assert value.startswith("skipped: ") and len(value) > len("skipped: "), key
        else:
            assert value and all(path.is_file() and out_dir in path.parents for path in value), key
    assert report["fig2 config=adaptive"] == [out_dir / "fig2_adaptive_t100.png"]
    assert report["table3 s=4 mu=0.0"] == [out_dir / "tables" / "table3_s4.tex"]
    assert "fig2 not applicable" in report["fig2 config=staticlow"]  # no detailed staticlow run
    assert "s > 0" in report["table3 mu=0.05"]  # no runs at μ = 0.05


def test_the_skip_warnings_are_still_emitted(full):
    emitted = {str(w.message) for w in full.caught if w.category is FigureNotApplicable}
    for value in full.report.values():
        if isinstance(value, str):
            assert set(value.removeprefix("skipped: ").split("; ")) <= emitted


def test_the_skip_reason_is_kept_when_the_warnings_are_ignored(sim_dir, tmp_path):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        report = make_all(sim_dir, tmp_path, figs=["fig2"])
    assert "detailed staticlow" in report["fig2 config=staticlow"]


def test_module_filters_apply_as_for_a_direct_call(sim_dir, tmp_path):
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", module=r"anb\.analysis")
        assert fig2(sim_dir, config="staticlow") is None
        direct = {w.category for w in caught}
        make_all(sim_dir, tmp_path, figs=["fig2 config=staticlow"])
    assert FigureNotApplicable not in direct
    assert {w.category for w in caught} == direct


def test_the_output_folder_holds_exactly_the_reported_files(full):
    report, out_dir = full.report, full.out_dir
    reported = {path.relative_to(out_dir) for value in report.values() if not isinstance(value, str) for path in value}
    assert written(out_dir) == reported
    assert all(path.suffix == ".png" for path in reported if path.parent == Path("."))


def test_the_printed_summary_lists_every_item(full):
    n_skipped = sum(isinstance(value, str) for value in full.report.values())
    assert f"{len(full.report) - n_skipped} written, {n_skipped} skipped, 0 failed" in full.printed
    for key in full.report:
        assert key in full.printed


def test_each_csv_is_read_at_most_once(full):
    assert full.loader.reads and set(full.loader.reads.values()) == {1}


def test_the_default_loader_reads_each_csv_once(sim_dir, tmp_path, monkeypatch):
    reads = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: reads.append(Path(path)) or real(path, *a, **k))

    run(sim_dir, tmp_path, figs=["fig3", "fig4", "fig5", "figA4", "table3", "tableA5"])

    assert reads and len(reads) == len(set(reads))


# Selecting items

def test_figs_restricts_the_run_to_the_named_items(sim_dir, tmp_path):
    report, _, _ = run(sim_dir, tmp_path, figs=["fig2", "tableA5 mu=0.0"])
    assert list(report) == [*FIG2_KEYS, "tableA5 mu=0.0"]


def test_an_unknown_item_is_an_error(sim_dir, tmp_path):
    with pytest.raises(ValueError, match="fig99"):
        make_all(sim_dir, tmp_path, figs=["fig99"])


def test_no_tables(sim_dir, tmp_path):
    report, _, _ = run(sim_dir, tmp_path, figs=["fig2", "table3", "tableA5 mu=0.0"], tables=False)
    assert list(report) == FIG2_KEYS
    assert not (tmp_path / "tables").exists()


def test_formats_and_mus_are_passed_on(sim_dir, tmp_path):
    report, _, _ = run(sim_dir, tmp_path, figs=["fig2", "table3"], formats=("png", "pdf"), mus=(0.005,))
    assert report["fig2 config=adaptive"] == [tmp_path / "fig2_adaptive_t100.png", tmp_path / "fig2_adaptive_t100.pdf"]
    assert list(report)[3:] == ["table3 s=4 mu=0.005"]


def test_a_pressure_level_with_too_few_runs_is_skipped_with_that_reason(sim_dir, tmp_path):
    report, _, _ = run(sim_dir, tmp_path, figs=["table3"], mus=(0.005,), min_seeds=3)  # s = 4 has seed 1 only
    assert list(report) == ["table3 s=4 mu=0.005"]
    assert report["table3 s=4 mu=0.005"].startswith("skipped: table3 not applicable: needs at least 3 run(s)")


def test_an_unknown_format_is_an_error(sim_dir, tmp_path):
    with pytest.raises(ValueError, match="svg"):
        make_all(sim_dir, tmp_path, formats=("svg",))


# Unexpected errors

def test_an_error_in_one_item_is_warned_about_and_the_others_still_run(broken_dir, tmp_path):
    report, printed, caught = run(broken_dir, tmp_path, figs=["fig2", "table3"])

    assert report["fig2 config=adaptive"].startswith("error: ValueError")
    assert report["fig2 config=staticlow"].startswith("skipped: ")
    assert report["table3 s=4 mu=0.0"] == [tmp_path / "tables" / "table3_s4.tex"]
    [failure] = [w for w in caught if w.category is FigureFailed]
    assert "fig2 config=adaptive" in str(failure.message) and "Traceback" in str(failure.message)
    assert "1 failed" in printed


def test_strict_raises_the_error(broken_dir, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            make_all(broken_dir, tmp_path, figs=["fig2"], strict=True)


# CLI

def test_the_cli_writes_the_same_files_as_the_api_call(sim_dir, tmp_path):
    api, cli = tmp_path / "api", tmp_path / "cli"
    report, _, _ = run(sim_dir, api, figs=["fig2", "table3", "tableA5"], formats=("png", "pdf"))

    done = subprocess.run([sys.executable, "-m", "anb.analysis", str(sim_dir), str(cli), "--figs",
                           "fig2,table3,tableA5", "--formats", "png,pdf"], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert written(cli) == written(api) and written(api)
    for path in written(api):
        if path.suffix == ".tex":
            assert (cli / path).read_text() == (api / path).read_text()
    for key in report:
        assert key in done.stdout


def test_the_cli_takes_the_other_arguments(sim_dir, tmp_path, capsys):
    """The M/N/τ variant has two adaptive runs at s = 4 for μ = 0 and one (seed 1) for μ = 0.2."""
    with_tables, no_tables = tmp_path / "tables", tmp_path / "no_tables"
    args = ["--figs", "fig2,table3", "--mus", "0,0.2", "--base", "M=6", "N=50", "tau=2", "--min-seeds", "2"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main([str(sim_dir), str(with_tables), *args]) == 0
        printed = capsys.readouterr().out
        assert main([str(sim_dir), str(no_tables), *args, "--no-tables"]) == 0
    suffix = "_M6.00_N50.00_tau2.00"
    assert written(with_tables) == {Path(f"fig2_adaptive_t100{suffix}.png"), Path(f"tables/table3_s4{suffix}.tex")}
    assert written(no_tables) == {Path(f"fig2_adaptive_t100{suffix}.png")}
    [line] = [line for line in printed.splitlines() if line.split()[:3] == ["table3", "s=4", "mu=0.2"]]
    assert "skipped: table3 not applicable: needs at least 2 run(s)" in line  # written with the default --min-seeds
    assert "mu=0.005" not in printed  # a default μ


def test_the_cli_exits_with_1_after_an_error_and_raises_with_strict(broken_dir, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            assert main([str(broken_dir), str(tmp_path), "--figs", "fig2"]) == 1
            with pytest.raises(ValueError):
                main([str(broken_dir), str(tmp_path), "--figs", "fig2", "--strict"])
