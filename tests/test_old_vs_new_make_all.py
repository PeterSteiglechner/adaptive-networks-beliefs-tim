"""`make_all` on `simOut_old_model`: every paper figure and table the folder allows, without errors, each CSV read once.

Run with `pytest -m slow -s tests/test_old_vs_new_make_all.py` (prints the report and the peak memory). Skipped when
`simOut_old_model` is missing. The numbers are compared with the old scripts in the other `test_old_vs_new_*.py`;
this checks that the whole run goes through. The folder has no detailed static runs (Fig. A1/A2) and no
sensitivity runs (Fig. A8), so those are skipped.
"""

import contextlib
import io
import resource
import time
import warnings

import matplotlib
import pytest

matplotlib.use("Agg")

from anb.analysis import Loader, index_folder, make_all  # noqa: E402
from old_scripts import OLD_SIM_DIR  # noqa: E402

pytestmark = pytest.mark.slow

MUS = ("0.0", "0.005", "0.05")
PRESSURES = (1, 2, 4, 8, 16)
FIGURES = [
    "fig2_adaptive_t100", "fig3_responseFrequencies", "AppendixFig_exampleSim_adaptive_seed1_overPressure",
    *(f"fig4_mu{mu}" for mu in MUS), *(f"fig5_metricChange_mu{mu}" for mu in MUS),
    "AppendixFig_metricCorrelations", "AppendixFig_metricCorrelations_mu0.005", "AppendixFig_metricCorrelations_mu0.05",
    "fig6_socialAdaptation", "AppendixFig_edgeweight_distributions",
]
TABLES = [
    f"{stem}{'' if mu == '0.0' else f'_mu{mu}'}" for mu in MUS
    for stem in (*(f"table3_s{s}" for s in PRESSURES), "tableA5_metricChange", "tableA6_logit_rRC", "tableA7_logit_Rr")
]
SKIPPED = ["fig2 config=staticlow", "fig2 config=statichigh", "figA8"]


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    if not OLD_SIM_DIR.is_dir():
        pytest.skip(f"{OLD_SIM_DIR.name} is missing")
    out_dir = tmp_path_factory.mktemp("make_all_old")
    index, loader, printed = index_folder(OLD_SIM_DIR), Loader(whole=True), io.StringIO()
    start = time.perf_counter()
    with warnings.catch_warnings(), contextlib.redirect_stdout(printed):
        warnings.simplefilter("ignore")
        report = make_all(OLD_SIM_DIR, out_dir, index=index, loader=loader)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print(f"\n{printed.getvalue()}\nmake_all: {time.perf_counter() - start:.0f} s, peak memory {peak:.1f} GB, "
          f"{len(loader.reads)} of {len(index)} CSVs read")
    return report, out_dir, loader


def test_nothing_fails(made):
    report = made[0]
    assert not {key: value for key, value in report.items() if isinstance(value, str) and value.startswith("error")}


def test_every_paper_figure_and_table_is_written(made):
    report, out_dir, _ = made
    for stem in FIGURES:
        assert (out_dir / f"{stem}.png").is_file(), stem
    for stem in TABLES:
        assert (out_dir / "tables" / f"{stem}.tex").is_file(), stem
    assert sum(not isinstance(value, str) for value in report.values()) == len(FIGURES) + len(TABLES)


def test_only_what_the_folder_lacks_is_skipped(made):
    report = made[0]
    assert [key for key, value in report.items() if isinstance(value, str)] == SKIPPED
    assert "detailed staticlow" in report["fig2 config=staticlow"]


def test_each_csv_is_read_once(made):
    loader = made[2]
    assert set(loader.reads.values()) == {1}
