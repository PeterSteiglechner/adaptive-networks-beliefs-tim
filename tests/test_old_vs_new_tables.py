"""Old vs new (seam 3): the tables and `greedy_uncorrelated_subset` against `analyse_BNmetrics.py` (μ = 0, 100 seeds).

The old script printed its LaTeX to stdout, among statsmodels' fit messages and other prints; the harness captures
it, and the tables are cut out of it. The comparison ignores whitespace. Run with
`pytest -m slow tests/test_old_vs_new_tables.py`. Skipped when `simOut_old_model` is missing.
"""

import re

import pytest

from anb.analysis import Loader, greedy_uncorrelated_subset, index_folder, table3, tableA5, tableA6, tableA7
from anb.analysis.style import METRIC_LABELS
from analysis_helpers import squash
from old_scripts import OLD_SIM_DIR

pytestmark = pytest.mark.slow

SCRIPT = "analyse_BNmetrics.py"
TABLE3_START = "% " + "#" * 50 + "\n% time"
TABLEA5_START = "% Change over time"
# Table A5's metric rows hold compliant, resilient (3 windows each), then the old cell interleaved the resistant
# columns with the s = 0 reference ones (R95, ref95, R145, ref145, R195, ref195). The package puts the reference
# last, as the paper does: these are the old cell positions in the new order.
TABLEA5_ORDER = [0, 1, 2, 3, 4, 5, 6, 8, 10, 7, 9, 11]


@pytest.fixture(scope="module")
def old(old_script):
    result = old_script(SCRIPT)
    if "chosen" not in result.globals:
        pytest.fail(result.report())
    return result


@pytest.fixture(scope="module")
def shared():
    return dict(index=index_folder(OLD_SIM_DIR), loader=Loader())


def old_table3(stdout: str) -> dict[int, str]:
    """Tables 3 and A1–A4 from the old stdout, by s."""
    section = stdout[stdout.index(TABLE3_START):stdout.index(TABLEA5_START)]
    blocks = [TABLE3_START + block for block in section.split(TABLE3_START)[1:]]
    return {int(re.search(r"% s_ext = (\d+)", block).group(1)): block for block in blocks}


def old_tableA5(stdout: str) -> str:
    """Table A5 from the old stdout, with its columns in the package's (the paper's) order."""
    start = stdout.index(TABLEA5_START)
    proportion = stdout.index("\nproportion", start)
    section = stdout[start:stdout.index("\\\\", proportion) + 2]
    lines = []
    for line in section.splitlines():
        cells = line.split("&")
        if line.startswith("  ") and len(cells) == 14:  # a metric row: label, 12 cells, the row end
            cells = [cells[0], *(cells[1 + k] for k in TABLEA5_ORDER), cells[13]]
        lines.append("&".join(cells))
    return "\n".join(lines)


def old_fit_tables(stdout: str) -> list[str]:
    """Tables A6 and A7 from the old stdout."""
    return re.findall(r"\\begin\{table\}.*?\\end\{table\}", stdout, flags=re.DOTALL)


def test_table3_matches_the_old_stdout(old, shared):
    expected = old_table3(old.stdout)
    assert list(expected) == [1, 2, 4, 8, 16]
    for s, latex in expected.items():
        table = table3(OLD_SIM_DIR, s=s, **shared)
        assert table.name == f"table3_s{s}"
        assert squash(table) == squash(latex), f"s = {s}"


def test_tableA5_matches_the_old_stdout_with_the_reference_columns_last(old, shared):
    table = tableA5(OLD_SIM_DIR, **shared)
    expected = old_tableA5(old.stdout)
    assert table.name == "tableA5_metricChange"
    assert squash(table) == squash(expected)


def test_fit_tables_match_the_old_stdout(old, shared):
    rRC, Rr = old_fit_tables(old.stdout)
    assert squash(tableA6(OLD_SIM_DIR, **shared)) == squash(rRC)
    assert squash(tableA7(OLD_SIM_DIR, **shared)) == squash(Rr)


def test_greedy_subset_matches_the_old_order(old, shared):
    order = greedy_uncorrelated_subset(OLD_SIM_DIR, **shared)
    assert [METRIC_LABELS[m] for m in order.metric] == old.globals["chosen"]
