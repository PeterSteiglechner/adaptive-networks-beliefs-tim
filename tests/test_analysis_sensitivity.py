"""`anb.analysis` Fig. A8 (OFAT sensitivity), on the synthetic folder of `sim_folder.py`.

The fixture's sensitivity variants (`sim_folder.SENSITIVITY`) are seed 0 at s = 4 of staticlow, statichigh and
adaptive: the M row is complete (M ∈ {5, 15}), the τ row has τ = 2 but no τ = 10, and the other rows have nothing. Its
staticlow base at s = 4 has only seed 0. A second folder (`full_grid`) links the base runs under every OFAT name, so all
seven rows exist.
"""

import string
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim_folder  # noqa: E402
from anb.analysis import (  # noqa: E402
    Baseline, FewSeedsWarning, FigureNotApplicable, FigureResult, Loader, MissingLevelsWarning, figA8,
)
from anb.io import legacy_filename  # noqa: E402

RESPONSES = ["compliant", "resilient", "resistant", "latecompliant"]
COLUMNS = ("staticlow", "statichigh", "adaptive")
# Each OFAT row's variants as (config fields, value on the x axis), on top of the paper base (N = 100 for p·N).
OFAT = {
    "M": [(dict(M=5), 5), (dict(M=15), 15)],
    "N": [(dict(N=50, link_prob=0.2), 50), (dict(N=200, link_prob=0.05), 200)],
    "beta": [(dict(beta=1.5), 1.5), (dict(beta=6.0), 6.0)],
    "p": [(dict(link_prob=0.05), 5), (dict(link_prob=0.2), 20)],
    "tau": [(dict(tau=2), 2), (dict(tau=10), 10)],
    "eps": [(dict(eps=0.5), 0.5), (dict(eps=2.0), 2.0)],
    "init_w": [(dict(init_w=0.1), 0.1), (dict(init_w=0.4), 0.4)],
}
BASE_VALUE = {"M": 10, "N": 100, "beta": 3.0, "p": 10, "tau": 1, "eps": 1.0, "init_w": 0.2}


def expected_warnings(func, *args, **kwargs):
    """`func(...)` and the warnings it gave (the stray-file warning of the index left out)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, [w for w in caught if "legacy naming scheme" not in str(w.message)]


def stem(config, seed, **fields) -> str:
    init_w, eps, mu, fixed = sim_folder.CONFIGS[config]
    params = dict(link_prob=0.1, init_w=init_w, beta=3.0, rho=1 / 3, eps=eps, mu=mu, fixedBNat100=fixed,
                  ext_strength=4, seed=seed, lam=0.0, n_events=1)
    return legacy_filename({**params, **fields}, False)


def frequencies(path) -> pd.Series:
    df = pd.read_csv(path)
    counts = df.loc[df.t == 95.5, RESPONSES].sum().astype(float)
    return counts / counts.sum()  # NaN for a run without a negative response, as in the old script


def visible(result) -> list:
    return [ax for ax in result.figure.axes if ax.axison]


@pytest.fixture(scope="module")
def paper_figA8(sim_dir):
    result, caught = expected_warnings(figA8, sim_dir)
    yield result, caught
    plt.close(result.figure)


@pytest.fixture(scope="module")
def full_grid(sim_dir, tmp_path_factory):
    """Every OFAT name (and the base) as a link to the fixture's base run of the same config, seed 0, s = 4."""
    out = tmp_path_factory.mktemp("full_grid")
    for config in COLUMNS:
        source = sim_dir / f"{stem(config, 0)}.csv"
        (out / source.name).symlink_to(source)
        for row, variants in OFAT.items():
            if row in ("eps", "init_w") and config != "adaptive":
                continue
            for fields, _ in variants:
                (out / f"{stem(config, 0, **fields)}.csv").symlink_to(source)
    return out


# The fixture: one complete row

def test_figA8_returns_a_figure_result_with_the_old_name(paper_figA8):
    result, _ = paper_figA8
    assert isinstance(result, FigureResult)
    assert result.name == "AppendixFig_ofat"
    assert len(result.figure.axes) == 7 * 3  # the grid is kept
    assert len(visible(result)) == 3  # only the M row is drawn


def test_the_rows_without_runs_warn_and_are_left_out(paper_figA8):
    result, caught = paper_figA8
    categories = {w.category for w in caught}
    assert {MissingLevelsWarning, FewSeedsWarning} <= categories
    assert FigureNotApplicable not in categories
    [missing] = [str(w.message) for w in caught if w.category is MissingLevelsWarning]
    for row in ("N", "beta", "p", "tau", "eps", "init_w"):
        assert f"{row}:" in missing
    assert "tau=10" in missing and "tau=2," not in missing  # τ = 2 exists, τ = 10 doesn't
    assert set(result.data.row) == {"M"}


def test_the_present_row_holds_the_per_simulation_frequencies(paper_figA8, sim_dir):
    data = paper_figA8[0].data
    assert list(data.columns) == ["row", "config", "value", "base", "seed", "response", "frequency"]
    runs = data[["config", "value", "seed"]].drop_duplicates()
    assert sorted(map(tuple, runs.to_numpy())) == sorted(
        [(config, M, 0) for config in COLUMNS for M in (5, 15)]  # the `_M` suffixes, found through the index
        + [("staticlow", 10, 0)] + [(config, 10, k) for config in ("statichigh", "adaptive") for k in (0, 1, 2)])
    assert set(data.loc[data.base, "value"]) == {10}  # M = 6 (the M/N/τ variant of fig2) is not an M-row run
    for (config, value, seed), group in data.groupby(["config", "value", "seed"]):
        fields = {} if value == 10 else dict(M=int(value))
        expected = frequencies(sim_dir / f"{stem(config, seed, **fields)}.csv")
        got = group.set_index("response").frequency[RESPONSES]
        np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())


def test_the_drawn_row_is_titled_lettered_and_labelled(paper_figA8):
    result, _ = paper_figA8
    axes = visible(result)
    assert [ax.get_title() for ax in axes] == [r"static ($\omega_0=0.2$)", r"static ($\omega_0=0.8$)", "adaptive"]
    letters = [t.get_text() for ax in axes for t in ax.texts if len(t.get_text()) == 1]
    assert letters == ["A", "B", "C"]
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["5", "10", "15"]
    assert any("nr of beliefs" in t.get_text() for t in axes[2].texts)


def test_a_missing_base_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="base staticlow"):
        assert figA8(sim_dir, seeds=[1]) is None  # staticlow at s = 4 has only seed 0
    with pytest.warns(FigureNotApplicable, match="staticlow"):
        assert figA8(sim_dir, base=Baseline(link_prob=0.2)) is None  # that base has only adaptive runs


def test_no_complete_row_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="tau=10"):
        assert figA8(sim_dir, rows=["tau"]) is None


def test_a_row_subset_draws_only_those_rows_without_warning_about_the_others(sim_dir):
    result, caught = expected_warnings(figA8, sim_dir, rows=["M"])
    assert MissingLevelsWarning not in {w.category for w in caught}
    assert set(result.data.row) == {"M"}
    plt.close(result.figure)
    with pytest.raises(ValueError, match="rho"):
        figA8(sim_dir, rows=["M", "rho"])


def test_figA8_writes_nothing_prints_nothing_and_leaves_the_global_rng(sim_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before, state = sorted(sim_dir.rglob("*")), np.random.get_state()
    result, _ = expected_warnings(figA8, sim_dir)
    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())
    assert sorted(sim_dir.rglob("*")) == before
    assert np.all(np.random.get_state()[1] == state[1])
    plt.close(result.figure)


def test_a_shared_loader_reads_each_run_once(sim_dir):
    loader = Loader()
    first, _ = expected_warnings(figA8, sim_dir, loader=loader)
    cached = loader.cached
    second, _ = expected_warnings(figA8, sim_dir, loader=loader)
    assert loader.cached == cached == first.data[["config", "value", "seed"]].drop_duplicates().shape[0]
    pd.testing.assert_frame_equal(first.data, second.data)
    plt.close("all")


# The full grid

@pytest.fixture(scope="module")
def full_figA8(full_grid):
    result, caught = expected_warnings(figA8, full_grid)
    yield result, caught
    plt.close(result.figure)


def test_the_full_grid_draws_all_seven_rows(full_figA8):
    result, caught = full_figA8
    assert {w.category for w in caught} == {FewSeedsWarning}
    assert list(result.data.row.unique()) == list(OFAT)
    axes = visible(result)
    assert len(axes) == 5 * 3 + 2  # ε and ω₀ only for the adaptive config
    letters = [t.get_text() for ax in axes for t in ax.texts if len(t.get_text()) == 1]
    assert letters == list(string.ascii_uppercase[:17])


@pytest.mark.parametrize("row", list(OFAT))
def test_each_row_has_the_base_and_its_two_variants(full_figA8, row):
    data = full_figA8[0].data
    rows = data[data.row == row]
    configs = ("adaptive",) if row in ("eps", "init_w") else COLUMNS
    assert tuple(rows.config.unique()) == configs
    for config in configs:
        group = rows[rows.config == config]
        values = sorted(group.value.unique())
        assert values == sorted([BASE_VALUE[row]] + [value for _, value in OFAT[row]])  # p as p·N
        assert set(group.loc[group.base, "value"]) == {BASE_VALUE[row]}
        assert len(group) == len(values) * len(RESPONSES)  # one seed each, the base not counted twice


def test_the_tick_labels_are_the_values(full_figA8):
    axes = visible(full_figA8[0])
    first = {row: axes[3 * k] for k, row in enumerate(("M", "N", "beta", "p", "tau"))}
    first.update(eps=axes[15], init_w=axes[16])
    labels = {row: [t.get_text() for t in ax.get_xticklabels()] for row, ax in first.items()}
    assert labels == {"M": ["5", "10", "15"], "N": ["50", "100", "200"], "beta": ["1.5", "3.0", "6.0"],
                      "p": ["5", "10", "20"], "tau": ["1", "2", "10"], "eps": ["0.5", "1.0", "2.0"],
                      "init_w": ["0.1", "0.2", "0.4"]}


def test_a_non_paper_base_gets_a_suffix_and_drops_the_variant_it_equals(full_grid):
    result, _ = expected_warnings(figA8, full_grid, base=Baseline(beta=6.0), rows=["beta"])
    assert result.name == "AppendixFig_ofat_beta6.00"
    assert sorted(result.data.value.unique()) == [1.5, 6.0]
    plt.close(result.figure)


def test_save_writes_the_requested_formats(paper_figA8, tmp_path):
    paths = paper_figA8[0].save(tmp_path / "new", formats=("png",))
    assert [p.name for p in paths] == ["AppendixFig_ofat.png"]
    assert paths[0].is_file()
