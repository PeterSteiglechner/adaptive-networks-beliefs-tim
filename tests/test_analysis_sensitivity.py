"""`anb.analysis` Fig. A8 (OFAT sensitivity), on the synthetic folder of `sim_folder.py`.

The fixture's sensitivity variants (`sim_folder.SENSITIVITY`) are seed 0 at s = 4 of staticlow, statichigh and
adaptive: the M and N rows are complete (M ∈ {5, 15}; N ∈ {50, 200} with p = 10/N), the τ row has τ = 2 but no
τ = 10, and the other rows have nothing. Its staticlow base at s = 4 has only seed 0. A second folder (`full_grid`)
links the base runs under every OFAT name, so all seven rows exist.
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


def caught_warnings(func, *args, **kwargs):
    """`func(...)` and the list of warnings it gave (the stray-file warning of the index left out)."""
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
    result, caught = caught_warnings(figA8, sim_dir)
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


# The fixture: two complete rows

def test_figA8_returns_a_figure_result_with_the_old_name(paper_figA8):
    result, _ = paper_figA8
    assert isinstance(result, FigureResult)
    assert result.name == "AppendixFig_ofat"
    assert len(result.figure.axes) == 7 * 3  # the grid is kept
    assert len(visible(result)) == 6  # only the M and N rows are drawn


def test_the_rows_without_runs_warn_and_are_left_out(paper_figA8):
    result, caught = paper_figA8
    categories = {w.category for w in caught}
    assert {MissingLevelsWarning, FewSeedsWarning} <= categories
    assert FigureNotApplicable not in categories
    [missing] = [str(w.message) for w in caught if w.category is MissingLevelsWarning]
    for row in ("beta", "p", "tau", "eps", "init_w"):
        assert f"{row}:" in missing
    assert "M:" not in missing and "N:" not in missing
    assert "tau=10" in missing and "tau=2," not in missing  # τ = 2 exists, τ = 10 doesn't
    assert set(result.data.row) == {"M", "N"}


@pytest.mark.parametrize("row", ["M", "N"])
def test_the_present_rows_hold_the_per_simulation_frequencies(paper_figA8, sim_dir, row):
    data = paper_figA8[0].data
    assert list(data.columns) == ["row", "config", "value", "base", "seed", "response", "frequency"]
    data = data[data.row == row]
    base = BASE_VALUE[row]
    runs = data[["config", "value", "seed"]].drop_duplicates()
    assert sorted(map(tuple, runs.to_numpy())) == sorted(
        [(config, value, 0) for config in COLUMNS for _, value in OFAT[row]]  # the `_M`/`_N` suffixes, via the index
        + [("staticlow", base, 0)] + [(config, base, k) for config in ("statichigh", "adaptive") for k in (0, 1, 2)])
    assert set(data.loc[data.base, "value"]) == {base}  # M = 6, N = 50 (the M/N/τ variant of fig2) is no OFAT run
    variants = {value: fields for fields, value in OFAT[row]}
    for (config, value, seed), group in data.groupby(["config", "value", "seed"]):
        fields = {} if value == base else variants[value]
        expected = frequencies(sim_dir / f"{stem(config, seed, **fields)}.csv")
        got = group.set_index("response").frequency[RESPONSES]
        np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())


def test_the_drawn_rows_are_titled_lettered_and_labelled(paper_figA8):
    result, _ = paper_figA8
    axes = visible(result)
    titles = [r"static ($\omega_0=0.2$)", r"static ($\omega_0=0.8$)", "adaptive"]
    assert [ax.get_title() for ax in axes] == titles + ["", "", ""]  # only the top drawn row is titled
    letters = [t.get_text() for ax in axes for t in ax.texts if len(t.get_text()) == 1]
    assert letters == ["A", "B", "C", "D", "E", "F"]
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["5", "10", "15"]
    assert [t.get_text() for t in axes[3].get_xticklabels()] == ["50", "100", "200"]
    assert any("nr of beliefs" in t.get_text() for t in axes[2].texts)
    assert any("nr of agents" in t.get_text() for t in axes[5].texts)


def test_a_missing_base_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="base staticlow"):
        assert figA8(sim_dir, seeds=[1]) is None  # staticlow at s = 4 has only seed 0
    with pytest.warns(FigureNotApplicable, match="staticlow"):
        assert figA8(sim_dir, base=Baseline(link_prob=0.2)) is None  # that base has only adaptive runs


def test_no_complete_row_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="tau=10"):
        assert figA8(sim_dir, rows=["tau"]) is None


def test_a_row_subset_draws_only_those_rows_without_warning_about_the_others(sim_dir):
    result, caught = caught_warnings(figA8, sim_dir, rows=["M"])
    assert MissingLevelsWarning not in {w.category for w in caught}
    assert set(result.data.row) == {"M"}
    plt.close(result.figure)
    with pytest.raises(ValueError, match="rho"):
        figA8(sim_dir, rows=["M", "rho"])


def test_figA8_writes_nothing_prints_nothing_and_leaves_the_global_rng(sim_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before, state = sorted(sim_dir.rglob("*")), np.random.get_state()
    result, _ = caught_warnings(figA8, sim_dir)
    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())
    assert sorted(sim_dir.rglob("*")) == before
    assert np.all(np.random.get_state()[1] == state[1])
    plt.close(result.figure)


def test_a_shared_loader_reads_each_run_once(sim_dir):
    loader = Loader()
    first, _ = caught_warnings(figA8, sim_dir, loader=loader)
    second, _ = caught_warnings(figA8, sim_dir, loader=loader)
    data = first.data
    n_runs = (len(data[data.base].drop_duplicates(["config", "seed"]))  # the base's runs, shared by the rows
              + len(data[~data.base].drop_duplicates(["row", "config", "value", "seed"])))
    assert len(loader.reads) == n_runs
    assert set(loader.reads.values()) == {1}  # each read from disk once, over both calls
    pd.testing.assert_frame_equal(first.data, second.data)
    plt.close("all")


# The full grid

@pytest.fixture(scope="module")
def full_figA8(full_grid):
    result, caught = caught_warnings(figA8, full_grid)
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
    result, _ = caught_warnings(figA8, full_grid, base=Baseline(beta=6.0), rows=["beta"])
    assert result.name == "AppendixFig_ofat_beta6.00"
    assert sorted(result.data.value.unique()) == [1.5, 6.0]
    plt.close(result.figure)


# A base with N = 50 (p = 0.1), its runs linked to the fixture's N-row runs

def link_n50_base(sim_dir, out, names):
    """Each of `names` (config fields) under `out`, for the three configs, linked to the fixture's N = 50 run."""
    for config in COLUMNS:
        source = sim_dir / f"{stem(config, 0, N=50, link_prob=0.2)}.csv"
        for fields in names:
            (out / f"{stem(config, 0, **fields)}.csv").symlink_to(source)


def test_a_variant_shown_at_the_bases_value_is_dropped_with_a_warning(sim_dir, tmp_path):
    # The N row's N = 50, p = 0.2 differs from this base (p = 0.1) but would share its tick, N = 50.
    link_n50_base(sim_dir, tmp_path, [dict(N=50), dict(N=50, link_prob=0.2)])
    for config in COLUMNS:
        name = f"{stem(config, 0, N=200, link_prob=0.05)}.csv"
        (tmp_path / name).symlink_to(sim_dir / name)
    result, caught = caught_warnings(figA8, tmp_path, base=Baseline(N=50), rows=["N"])
    [dropped] = [str(w.message) for w in caught if w.category is MissingLevelsWarning]
    assert all(f"{config} N=50, link_prob=0.2" in dropped for config in COLUMNS)
    for config in COLUMNS:
        group = result.data[result.data.config == config]
        assert sorted(group.value.unique()) == [50, 200]
        assert set(group.loc[group.base, "value"]) == {50}
        assert len(group) == 2 * len(RESPONSES)  # the base and N = 200: the variant isn't pooled with the base
    plt.close(result.figure)


def test_p_is_shown_as_the_mean_number_of_contacts_unrounded(sim_dir, tmp_path):
    link_n50_base(sim_dir, tmp_path, [dict(N=50, link_prob=p) for p in (0.05, 0.1, 0.2)])
    result, _ = caught_warnings(figA8, tmp_path, base=Baseline(N=50), rows=["p"])
    assert sorted(result.data.value.unique()) == [2.5, 5, 10]  # p·N with N = 50, not 2 for p = 0.05
    assert [t.get_text() for t in visible(result)[0].get_xticklabels()] == ["2.5", "5", "10"]
    plt.close(result.figure)


def test_save_writes_the_requested_formats(paper_figA8, tmp_path):
    paths = paper_figA8[0].save(tmp_path / "new", formats=("png",))
    assert [p.name for p in paths] == ["AppendixFig_ofat.png"]
    assert paths[0].is_file()
