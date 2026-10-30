"""`anb.analysis` core: the base config, run selection and its warnings, the loader and `FigureResult.save`."""

import warnings
from dataclasses import FrozenInstanceError

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim_folder  # noqa: E402
from anb.analysis import (  # noqa: E402
    PAPER, Baseline, Cell, FewSeedsWarning, FigureNotApplicable, FigureResult, Loader, index_folder, runs, select,
)


@pytest.fixture(scope="module")
def index(sim_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the stray files
        return index_folder(sim_dir)


# Baseline

def test_the_default_base_is_the_papers_adaptive_baseline():
    assert Baseline() == PAPER
    assert (PAPER.link_prob, PAPER.beta, PAPER.rho, PAPER.init_w, PAPER.eps, PAPER.mu, PAPER.fixedBNat100) == \
        (0.1, 3.0, 0.33, 0.2, 1.0, 0.0, False)
    assert (PAPER.lam, PAPER.n_events, PAPER.M, PAPER.N, PAPER.tau) == (0.0, 1, 10, 100, 1)
    with pytest.raises(FrozenInstanceError):
        PAPER.N = 50


@pytest.mark.parametrize("config, values", [
    ("staticlow", (0.2, 0.0, 0.0, False)),
    ("statichigh", (0.8, 0.0, 0.0, False)),
    ("adaptive2static", (0.2, 1.0, 0.0, True)),
    ("adaptive", (0.2, 1.0, 0.0, False)),
])
def test_the_four_configs_replace_only_the_bn_fields(config, values):
    base = Baseline(link_prob=0.2, mu=0.05, N=50)
    derived = base.config(config)

    assert (derived.init_w, derived.eps, derived.mu, derived.fixedBNat100) == values
    assert (derived.link_prob, derived.N) == (0.2, 50)


def test_an_unknown_config_is_an_error():
    with pytest.raises(ValueError, match="staticlow"):
        PAPER.config("static")


def test_the_paper_base_has_no_name_suffix():
    assert PAPER.suffix() == ""
    assert Baseline(rho=1 / 3).suffix() == ""  # compared as rounded in file names


@pytest.mark.parametrize("base, suffix", [
    (Baseline(link_prob=0.2), "_link_prob0.20"),
    (Baseline(M=15), "_M15.00"),
    (Baseline(mu=0.005), "_mu0.005"),
    (Baseline(fixedBNat100=True), "_fixedBNat100"),
    (Baseline(lam=0.05, n_events=2), "_lambda0.0500_2events"),
    (Baseline(tau=2, link_prob=0.2, N=50), "_link_prob0.20_N50.00_tau2.00"),  # in field order
])
def test_a_non_paper_base_appends_its_differing_fields_as_legacy_tokens(base, suffix):
    assert base.suffix() == suffix


def test_the_suffix_can_leave_out_fields_a_figure_sets_itself():
    assert Baseline(mu=0.005, link_prob=0.2).suffix(ignore=("init_w", "eps", "mu", "fixedBNat100")) == \
        "_link_prob0.20"


# runs / select

def test_runs_match_the_base_with_the_filename_rounding(index):
    found = runs(index, Baseline(rho=1 / 3).config("staticlow"), s=1)

    assert sorted(found.seed) == list(sim_folder.SEEDS)
    assert not found.detailed.any()
    assert (found.N == 100).all()


def test_runs_ignore_runs_that_differ_from_the_base(index):
    assert set(runs(index, PAPER, s=4).seed) == {0, 1, 2}  # not the link_prob 0.2 or M/N/tau runs
    assert set(runs(index, Baseline(link_prob=0.2), s=4).seed) == {0, 1}
    assert set(runs(index, Baseline(**sim_folder.VARIANT), s=4, detailed=True).seed) == {2}


def test_select_returns_the_matching_runs_per_cell(index):
    cells = [Cell("low s=1", PAPER.config("staticlow"), 1), Cell("high s=4", PAPER.config("statichigh"), 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selected = select(index, cells, item="test", paper_seeds=3)

    assert list(selected) == ["low s=1", "high s=4"]
    assert all(sorted(frame.seed) == [0, 1, 2] for frame in selected.values())


def test_select_restricts_to_the_requested_seeds(index):
    selected = select(index, [Cell("one", PAPER, 4, detailed=True, seeds=(2,))], item="test", paper_seeds=1)

    assert list(selected["one"].seed) == [2]


def test_a_missing_cell_is_not_applicable_and_named(index):
    cells = [Cell("a2s s=1", PAPER.config("adaptive2static"), 1), Cell("low s=1", PAPER.config("staticlow"), 1)]

    with pytest.warns(FigureNotApplicable, match="a2s s=1") as caught:
        assert select(index, cells, item="figX", paper_seeds=1) is None

    assert len(caught) == 1
    assert "figX" in str(caught[0].message) and "low s=1" not in str(caught[0].message)


def test_fewer_seeds_than_the_paper_warn_but_select(index):
    cells = [Cell("low s=4", PAPER.config("staticlow"), 4), Cell("high s=4", PAPER.config("statichigh"), 4)]

    with pytest.warns(FewSeedsWarning) as caught:
        selected = select(index, cells, item="figX", paper_seeds=100)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "low s=4: 1" in message and "high s=4: 3" in message and "100" in message
    assert len(selected["low s=4"]) == 1


def test_fewer_than_min_seeds_is_not_applicable(index):
    cells = [Cell("low s=4", PAPER.config("staticlow"), 4)]

    with pytest.warns(FigureNotApplicable, match="low s=4"):
        assert select(index, cells, item="figX", paper_seeds=1, min_seeds=2) is None


def test_the_warnings_are_user_warnings():
    assert issubclass(FigureNotApplicable, UserWarning)
    assert issubclass(FewSeedsWarning, UserWarning)


# Loader

def test_the_loader_derives_the_layout_from_the_data(index):
    path = runs(index, Baseline(**sim_folder.VARIANT), s=4, detailed=True).path.iloc[0]

    run = Loader().load(path, t=[0, 100], columns=["tb_tot"], beliefs=True, edges=True)

    assert (run.layout.M, run.layout.N, run.layout.tau) == (6, 50, 2)
    assert run.layout.beliefs == ["0", "1", "2", "3", "4", "5"]
    assert run.layout.edges[:6] == ["w01", "w02", "w03", "w04", "w05", "w12"] and len(run.layout.edges) == 15
    assert list(run.frame.columns) == ["t", "id", "tb_tot", *run.layout.beliefs, *run.layout.edges]
    assert sorted(run.frame.t.unique()) == [0, 100]
    assert run.frame.id.dtype.kind == "i" and run.frame.id.nunique() == 50


def test_a_shared_loader_reads_each_request_once(index, monkeypatch):
    path = runs(index, PAPER, s=4).path.iloc[0]
    loader = Loader()
    reads = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: reads.append(a) or real(*a, **k))

    first = loader.load(path, columns=["tb_tot"])
    n_reads = len(reads)
    second = loader.load(path, columns=["tb_tot"])

    assert len(reads) == n_reads
    assert second.frame is first.frame


# FigureResult

def make_result(name="fig_test"):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return FigureResult(fig, pd.DataFrame({"x": [0, 1], "y": [0, 1]}), name)


@pytest.mark.parametrize("formats, suffixes", [(("png",), [".png"]), (("png", "pdf"), [".pdf", ".png"]),
                                               (("pdf",), [".pdf"])])
def test_save_writes_exactly_the_requested_formats_into_a_new_directory(tmp_path, formats, suffixes):
    out_dir = tmp_path / "not" / "yet"
    result = make_result()

    paths = result.save(out_dir, formats=formats)

    assert sorted(p.suffix for p in paths) == suffixes
    assert sorted(out_dir.iterdir()) == sorted(paths)
    assert all(p.stem == "fig_test" for p in paths)
    plt.close(result.figure)


def test_save_defaults_to_png_at_600_dpi(tmp_path):
    result = make_result()

    (path,) = result.save(tmp_path)

    from PIL import Image
    with Image.open(path) as image:
        assert round(image.info["dpi"][0]) == 600
    plt.close(result.figure)


def test_save_rejects_unknown_formats(tmp_path):
    result = make_result()
    with pytest.raises(ValueError, match="svg"):
        result.save(tmp_path, formats=("svg",))
    assert not list(tmp_path.iterdir())
    plt.close(result.figure)
