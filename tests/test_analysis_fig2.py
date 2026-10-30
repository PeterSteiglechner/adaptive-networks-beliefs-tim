"""`anb.analysis.fig2` (Fig. 2, A1, A2) and its summaries, on the synthetic folder of `sim_folder.py`."""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim_folder  # noqa: E402
from anb.analysis import (  # noqa: E402
    Baseline, ExampleAgentFallback, FigureNotApplicable, FigureResult, Loader, example_agent_stats, fig2,
    focal_sign_counts,
)

VARIANT = Baseline(**sim_folder.VARIANT)  # M 6, N 50, tau 2


def quietly(func, *args, **kwargs):
    """`func(...)`, failing on any warning other than the stray-file one of the index."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=".*don't follow the legacy naming scheme")
        return func(*args, **kwargs)


@pytest.fixture(scope="module")
def paper_fig2(sim_dir):
    result = quietly(fig2, sim_dir)
    yield result
    plt.close(result.figure)


def test_fig2_returns_a_figure_result_with_the_old_name(paper_fig2):
    assert isinstance(paper_fig2, FigureResult)
    assert isinstance(paper_fig2.figure, matplotlib.figure.Figure)
    assert paper_fig2.name == "fig2_adaptive_t100"


def test_fig2_data_holds_the_trajectories_and_the_example_bns(paper_fig2, sim_dir):
    data = paper_fig2.data
    run = pd.read_csv(next((sim_dir / "detailed").glob("*link_prob0.10*_ext_strength4_seed2_detailed.csv")))

    focal = data[data.quantity == "focal"]
    assert sorted(focal.agent.unique()) == list(range(100))
    assert sorted(focal.t.unique()) == list(range(101))
    expected = run[run.t <= 100].set_index(["t", "id"])["0"]
    got = focal.set_index(["t", "agent"])["value"]
    np.testing.assert_array_equal(got.sort_index().to_numpy(), expected.sort_index().to_numpy())

    for quantity, columns in (("belief", [str(m) for m in range(10)]),
                              ("edge", [c for c in run.columns if c.startswith("w")])):
        rows = data[data.quantity == quantity]
        assert sorted(rows.agent.unique()) == [61, 79]
        assert sorted(rows.t.unique()) == [0, 100]
        assert sorted(rows["item"].unique()) == sorted(columns)
        for (agent, t), group in rows.groupby(["agent", "t"]):
            row = run[(run.id == agent) & (run.t == t)].iloc[0]
            np.testing.assert_array_equal(group.set_index("item").value[columns].to_numpy(),
                                          row[columns].to_numpy(float))


def test_fig2_has_the_papers_panels(paper_fig2):
    axes = paper_fig2.figure.axes
    assert len(axes) == 6  # trajectories, histogram, two BNs per example agent
    texts = [t.get_text() for t in paper_fig2.figure.texts]
    assert "agent 61" in texts and "agent 79" in texts
    assert axes[0].get_xlabel() == "time $t$"


def test_fig2_writes_nothing_and_prints_nothing(sim_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before = sorted(p for p in sim_dir.rglob("*"))

    result = quietly(fig2, sim_dir)

    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())
    assert sorted(p for p in sim_dir.rglob("*")) == before
    plt.close(result.figure)


def test_a_missing_detailed_run_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="staticlow"):
        assert fig2(sim_dir, config="staticlow") is None
    with pytest.warns(FigureNotApplicable, match="seed=1"):
        assert fig2(sim_dir, seed=1) is None


@pytest.mark.parametrize("config, name", [("staticlow", "fig2_staticlow_t100"), ("statichigh", "fig2_statichigh_t100"),
                                          ("adaptive2static", "fig2_adaptive2static_t100")])
def test_the_name_follows_the_config(sim_dir, config, name, tmp_path):
    """The fixture has no detailed static runs, so write one under this name's config into a copy."""
    from anb.io import legacy_filename
    src = next((sim_dir / "detailed").glob("*link_prob0.10*_ext_strength4_seed2_detailed.csv"))
    init_w, eps, mu, fixed = sim_folder.CONFIGS[config]
    stem = legacy_filename(dict(link_prob=0.1, init_w=init_w, beta=3.0, rho=1 / 3, eps=eps, mu=mu,
                                fixedBNat100=fixed, ext_strength=4, seed=2, lam=0.0, n_events=1), True)
    (tmp_path / "detailed").mkdir()
    (tmp_path / f"{stem}_detailed.csv").write_bytes(src.read_bytes())

    result = quietly(fig2, tmp_path, config=config)

    assert result.name == name
    plt.close(result.figure)


def test_a_non_paper_base_appends_its_fields_to_the_name(sim_dir):
    result = quietly(fig2, sim_dir, base=Baseline(link_prob=0.2))

    assert result.name == "fig2_adaptive_t100_link_prob0.20"
    plt.close(result.figure)


def test_an_n50_run_plots_and_the_missing_example_agents_fall_back(sim_dir):
    with pytest.warns(ExampleAgentFallback) as caught:
        result = fig2(sim_dir, base=VARIANT)

    messages = " ".join(str(w.message) for w in caught if w.category is ExampleAgentFallback)
    assert "61" in messages and "79" in messages
    agents = sorted(result.data.loc[result.data.quantity == "edge", "agent"].unique())
    assert len(agents) == 2 and all(0 <= a < 50 for a in agents)
    assert sorted(result.data.loc[result.data.quantity == "belief", "item"].unique()) == list("012345")
    assert result.data.loc[result.data.quantity == "edge", "item"].nunique() == 15
    assert result.name == "fig2_adaptive_t100_M6.00_N50.00_tau2.00"
    plt.close(result.figure)


def test_the_fallback_is_seeded(sim_dir):
    with pytest.warns(ExampleAgentFallback):
        first = fig2(sim_dir, base=VARIANT)
    with pytest.warns(ExampleAgentFallback):
        second = fig2(sim_dir, base=VARIANT)

    pick = lambda r: sorted(r.data.loc[r.data.quantity == "edge", "agent"].unique())  # noqa: E731
    assert pick(first) == pick(second)
    plt.close(first.figure), plt.close(second.figure)


def test_only_the_missing_agent_falls_back(sim_dir):
    with pytest.warns(ExampleAgentFallback, match="120") as caught:
        result = fig2(sim_dir, agents=(61, 120))

    assert len([w for w in caught if w.category is ExampleAgentFallback]) == 1
    agents = set(result.data.loc[result.data.quantity == "edge", "agent"].unique())
    assert 61 in agents and len(agents) == 2
    plt.close(result.figure)


def test_fig2_needs_two_example_agents(sim_dir):
    with pytest.raises(ValueError, match="two"):
        fig2(sim_dir, agents=(61,))


def test_a_shared_loader_is_used(sim_dir):
    loader = Loader()
    result = quietly(fig2, sim_dir, loader=loader)
    assert loader.cached
    plt.close(result.figure)


# Summaries

def test_example_agent_stats_at_t100(sim_dir):
    stats = quietly(example_agent_stats, sim_dir)
    run = pd.read_csv(next((sim_dir / "detailed").glob("*link_prob0.10*_ext_strength4_seed2_detailed.csv")))
    at100 = run[run.t == 100].set_index("id")

    assert isinstance(stats, pd.DataFrame)
    assert list(stats.index) == [61, 79]
    assert list(stats.columns) == ["tb_tot", "tb_foc", "Hpers"]
    np.testing.assert_array_equal(stats.to_numpy(), at100.loc[[61, 79], ["tb_tot", "tb_foc", "Hpers"]].to_numpy())


def test_focal_sign_counts_at_t100(sim_dir):
    counts = quietly(focal_sign_counts, sim_dir)
    run = pd.read_csv(next((sim_dir / "detailed").glob("*link_prob0.10*_ext_strength4_seed2_detailed.csv")))
    signs = np.sign(run.loc[run.t == 100, "0"])

    assert isinstance(counts, pd.DataFrame)
    assert list(counts.index) == [-1, 0, 1]
    assert counts["count"].sum() == 100
    assert counts.loc[-1, "count"] == (signs < 0).sum() and counts.loc[1, "count"] == (signs > 0).sum()


def test_summaries_are_not_applicable_without_the_run(sim_dir):
    with pytest.warns(FigureNotApplicable):
        assert example_agent_stats(sim_dir, config="statichigh") is None
    with pytest.warns(FigureNotApplicable):
        assert focal_sign_counts(sim_dir, s=1) is None
