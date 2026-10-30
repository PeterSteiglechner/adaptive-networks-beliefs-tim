"""`anb.analysis` Fig. 6 and Fig. A7 (social adaptation), on the synthetic folder of `sim_folder.py`.

The fixture's paper base has the adaptive config at s = 4 for μ = 0 (seeds {0, 1, 2}) and μ ∈ {0.005, 0.2} (seed 1
only, `sim_folder.SOCIAL`); the M/N/τ variant has μ = 0 (seeds {0, 1}) and μ = 0.2 (seed 1). Fig. 6's and Fig. A7's
example simulations are therefore passed as seed 1 (the paper's Fig. A7 uses seed 99).
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim_folder  # noqa: E402
from anb.analysis import (  # noqa: E402
    Baseline, FewSeedsWarning, FigureNotApplicable, FigureResult, MissingLevelsWarning, fig6, figA7,
)
from anb.analysis.social import FIG6_METRICS, PAPER_MUS  # noqa: E402
from anb.io import legacy_filename  # noqa: E402

SEED = sim_folder.SOCIAL["seed"]
MUS = (0.0, *sim_folder.SOCIAL["mus"])  # the paper base's μ values at s = 4
VARIANT = Baseline(**sim_folder.VARIANT)
PLOTTED = ["compliant", "resilient", "resistant"]


def expected_warnings(func, *args, **kwargs):
    """`func(...)` and the categories it warned with (the stray-file warning of the index left out)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, {w.category for w in caught if "legacy naming scheme" not in str(w.message)}


def adaptive_stem(mu, seed, s=4, **sizes) -> str:
    return legacy_filename(dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=1 / 3, eps=1.0, mu=mu, fixedBNat100=False,
                                ext_strength=s, seed=seed, lam=0.0, n_events=1, **sizes), False)


def adaptive_csv(sim_dir, mu, seed, s=4, **sizes) -> pd.DataFrame:
    return pd.read_csv(sim_dir / f"{adaptive_stem(mu, seed, s, **sizes)}.csv")


def seeds_of(mu):
    return sim_folder.SEEDS if mu == 0 else (SEED,)


def edge_columns(df):
    return [c for c in df.columns if c.startswith("w") and c[1:].isdigit()]


def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def axis_with_ylabel(fig, label):
    [ax] = [ax for ax in fig.axes if ax.get_ylabel() == label]
    return ax


@pytest.fixture(scope="module")
def paper_fig6(sim_dir):
    result, categories = expected_warnings(fig6, sim_dir)
    yield result, categories
    plt.close(result.figure)


# Fig. 6

def test_fig6_returns_a_figure_result_with_the_old_name(paper_fig6):
    result, categories = paper_fig6
    assert isinstance(result, FigureResult)
    assert result.name == "fig6_socialAdaptation"
    assert len(result.figure.axes) == 8 + 8 + 3  # 2 × 8 example BNs, the spacer, panels C and D
    assert MissingLevelsWarning in categories  # 3 of the paper's 11 μ values
    assert FewSeedsWarning in categories
    assert FigureNotApplicable not in categories


def test_fig6_uses_the_paper_mu_values_present(paper_fig6):
    data = paper_fig6[0].data
    assert sorted(data.mu.unique()) == list(MUS)
    ax = axis_with_ylabel(paper_fig6[0].figure, "relative response frequency")
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(mu) for mu in MUS]
    assert set(MUS) <= set(PAPER_MUS)


def test_fig6_heterogeneity_is_the_mean_distance_of_the_edge_vectors_z_scored_over_runs(paper_fig6, sim_dir):
    data = paper_fig6[0].data
    rows = data[data.quantity == "H"].sort_values(["mu", "seed"], ignore_index=True)
    expected = []
    for mu in MUS:
        for seed in seeds_of(mu):
            df = adaptive_csv(sim_dir, mu, seed)
            expected.append(pdist(df.loc[df.t == 100, edge_columns(df)].to_numpy(), metric="euclidean").mean())
    expected = pd.Series(expected)
    assert (rows.panel == "C").all()
    np.testing.assert_allclose(rows.value, expected)
    np.testing.assert_allclose(rows.z, zscore(expected))


def test_fig6_metrics_are_z_scored_over_every_agent_and_mu(paper_fig6, sim_dir):
    data = paper_fig6[0].data
    before = pd.concat([adaptive_csv(sim_dir, mu, seed).query("t == 95.5").assign(mu=mu, seed=seed)
                        for mu in MUS for seed in seeds_of(mu)], ignore_index=True)
    for metric in FIG6_METRICS:
        rows = data[data.quantity == metric].sort_values(["mu", "seed", "id"], ignore_index=True)
        expected = before.sort_values(["mu", "seed", "id"], ignore_index=True)[metric]
        assert len(rows) == 5 * 100  # 5 runs × N
        np.testing.assert_allclose(rows.value, expected, equal_nan=True)
        np.testing.assert_allclose(rows.z, zscore(expected), equal_nan=True)


def test_fig6_frequencies_leave_latecompliant_out_of_the_denominator(paper_fig6, sim_dir):
    data = paper_fig6[0].data
    rows = data[data.quantity == "frequency"]
    assert set(rows.response) == set(PLOTTED)
    assert (rows.panel == "D").all()
    for mu in MUS:
        for seed in seeds_of(mu):
            df = adaptive_csv(sim_dir, mu, seed)
            counts = df.loc[df.t == 95.5, PLOTTED].astype(float).sum()
            got = rows[(rows.mu == mu) & (rows.seed == seed)].set_index("response").value
            np.testing.assert_allclose(got.loc[PLOTTED], counts / counts.sum())


def test_fig6_can_count_latecompliant_too(sim_dir):
    result, _ = expected_warnings(fig6, sim_dir, denominator=(*PLOTTED, "latecompliant"))
    rows = result.data[(result.data.quantity == "frequency") & (result.data.mu == 0.2)]
    df = adaptive_csv(sim_dir, 0.2, SEED)
    counts = df.loc[df.t == 95.5, [*PLOTTED, "latecompliant"]].astype(float).sum()
    np.testing.assert_allclose(rows.set_index("response").value.loc[PLOTTED], counts[PLOTTED] / counts.sum())
    plt.close(result.figure)
    with pytest.raises(ValueError, match="denominator"):
        fig6(sim_dir, denominator=("compliant", "resilient"))


def test_fig6_panel_d_tick_labels_are_the_tick_positions_as_percentages(paper_fig6):
    """The bug fix: the old script set the labels on unfixed ticks, so they read 1.25× too large."""
    ax = axis_with_ylabel(paper_fig6[0].figure, "relative response frequency")
    ticks = ax.get_yticks()
    assert list(ticks) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert [t.get_text() for t in ax.get_yticklabels()] == [rf"${round(x * 100)}\,\%$" for x in ticks]


def test_fig6_example_networks_are_drawn_with_a_local_rng_over_n(paper_fig6, sim_dir):
    data = paper_fig6[0].data
    rs = np.random.RandomState(42)
    agents = [rs.randint(100) for _ in range(8)]
    for panel, mu in zip("AB", sim_folder.SOCIAL["mus"]):
        rows = data[data.panel == panel]
        assert set(rows.mu) == {mu} and set(rows.seed) == {SEED}
        assert rows.id.to_numpy()[::45].tolist() == agents  # 45 edges per example network, in draw order
        df = adaptive_csv(sim_dir, mu, SEED).query("t == 100").set_index("id")
        for agent in set(agents):
            got = rows[rows.id == agent].drop_duplicates("quantity").set_index("quantity").value
            np.testing.assert_allclose(got, df.loc[agent, got.index].to_numpy(float))
    state = np.random.get_state()
    fig6(sim_dir, agent_rng_seed=0)
    assert np.array_equal(np.random.get_state()[1], state[1])  # the global RNG is left alone
    plt.close("all")


def test_fig6_without_an_example_mu_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="0.05"):
        assert fig6(sim_dir, example_mus=(0.005, 0.05)) is None


@pytest.fixture
def extra_mu_dir(sim_dir, tmp_path):
    """The fixture's runs (linked) plus μ = 0.05 at s = 4 for seeds {0, 2}, so that μ = 0.005 and 0.2 can drop out."""
    for path in sim_dir.glob("*.csv"):
        (tmp_path / path.name).symlink_to(path)
    source = sim_dir / f"{adaptive_stem(0.005, SEED)}.csv"
    for seed in (0, 2):
        (tmp_path / f"{adaptive_stem(0.05, seed)}.csv").symlink_to(source)
    return tmp_path


@pytest.mark.parametrize("kwargs", [dict(min_seeds=2), dict(seeds=(0, 2))], ids=["min_seeds", "seeds"])
def test_fig6_with_an_example_mu_left_off_the_x_axis_is_not_applicable(extra_mu_dir, kwargs):
    with pytest.warns(FigureNotApplicable, match="0.005"):
        assert fig6(extra_mu_dir, **kwargs) is None


def test_fig6_matches_the_example_mus_to_the_plotted_mus_with_a_tolerance(sim_dir):
    result, categories = expected_warnings(fig6, sim_dir, mus=[0.0, 0.005, 0.05, 0.2 + 1e-12])
    assert FigureNotApplicable not in categories
    assert set(result.data[result.data.panel == "B"].mu) == {0.2 + 1e-12}  # the plotted value
    plt.close(result.figure)


def test_fig6_without_the_example_seed_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="seed=2"):
        assert fig6(sim_dir, example_seed=2) is None


def test_fig6_needs_two_mu_values(sim_dir):
    with pytest.warns(FigureNotApplicable, match="two"):
        assert fig6(sim_dir, base=Baseline(link_prob=0.2)) is None  # μ = 0 only


def test_fig6_with_listed_mus_warns_about_the_missing_ones(sim_dir):
    result, categories = expected_warnings(fig6, sim_dir, mus=(0.0, 0.005, 0.05, 0.2))
    assert MissingLevelsWarning in categories
    assert sorted(result.data.mu.unique()) == list(MUS)
    plt.close(result.figure)
    with pytest.raises(ValueError, match="example"):
        fig6(sim_dir, mus=(0.0, 0.005))


def test_fig6_renders_for_an_m_n_tau_variant(sim_dir):
    result, categories = expected_warnings(fig6, sim_dir, base=VARIANT, example_mus=(0.0, 0.2))
    assert FigureNotApplicable not in categories
    assert result.name == "fig6_socialAdaptation_M6.00_N50.00_tau2.00"
    data = result.data
    assert sorted(data.mu.unique()) == [0.0, 0.2]
    examples = data[data.panel.isin(["A", "B"])]
    assert examples.id.max() < 50
    assert examples.quantity.nunique() == 15  # the edges of M = 6 beliefs
    assert len(data[data.quantity == "tb_foc"]) == 3 * 50
    plt.close(result.figure)


# Fig. A7

@pytest.fixture(scope="module")
def paper_figA7(sim_dir):
    result, categories = expected_warnings(figA7, sim_dir, seed=SEED)
    yield result, categories
    plt.close(result.figure)


def test_figA7_returns_a_figure_result_with_the_old_name(paper_figA7):
    result, categories = paper_figA7
    assert isinstance(result, FigureResult)
    assert result.name == "AppendixFig_edgeweight_distributions"
    assert len(result.figure.axes) == 3
    assert [ax.get_title(loc="center") for ax in result.figure.axes] == [rf"$\mu={mu}$" for mu in MUS]
    assert FigureNotApplicable not in categories


def test_figA7_data_holds_every_edge_weight_at_t100(paper_figA7, sim_dir):
    data = paper_figA7[0].data
    assert list(data.columns) == ["mu", "edge", "id", "value"]
    for mu in MUS:
        df = adaptive_csv(sim_dir, mu, SEED).query("t == 100")
        long = df.melt(id_vars="id", value_vars=edge_columns(df), var_name="edge")
        got = data[data.mu == mu].reset_index(drop=True)
        assert len(got) == 45 * 100
        np.testing.assert_allclose(got.value, long.value)
        assert got.edge.tolist() == long.edge.tolist()


def test_figA7_without_its_seed_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="seed=99"):
        assert figA7(sim_dir) is None  # the paper's seed 99 isn't in the fixture


def test_figA7_renders_for_an_m_n_tau_variant(sim_dir):
    result, categories = expected_warnings(figA7, sim_dir, base=VARIANT, seed=SEED, mus=(0.0, 0.2))
    assert FigureNotApplicable not in categories
    assert result.name == "AppendixFig_edgeweight_distributions_M6.00_N50.00_tau2.00"
    assert len(result.figure.axes) == 2
    assert result.data.edge.nunique() == 15
    assert len(result.data) == 2 * 15 * 50
    plt.close(result.figure)
