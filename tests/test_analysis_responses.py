"""`anb.analysis` Fig. 3, Fig. A3 and the polarisation summary, on the synthetic folder of `sim_folder.py`.

The fixture has s ∈ {0, 1, 4} × seeds {0, 1, 2} for the four configs, no `adaptive2static` at s = 1 and only seed 0
of `staticlow` at s = 4, so Fig. 3 has one complete pressure level (s = 4). Its one detailed run is seed 2.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim_folder  # noqa: E402
from analysis_helpers import expected_warnings  # noqa: E402
from anb.analysis import (  # noqa: E402
    Baseline, ExampleAgentFallback, FewSeedsWarning, FigureNotApplicable, FigureResult, MissingLevelsWarning,
    fig3, figA3, polarisation_summary,
)
from anb.io import legacy_filename  # noqa: E402

RESPONSES = ["compliant", "resilient", "resistant", "latecompliant"]
EXAMPLE = dict(example_seed=sim_folder.DETAILED["seed"])  # the fixture's detailed run is seed 2, not the paper's 1


def ensemble_csv(sim_dir, config, s, seed, link_prob=0.1) -> pd.DataFrame:
    init_w, eps, mu, fixed = sim_folder.CONFIGS[config]
    stem = legacy_filename(dict(link_prob=link_prob, init_w=init_w, beta=3.0, rho=1 / 3, eps=eps, mu=mu,
                                fixedBNat100=fixed, ext_strength=s, seed=seed, lam=0.0, n_events=1), False)
    return pd.read_csv(sim_dir / f"{stem}.csv")


@pytest.fixture(scope="module")
def paper_fig3(sim_dir):
    result, categories = expected_warnings(fig3, sim_dir, **EXAMPLE)
    yield result, categories
    plt.close(result.figure)


# Fig. 3

def test_fig3_returns_a_figure_result_with_the_old_name(paper_fig3):
    result, _ = paper_fig3
    assert isinstance(result, FigureResult)
    assert result.name == "fig3_responseFrequencies"
    assert len(result.figure.axes) == 5  # the example simulation and one panel per config


def test_fig3_warns_about_missing_levels_and_few_seeds_but_renders(paper_fig3):
    result, categories = paper_fig3
    assert {MissingLevelsWarning, FewSeedsWarning} <= categories
    assert FigureNotApplicable not in categories
    assert sorted(result.data.s.unique()) == [4]  # s = 1 lacks adaptive2static; 2, 8, 16 don't exist


def test_the_missing_level_warning_names_the_levels(sim_dir):
    with pytest.warns(MissingLevelsWarning, match=r"s = 1, 2, 8, 16") as caught:
        fig3(sim_dir, **EXAMPLE)
    assert any("adaptive2static" in str(w.message) for w in caught if w.category is MissingLevelsWarning)
    plt.close("all")


def test_fig3_data_holds_the_per_simulation_relative_frequencies(paper_fig3, sim_dir):
    data = paper_fig3[0].data
    assert list(data.columns) == ["config", "s", "seed", "response", "frequency"]
    runs = data[["config", "seed"]].drop_duplicates()
    assert sorted(map(tuple, runs.to_numpy())) == sorted(
        [("adaptive", k) for k in (0, 1, 2)] + [("adaptive2static", k) for k in (0, 1, 2)]
        + [("statichigh", k) for k in (0, 1, 2)] + [("staticlow", 0)])
    for (config, seed), group in data.groupby(["config", "seed"]):
        df = ensemble_csv(sim_dir, config, 4, seed)
        counts = df.loc[df.t == 95.5, RESPONSES].sum()
        expected = counts / counts.sum()  # the denominator includes latecompliant, as in the paper
        got = group.set_index("response").frequency[RESPONSES]
        np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())


def test_fig3_restricted_to_a_seed_subset(sim_dir):
    result, _ = expected_warnings(fig3, sim_dir, seeds=[0, 1], **EXAMPLE)
    assert sorted(result.data.seed.unique()) == [0, 1]
    plt.close(result.figure)
    with pytest.warns(FigureNotApplicable):  # staticlow at s = 4 has only seed 0
        assert fig3(sim_dir, seeds=[1, 2], **EXAMPLE) is None


def test_a_missing_config_is_not_applicable(sim_dir):
    """The `link_prob 0.2` base has only adaptive runs."""
    with pytest.warns(FigureNotApplicable, match="staticlow"):
        assert fig3(sim_dir, base=Baseline(link_prob=0.2), **EXAMPLE) is None


def test_a_missing_example_run_is_not_applicable(sim_dir):
    with pytest.warns(FigureNotApplicable, match="seed=1"):
        assert fig3(sim_dir) is None  # the paper's example seed 1 has no detailed run in the fixture


def test_a_wrong_example_agent_falls_back_to_one_of_the_right_type(sim_dir):
    with pytest.warns(ExampleAgentFallback, match="999"):
        result = fig3(sim_dir, example_agents={"resistant": 999, "resilient": None, "compliant": None}, **EXAMPLE)
    plt.close(result.figure)


def test_fig3_writes_nothing_and_prints_nothing(sim_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before = sorted(sim_dir.rglob("*"))
    result, _ = expected_warnings(fig3, sim_dir, **EXAMPLE)
    assert capsys.readouterr() == ("", "")
    assert not list(tmp_path.iterdir())
    assert sorted(sim_dir.rglob("*")) == before
    plt.close(result.figure)


# Fig. A3

@pytest.fixture(scope="module")
def six_pressures(sim_dir, tmp_path_factory):
    """A folder with the fixture's detailed adaptive run copied under each of the six paper pressures, seed 1."""
    src = next((sim_dir / "detailed").glob("*link_prob0.10*_ext_strength4_seed2_detailed.csv"))
    out = tmp_path_factory.mktemp("six_pressures")
    (out / "detailed").mkdir()
    for s in (0, 1, 2, 4, 8, 16):
        stem = legacy_filename(dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=1 / 3, eps=1.0, mu=0.0,
                                    fixedBNat100=False, ext_strength=s, seed=1, lam=0.0, n_events=1), True)
        (out / f"{stem}_detailed.csv").write_bytes(src.read_bytes())
    return out, pd.read_csv(src)


def highlighted(result) -> pd.DataFrame:
    rows = result.data[result.data.highlight != ""]
    return rows[["s", "highlight", "agent"]].drop_duplicates().reset_index(drop=True)


def test_figA3_draws_its_agents_with_a_local_random_state(six_pressures):
    folder, run = six_pressures
    state = np.random.get_state()
    result, categories = expected_warnings(figA3, folder)

    assert np.all(np.random.get_state()[1] == state[1])  # the global RNG is untouched
    assert categories == set()
    assert result.name == "AppendixFig_exampleSim_adaptive_seed1_overPressure"
    assert len(result.figure.axes) == 6
    rs = np.random.RandomState(14)  # the old script's draws: one shared stream, in panel and type order
    expected = []
    for s in (0, 1, 2, 4, 8, 16):
        for name in ("resistant", "resilient", "compliant"):
            ids = run.loc[run[name] == 1, "id"]
            if len(ids):
                expected.append((s, name, int(ids.sample(random_state=rs).iloc[0])))
    assert sorted(map(tuple, highlighted(result).to_numpy())) == sorted(expected)
    plt.close(result.figure)


def test_figA3_data_holds_every_trajectory(six_pressures):
    folder, run = six_pressures
    result, _ = expected_warnings(figA3, folder)
    assert list(result.data.columns) == ["s", "t", "agent", "value", "highlight"]
    panel = result.data[result.data.s == 8].pivot(index="t", columns="agent", values="value")
    expected = run.pivot_table(index="t", columns="id", values="0")
    np.testing.assert_array_equal(panel.to_numpy(), expected.to_numpy())
    plt.close(result.figure)


def test_another_rng_seed_draws_other_agents(six_pressures):
    folder, _ = six_pressures
    first, _ = expected_warnings(figA3, folder)
    other, _ = expected_warnings(figA3, folder, rng_seed=15)
    assert not highlighted(first).equals(highlighted(other))
    plt.close(first.figure), plt.close(other.figure)


def test_figA3_needs_all_six_pressures(sim_dir):
    with pytest.warns(FigureNotApplicable, match="s=0"):
        assert figA3(sim_dir, seed=2) is None  # the fixture has the detailed run at s = 4 only


# Polarisation summary

def test_polarisation_summary_per_config(sim_dir):
    summary, categories = expected_warnings(polarisation_summary, sim_dir)
    assert MissingLevelsWarning in categories  # s = 1 lacks adaptive2static, so only s = 0 and 4 are used
    assert list(summary.index) == ["staticlow", "statichigh", "adaptive2static", "adaptive"]

    for config in summary.index:
        rows = []
        seeds = {0: (0, 1, 2), 4: (0,) if config == "staticlow" else (0, 1, 2)}
        for s, ks in seeds.items():
            for seed in ks:
                df = ensemble_csv(sim_dir, config, s, seed)
                onset, before = df[df.t == 100], df[df.t == 95.5]
                W = onset[[c for c in df.columns if c.startswith("w")]].to_numpy()
                d = pdist(W, metric="cityblock")
                rows.append(dict(std_focal=onset["0"].std(), abs_edge=np.abs(W).mean(), mean_edge=W.mean(),
                                 groupishness=0 if sim_folder.CONFIGS[config][1] == 0 else d.std() / d.mean(),
                                 nr_negs=(before["0"] < 0).sum()))  # the 95.5 row: the bug fix
        expected = pd.DataFrame(rows)
        for metric in expected.columns:
            assert summary.loc[config, (metric, "mean")] == pytest.approx(expected[metric].mean())
            assert summary.loc[config, (metric, "std")] == pytest.approx(expected[metric].std())
        assert summary.loc[config, ("n_runs", "")] == len(expected)
    assert (summary.loc[["staticlow", "statichigh"], ("groupishness", "mean")] == 0).all()


def test_polarisation_summary_needs_the_four_configs(sim_dir):
    with pytest.warns(FigureNotApplicable):
        assert polarisation_summary(sim_dir, base=Baseline(link_prob=0.2)) is None


def test_the_denominator_can_leave_out_latecompliant(sim_dir):
    """As Fig. 6 does; the frequencies then sum to one over the three plotted types."""
    result, _ = expected_warnings(fig3, sim_dir, denominator=("compliant", "resilient", "resistant"), **EXAMPLE)
    assert set(result.data.response) == {"compliant", "resilient", "resistant"}
    sums = result.data.groupby(["config", "seed"]).frequency.sum().dropna()
    np.testing.assert_allclose(sums, 1.0)
    plt.close(result.figure)
    with pytest.raises(ValueError, match="denominator"):
        fig3(sim_dir, denominator=("compliant", "resistant"), **EXAMPLE)
