"""The naming scheme in both directions (`legacy_filename` ↔ `parse_filename`) and `anb.analysis.index_folder`."""

import warnings

import pandas as pd
import pytest

from anb.agents import ANBAgent
from anb.analysis import index_folder
from anb.io import is_detailed, legacy_filename, parse_filename
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig

BASE = dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=0.33, eps=1.0, mu=0.0, fixedBNat100=False, ext_strength=4,
            seed=3, lam=0.0, n_events=1)
DEFAULTS = dict(M=10, N=100, tau=1)


def params(**changes) -> dict:
    return {**BASE, **changes}


ROUND_TRIPS = [
    dict(),
    dict(fixedBNat100=True),
    dict(n_events=2),
    dict(lam=0.05),
    dict(M=5),
    dict(N=200),
    dict(tau=2),
    dict(M=5, N=50, tau=3, n_events=2, lam=0.0123, fixedBNat100=True, mu=0.005, ext_strength=16),
    dict(ext_strength=2.5),
]


@pytest.mark.parametrize("detail", [False, True])
@pytest.mark.parametrize("changes", ROUND_TRIPS)
def test_parse_filename_inverts_legacy_filename(changes, detail):
    p = params(**changes)
    name = legacy_filename(p, detail)

    for written in (name, name + ".csv", name + ("_detailed.csv" if detail else ".csv")):
        cell, seed = parse_filename(written)
        assert seed == p["seed"]
        assert cell == pytest.approx({k: v for k, v in {**DEFAULTS, **p}.items() if k != "seed"})
        assert is_detailed(written) is detail
        assert legacy_filename({**cell, "seed": seed}, detail) == name


def test_absent_suffixes_take_the_defaults():
    cell, _ = parse_filename(legacy_filename(params(), False))

    assert (cell["M"], cell["N"], cell["tau"], cell["lam"], cell["n_events"]) == (10, 100, 1, 0.0, 1)
    assert cell["fixedBNat100"] is False


@pytest.mark.parametrize("changes, suffix", [
    (dict(M=5), "_seed3_M5.00"),
    (dict(N=200), "_seed3_N200.00"),
    (dict(tau=2), "_seed3_tau2.00"),
    (dict(M=5, N=50, tau=3, n_events=2, lam=0.05), "_seed3_M5.00_N50.00_tau3.00_2events_lambda0.0500"),
])
def test_m_n_tau_suffixes_follow_the_seed_in_the_order_the_sensitivity_script_reads(changes, suffix):
    assert legacy_filename(params(**changes), False).endswith(suffix)


def test_default_m_n_tau_leave_the_baseline_name_unchanged():
    assert legacy_filename(params(**DEFAULTS), False) == legacy_filename(params(), False)


def test_detailed_name_in_the_detailed_folder_or_with_the_suffix():
    stem = legacy_filename(params(), False)

    assert is_detailed(f"some/where/detailed/{stem}_detailed.csv")
    assert is_detailed(f"{stem}_detailed.csv")
    assert not is_detailed(f"some/where/{stem}.csv")


@pytest.mark.parametrize("name", ["sim_foo.csv", "notes.csv", legacy_filename(params(), False) + "_extra",
                                  legacy_filename(params(), False) + "_tau2.50"])
def test_parse_filename_rejects_other_names(name):
    with pytest.raises(ValueError, match="legacy"):
        parse_filename(name)


def test_saved_runs_carry_the_specs_m_n_tau_in_the_name_but_not_as_columns(tmp_path):
    agent = ANBAgent(beta=3.0, eps=1.0, init_w=0.2, tau=2, n_beliefs=4)
    spec = ModelSpec(agent, ErdosRenyiNetwork, {"link_prob": 0.1}, n_agents=12, rho=1 / 3)
    testbed = Testbed(spec, TestbedConfig(PressureSchedule.single_event(4)))

    path = testbed.output_path(5, tmp_path)
    assert path.name.endswith("_seed5_M4.00_N12.00_tau2.00.csv")
    assert path == testbed.run(5).save(tmp_path)
    assert not {"M", "N", "tau"} & set(pd.read_csv(path, nrows=1).columns)


# index_folder

INDEX_COLUMNS = ["link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100", "s", "seed", "lam", "n_events",
                 "M", "N", "tau", "detailed", "path"]


def touch(folder, name):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("t,id\n")
    return path


def test_index_folder_has_one_row_per_parseable_csv(tmp_path):
    expected = [
        (params(seed=0), False),
        (params(seed=1, ext_strength=8, fixedBNat100=True), False),
        (params(seed=0, M=5), False),
        (params(seed=2, n_events=2, lam=0.01), False),
        (params(seed=1), True),
    ]
    paths = [touch(tmp_path, legacy_filename(p, d) + ("_detailed" if d else "") + ".csv") for p, d in expected]
    (tmp_path / "readme.txt").write_text("not a csv")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        index = index_folder(tmp_path)

    assert list(index.columns) == INDEX_COLUMNS
    assert len(index) == len(expected)
    rows = {row.path: row for row in index.itertuples()}
    for (p, detail), path in zip(expected, paths):
        row = rows[path]
        assert row.detailed is detail
        assert row.s == p["ext_strength"]
        assert row.seed == p["seed"]
        assert (row.M, row.N, row.tau) == tuple({**DEFAULTS, **p}[k] for k in ("M", "N", "tau"))
        assert (row.lam, row.n_events, row.fixedBNat100) == (p["lam"], p["n_events"], p["fixedBNat100"])


def test_index_folder_warns_once_about_unparseable_files(tmp_path):
    touch(tmp_path, legacy_filename(params(), False) + ".csv")
    touch(tmp_path, "sim_typo_fixedBNatt100.csv")
    touch(tmp_path, "detailed/other.csv")

    with pytest.warns(UserWarning) as caught:
        index = index_folder(tmp_path)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "2" in message and "sim_typo_fixedBNatt100.csv" in message
    assert len(index) == 1


def test_index_folder_of_an_empty_folder_has_the_columns_and_no_rows(tmp_path):
    index = index_folder(tmp_path)

    assert list(index.columns) == INDEX_COLUMNS
    assert index.empty
