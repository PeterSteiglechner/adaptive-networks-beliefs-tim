"""`anb.comparison.compare_folders` on synthetic ensemble-format CSVs: pairing, cell parsing and the pooled table."""

import numpy as np
import pandas as pd
import pytest

from anb.comparison import METRICS, RESPONSES, WINDOWS, cell_label, compare_folders, parse_filename
from anb.io import legacy_filename
from anb.metrics import RESPONSE_COLUMNS

N_AGENTS = 20
BASE = dict(link_prob=0.1, init_w=0.2, beta=3.0, rho=1 / 3, eps=1.0, mu=0.0, fixedBNat100=False, ext_strength=4,
            seed=0, lam=0.0, n_events=1)


def params(**changes) -> dict:
    return {**BASE, **changes}


def write_run(folder, run_params: dict, shares, rng) -> None:
    """One ensemble CSV: the three window rows per agent, response dummies on the t = 95.5 row only."""
    labels = rng.choice(len(RESPONSES), size=N_AGENTS, p=shares)
    rows = []
    for t in WINDOWS.values():
        frame = pd.DataFrame({"t": t, "id": range(N_AGENTS), **{m: rng.normal(size=N_AGENTS) for m in METRICS}})
        for k, name in enumerate(RESPONSE_COLUMNS):
            frame[name] = (labels == k).astype(float) if t == WINDOWS["before"] else np.nan
        rows.append(frame)
    folder.mkdir(exist_ok=True)
    pd.concat(rows).to_csv(folder / f"{legacy_filename(run_params, detail=False)}.csv", index=False)


def write_cell(folder, cell_params: dict, shares, rng, seeds=range(15)) -> None:
    for seed in seeds:
        write_run(folder, {**cell_params, "seed": seed}, shares, rng)


UNIFORM = np.full(len(RESPONSES), 1 / len(RESPONSES))
SHIFTED = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45])


@pytest.mark.parametrize("changes, expected", [
    (dict(init_w=0.2, eps=0.0), "static0.2"),
    (dict(init_w=0.8, eps=0.0), "static0.8"),
    (dict(eps=1.0, fixedBNat100=True), "adaptive-static"),
    (dict(), "adaptive"),
    (dict(mu=0.1), "mu=0.1"),
    (dict(mu=0.001, eps=0.5), "eps=0.5, mu=0.001"),
    (dict(n_events=2), "n_events=2"),
])
def test_filenames_round_trip_into_labelled_cells(changes, expected):
    p = params(**changes)
    name = legacy_filename(p, detail=False)

    cell, seed = parse_filename(name)

    assert cell_label(cell) == expected
    assert seed == p["seed"]
    assert legacy_filename({**cell, "seed": seed}, detail=False) == name


def test_the_fixed_token_and_lambda_survive_parsing():
    p = params(fixedBNat100=True, lam=0.0123, n_events=2, ext_strength=16)
    name = legacy_filename(p, detail=False)

    cell, _ = parse_filename(name)

    assert cell["fixedBNat100"] is True
    assert cell["lam"] == pytest.approx(0.0123)
    assert cell["n_events"] == 2
    assert cell["ext_strength"] == 16


def test_files_on_one_side_only_are_reported_and_the_rest_is_compared(tmp_path):
    rng = np.random.default_rng(0)
    old, new = tmp_path / "old", tmp_path / "new"
    write_cell(old, params(), UNIFORM, rng, seeds=range(10))
    write_cell(new, params(), UNIFORM, rng, seeds=range(1, 11))

    result = compare_folders(old, new, n_permutations=200)

    assert len(result.pairing.matched) == 9
    assert [legacy_filename({**params(seed=0)}, False)] == result.pairing.only_old
    assert [legacy_filename({**params(seed=10)}, False)] == result.pairing.only_new
    assert len(result.old) == len(result.new) == 9
    assert len(result.pooled_responses) == 1


def test_an_injected_shift_in_one_cell_flags_exactly_that_cell(tmp_path):
    rng = np.random.default_rng(1)
    old, new = tmp_path / "old", tmp_path / "new"
    cells = [params(ext_strength=0), params(ext_strength=4), params(mu=0.1)]
    for cell in cells:
        write_cell(old, cell, UNIFORM, rng)
        write_cell(new, cell, SHIFTED if cell["mu"] == 0.1 else UNIFORM, rng)

    table = compare_folders(old, new, n_permutations=500).pooled_responses

    assert len(table) == 3
    assert list(table[table.fail].config) == ["mu=0.1"]


def test_the_table_is_ranked_by_cramers_v_with_q_alongside(tmp_path):
    rng = np.random.default_rng(2)
    old, new = tmp_path / "old", tmp_path / "new"
    mild = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3])
    for cell, shares in [(params(ext_strength=0), UNIFORM), (params(ext_strength=1), mild),
                         (params(ext_strength=2), SHIFTED)]:
        write_cell(old, cell, UNIFORM, rng)
        write_cell(new, cell, shares, rng)

    table = compare_folders(old, new, n_permutations=200).pooled_responses

    assert list(table.pressure) == [2, 1, 0]
    assert table.cramers_v.is_monotonic_decreasing
    assert {"q", "fail", "p", "chi2"} <= set(table.columns)


def test_run_summaries_carry_the_cell_and_response_counts(tmp_path):
    rng = np.random.default_rng(3)
    old, new = tmp_path / "old", tmp_path / "new"
    for folder in (old, new):
        write_cell(folder, params(ext_strength=8), UNIFORM, rng, seeds=range(3))

    result = compare_folders(old, new, n_permutations=50)

    assert set(result.old.config) == {"adaptive"}
    assert set(result.old.pressure) == {8}
    assert sorted(result.old.seed) == [0, 1, 2]
    assert (result.old[list(RESPONSES)].sum(axis=1) == N_AGENTS).all()


def test_folders_without_common_files_are_an_error(tmp_path):
    rng = np.random.default_rng(4)
    old, new = tmp_path / "old", tmp_path / "new"
    write_cell(old, params(), UNIFORM, rng, seeds=[0])
    write_cell(new, params(), UNIFORM, rng, seeds=[1])

    with pytest.raises(ValueError, match="no CSV"):
        compare_folders(old, new)
