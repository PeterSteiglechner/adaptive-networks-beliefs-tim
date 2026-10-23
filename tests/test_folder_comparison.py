"""`anb.comparison.compare_folders` on synthetic ensemble-format CSVs: pairing, cell parsing and the pooled table."""

import os

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


def write_run(folder, run_params: dict, shares, rng, labels=None, shift=None) -> None:
    """One ensemble CSV: the three window rows plus one tracked step (t = 50) per agent, response dummies on the
    t = 95.5 row only. `labels` fixes each agent's response index; `shift = (window, metric, amount)` adds to a metric."""
    if labels is None:
        labels = rng.choice(len(RESPONSES), size=N_AGENTS, p=shares)
    rows = []
    for t in (50, *WINDOWS.values()):
        frame = pd.DataFrame({"t": t, "id": range(N_AGENTS), **{m: rng.normal(size=N_AGENTS) for m in METRICS}})
        if shift and t == WINDOWS[shift[0]]:
            frame[shift[1]] += shift[2]
        for k, name in enumerate(RESPONSE_COLUMNS):
            frame[name] = (labels == k).astype(float) if t == WINDOWS["before"] else np.nan
        rows.append(frame)
    folder.mkdir(exist_ok=True)
    pd.concat(rows).to_csv(folder / f"{legacy_filename(run_params, detail=False)}.csv", index=False)


def write_cell(folder, cell_params: dict, shares, rng, seeds=range(15), shift=None) -> None:
    for seed in seeds:
        write_run(folder, {**cell_params, "seed": seed}, shares, rng, shift=shift)


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


# Conditional responses, lean, metrics and the combined table

# Positive-before shares move (0.5 → 0.8) while the split among the negative-before classes stays put.
NEGATIVE_SPLIT = np.array([0.4, 0.3, 0.2, 0.1])
LEAN_OLD = np.array([0.25, 0.25, *(0.5 * NEGATIVE_SPLIT), 0.0])
LEAN_NEW = np.array([0.4, 0.4, *(0.2 * NEGATIVE_SPLIT), 0.0])


def test_conditional_counts_and_the_lean_come_from_the_labels_of_a_hand_built_run(tmp_path):
    # persistPos, nonpersistPos, compliant, resilient, resistant, latecompliant, unlabelled
    labels = np.repeat(np.arange(len(RESPONSES)), [5, 3, 4, 2, 3, 1, 2])
    for folder in (tmp_path / "old", tmp_path / "new"):
        write_run(folder, params(), None, np.random.default_rng(0), labels=labels)

    result = compare_folders(tmp_path / "old", tmp_path / "new", n_permutations=20)

    row = result.old.iloc[0]
    assert [row[c] for c in ("compliant", "resilient", "resistant", "latecompliant")] == [4, 2, 3, 1]
    assert row.lean == pytest.approx(8 / 20)
    assert row["x_focal@t50"] == pytest.approx(
        pd.read_csv(next((tmp_path / "old").glob("*.csv"))).query("t == 50").x_focal.mean())
    assert len(result.conditional_responses) == 1 and len(result.lean) == 1


def test_a_lean_only_shift_is_flagged_in_lean_and_pooled_but_not_conditional(tmp_path):
    rng = np.random.default_rng(5)
    old, new = tmp_path / "old", tmp_path / "new"
    cells = [params(ext_strength=0), params(ext_strength=4), params(mu=0.1)]
    for cell in cells:
        write_cell(old, cell, UNIFORM, rng)
        write_cell(new, cell, UNIFORM, rng)
    write_cell(old, params(ext_strength=16), LEAN_OLD, rng, seeds=range(30))
    write_cell(new, params(ext_strength=16), LEAN_NEW, rng, seeds=range(30))

    result = compare_folders(old, new, n_permutations=500)

    assert list(result.lean[result.lean.fail].pressure) == [16]
    assert list(result.pooled_responses[result.pooled_responses.fail].pressure) == [16]
    assert not result.conditional_responses.fail.any()
    lean_row = result.lean[result.lean.pressure == 16].iloc[0]
    assert lean_row.lean_legacy < lean_row.lean_anb and lean_row.gap_se > 3 and "ks" not in result.lean.columns


def test_a_conditional_shift_is_flagged_in_the_conditional_table(tmp_path):
    rng = np.random.default_rng(6)
    old, new = tmp_path / "old", tmp_path / "new"
    shifted = np.array([0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.0])
    write_cell(old, params(), LEAN_OLD, rng, seeds=range(30))
    write_cell(new, params(), shifted, rng, seeds=range(30))

    table = compare_folders(old, new, n_permutations=500).conditional_responses

    assert table.fail.all()
    assert "unlabelled_anb" not in table.columns and "latecompliant_anb" in table.columns


def test_a_metric_shift_is_flagged_only_for_its_cell_and_window(tmp_path):
    rng = np.random.default_rng(7)
    old, new = tmp_path / "old", tmp_path / "new"
    for cell in [params(ext_strength=0), params(ext_strength=4)]:
        write_cell(old, cell, UNIFORM, rng)
        write_cell(new, cell, UNIFORM, rng,
                   shift=("during", "bc_foc", 2.0) if cell["ext_strength"] == 4 else None)

    table = compare_folders(old, new, n_permutations=50).metrics

    flagged = table[table.fail]
    assert len(table) == 2 * len(WINDOWS) * len(METRICS)
    assert list(zip(flagged.pressure, flagged.window, flagged.metric)) == [(4, "during", "bc_foc")]
    assert table.iloc[0].smd > 1 and {"ks", "ci_low", "ci_high"} <= set(table.columns)


def test_every_table_is_ranked_by_effect_size_and_correction_is_within_each_family(tmp_path):
    rng = np.random.default_rng(8)
    old, new = tmp_path / "old", tmp_path / "new"
    for cell, shift in [(params(ext_strength=0), None), (params(ext_strength=4), ("during", "bc_foc", 1.5)),
                        (params(ext_strength=16), ("after", "Hsoc", 3.0))]:
        write_cell(old, cell, LEAN_OLD, rng)
        write_cell(new, cell, LEAN_NEW if cell["ext_strength"] else LEAN_OLD, rng, shift=shift)

    result = compare_folders(old, new, n_permutations=200)

    assert result.pooled_responses.cramers_v.is_monotonic_decreasing
    assert result.conditional_responses.cramers_v.is_monotonic_decreasing
    assert result.lean.smd.abs().is_monotonic_decreasing
    assert result.metrics.smd.abs().is_monotonic_decreasing
    # BH within the family: re-correcting the metric p-values alone reproduces q.
    from statsmodels.stats.multitest import multipletests
    expected = multipletests(result.metrics.p, method="fdr_bh")[1]
    assert result.metrics.q.to_numpy() == pytest.approx(expected)


def test_the_combined_table_merges_all_four_families_ranked_by_effect(tmp_path):
    rng = np.random.default_rng(9)
    old, new = tmp_path / "old", tmp_path / "new"
    write_cell(old, params(), LEAN_OLD, rng)
    write_cell(new, params(), LEAN_NEW, rng, shift=("after", "bc_foc", 2.0))

    result = compare_folders(old, new, n_permutations=100)
    top = result.top_differences

    assert set(top.family) == {"pooled responses", "conditional responses", "lean", "metrics"}
    assert len(top) == sum(len(t) for t in (result.pooled_responses, result.conditional_responses, result.lean,
                                            result.metrics))
    assert top.effect.is_monotonic_decreasing
    assert {"config", "pressure", "detail", "effect", "q", "fail"} <= set(top.columns)


# Summary cache

@pytest.fixture
def csv_reads(monkeypatch):
    """Counts the CSV files `compare_folders` reads (in-process, so use `n_jobs=1`)."""
    reads = []
    original = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: (reads.append(path), original(path, *a, **k))[1])
    return reads


def two_folders(tmp_path, seed=10):
    rng = np.random.default_rng(seed)
    old, new = tmp_path / "old", tmp_path / "new"
    for cell in (params(ext_strength=4), params(mu=0.1)):
        write_cell(old, cell, UNIFORM, rng, seeds=range(4))
        write_cell(new, cell, UNIFORM, rng, seeds=range(4))
    return old, new


def compare(old, new, cache_dir=None):
    return compare_folders(old, new, n_permutations=50, cache_dir=cache_dir, n_jobs=1)


def test_a_second_comparison_of_unchanged_folders_reads_no_csv(tmp_path, csv_reads):
    old, new = two_folders(tmp_path)
    compare(old, new, tmp_path / "cache")
    assert len(csv_reads) == 16

    csv_reads.clear()
    compare(old, new, tmp_path / "cache")

    assert csv_reads == []


@pytest.mark.parametrize("change", ["touch", "add", "remove"])
def test_changing_a_cell_rebuilds_only_that_cell(tmp_path, csv_reads, change):
    old, new = two_folders(tmp_path)
    cache = tmp_path / "cache"
    compare(old, new, cache)
    victim = old / f"{legacy_filename(params(mu=0.1, seed=0), detail=False)}.csv"
    if change == "touch":
        victim.write_text(victim.read_text() + "\n")  # new mtime and size
    elif change == "add":
        for folder in (old, new):  # the new run must be paired to count
            write_run(folder, params(mu=0.1, seed=99), UNIFORM, np.random.default_rng(0))
    else:
        victim.unlink()
    csv_reads.clear()

    compare(old, new, cache)

    mu_cell = [r for r in csv_reads if "mu0.100" in str(r)]
    assert len(mu_cell) == len(csv_reads) > 0  # the other cell stays cached
    assert len(csv_reads) == {"touch": 4, "add": 10, "remove": 6}[change]


def test_results_are_the_same_with_and_without_the_cache(tmp_path):
    old, new = two_folders(tmp_path)

    plain = compare(old, new)
    fresh = compare(old, new, tmp_path / "cache")
    cached = compare(old, new, tmp_path / "cache")

    for other in (fresh, cached):
        pd.testing.assert_frame_equal(plain.old, other.old)
        pd.testing.assert_frame_equal(plain.new, other.new)
        pd.testing.assert_frame_equal(plain.top_differences, other.top_differences)


def test_reading_in_parallel_gives_the_same_summaries(tmp_path):
    old, new = two_folders(tmp_path)

    serial = compare(old, new)
    parallel = compare_folders(old, new, n_permutations=50, n_jobs=2)

    pd.testing.assert_frame_equal(serial.old, parallel.old)
    pd.testing.assert_frame_equal(serial.new, parallel.new)


def test_a_changed_modification_time_alone_invalidates_the_cell(tmp_path, csv_reads):
    old, new = two_folders(tmp_path)
    compare(old, new, tmp_path / "cache")
    victim = next(old.glob("*mu0.100*seed0.csv"))
    stamp = victim.stat().st_mtime_ns + 10**9
    os.utime(victim, ns=(stamp, stamp))
    csv_reads.clear()

    compare(old, new, tmp_path / "cache")

    assert len(csv_reads) == 4


def test_unreadable_cache_files_are_rebuilt(tmp_path, csv_reads):
    old, new = two_folders(tmp_path)
    cache = tmp_path / "cache"
    expected = compare(old, new, cache)
    for path in cache.iterdir():
        path.write_bytes(b"not a pickle")
    csv_reads.clear()

    result = compare(old, new, cache)

    assert len(csv_reads) == 16
    pd.testing.assert_frame_equal(expected.old, result.old)
