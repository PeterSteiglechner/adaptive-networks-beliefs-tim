"""The statistics of the on-demand validation (`anb.comparison`): calibrated under the null, sensitive to shifts."""

import numpy as np
import pandas as pd
import pytest

from anb.comparison import RESPONSES, compare_responses, compare_values, correct, summarise_run

K = len(RESPONSES)


def run_counts(rng, shares, n_runs=100, n_agents=100):
    return rng.multinomial(n_agents, shares, size=n_runs)


def test_identical_response_counts_give_chi2_zero_and_p_one():
    counts = run_counts(np.random.default_rng(0), np.full(K, 1 / K))

    result = compare_responses(counts, counts.copy(), np.random.default_rng(1), n_permutations=500)

    assert result["chi2"] == pytest.approx(0.0)
    assert result["p"] == 1.0


def test_shifted_response_frequencies_are_detected():
    rng = np.random.default_rng(2)
    shares = np.full(K, 1 / K)
    shifted = shares + np.array([0.04, -0.04, 0, 0, 0, 0, 0])

    result = compare_responses(run_counts(rng, shares), run_counts(rng, shifted), rng, n_permutations=1000)

    assert result["p"] < 0.01


def test_response_p_values_are_calibrated_when_agents_in_a_run_move_together():
    """Each run puts all its agents in one category, so pooled agent counts are 100× overdispersed. A naive
    χ² would almost always reject; the run-level permutation keeps the false-positive rate near α."""
    rng = np.random.default_rng(3)
    shares = np.full(K, 1 / K)

    def clustered():
        return 100 * np.eye(K, dtype=int)[rng.choice(K, size=100, p=shares)]

    ps = [compare_responses(clustered(), clustered(), rng, n_permutations=200)["p"] for _ in range(40)]

    assert np.mean(np.array(ps) < 0.05) <= 0.15


def test_a_single_category_on_both_sides_is_no_difference():
    counts = np.zeros((10, K), dtype=int)
    counts[:, 0] = 100

    assert compare_responses(counts, counts, np.random.default_rng(0))["p"] == 1.0


def test_value_comparison_detects_a_shift_and_brackets_it():
    rng = np.random.default_rng(4)
    result = compare_values(rng.normal(0, 1, 100), rng.normal(1, 1, 100))

    assert result["p"] < 1e-6
    assert result["ci_low"] < 1 < result["ci_high"]


def test_equal_constants_and_nans_are_no_difference():
    """Edge metrics of a static BN are one constant on both sides; clust may be NaN for some runs."""
    result = compare_values(np.full(100, 0.2), np.r_[np.full(99, 0.2), np.nan])

    assert result["p"] == 1.0
    assert [result[k] for k in ("diff", "ci_low", "ci_high")] == pytest.approx([0, 0, 0], abs=1e-12)


def test_correction_flags_only_the_real_deviation_among_many_nulls():
    rng = np.random.default_rng(5)
    rows = [compare_values(rng.normal(0, 1, 100), rng.normal(0, 1, 100)) for _ in range(200)]
    rows.append(compare_values(rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)))

    failed = correct(pd.DataFrame(rows)).fail.to_numpy()

    assert failed[-1]
    assert failed[:-1].sum() <= 2


def test_an_untestable_row_leaves_the_correction_of_the_others_intact():
    """A side with one run has no SE, so no p; it must not turn every q of the family into NaN."""
    rng = np.random.default_rng(6)
    rows = [compare_values(rng.normal(0, 1, 100), rng.normal(3, 1, 100)), compare_values([0.1], [0.2, 0.3])]

    table = correct(pd.DataFrame(rows))

    assert np.isnan(table.p[1]) and np.isnan(table.q[1]) and not table.fail[1]
    assert table.fail[0] and table.q[0] == pytest.approx(table.p[0])


def test_a_metric_nan_in_every_run_gives_nan_instead_of_an_error():
    result = compare_values(np.full(10, np.nan), np.full(10, np.nan))

    assert all(np.isnan(result[k]) for k in ("p", "ks", "diff", "smd"))


def test_summarise_run_counts_responses_and_unlabelled_agents():
    """Three agents on the 95.5 row: compliant, resistant, and one with no label (dummies stored as objects)."""
    dummies = {name: [False, False, False] for name in RESPONSES[:-1]}
    dummies["compliant"] = [True, False, False]
    dummies["resistant"] = [False, True, False]
    rows = []
    for t in (0.0, 95.5, 145.5, 195.5):
        frame = pd.DataFrame({"t": t, "id": [0, 1, 2], "x_focal": [0.1, 0.2, np.nan]})
        frame = frame.assign(**{m: 1.0 for m in ("n_nbs", "Hpers", "Hpersfoc", "Hsoc", "Hext", "tb_tot",
                                                   "tb_foc", "absOm_tot", "absOm_foc", "clust", "bc_foc",
                                                   "expI", "extr_nonfoc")})
        for name in RESPONSES[:-1]:
            frame[name] = pd.Series(dummies[name], dtype=object) if t == 95.5 else np.nan
        rows.append(frame)

    summary = summarise_run(pd.concat(rows, ignore_index=True))

    assert (summary["compliant"], summary["resistant"], summary["unlabelled"]) == (1, 1, 1)
    assert summary["x_focal@during"] == pytest.approx(0.15)
