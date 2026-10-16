"""`legacy_filename` must reproduce `run_one`'s naming. Expected strings come from legacy's own `run_one`
(with the simulation stubbed out, in a scratch cwd, so only an empty CSV is written), not from hand-typed expectations."""

import itertools

import pandas as pd
import pytest

from legacy_oracle import load_legacy

from anb.io import legacy_filename

BN_CONFIGS = [  # link_prob, init_w, beta, rho, eps, mu, fixedBNat100: the 4 paper configurations
    (0.0, 0.2, 0.0, 0.0, 1.0, 0.0, False),
    (0.0, 0.2, 1.0, 0.5, 1.0, 0.0, False),
    (0.1, 0.2, 2.0, 0.0, 1.0, 0.0, True),
    (0.1, 0.2, 2.0, 0.5, 1.0, 0.0, False),
]
SENSITIVITY = [
    (0.1, 0.1, 2.0, 0.5, 0.5, 0.0, False),
    (0.1, 0.4, 2.0, 0.5, 2.0, 0.0, False),
    (0.1, 0.2, 2.0, 0.5, 1.0, 0.025, False),
    (0.1, 0.2, 2.0, 0.5, 1.0, 0.1, True),
]
PRESSURES = [0, 1, 2, 4, 8, 16]
KEYS = ["link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100"]


def legacy_name(combo, pressure, seed, detail, lam=0.0, two_events=False, tmp_path=None, monkeypatch=None):
    legacy = load_legacy(detail=detail, two_events=two_events)
    legacy.lam = lam
    legacy.run_simulation = lambda params: pd.DataFrame()
    monkeypatch.chdir(tmp_path)
    return legacy.run_one(seed, *combo, pressure)


def params_for(combo, pressure, seed, lam=0.0, n_events=1):
    return dict(zip(KEYS, combo)) | {"ext_strength": pressure, "seed": seed, "lam": lam, "n_events": n_events}


@pytest.mark.parametrize("detail", [False, True])
def test_matches_run_one_for_paper_and_sensitivity_sweeps(detail, tmp_path, monkeypatch):
    for combo, pressure, seed in itertools.product(BN_CONFIGS + SENSITIVITY, PRESSURES, [0, 1, 25, 99]):
        expected = legacy_name(combo, pressure, seed, detail, tmp_path=tmp_path, monkeypatch=monkeypatch)
        assert legacy_filename(params_for(combo, pressure, seed), detail) == expected


@pytest.mark.parametrize("fixed", [False, True])
def test_fixedBNat100_token(fixed, tmp_path, monkeypatch):
    combo = (0.1, 0.2, 2.0, 0.5, 1.0, 0.0, fixed)
    expected = legacy_name(combo, 4, 1, False, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert legacy_filename(params_for(combo, 4, 1), False) == expected
    assert ("fixedBNat100" in expected) == fixed


@pytest.mark.parametrize("lam", [0.0, 0.05, 0.123456])
def test_lam_variant(lam, tmp_path, monkeypatch):
    expected = legacy_name(BN_CONFIGS[3], 4, 1, False, lam=lam, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert legacy_filename(params_for(BN_CONFIGS[3], 4, 1, lam=lam), False) == expected
    assert ("_lambda" in expected) == (lam > 0)


@pytest.mark.parametrize("detail", [False, True])
def test_two_event_variant(detail, tmp_path, monkeypatch):
    expected = legacy_name(
        BN_CONFIGS[3], 4, 1, detail, lam=0.05, two_events=True, tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    got = legacy_filename(params_for(BN_CONFIGS[3], 4, 1, lam=0.05, n_events=2), detail)
    assert got == expected
    assert "_2events_lambda" in expected


def test_detail_selects_directory():
    p = params_for(BN_CONFIGS[3], 4, 1)
    assert legacy_filename(p, False).startswith("simOut/sim_")
    assert legacy_filename(p, True).startswith("simOut/detailed/sim_")


def test_does_not_add_M_N_tau_suffixes():
    name = legacy_filename(params_for(BN_CONFIGS[3], 4, 1), False)
    assert not any(s in name for s in ("_M", "_N", "_tau"))
