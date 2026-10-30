"""Old vs new (seam 3): `figA8` against `analyse_sensitivity.py` (100 seeds).

Run with `pytest -m slow -s tests/test_old_vs_new_figA8.py`. Skipped when `simOut_old_model` is missing.

`simOut_old_model` holds no sensitivity runs (legacy never wrote the `_M`/`_N`/`_tau` suffixes, known issue #4), so the
comparison on it skips with that reason. The stand-in comparison runs the unchanged script on a folder that links
every name it reads to a baseline run of the same BN config, each name to a different seed (a variant read from the
wrong file would show), and compares the frequencies with the package's on the same folder.
"""

import numpy as np
import pandas as pd
import pytest

from anb.analysis import Loader, figA8, index_folder
from anb.analysis.sensitivity import COLUMNS, ROWS, S
from anb.analysis.style import CONFIG_LABELS
from old_scripts import OLD_SIM_DIR, png_difference, run_old_script

pytestmark = pytest.mark.slow

SCRIPT = "analyse_sensitivity.py"
SEEDS = range(100)
RESPONSES = ["compliant", "resilient", "resistant", "latecompliant"]
CONFIG_OF_LABEL = {label: config for config, label in CONFIG_LABELS.items()}
STATIC_AND_ADAPTIVE = [(0.2, 0.0), (0.8, 0.0), (0.2, 1.0)]  # (init_w, eps) of the grid's columns


def old_combis() -> list[tuple]:
    """The old script's `param_combis` as (exp, M, N, init_w, eps, beta, p, tau), in its order."""
    combis = [("base", 10, 100, w, e, 3.0, 0.1, 1) for w, e in STATIC_AND_ADAPTIVE]
    combis += [("M", M, 100, w, e, 3.0, 0.1, 1) for w, e in STATIC_AND_ADAPTIVE for M in (5, 15)]
    combis += [("N", 10, N, w, e, 3.0, p, 1) for w, e in STATIC_AND_ADAPTIVE for N, p in ((50, 0.2), (200, 0.05))]
    combis += [("beta", 10, 100, w, e, beta, 0.1, 1) for w, e in STATIC_AND_ADAPTIVE for beta in (1.5, 6.0)]
    combis += [("p", 10, 100, w, e, 3.0, p, 1) for w, e in STATIC_AND_ADAPTIVE for p in (0.05, 0.2)]
    combis += [("tau", 10, 100, w, e, 3.0, 0.1, tau) for w, e in STATIC_AND_ADAPTIVE for tau in (2.0, 10.0)]
    combis += [("eps", 10, 100, 0.2, e, 3.0, 0.1, 1) for e in (0.5, 1.0, 2.0)]
    combis += [("init_w", 10, 100, w, 1.0, 3.0, 0.1, 1) for w in (0.1, 0.2, 0.4)]
    return combis


def old_name(exp, M, N, init_w, eps, beta, p, tau, seed) -> str:
    """The file name the old script reads (its f-string, `fixedBNatt100` False)."""
    addon = f"_M{M:.2f}" if exp == "M" else f"_N{N:.2f}" if exp == "N" else f"_tau{tau:.2f}" if exp == "tau" else ""
    return (f"sim_link_prob{p:.2f}_init_w{init_w:.2f}_beta{beta:.2f}_rho0.33_eps{eps:.2f}_mu0.000_ext_strength{S}"
            f"_seed{seed}{addon}.csv")


def old_frequencies(relres: pd.DataFrame) -> pd.DataFrame:
    """The old `relres` as the package's `data` (row, config, value, seed, response, frequency), each row with the
    base's runs and without the duplicates (the script reads the ε = 1 and ω₀ = 0.2 runs as variant and as base)."""
    relres = relres.assign(config=relres["name"].map(CONFIG_OF_LABEL), seed=relres.index % len(SEEDS))
    blocks = []
    for row, (configs, _) in ROWS.items():
        subset = relres[relres.exp.isin([row, "base"]) & relres.config.isin(configs)]
        block = subset.assign(row=row, value=subset[row])[["row", "config", "value", "seed", *RESPONSES]]
        blocks.append(block.drop_duplicates(["config", "value", "seed"]))
    frame = pd.concat(blocks).melt(id_vars=["row", "config", "value", "seed"], value_vars=RESPONSES,
                                   var_name="response", value_name="frequency")
    return frame


def assert_same_frequencies(new: pd.DataFrame, old: pd.DataFrame):
    key = ["row", "config", "value", "seed", "response"]
    old, new = old.sort_values(key, ignore_index=True), new.sort_values(key, ignore_index=True)
    pd.testing.assert_frame_equal(new[key], old[key], check_dtype=False)
    np.testing.assert_array_equal(new.frequency.to_numpy(), old.frequency.to_numpy())


# simOut_old_model

def test_figA8_matches_the_old_script_on_simOut_old_model(old_script):
    index = index_folder(OLD_SIM_DIR)
    base, others = ({old_name(*combi, seed) for combi in combis for seed in SEEDS}
                    for combis in (old_combis()[:3], old_combis()[3:]))
    variants = (others - base) & {path.name for path in index.path}  # the ε = 1, ω₀ = 0.2 names are the base's
    if not variants:
        pytest.skip(f"{OLD_SIM_DIR.name} holds no sensitivity runs (legacy never wrote the _M/_N/_tau suffixes, "
                    "known issue #4); see the stand-in comparison")
    old = old_script(SCRIPT)
    if "relres" not in old.globals:
        pytest.fail(old.report())
    new = figA8(OLD_SIM_DIR, seeds=SEEDS, index=index, loader=Loader())
    assert_same_frequencies(new.data, old_frequencies(old.globals["relres"]))


# Stand-in: baseline runs linked under the OFAT names

@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    if not OLD_SIM_DIR.is_dir():
        pytest.skip(f"{OLD_SIM_DIR.name} is missing")
    out = tmp_path_factory.mktemp("ofat_stand_in")
    links = {}
    for k, combi in enumerate(old_combis()):
        exp, _, _, init_w, eps, *_ = combi
        init_w, eps = (0.2, 1.0) if exp in ("eps", "init_w") else (init_w, eps)  # the config's baseline run
        for seed in SEEDS:
            name = old_name(*combi, seed)
            if name not in links:  # the base names first, at their own seed; every other name shifted by its own k
                shift = 0 if combi[0] == "base" else k
                links[name] = old_name("base", 10, 100, init_w, eps, 3.0, 0.1, 1, (seed + shift) % len(SEEDS))
    for name, source in links.items():
        (out / name).symlink_to((OLD_SIM_DIR / source).resolve())
    return out


@pytest.fixture(scope="module")
def old_on_stand_in(stand_in, tmp_path_factory):
    result = run_old_script(SCRIPT, tmp_path_factory.mktemp("analyse_sensitivity"), stand_in)
    if "relres" not in result.globals:
        pytest.fail(result.report())
    return result


@pytest.fixture(scope="module")
def new_on_stand_in(stand_in):
    return figA8(stand_in, index=index_folder(stand_in), loader=Loader())


def test_the_old_script_ran_on_the_stand_in(old_on_stand_in):
    assert old_on_stand_in.ok, old_on_stand_in.report()


def test_figA8_frequencies_match_the_old_script_exactly(old_on_stand_in, new_on_stand_in):
    assert list(new_on_stand_in.data.row.unique()) == list(ROWS)
    assert new_on_stand_in.data.seed.nunique() == len(SEEDS)
    assert set(new_on_stand_in.data.config) == set(COLUMNS)
    assert_same_frequencies(new_on_stand_in.data, old_frequencies(old_on_stand_in.globals["relres"]))


def test_figA8_png_is_close(old_on_stand_in, new_on_stand_in, tmp_path):
    [path] = new_on_stand_in.save(tmp_path)
    diff = png_difference(path, old_on_stand_in.pngs[f"{new_on_stand_in.name}.png"])
    print(f"\nfigA8: {diff:.2%} of the pixels differ (jitter, and the old script's doubled ε = 1, ω₀ = 0.2 dots)")
