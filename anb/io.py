"""Output naming and parameter columns compatible with the legacy `analyse_*.py` scripts."""

import re
from pathlib import Path

import numpy as np

# The parameters legacy `run_one` appends as columns, in its order. Also the order of its filename tokens.
LEGACY_PARAM_COLUMNS = ("link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100", "ext_strength", "seed")

# Parameters legacy encodes only in the filename. They become columns only when they are per-agent.
FILENAME_ONLY_PARAMS = ("lam", "n_events", "M", "N", "tau")

# Beliefs, agents and memory length of the paper's setup. `legacy_filename` writes `_M`/`_N`/`_tau` only for others.
SIZE_DEFAULTS = {"M": 10, "N": 100, "tau": 1}

# The `learning_until` that legacy's `fixedBNat100` flag stands for. Any other gate has no legacy name.
LEGACY_LEARNING_UNTIL = 101


def parameter_columns(params: dict) -> list[str]:
    """The output columns for `params`: the legacy ones in legacy order, then any others in `params` order.

    A scalar `lam`, `n_events`, `M`, `N` or `tau` gets no column (legacy writes them only into the filename).
    """
    extra = [
        k for k in params
        if k not in LEGACY_PARAM_COLUMNS and not (k in FILENAME_ONLY_PARAMS and np.ndim(params[k]) == 0)
    ]
    return [k for k in LEGACY_PARAM_COLUMNS if k in params] + extra


def _unformatted(v) -> str:
    """`v` as legacy's unformatted f-string writes it. Legacy passes ints, so an integral float (4.0) is written 4."""
    return str(int(v)) if float(v).is_integer() else str(v)


def legacy_filename(params: dict, detail: bool) -> str:
    """The path (without `.csv`) that legacy `run_one` writes for `params`, relative to its output folder.

    `params` holds the nine `run_one` keys (`link_prob`, `init_w`, `beta`, `rho`, `eps`, `mu`,
    `fixedBNat100`, `ext_strength`, `seed`) plus `lam` and `n_events`, all scalars, and optionally `M`, `N` and
    `tau`. Each of those three that differs from `SIZE_DEFAULTS` adds a `_M{M:.2f}`/`_N{N:.2f}`/`_tau{tau:.2f}`
    suffix right after the seed, the names `analyse_sensitivity.py` reads (legacy itself never writes them, known
    issue #4). Detail-mode files additionally end in
    `_detailed` before `.csv`; that suffix is not part of this name, as in `run_one`'s return value. Legacy's
    `simOut_old_model/` prefix is left off too, so `SimulationResult.save` picks the folder (`simOut/` by default).

    Raises `ValueError` for parameters legacy can't name: a missing key (any network but Erdős–Rényi), an
    extra one (`learning_until` ≠ 101, a `fixed` mask, other network kwargs) or a per-agent value.
    """
    known = (*LEGACY_PARAM_COLUMNS, *FILENAME_ONLY_PARAMS)
    params = {**SIZE_DEFAULTS, **params}
    missing = [k for k in known if k not in params]
    extra = [k for k in params if k not in known]
    if missing or extra:
        problems = ([f"missing: {', '.join(missing)}"] if missing else []) + \
                   ([f"no legacy token for: {', '.join(extra)}"] if extra else [])
        raise ValueError(f"legacy filenames only exist for the paper's setup ({'; '.join(problems)}); "
                         "save with legacy_names=False")
    per_agent = [k for k in known if np.ndim(params[k]) != 0]
    if per_agent:
        raise ValueError(f"legacy filenames need uniform parameters; per-agent: {', '.join(per_agent)}")
    parts = []
    for k in LEGACY_PARAM_COLUMNS:
        v = params[k]
        if k in ("seed", "ext_strength"):
            parts.append(f"{k}{_unformatted(v)}")
        elif k == "mu":
            parts.append(f"{k}{v:.3f}")
        elif k == "fixedBNat100":
            parts.append(k if v else "")
        else:
            parts.append(f"{k}{v:.2f}")
    name = ("detailed/" if detail else "") + "sim_" + "_".join(parts)
    name = name.replace("__", "_")
    for k, default in SIZE_DEFAULTS.items():
        if params[k] != default:
            name += f"_{k}{params[k]:.2f}"
    if params["n_events"] == 2:
        name += "_2events"
    if params["lam"] > 0:
        name += f"_lambda{params['lam']:.4f}"
    return name


_FILENAME = re.compile(
    r"sim_link_prob(?P<link_prob>[\d.]+)_init_w(?P<init_w>[\d.]+)_beta(?P<beta>[\d.]+)_rho(?P<rho>[\d.]+)"
    r"_eps(?P<eps>[\d.]+)_mu(?P<mu>[\d.]+)(?P<fixed>_fixedBNat100)?_ext_strength(?P<ext_strength>[\d.]+)"
    r"_seed(?P<seed>\d+)(?:_M(?P<M>[\d.]+))?(?:_N(?P<N>[\d.]+))?(?:_tau(?P<tau>[\d.]+))?"
    r"(?P<events>_2events)?(?:_lambda(?P<lam>[\d.]+))?(?:_detailed)?")


def _stem(name) -> str:
    return Path(name).name.removesuffix(".csv")


def parse_filename(name) -> tuple[dict, int]:
    """Inverse of `legacy_filename`: a file name or path (with or without `.csv`) → (cell parameters, seed).

    The cell holds every parameter except the seed, as `legacy_filename` takes them, with `M`, `N` and `tau` at
    `SIZE_DEFAULTS` when their suffix is absent. Values are as rounded in the name (`rho = 0.33`). Detailed names
    (in `detailed/`, ending in `_detailed`) parse too; `is_detailed` tells them apart. Raises `ValueError` for any
    other name.
    """
    match = _FILENAME.fullmatch(_stem(name))
    if match is None:
        raise ValueError(f"not a legacy simulation file name: {name}")
    g = match.groupdict()
    ext = float(g["ext_strength"])
    cell = dict(link_prob=float(g["link_prob"]), init_w=float(g["init_w"]), beta=float(g["beta"]),
                rho=float(g["rho"]), eps=float(g["eps"]), mu=float(g["mu"]), fixedBNat100=g["fixed"] is not None,
                ext_strength=int(ext) if ext.is_integer() else ext, lam=float(g["lam"] or 0.0),
                n_events=2 if g["events"] else 1,
                **{k: default if g[k] is None else float(g[k]) for k, default in SIZE_DEFAULTS.items()})
    sizes = [k for k in SIZE_DEFAULTS if not float(cell[k]).is_integer()]
    if sizes:
        raise ValueError(f"not a legacy simulation file name ({', '.join(sizes)} must be whole): {name}")
    cell.update({k: int(cell[k]) for k in SIZE_DEFAULTS})
    return cell, int(g["seed"])


def is_detailed(name) -> bool:
    """Whether a file name or path is a detail-mode run's: under a `detailed/` folder or ending in `_detailed`."""
    return Path(name).parent.name == "detailed" or _stem(name).endswith("_detailed")
