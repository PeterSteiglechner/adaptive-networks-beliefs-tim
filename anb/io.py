"""Output naming and parameter columns compatible with the legacy `analyse_*.py` scripts."""

import numpy as np

# The parameters legacy `run_one` appends as columns, in its order. Also the order of its filename tokens.
LEGACY_PARAM_COLUMNS = ("link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100", "ext_strength", "seed")

# Parameters legacy encodes only in the filename. They become columns only when they are per-agent.
FILENAME_ONLY_PARAMS = ("lam", "n_events")

# The `learning_until` that legacy's `fixedBNat100` flag stands for. Any other gate has no legacy name.
LEGACY_LEARNING_UNTIL = 101


def parameter_columns(params: dict) -> list[str]:
    """The output columns for `params`: the legacy ones in legacy order, then any others in `params` order.

    A scalar `lam` or `n_events` gets no column (legacy writes them only into the filename).
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
    `fixedBNat100`, `ext_strength`, `seed`) plus `lam` and `n_events`, all scalars. As in legacy, the
    `_M`/`_N`/`_tau` suffixes are not added (known issue #4). Detail-mode files additionally end in
    `_detailed` before `.csv`; that suffix is not part of this name, as in `run_one`'s return value. Legacy's
    `simOut_old_model/` prefix is left off too, so `SimulationResult.save` picks the folder (`simOut/` by default).

    Raises `ValueError` for parameters legacy can't name: a missing key (any network but Erdős–Rényi), an
    extra one (`learning_until` ≠ 101, a `fixed` mask, other network kwargs) or a per-agent value.
    """
    known = (*LEGACY_PARAM_COLUMNS, *FILENAME_ONLY_PARAMS)
    missing = [k for k in known if k not in params]
    extra = [k for k in params if k not in known]
    if missing or extra:
        problems = ([f"missing: {', '.join(missing)}"] if missing else []) + \
                   ([f"no legacy token for: {', '.join(extra)}"] if extra else [])
        raise ValueError(f"legacy filenames only exist for the paper's setup ({'; '.join(problems)}); "
                         "save with legacy_names=False")
    per_agent = [k for k in (*LEGACY_PARAM_COLUMNS, *FILENAME_ONLY_PARAMS) if np.ndim(params[k]) != 0]
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
    if params["n_events"] == 2:
        name += "_2events"
    if params["lam"] > 0:
        name += f"_lambda{params['lam']:.4f}"
    return name
