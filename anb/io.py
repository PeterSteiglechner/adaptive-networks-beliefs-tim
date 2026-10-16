"""Output naming and parameter columns compatible with the legacy `analyse_*.py` scripts."""

import numpy as np

# The parameters legacy `run_one` appends as columns, in its order. Also the order of its filename tokens.
LEGACY_PARAM_COLUMNS = ("link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100", "ext_strength", "seed")

# Parameters legacy encodes only in the filename. They become columns only when they are per-agent.
FILENAME_ONLY_PARAMS = ("lam", "n_events")


def parameter_columns(params: dict) -> list[str]:
    """The output columns for `params`: the legacy ones in legacy order, then any others in `params` order.

    A scalar `lam` or `n_events` gets no column (legacy writes them only into the filename).
    """
    extra = [
        k for k in params
        if k not in LEGACY_PARAM_COLUMNS and not (k in FILENAME_ONLY_PARAMS and np.ndim(params[k]) == 0)
    ]
    return [k for k in LEGACY_PARAM_COLUMNS if k in params] + extra


def legacy_filename(params: dict, detail: bool) -> str:
    """The path (without `.csv`) that legacy `run_one` writes for `params`.

    `params` holds the nine `run_one` keys (`link_prob`, `init_w`, `beta`, `rho`, `eps`, `mu`,
    `fixedBNat100`, `ext_strength`, `seed`) plus `lam` and `n_events`, all scalars. As in legacy, the
    `_M`/`_N`/`_tau` suffixes are not added (known issue #4). Detail-mode files additionally end in
    `_detailed` before `.csv`; that suffix is not part of this name, as in `run_one`'s return value.
    """
    per_agent = [k for k in (*LEGACY_PARAM_COLUMNS, *FILENAME_ONLY_PARAMS) if np.ndim(params[k]) != 0]
    if per_agent:
        raise ValueError(f"legacy filenames need uniform parameters; per-agent: {', '.join(per_agent)}")
    parts = []
    for k in LEGACY_PARAM_COLUMNS:
        v = params[k]
        if k in ("seed", "ext_strength"):
            parts.append(f"{k}{v}")
        elif k == "mu":
            parts.append(f"{k}{v:.3f}")
        elif k == "fixedBNat100":
            parts.append(k if v else "")
        else:
            parts.append(f"{k}{v:.2f}")
    name = "simOut/" + ("detailed/" if detail else "") + "sim_" + "_".join(parts)
    name = name.replace("__", "_")
    if params["n_events"] == 2:
        name += "_2events"
    if params["lam"] > 0:
        name += f"_lambda{params['lam']:.4f}"
    return name
