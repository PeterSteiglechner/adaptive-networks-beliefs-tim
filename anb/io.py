"""Output naming compatible with the legacy `analyse_*.py` scripts."""


def legacy_filename(params: dict, detail: bool) -> str:
    """The path (without `.csv`) that legacy `run_one` writes for `params`.

    `params` holds the nine `run_one` keys (`link_prob`, `init_w`, `beta`, `rho`, `eps`, `mu`,
    `fixedBNat100`, `ext_strength`, `seed`) plus `lam` and `n_events`. As in legacy, the `_M`/`_N`/`_tau`
    suffixes are not added (known issue #4).
    """
    parts = []
    for k in ("link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100", "ext_strength", "seed"):
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
