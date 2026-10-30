"""A small synthetic simulation folder for the `anb.analysis` tests, written with `Testbed.save_ensemble`.

What it holds (all on the paper's time structure, T = 200, pressure at t = 101…150):

- the paper base (`link_prob 0.1`, N = 100), ensemble runs of the four BN configs × s ∈ {0, 1, 4} × seeds {0, 1, 2},
  with two deliberate gaps (`GAPS`): no `adaptive2static` at s = 1, and `staticlow` at s = 4 for seed 0 only;
- one detailed run: adaptive, s = 4, seed 2 (Fig. 2's example simulation);
- one M/N/τ variant (`VARIANT`: M = 6, N = 50, τ = 2), adaptive at s = 4: ensemble seeds {0, 1} and detailed seed 2;
- one non-paper base (`link_prob 0.2`), adaptive at s = 4: ensemble seeds {0, 1} and detailed seed 2;
- social adaptation (`SOCIAL`), adaptive at s = 4, ensemble seed 1 only: μ ∈ {0.005, 0.2} for the paper base (Fig. 6's
  example μ values) and μ = 0.2 for the M/N/τ variant;
- Fig. A8's single-parameter variants of the paper base (`SENSITIVITY`), staticlow, statichigh and adaptive at s = 4,
  seed 0: the M row and the N row complete (M ∈ {5, 15}; N ∈ {50, 200} with p = 10/N), the τ row only half (τ = 2,
  no τ = 10);
- stray files that don't follow the naming scheme (`STRAY`).

Run as a script (`python tests/sim_folder.py OUT_DIR`) to build it in a fresh interpreter, which `conftest.py`
does when the parent runs with `NUMBA_DISABLE_JIT=1` (the kernel would take minutes per run without the JIT).
"""

import sys
from pathlib import Path

from anb.agents import ANBAgent
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig

CONFIGS = {  # (init_w, eps, mu, fixedBNat100), as in docu/analysis-package.md
    "staticlow": (0.2, 0.0, 0.0, False),
    "statichigh": (0.8, 0.0, 0.0, False),
    "adaptive2static": (0.2, 1.0, 0.0, True),
    "adaptive": (0.2, 1.0, 0.0, False),
}
PRESSURES = (0, 1, 4)
SEEDS = (0, 1, 2)
GAPS = {("adaptive2static", 1): (), ("staticlow", 4): (0,)}  # (config, s) → the seeds that do exist
DETAILED = dict(config="adaptive", s=4, seed=2)
VARIANT = dict(M=6, N=50, tau=2)
OTHER_LINK_PROB = 0.2
SOCIAL = dict(mus=(0.005, 0.2), variant_mus=(0.2,), seed=1)
SENSITIVITY = (dict(M=5), dict(M=15), dict(N=50, link_prob=0.2), dict(N=200, link_prob=0.05), dict(tau=2))
SENSITIVITY_CONFIGS = ("staticlow", "statichigh", "adaptive")
STRAY = ("notes.csv", "detailed/sim_typo_fixedBNatt100.csv")


def testbed(config="adaptive", s=4, *, link_prob=0.1, M=10, N=100, tau=1, detail=False, mu=None) -> Testbed:
    """One config's testbed; `mu` replaces the config's μ."""
    init_w, eps, config_mu, fixed = CONFIGS[config]
    mu = config_mu if mu is None else mu
    agent = ANBAgent(beta=3.0, eps=eps, mu=mu, init_w=init_w, learning_until=101 if fixed else None, tau=tau,
                     n_beliefs=M)
    spec = ModelSpec(agent, ErdosRenyiNetwork, {"link_prob": link_prob}, n_agents=N, rho=1 / 3)
    return Testbed(spec, TestbedConfig(PressureSchedule.single_event(s), detail=detail))


def build(out_dir) -> Path:
    out_dir = Path(out_dir)
    for config in CONFIGS:
        for s in PRESSURES:
            testbed(config, s).save_ensemble(GAPS.get((config, s), SEEDS), out_dir)
    detailed = dict(DETAILED)
    seed = detailed.pop("seed")
    testbed(**detailed, detail=True).save_ensemble([seed], out_dir)
    for kwargs in (VARIANT, dict(link_prob=OTHER_LINK_PROB)):
        testbed("adaptive", 4, **kwargs).save_ensemble([0, 1], out_dir)
        testbed("adaptive", 4, **kwargs, detail=True).save_ensemble([2], out_dir)
    for mus, kwargs in ((SOCIAL["mus"], {}), (SOCIAL["variant_mus"], VARIANT)):
        for mu in mus:
            testbed("adaptive", 4, mu=mu, **kwargs).save_ensemble([SOCIAL["seed"]], out_dir)
    for kwargs in SENSITIVITY:
        for config in SENSITIVITY_CONFIGS:
            testbed(config, 4, **kwargs).save_ensemble([0], out_dir)
    for name in STRAY:
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text("t,id\n")
    return out_dir


if __name__ == "__main__":
    build(sys.argv[1])
