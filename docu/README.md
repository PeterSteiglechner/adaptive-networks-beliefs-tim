# Documentation: Adaptive Networks of Beliefs

Code documentation for the ANB model and its analysis pipeline. The theory and results are in the paper:

> Steiglechner, P., Poulsen, V. M., Galesic, M., & Olsson, H. (2026). *Heterogeneous, Adaptive Networks of Beliefs.*
> PsyArXiv. https://doi.org/10.31234/osf.io/uxkg6_v1. Full text: [ANB.md](ANB.md).

| Document | Contents |
|----------|----------|
| [model.md](model.md) | How `ANBmodel.py` implements the theory: parameters, data layout, belief and edge updating, metrics, response classification, output format, caveats |
| [parameter-cheatsheet.md](parameter-cheatsheet.md) | Every parameter at a glance: what it does, which equation it enters, its legacy and `anb` names, baselines |
| [system-overview.md](system-overview.md) | Mermaid diagrams: the config → simulation → CSV → analysis → figures pipeline, and the inside of one simulation run |
| [running-simulations.md](running-simulations.md) | How to configure runs, filename conventions, and exactly which simulations each analysis script needs |
| [known-issues.md](known-issues.md) | Open bugs and inconsistencies in the model and analysis code, each with location, impact and a suggested fix |
| [analysis.md](analysis.md) | What each `analyse_*.py` script computes and which paper figure or table it produces |
| [adr-001-numba-over-gpu.md](adr-001-numba-over-gpu.md) | Design decision: speed up the simulation with Numba on the CPU instead of a GPU |
| [adr-002-testbed-architecture.md](adr-002-testbed-architecture.md) | Design decision: refactor into an `anb/` package with Agent/Network ABCs, an injectable model and a testbed |
| [class-design.md](class-design.md) | Mermaid class diagrams of the `anb/` package (one namespace per module), responsibilities of each class, and the resolution of ADR-002's open points |
| [spec-anb-package.md](spec-anb-package.md) | The buildable spec the `anb/` package was implemented from: user stories, module decisions, test seams |
| [validation.md](validation.md) | Statistical validation of `anb` against `ANBmodel.py`, comparing two output folders (`compare_old_new.py`: tables, ablations, reruns, shared starts), the benchmark, and the deliberate differences awaiting supervisor sign-off |

## The model in one paragraph

Each agent holds $M=10$ beliefs on $[-1,1]$, connected by a fully connected, weighted **belief network** (BN).
Belief 0 is the **focal belief**. Agents observe their social contacts' focal beliefs and feel social dissonance
when they disagree. Beliefs update stochastically to reduce **dissonance**: personal dissonance within the BN,
plus social dissonance and external dissonance on the focal belief (Glauber dynamics with attention $\beta$).
The BN itself adapts through **internal adaptation** (Hebbian: beliefs that change together get more strongly
related, rate $\epsilon$) and **social adaptation** (convergence to the neighbours' average BN, rate $\mu$).
After a warm-up phase ($t=0$–$100$), a temporary **external pressure** $s$ pushes the focal belief towards $+1$
($t=101$–$150$). Agents with a negative focal belief before the pressure are classified as **compliant**
(lasting change), **resilient** (temporary change) or **resistant** (no change).

## Key results the code reproduces

- Static BNs give homogeneous responses: weak edges lead to full compliance, strong edges to resistance or
  resilience. Adaptive BNs become heterogeneous, so all three responses coexist under the same pressure (Fig. 3).
- Non-compliance is associated with higher balance around the focal belief and higher BN clustering. Resistance,
  compared to resilience, is associated with a more strongly connected and more balanced focal belief and with
  fewer social contacts (Fig. 4, Table 3).
- The pressure also reshapes the BN. Compliers' BNs become more connected, and resisters' focal beliefs become
  more central (Fig. 5).
- Social adaptation reduces BN heterogeneity and affects responses non-monotonically: weak $\mu$ increases
  compliance, and strong $\mu$ increases resilience (Fig. 6).

## Quick start

```bash
conda activate anb                # or `uv sync` and prefix the commands with `uv run`
python ANBmodel.py                # the frozen reference model; configure runs first, see running-simulations.md
python analyse_timeseries.py      # then the other analysis scripts, see analysis.md
```

New runs should use the `anb/` package, which is about 46× faster and is configured from Python instead of by
editing source. `../notebooks/demo.ipynb` walks through it. A run saved with `save("simOut")` (the default) lands in
`simOut/` under the legacy filename, so the analysis scripts read it unchanged:

```python
from anb.agents import ANBAgent
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig

spec = ModelSpec(ANBAgent(beta=3.0, eps=1.0, mu=0.0, init_w=0.2), ErdosRenyiNetwork, {"link_prob": 0.1})
testbed = Testbed(spec, TestbedConfig(PressureSchedule.single_event(4)))
testbed.save_ensemble(range(100), "simOut", n_jobs=-1)   # simOut/sim_link_prob0.10_..._seed{n}.csv
```

`save_ensemble` writes each seed's CSV as soon as it finishes and returns the paths. `run_ensemble` returns the
results in memory instead. Both log progress once you call `logging.basicConfig(level=logging.INFO)`.
`save_ensemble(..., skip_existing=True)` resumes a crashed sweep. See [running-simulations.md](running-simulations.md#with-the-anb-package)
for both, including why `skip_existing` is unsafe after a model change.

The adaptive→static configuration is `ANBAgent(..., learning_until=101)`. Until the supervisor signs off on the
differences in [validation.md](validation.md), keep paper results on `ANBmodel.py`.

## Code ↔ paper notation

| Paper | Code |
|-------|------|
| $x_m$, $x_{foc}$ | columns `"0"`…`"9"`, focal = `"0"` / `x_focal` |
| $\omega_{mn}$ | columns `w{m}{n}` with `m < n` |
| $N$, $M$, $p$, $\tau$ | `n_agents`, `M`, `link_prob`, `tau` |
| $\beta$, $\rho$, $\epsilon$, $\mu$, $\omega_0$, $s$ | `beta`, `rho`, `eps`, `mu`, `init_w`, `ext_strength` |
| $D$ (dissonance) | `Hpers`, `Hpersfoc`, `Hsoc`, `Hext` (H = "energy") |
| balance / connectedness / clustering / centrality | `tb_*`, `absOm_*`, `clust`, `bc_foc` |
| static / adaptive→static / adaptive | `eps=0` / `fixedBNat100=True` / `eps=1` |
