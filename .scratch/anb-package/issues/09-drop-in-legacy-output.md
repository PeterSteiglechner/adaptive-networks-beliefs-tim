# 09: Drop-in legacy output

**What to build:** An analysis-script user points an unchanged `analyse_*.py` script at CSVs written by the new package, and it runs. The default `Testbed` metric set is all 14 legacy metrics. `SimulationResult.to_frame()` appends the parameter columns from `spec.params()` plus the schedule info:

- uniform parameters are scalar columns, as legacy writes them
- per-agent parameters are per-agent columns of the same name, joined on `id`
- `fixedBNat100` is derived from `learning_until`

`SimulationResult.save(directory, legacy_names=True)` writes the CSV under `legacy_filename`. See the spec `docu/spec-anb-package.md`, "Output compatibility" (stories 21–27).

**Blocked by:** 03 (`legacy_filename`), 06 (Belief-network structure metrics), 07 (`FocalBetweenness` metric), 08 (Response classification, ensemble mode, `run_ensemble`)

**Status:** done

- [x] The column set **and order** equal legacy's output exactly, in both ensemble and detail mode (compared against a legacy-produced frame)
- [x] Homogeneous runs produce the scalar legacy parameter columns unchanged, so old and new CSVs concatenate cleanly
- [x] A heterogeneous run (for example per-agent β) produces a per-agent `beta` column matching each agent's value
- [x] `save()` writes to the legacy path and filename in both modes
- [x] Demo: a small batch of new ensemble CSVs, enough for one analysis script to read, is generated, and that script runs unchanged on them (written up in the ticket or PR, not as a default test)

## Demo (2026-10-16)

The runs `analyse_responseFrequencies.py` reads (4 BN configs × s ∈ {0, 1, 2, 4, 8, 16} × seeds 0–19 in ensemble
mode, plus the adaptive config × all six pressures × seed 1 in detail mode: 486 CSVs, N = 100, T = 200) were
generated with the package (`Testbed(...).run(seed).save(".")`, joblib over 11 cores, about 2.5 min on the
pure-Python kernel) in a scratch directory. The **unchanged** script, run from that directory, exits 0 and writes
`figs/fig3_responseFrequencies.{png,pdf}` and `figs/AppendixFig_exampleSim_adaptive_seed1_overPressure.{png,pdf}`.
Fig. 3 D/E has the paper's qualitative shape. Statistical agreement with legacy is ticket 11.

Generator (run in an empty directory, then `MPLBACKEND=Agg python <repo>/analyse_responseFrequencies.py` there):

```python
"""Ticket 09 demo: the runs analyse_responseFrequencies.py reads, written by the anb package into ./simOut/."""

import itertools
import sys

from joblib import Parallel, delayed

from anb.agents import ANBAgent
from anb.model import ModelSpec
from anb.networks import ErdosRenyiNetwork
from anb.testbed import PressureSchedule, Testbed, TestbedConfig

BN_CONFIGS = [(0.2, 0.0, 0.0, False), (0.8, 0.0, 0.0, False), (0.2, 1.0, 0.0, True), (0.2, 1.0, 0.0, False)]
PRESSURES = [0, 1, 2, 4, 8, 16]


def one(init_w, eps, mu, fixed, pressure, seed, detail):
    agent = ANBAgent(beta=3.0, eps=eps, mu=mu, init_w=init_w, learning_until=101 if fixed else None)
    spec = ModelSpec(agent, ErdosRenyiNetwork, {"link_prob": 0.1}, n_agents=100, rho=1 / 3)
    config = TestbedConfig(PressureSchedule.single_event(pressure), detail=detail)
    return str(Testbed(spec, config).run(seed).save("."))


jobs = [(*bn, s, seed, False) for bn, s, seed in itertools.product(BN_CONFIGS, PRESSURES, range(20))]
jobs += [(*BN_CONFIGS[3], s, 1, True) for s in PRESSURES]
paths = Parallel(n_jobs=-2)(delayed(one)(*job) for job in jobs)
print(len(paths), "files, e.g.", paths[0], paths[-1], file=sys.stderr)
```

Notes on decisions:
- `lam` and `n_events` are filename-only in legacy, so a scalar value gets no column (the column set stays
  exactly legacy's). A per-agent `lam` becomes a per-agent column, so the value isn't lost. Other parameters
  not in legacy's nine (e.g. extra network kwargs) are appended after `seed`.
- `save()` writes the index like legacy's `to_csv`, so read-back frames have the same leading `Unnamed: 0`.
- `legacy_filename` raises `ValueError` for per-agent parameters. Use `save(dir, legacy_names=False)`
  (`seed{seed}.csv`) for heterogeneous runs.
- `SimulationResult` gained `detail: bool` so `save()` can choose the legacy directory and `_detailed` suffix.
