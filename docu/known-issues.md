# Known Issues

These problems turned up while documenting `ANBmodel.py` and the analysis scripts. None of them has been
fixed. Each entry gives the location, what goes wrong, and a suggested fix.

Issues are ordered by impact:

- **Results**: can change numbers reported in the paper, or their interpretation.
- **Reproducibility**: the same inputs may not give the same outputs.
- **Breaks the pipeline**: a script crashes or cannot find its input.
- **Minor**: inconsistencies, typos, or traps for future changes.

| # | Issue | Category | Where |
|---|-------|----------|-------|
| 1 | `Hsoc` metric uses the neighbour mean, the model uses the sum | Results | `ANBmodel.py:233` |
| 2 | In-place shuffle of module-level lists carries state between runs | Reproducibility | `ANBmodel.py:179`, `:439` |
| 3 | README run table doesn't match what the scripts read | Breaks the pipeline | `README.md`, analysis scripts |
| 4 | Sensitivity analysis expects filename suffixes the model never writes | Breaks the pipeline | `ANBmodel.py:498–521`, `analyse_sensitivity.py:114–124` |
| 5 | `scikit-learn` / `statsmodels` missing from dependencies | Breaks the pipeline | `pyproject.toml`, `analyse_BNmetrics.py:443–444` |
| 6 | `analyse_BNmetrics.py` saves into `figs/` before creating it | Breaks the pipeline | `analyse_BNmetrics.py:388`, `:791` |
| 7 | `detail` / `track_times` exist only under `__main__` | Breaks the pipeline | `ANBmodel.py:331`, `:467`, `:573–584` |
| 8 | Two `fixedBNat100` flags; the BN still adapts at t = 101 | Minor | `ANBmodel.py:24`, `:448–457` |
| 9 | Boundary cases in the response classification | Minor | `ANBmodel.py:332–349`, `:402–411` |
| 10 | `tau` must be an integer, but the sensitivity script uses floats | Minor | `ANBmodel.py:129`, `:166` |
| 11 | `if True:` always overwrites existing results | Minor | `ANBmodel.py:525` |
| 12 | Small typos and leftovers in the analysis scripts | Minor | various |

---

## 1. `Hsoc` uses the neighbour mean, the model uses the sum

**Category:** Results

**Where:** `get_energies`, [ANBmodel.py:229-238](../ANBmodel.py#L229-L238), compared with the belief
update at [ANBmodel.py:441-447](../ANBmodel.py#L441-L447).

**What happens.** The belief update follows the paper's eq. 1, where social dissonance is
$-\rho\sum_{k}x_{foc}x^{(k)}_{foc}$ (the **sum** over contacts):

```python
summed_social_beliefs = 0 if len(social_beliefs) == 0 else np.sum(social_beliefs)
```

The recorded metric `Hsoc` uses the **mean**:

```python
-params["rho"] * agents[n][beliefids][focal] * np.nanmean(agents[nb_list[n]][:, beliefids[focal]])
```

**Impact.** The model dynamics are correct. The reported "dissonance focal social" values (paper Table 2/3,
Fig. 4, Fig. A4) are smaller than eq. 1 by a factor of $|\mathcal K|$, which is about 10 on average.
Two consequences:

- The magnitudes in Table 3 (≈ −0.05) can't be compared directly with the BN dissonances, which use
  sums (≈ −3 to −22).
- Because $|\mathcal K|$ differs between agents, the ranking of agents by `Hsoc` can differ from a
  ranking by the eq. 1 term. That can change the Cohen's d values for this metric.

**Fix.** Either replace `np.nanmean` with `np.sum` so the metric matches eq. 1, or keep the mean and state
in Table 2 that the metric is per contact. Re-running the analysis is only needed for the `Hsoc` rows.

---

## 2. In-place shuffle of module-level lists carries state between runs

**Category:** Reproducibility

**Where:** `np.random.shuffle(beliefupdate_order)` at [ANBmodel.py:179](../ANBmodel.py#L179) and
`np.random.shuffle(agentids)` at [ANBmodel.py:439](../ANBmodel.py#L439). Both lists are module-level
globals ([ANBmodel.py:59](../ANBmodel.py#L59), [ANBmodel.py:64](../ANBmodel.py#L64)).

**What happens.** `np.random.shuffle` permutes the list **in place**, and the result depends on the
list's current order. The lists are never reset at the start of `run_simulation`. If two simulations
run in the same Python process, the second run starts from whatever order the first one left behind.
The seed alone therefore doesn't determine the update order.

**Impact.** Within one process this is certain: running seed 0 and then seed 1 gives a different result
for seed 1 than running seed 1 alone. Under `joblib`, it depends on how runs are assigned to worker
processes and on how the functions are serialised. This is untested, but bit-for-bit reproducibility of
a given seed is probably not guaranteed. Ensemble statistics are not biased, because the order is still
uniformly random.

**Fix.** Reset the order at the start of each run, or avoid the shared state altogether:

```python
# in run_simulation, per time step
for n in np.random.permutation(n_agents):
    ...
# in update_belief
for dim in np.random.permutation(M):
    ...
```

Using a per-run `rng = np.random.default_rng(seed)` everywhere would be cleaner still.

---

## 3. README run table doesn't match what the scripts read

**Category:** Breaks the pipeline

**Where:** the Configuration table in [README.md](../README.md), compared with the hard-coded inputs of
the analysis scripts.

**What happens.** The README says a single detailed simulation uses `seeds=[1]` and `pressures=[4]`,
and that the ensemble uses `pressures=[1,2,4,8,16]`. The scripts need more than that:

| Script | What it reads | Not covered by the README |
|--------|---------------|---------------------------|
| `analyse_timeseries.py:57` | detailed, `seed = 2`, s = 4 | seed 2 |
| `analyse_responseFrequencies.py:184, 460` | detailed, seed 1, **s ∈ {0,1,2,4,8,16}** | five more pressure levels |
| `analyse_responseFrequencies.py:55` | ensemble, **s = 0** included | s = 0 |
| `analyse_BNmetrics.py:161` | ensemble, **s = 0** as the no-pressure reference | s = 0 |
| `analyse_responseFrequencies.py:56` | `seeds = range(20)` | inconsistent with 100 seeds elsewhere and in the paper |

**Impact.** If you follow the README, `analyse_timeseries.py` and `analyse_responseFrequencies.py`
stop with `FileNotFoundError`. `analyse_responseFrequencies.py` also reports response frequencies from
20 seeds while the paper reports 100.

**Fix.** Update the README table (see the recipes in
[running-simulations.md](running-simulations.md#recipes)) and set `seeds = list(range(100))` in
`analyse_responseFrequencies.py`.

---

## 4. Sensitivity analysis expects filename suffixes the model never writes

**Category:** Breaks the pipeline

**Where:** `run_one` builds the filename at [ANBmodel.py:498-521](../ANBmodel.py#L498-L521).
`analyse_sensitivity.py` reads at [analyse_sensitivity.py:114-124](../analyse_sensitivity.py#L114-L124).

**What happens.** `M`, `n_agents` and `tau` are module constants and are not part of the filename.
`analyse_sensitivity.py` expects `..._seed{seed}_M5.00.csv`, `..._N200.00.csv` and `..._tau2.00.csv`.

**Impact.** Changing `M`, `N` or `tau` produces files with the same name as the baseline. That has two effects:

- They **overwrite the baseline results**, because of issue 11.
- The sensitivity script can't find them.

**Fix.** Append the suffixes in `run_one` whenever a constant differs from its default:

```python
fname += f"_M{M:.2f}" if M != 10 else ""
fname += f"_N{n_agents:.2f}" if n_agents != 100 else ""
fname += f"_tau{tau:.2f}" if tau != 1 else ""
```

A better long-term fix is to pass `M`, `N` and `tau` as run parameters instead of editing module constants.

---

## 5. `scikit-learn` / `statsmodels` missing from dependencies

**Category:** Breaks the pipeline

**Where:** [pyproject.toml](../pyproject.toml). The packages are imported at
[analyse_BNmetrics.py:443-444](../analyse_BNmetrics.py#L443-L444).

**Impact.** After a clean `uv sync`, `analyse_BNmetrics.py` fails with `ModuleNotFoundError` halfway
through, after the slow data-loading step.

**Fix.** Run `uv add scikit-learn statsmodels`. Optionally also add `matplotlib` explicitly. It currently
only comes in through `seaborn`, yet every script imports it directly. `xarray` is listed but not used anywhere.

---

## 6. `analyse_BNmetrics.py` saves into `figs/` before creating it

**Category:** Breaks the pipeline

**Where:** `plt.savefig("figs/fig5_...")` at [analyse_BNmetrics.py:388](../analyse_BNmetrics.py#L388).
The `os.mkdir("figs")` only comes later, at [analyse_BNmetrics.py:791](../analyse_BNmetrics.py#L791).

**Impact.** On a fresh checkout, or if `analyse_BNmetrics.py` runs before the other scripts, it crashes
with `FileNotFoundError`.

**Fix.** Put `os.makedirs("figs", exist_ok=True)` once at the top of every analysis script.

---

## 7. `detail` / `track_times` exist only under `__main__`

**Category:** Breaks the pipeline (for reuse)

**Where:** These globals are defined at [ANBmodel.py:573-584](../ANBmodel.py#L573-L584), inside
`if __name__ == "__main__":`. They are read by `get_output` (`:331`), `run_simulation` (`:467`) and
`run_one` (`:484`, `:500`, `:531`).

**Impact.** `uv run ANBmodel.py` works. `from ANBmodel import run_one` followed by `run_one(...)` raises
`NameError: name 'detail' is not defined`. This matters for the planned project extensions (adaptive
social networks, influencers, etc.), which will probably import the model rather than edit its main block.

**Fix.** Give both defaults at module level, or better, pass them as arguments:

```python
def run_simulation(params, detail=False, track_times=None): ...
```

---

## 8. Two `fixedBNat100` flags; the BN still adapts at t = 101

**Category:** Minor

**Where:** The module-level flag is at [ANBmodel.py:24](../ANBmodel.py#L24). It is checked at
[ANBmodel.py:448](../ANBmodel.py#L448), and the per-run flag is checked at
[ANBmodel.py:453-457](../ANBmodel.py#L453-L457).

**What happens.**

- `if fixedBNat100 and (t >= ext_time[0]): pass` reads the **module-level** flag, which is always `False`.
  The branch is dead code. Only `params["fixedBNat100"]` has any effect.
- That per-run check uses `t > ext_time[0]`, i.e. `t > 101`. During t = 101, the first pressure step,
  the BN still adapts. The paper describes the BN as "adaptive until t = 100 and static thereafter".

**Impact.** The adaptive→static configuration has one step of adaptation under pressure. The effect on
results is probably negligible, but code and paper disagree.

**Fix.** Delete the module-level flag and the dead branch. Use `t >= ext_time[0]` if the BN should be
frozen from the first pressure step on.

---

## 9. Boundary cases in the response classification

**Category:** Minor

**Where:** [ANBmodel.py:332-349](../ANBmodel.py#L332-L349) (detailed mode) and
[ANBmodel.py:402-411](../ANBmodel.py#L402-L411) (ensemble mode).

**What happens.** The rules treat a window-mean of exactly 0 inconsistently:

- `compliant` requires `after >= 0`, but `latecompliant` requires `after > 0`. An agent with
  before < 0, during < 0 and an after-mean of exactly 0 gets **no** label.
- A before-mean of exactly 0 counts as "positive" (aligned), so that agent is excluded from the
  misaligned group.

**Impact.** This is rare, because a 10-step mean of beliefs on a 0.1 grid seldom equals exactly 0.
Unlabelled agents drop silently out of the response frequencies.

**Fix.** Use `>= 0` in `latecompliant` as well, or define one `positive = x >= 0` helper and use it everywhere.

---

## 10. `tau` must be an integer

**Category:** Minor

**Where:** `list(np.zeros(M)) * tau` at [ANBmodel.py:129](../ANBmodel.py#L129) and `reshape((tau, M))` at
[ANBmodel.py:166](../ANBmodel.py#L166).

**Impact.** `analyse_sensitivity.py` lists `tau` as `2.0` and `10.0`. If you set `tau = 2.0` in the model
to match, it crashes with `TypeError`.

**Fix.** Keep `tau` an `int` in the model and format it as `f"{tau:.2f}"` only in the filename.

---

## 11. `if True:` always overwrites existing results

**Category:** Minor

**Where:** [ANBmodel.py:525](../ANBmodel.py#L525). The file-exists check is commented out:
`if True:  # not os.path.isfile(fname+".csv"):`.

**Impact.** Every run recomputes and overwrites its CSV, which wastes time when a batch is resumed.
Together with issue 4, it silently replaces baseline results.

**Fix.** Restore the check, behind an explicit `overwrite` flag.

---

## 12. Small typos and leftovers in the analysis scripts

**Category:** Minor

| Where | What |
|-------|------|
| `README.md` | Lists `analsyse_sensitivity.py`; the file is `analyse_sensitivity.py`. |
| [analyse_socialAdaptation.py:327](../analyse_socialAdaptation.py#L327) | Filename token `'_fixedBNat1001.00'` should be `'_fixedBNat100'`. It is only reached if `fixedBNat100=True`. |
| [analyse_sensitivity.py:124](../analyse_sensitivity.py#L124) | Token `'_fixedBNatt100'` (double `t`) should be `'_fixedBNat100'`. Same condition. |
| [analyse_socialAdaptation.py:255](../analyse_socialAdaptation.py#L255) | Fig. A7 uses the leftover loop variables `seed = 99` and `s = 4`. It works, but the choice of example simulation is implicit. |
| [analyse_BNmetrics.py:446](../analyse_BNmetrics.py#L446) | `dependent_vars` holds the regression **predictors**. The dependent variable is `response_group`. |
| All analysis scripts | `M = 10`, the focal index and column names are hard-coded instead of being imported from `ANBmodel.py`. A model run with a different `M` (sensitivity) has a different number of columns. |
