# The ANB Model (`ANBmodel.py`)

This document explains how the Adaptive Networks of Beliefs (ANB) model from the paper
([ANB.md](ANB.md), Steiglechner, Poulsen, Galesic & Olsson, 2026) is implemented in
[`ANBmodel.py`](../ANBmodel.py). Each section names the paper's equation and the function
that implements it.

## 1. Overview

Each of the $N$ agents has:

- **beliefs** $X = \{x_1,\dots,x_M\}$, each on the discrete scale $\{-1,-0.9,\dots,1\}$ (21 values).
  Belief `0` is the **focal belief** $x_{foc}$. It is the only belief that neighbours can observe and the only one that external pressure acts on.
- a **belief network** $\Omega$: a fully connected, undirected, weighted network over the $M$ beliefs.
  The edge weights $\omega_{mn}$ are unbounded real numbers and can be positive or negative.

Agents sit on a fixed Erdős–Rényi **social network**. In each time step, every agent (in random order):

1. updates each of its $M$ beliefs in random order to reduce dissonance (Glauber dynamics, eq. 2),
2. updates its belief-network edges through internal (Hebbian) and social adaptation (eq. 3).

From $t=101$ to $t=150$, an **external pressure** of strength $s$ pushes the focal belief towards $+1$ (eq. 4).
Agents whose focal belief was negative before the pressure are then classified as
*compliant*, *resilient* or *resistant*, depending on what their focal belief does during and after the pressure.

## 2. Parameters

### Swept parameters (passed per run via `run_one`)

| Code name      | Paper symbol | Meaning                                                      | Baseline |
|----------------|--------------|--------------------------------------------------------------|----------|
| `link_prob`    | $p$          | Erdős–Rényi link probability of the social network           | `10/N = 0.1` |
| `init_w`       | $\omega_0$   | initial weight of every belief-network edge                  | `0.2`    |
| `beta`         | $\beta$      | attention to dissonance (inverse temperature)                | `3.0`    |
| `rho`          | $\rho$       | weight of the social coupling between focal beliefs         | `1/3`    |
| `eps`          | $\epsilon$   | internal (Hebbian) adaptation rate                           | `1.0` (`0` = static BN) |
| `mu`           | $\mu$        | social adaptation rate                                       | `0.0`    |
| `fixedBNat100` | –            | if `True`, the BN adapts until the pressure starts and is frozen afterwards | `False` |
| `ext_strength` | $s$          | strength of the external pressure on the focal belief        | `{0,1,2,4,8,16}` |
| `seed`         | –            | RNG seed (seeds the initial beliefs, the social network and the dynamics) | `0…` |

### Fixed module-level constants (top of the file)

| Constant              | Paper symbol | Value | Meaning |
|-----------------------|--------------|-------|---------|
| `M`                   | $M$          | 10    | number of beliefs, including the focal belief |
| `focal`, `ext_belief` | $foc$        | 0     | index of the focal belief |
| `n_agents`            | $N$          | 100   | number of agents |
| `tau`                 | $\tau$       | 1     | activation memory: number of past steps averaged into $\partial x_m$ |
| `lam`                 | $\lambda$    | 0.0   | optional edge-weight decay (`-lam * w` in the edge update). It is not part of the main paper model. |
| `two_external_events` | –            | False | if `True`: $T=300$, with a second pressure event at $t=201$–$250$ |
| `T`                   | –            | 200 (300) | last time step |
| `ext_time`            | –            | 101–150 (+ 201–250) | time steps with active pressure |
| `beforeRange`         | –            | 91–100  | window for "before" averages |
| `duringRange`         | –            | 141–150 | window for "during" averages |
| `afterRange`          | –            | 191–200 | window for "after" averages |
| `belief_options`      | $\mathcal{X}$ | `linspace(-1,1,21)` | belief space |

> To change `M`, `N` or `tau` (as in the sensitivity analysis), edit these constants directly.
> They are **not** part of the output filename, so you also need to change the filename manually
> (see [running-simulations.md](running-simulations.md#sensitivity-runs)).

## 3. Data layout: the agent matrix

The state of the whole population is a single NumPy array `agents` of shape
`(n_agents, n_columns)`. Each row is one agent, and its columns are, in order:

| Columns | Index variable | Content |
|---------|----------------|---------|
| `t`, `id` | `meta_cols` (0, 1) | time step, agent id |
| 14 metric columns | `metric_cols` (2–15) | metrics, filled by `fill_metrics` at tracked times |
| 6 response columns | `response_cols` (16–21) | response dummies, filled at the end by `get_output` |
| `"0"`…`"9"` | `beliefids` | the $M$ beliefs ($x_0 = x_{foc}$) |
| `w01`, `w02`, …, `w89` | `edgeids` | the $M(M-1)/2 = 45$ edge weights, ordered as `itertools.combinations(range(M), 2)` |
| (unnamed) | `delbeliefids` | the last $\tau$ belief changes $\Delta x$ (memory for Hebbian learning; not saved) |

Helper lookups:

- `edge_list` / `edge_arr`: list of `(m, n)` pairs with `m < n`.
- `edge_lookup[(m, n)]`: the edge's position in `edge_list`.
- `belief_neighbours[d]`: all belief indices except `d`.
- `adjacent_edge_ids[d]`: edge positions of all edges incident to belief `d`, in the same order as `belief_neighbours[d]`.

## 4. Initialisation

**`initialise_agents(init_w)`**: draws each belief uniformly from the 21 options and sets all 45 edges to
`init_w`. The belief-change memory starts at zero.

**`initialise_network(seed, link_prob)`**: re-seeds NumPy with `seed` and draws an undirected Erdős–Rényi
adjacency matrix (no self-loops). It returns `nb_list`, where `nb_list[i]` is the array of agent `i`'s neighbours.
The social network stays fixed for the whole run.

## 5. Dissonance and belief updating (paper eq. 1, 2, 4)

Paper eq. 1 (with eq. 4 during the pressure):

$$
D = -\sum_{m<n}\omega_{mn}x_mx_n \;-\; \rho\sum_{k\in\mathcal K}x_{foc}x^{(k)}_{foc} \;-\; s\,x_{foc}
$$

Only one belief $x_d$ changes at a time, so the change in dissonance from moving $x_d \to x'$ is linear in $(x'-x_d)$:

$$
\Delta D(x') = -(x' - x_d)\Big[\sum_{j\neq d}\omega_{dj}x_j \;+\; \mathbb 1_{d=foc}\big(s + \rho\sum_{k}x^{(k)}_{foc}\big)\Big]
$$

**`glauber_fast(dim, agent, ext_strength, beta, rho, summed_social_beliefs)`** evaluates $\Delta D$ for all 21
candidate values at once and returns

$$
p(x') \propto \frac{1}{1+e^{\beta\,\Delta D(x')}}
$$

normalised over all candidates, which is the paper's eq. 2. For non-focal beliefs the caller passes
`ext_strength = 0` and `summed_social_beliefs = 0`.

**`update_belief(agent, ext_strength, beta, rho, summed_social_beliefs)`**:

1. shuffles `beliefupdate_order` and updates each belief in turn. Each new value is sampled from
   `glauber_fast` and written back immediately, so later beliefs in the same step see the earlier updates.
2. shifts the $\tau$-step belief-change memory and stores the new change $\Delta x = x(t) - x(t-1)$.

> The social term uses the **sum** over neighbours' focal beliefs (as in eq. 1). The neighbours'
> focal beliefs are read from the current `agents` array, which is updated asynchronously: neighbours
> that have already moved in this time step contribute their new values.

## 6. Belief-network updating (paper eq. 3)

$$
\omega_{mn} \leftarrow \omega_{mn} + \epsilon\,\partial x_m\,\partial x_n + \mu\big(\bar\omega^{(\mathcal K)}_{mn} - \omega_{mn}\big) - \lambda\,\omega_{mn},
\qquad \partial x_m = \tfrac1\tau\sum_{i=0}^{\tau-1}\Delta x_m(t-i)
$$

**`update_edge_weights(agent, eps, mu, mean_social_edges)`** applies this rule to all 45 edges at once
(vectorised over `edge_arr`). `mean_social_edges` is the average edge vector of the agent's neighbours,
read from the current `agents` array. Notes:

- Agents without neighbours get `mu = 0`.
- With `fixedBNat100=True`, `eps` and `mu` are set to 0 for $t > 101$. The BN still adapts at $t=101$,
  the first pressure step, and is frozen from $t=102$ on.
- `lam` is read from `params_fixed["lam"]`, not from the per-run parameters.

## 7. Main loop: `run_simulation(params)`

```
seed RNG → initialise_agents → initialise_network (re-seeds with the same seed)
fill_metrics(t=0); snapshots = agents
for t in 1..T:
    s_t = ext_strength if t in ext_time else 0
    shuffle agent order
    for each agent n:
        update_belief(n, s_t, β, ρ, Σ_k x_foc^(k))
        update_edge_weights(n, ε, μ, mean of neighbours' edges)
    if t in track_times or t == T:
        fill_metrics(t); append to snapshots
return get_output(snapshots)
```

Updates are **asynchronous**: each agent's belief and edge update is written to `agents` immediately,
and agents later in the shuffled order see it.

## 8. Metrics (`get_energies`, `get_metrics`, `fill_metrics`)

These are computed for every agent at each tracked time step. They correspond to Table 2 of the paper.

| Column | Paper name (Table 2) | Definition in code |
|--------|----------------------|--------------------|
| `n_nbs` | nr social contacts $\lvert\mathcal K\rvert$ | `len(nb_list[n])` |
| `Hpers` | total personal (BN) dissonance | $-\sum_{m<n}\omega_{mn}x_mx_n$ (computed as $-\tfrac12\sum_m x_m\sum_{j\ne m}\omega_{mj}x_j$) |
| `Hpersfoc` | dissonance focal BN | $-x_{foc}\sum_{m\neq foc}\omega_{foc,m}x_m$ |
| *(derived)* `Hpersnonfoc` | dissonance non-focal BN | `Hpers - Hpersfoc`. This is computed in the analysis scripts, not saved. |
| `Hsoc` | dissonance focal social | $-\rho\,x_{foc}\cdot\mathrm{mean}_k\,x^{(k)}_{foc}$. **Note: this uses the mean, while eq. 1 and the belief update use the sum** (see §11). |
| `Hext` | external dissonance | $-s\,x_{foc}$ during pressure steps, else 0 |
| `tb_tot` | balance BN | number of balanced triangles, i.e. $\omega_{ab}\omega_{ac}\omega_{bc}>0$ (out of $\binom{M}{3}=120$) |
| `tb_foc` | balance focal BN | balanced triangles that contain the focal belief (out of $\binom{M-1}{2}=36$) |
| `absOm_tot` | connectedness BN | mean $\lvert\omega_{mn}\rvert$ over all 45 edges |
| `absOm_foc` | connectedness focal BN | mean $\lvert\omega_{foc,m}\rvert$ over the 9 focal edges |
| `clust` | clustering BN | Onnela et al. (2005) weighted clustering on $\lvert\omega\rvert$, with weights normalised by the agent's own max $\lvert\omega\rvert$, averaged over nodes |
| `bc_foc` | centrality focal BN | igraph betweenness of the focal node (unnormalised), with path lengths $1/(\lvert\omega\rvert+10^{-6})$ |
| `expI` | – (not used in the paper) | "expected influence" on the focal belief, $\sum_m\omega_{foc,m}x_m$, i.e. the local field |
| `x_focal` | focal belief | $x_{foc}$ |
| `extr_nonfoc` | extremity non-focal beliefs | $\tfrac{1}{M-1}\sum_{m\neq foc}\lvert x_m\rvert$ |

## 9. Response classification (`get_output`)

The focal belief of each agent is averaged over three windows:
`before` (t=91–100), `during` (t=141–150) and `after` (t=191–200). From these means the code builds 0/1 dummies:

| Column | Condition | Paper term |
|--------|-----------|------------|
| `persistPos`    | before ≥ 0, during ≥ 0, after ≥ 0 | aligned agents that stay positive |
| `nonpersistPos` | before ≥ 0, after < 0             | aligned agents that turn negative |
| `compliant`     | before < 0, during ≥ 0, after ≥ 0 | **compliance**: lasting change |
| `resilient`     | before < 0, during ≥ 0, after < 0 | **resilience**: temporary change |
| `resistant`     | before < 0, during < 0, after < 0 | **resistance**: no change |
| `latecompliant` | before < 0, during < 0, after > 0 | late compliance (negligible) |

The paper's response frequencies are computed only over the misaligned agents (before < 0), i.e. over
`compliant + resilient + resistant (+ latecompliant)`.

## 10. Output files

`run_one(...)` runs one simulation and writes a CSV. The parameter values are appended as constant columns.

**Filename pattern**

```
simOut/sim_link_prob{p:.2f}_init_w{w0:.2f}_beta{β:.2f}_rho{ρ:.2f}_eps{ε:.2f}_mu{μ:.3f}[_fixedBNat100]_ext_strength{s}_seed{seed}[_2events][_lambda{λ:.4f}].csv
simOut/detailed/sim_..._detailed.csv          # if detail=True
```

e.g. `simOut/sim_link_prob0.10_init_w0.20_beta3.00_rho0.33_eps1.00_mu0.000_ext_strength4_seed7.csv`.

**Ensemble mode (`detail = False`)**: a compact CSV with `n_agents` rows per stored time `t`:

| `t` value | Content |
|-----------|---------|
| `0`     | snapshot at initialisation |
| `95.5`  | mean over t=91–100 ("before"). **The response dummies are stored only in these rows.** |
| `100`   | snapshot just before the pressure |
| `145.5` | mean over t=141–150 ("during") |
| `195.5` | mean over t=191–200 ("after") |
| `200`   | final snapshot |
| `245.5`, `295.5`, `300` | only with `two_external_events` |

Columns: `t, id`, the metric columns, the belief columns `0…9`, the edge columns `w01…w89`, the response
columns (only on `t == 95.5` rows), and the run parameters.

**Detailed mode (`detail = True`)**: every time step for every agent (`(T+1)·N` rows). The response
dummies are written only on the rows of the last time step.

## 11. Implementation notes and caveats

These are things worth knowing before you extend the code:

1. **Globals defined under `__main__`.** `detail` and `track_times` are only set inside
   `if __name__ == "__main__":`, but `run_simulation`, `get_output` and `run_one` read them. If you
   `import ANBmodel` and call `run_one` from elsewhere, set `ANBmodel.detail` and `ANBmodel.track_times`
   first, or you get a `NameError`.
2. **Mean vs. sum in `Hsoc`.** The belief update uses $\rho\sum_k x^{(k)}_{foc}$ (eq. 1), but the
   recorded metric `Hsoc` uses the neighbour **mean**. The reported "dissonance focal social" values
   (e.g. Table 3) are therefore per-contact values, roughly $1/\lvert\mathcal K\rvert$ of eq. 1's term.
3. **`fixedBNat100` exists twice.** There is a module-level `fixedBNat100 = False` and a per-run
   `params["fixedBNat100"]`. The first `if` in the main loop checks the module-level flag, which is
   always `False`, so only the per-run flag has an effect.
4. **In-place shuffling of module-level lists.** `agentids` and `beliefupdate_order` are module-level
   lists that are shuffled in place. If a joblib worker runs several simulations, a run can start
   from a list order left over from the previous run. Bit-for-bit reproducibility of a given seed may
   therefore depend on how runs are distributed to worker processes. Statistical results are unaffected.
5. **Boundary cases in the classification.** `compliant` uses `after >= 0` and `latecompliant` uses
   `after > 0`. An agent with before < 0, during < 0 and an after-mean of exactly 0 gets no label.
   Agents with a before-mean of exactly 0 count as positive.
6. **Edge weights are unbounded** (with `lam = 0` and `mu = 0`), as intended by the paper (see the
   discussion of eq. 3).
7. **`tau` must be an integer**, because it is used in `list(...) * tau` and `reshape((tau, M))`.
8. **The `if True:` in `run_one`** always re-runs and overwrites existing files. The commented-out
   check would skip runs whose CSV already exists.
9. **Performance.** `get_metrics` builds an igraph graph per agent for betweenness, and `get_energies`
   uses Python loops. Metric computation dominates when `detail=True`, because it then runs at every step.
