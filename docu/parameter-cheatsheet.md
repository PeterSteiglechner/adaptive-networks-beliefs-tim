# Parameter cheatsheet

Every parameter of the ANB model in one place: what it does, which equation it belongs to, and where to set it,
in legacy `ANBmodel.py` and in the `anb/` package. The equations come from the paper ([ANB.md](ANB.md)). For the
derivations and implementation detail, see [model.md](model.md). The second half,
[Metrics](#metrics-what-is-measured-and-where-it-is-used), covers every output column and derived quantity and
links to the places in the analysis scripts that use it.

## The four equations

The other tables refer to these by number.

**Eq. 1 + 4: dissonance** (what an agent tries to minimise)

$$
D = -\underbrace{\sum_{m<n}\omega_{mn}\,x_m x_n}_{\text{personal}}
\;-\;\underbrace{\rho\sum_{k\in\mathcal K}x_{foc}\,x^{(k)}_{foc}}_{\text{social}}
\;-\;\underbrace{s\,x_{foc}}_{\text{external (eq. 4)}}
$$

The dynamics never compute $D$ itself. They use the local field of the belief being updated, which is the
part of $D$ that depends on $x_d$: $\sum_{j\ne d}\omega_{dj}x_j$, plus $s + \rho\sum_k x^{(k)}_{foc}$ if $d$ is
focal. The metrics compute the separate dissonance terms.

| Term | Legacy `ANBmodel.py` | `anb` |
|------|----------------------|-------|
| personal (local field) | [`glauber_fast`](../ANBmodel.py#L150), [L154–155](../ANBmodel.py#L154) | [`_sweep`](../anb/agents.py#L177), [L177–179](../anb/agents.py#L177) |
| social sum $\sum_k x^{(k)}_{foc}$ | [`run_simulation`, L441–444](../ANBmodel.py#L441) | [`_sweep`, L164–172](../anb/agents.py#L164) |
| social + external added to the focal field | [L156–157](../ANBmodel.py#L156), with zeros passed for non-focal beliefs in [`update_belief`, L183–188](../ANBmodel.py#L183) | [L180–181](../anb/agents.py#L180) |
| $s$ on or off at step $t$ (eq. 4) | [L438](../ANBmodel.py#L438) (`ext_time`) | [`PressureSchedule.strength_at`](../anb/testbed.py#L26), called in [`Testbed.run`](../anb/testbed.py#L166) |
| metric `Hpers` | [`get_energies`, L198](../ANBmodel.py#L198) | [`PersonalEnergy`](../anb/metrics.py#L64) |
| metric `Hpersfoc` | [`get_energies`, L213](../ANBmodel.py#L213) | [`FocalPersonalEnergy`](../anb/metrics.py#L74) |
| metric `Hsoc` (mean, not sum) | [`get_energies`, L229](../ANBmodel.py#L229) | [`SocialEnergy`](../anb/metrics.py#L84) (`aggregate="sum"` gives the eq. 1 term) |
| metric `Hext` | [`get_energies`, L221](../ANBmodel.py#L221) | [`ExternalEnergy`](../anb/metrics.py#L108) |

**Eq. 2: belief update** (Glauber, one belief at a time, over the 21 values in $\mathcal X$)

$$
p(x_m \to x') \propto \frac{1}{1+e^{\beta\,[D(x') - D(x_m)]}}
$$

| Step | Legacy `ANBmodel.py` | `anb` |
|------|----------------------|-------|
| $\Delta D$ for all 21 candidates | [`glauber_fast`, L154](../ANBmodel.py#L154) | [`_sweep`, L184](../anb/agents.py#L184) |
| Glauber probability with $\beta$ | [L159–160](../ANBmodel.py#L159) | [L185](../anb/agents.py#L185) (tanh form, so it can't overflow) |
| sampling and write-back | [`update_belief`, L189](../ANBmodel.py#L189) (`np.random.choice`) | [L187–191](../anb/agents.py#L187) (inverse CDF) |
| random belief order | [`update_belief`, L179](../ANBmodel.py#L179) | [L176](../anb/agents.py#L176) |
| random agent order | [`run_simulation`, L439](../ANBmodel.py#L439) | [L160](../anb/agents.py#L160) |
| `fixed` agents skip everything | – | [L161](../anb/agents.py#L161) |

**Eq. 3: belief-network (edge) update**

$$
\omega_{mn} \leftarrow \omega_{mn}
+ \underbrace{\epsilon\,\partial x_m\,\partial x_n}_{\text{internal (Hebbian)}}
+ \underbrace{\mu\,\big(\bar\omega^{(\mathcal K)}_{mn} - \omega_{mn}\big)}_{\text{social}}
- \underbrace{\lambda\,\omega_{mn}}_{\text{decay, not in paper}},
\qquad
\partial x_m = \frac1\tau\sum_{i=0}^{\tau-1}\Delta x_m(t-i)
$$

| Step | Legacy `ANBmodel.py` | `anb` |
|------|----------------------|-------|
| store $\Delta x$ in the $\tau$-step memory | [`update_belief`, L191–193](../ANBmodel.py#L191) | [`_sweep`, L192–196](../anb/agents.py#L192) |
| $\partial x_m$, the mean over $\tau$ | [`update_edge_weights`, L166](../ANBmodel.py#L166) | [L199–203](../anb/agents.py#L199) |
| neighbours' mean BN $\bar\omega^{(\mathcal K)}$ | [`run_simulation`, L458–462](../ANBmodel.py#L458) | [L204–212](../anb/agents.py#L204) |
| the update ($\epsilon$, $\mu$, $\lambda$ terms) | [`update_edge_weights`, L163–173](../ANBmodel.py#L163) | [L213–221](../anb/agents.py#L213) |
| $\mu = 0$ without neighbours | [L451](../ANBmodel.py#L451) | [L219](../anb/agents.py#L219) |
| freeze gate (`fixedBNat100` / `learning_until`) | [L448–457](../ANBmodel.py#L448) | [`ANBAgent.sweep`, L136](../anb/agents.py#L136), [L198](../anb/agents.py#L198) |

**Initial conditions and the social network** (no equation number in the paper)

| What | Legacy `ANBmodel.py` | `anb` |
|------|----------------------|-------|
| beliefs drawn from $\mathcal X$, edges set to $\omega_0$ | [`initialise_agents`](../ANBmodel.py#L117) | [`ANBAgent.init_state`](../anb/agents.py#L122) |
| Erdős–Rényi network with $p$ | [`initialise_network`](../ANBmodel.py#L134) | [`ErdosRenyiNetwork.build`](../anb/networks.py#L38) |
| one time step | [`run_simulation`, L437–469](../ANBmodel.py#L437) | [`ANBModel.step`](../anb/model.py#L21) → [`ANBAgent.sweep`](../anb/agents.py#L134) |

> The line numbers are from 2026-10-17. `ANBmodel.py` is frozen, so its links should stay valid. The `anb`
> links move whenever the package is edited. If one points to the wrong line, search for the function name.

## Quick lookup

| Symbol | Legacy name | `anb` name / where | Equation | Baseline | One-liner |
|--------|-------------|--------------------|----------|----------|-----------|
| $\beta$ | `beta` | `ANBAgent(beta=)` | 2 | `3.0` | How strongly agents avoid dissonance (inverse temperature) |
| $\rho$ | `rho` | `ModelSpec(rho=)` | 1 | `1/3` | Weight of each social link on the focal belief |
| $s$ | `ext_strength` | `PressureSchedule(strength)` | 4 | `4` (swept 0–16) | Strength of the external push of $x_{foc}$ towards +1 |
| $\epsilon$ | `eps` | `ANBAgent(eps=)` | 3 | `1.0` | Hebbian learning rate. `0` gives a static BN |
| $\mu$ | `mu` | `ANBAgent(mu=)` | 3 | `0.0` | Rate of copying the neighbours' mean BN |
| $\lambda$ | `lam` (module const) | `ANBAgent(lam=)` | 3 | `0.0` | Edge-weight decay towards 0. Not in the paper |
| $\tau$ | `tau` (module const) | `ANBAgent(tau=)` | 3 ($\partial x$) | `1` | Activation memory: steps averaged into $\partial x_m$. Must be an int |
| $\omega_0$ | `init_w` | `ANBAgent(init_w=)` | initial value of $\omega$ | `0.2` | Starting weight of all 45 BN edges |
| $p$ | `link_prob` | `ErdosRenyiNetwork`, `network_kwargs={"link_prob": …}` | defines $\mathcal K$ (1, 3) | `0.1` | Erdős–Rényi link probability, ≈ $pN$ = 10 contacts |
| $N$ | `n_agents` (module const) | `ModelSpec(n_agents=)` | size of the population | `100` | Number of agents |
| $M$ | `M` (module const) | `ANBAgent(n_beliefs=)` | number of $x_m$ terms | `10` | Beliefs per agent, incl. the focal one |
| $\mathcal X$ | `belief_options` | `ANBAgent(belief_options=)` | 2 (candidate values) | `linspace(-1,1,21)` | Belief space |
| – | `fixedBNat100` | `ANBAgent(learning_until=101)` | switches eq. 3 off | `False` | Adaptive→static: the BN freezes when the pressure starts |
| – | – | `ANBAgent(fixed=mask)` | skips eq. 2 and 3 | `None` | Per-agent mask of agents that never update (new in `anb`) |
| – | `seed` | `Testbed.run(seed)` | – | `0…99` | Seeds the initial beliefs, the network and the dynamics |

In `anb`, `beta`, `eps`, `mu`, `lam` and `init_w` can be a scalar or an `(N,)` array, one value per agent.

## Per parameter

### Belief dynamics (eq. 1, 2, 4)

**$\beta$, `beta`: attention to dissonance.** This is the inverse temperature in eq. 2. With high $\beta$, agents
almost always choose the value that lowers $D$ the most, so the dynamics are near-deterministic and close to
greedy. With low $\beta$, the choice is close to uniform random over $\mathcal X$, so beliefs are noisy. $\beta$
multiplies every dissonance term at once, so it sets the scale against which $\omega$, $\rho$ and $s$ are measured.

**$\rho$, `rho`: social coupling.** Only the focal belief has a social term. Each neighbour $k$ contributes
$\rho\,x^{(k)}_{foc}$ to the local field of $x_{foc}$. It is a **sum**, so an agent with more contacts feels more
social pull. The recorded metric `Hsoc` uses the neighbours' **mean** instead
([known-issues.md](known-issues.md)). Keep this in mind when you compare `Hsoc` across degrees.

**$s$, `ext_strength`: external pressure.** This term is active only in pressure steps ($t=101$–$150$, plus
$201$–$250$ with two events). It adds $+s$ to the local field of $x_{foc}$ and pushes it towards +1. You can
think of it as an extra BN node fixed at +1 with edge weight $s$ to the focal belief. It is the main x-axis of
the response plots (compliant / resilient / resistant).

**$\mathcal X$, `belief_options`.** These are the 21 candidate values Glauber samples from. Changing the
resolution changes how noisy individual steps are.

### Belief-network dynamics (eq. 3)

**$\epsilon$, `eps`: internal (Hebbian) adaptation.** Beliefs that change in the same direction in the same step
strengthen their edge, and beliefs that change in opposite directions weaken it ("fire together, wire
together"). This term is what makes BNs heterogeneous across agents.
- `eps = 0`: a static BN, and every agent keeps $\omega_0$ forever.
- `eps = 1`: the paper's adaptive case.

**$\mu$, `mu`: social adaptation.** The edges relax towards the neighbours' average edges
$\bar\omega^{(\mathcal K)}_{mn}$, which makes BNs within a neighbourhood more alike. Agents without neighbours
get $\mu = 0$. In the paper, weak $\mu$ raises compliance and strong $\mu$ raises resilience (Fig. 6). In `anb`,
$\bar\omega$ is weighted by the adjacency entries and divided by $\sum_k|A_{ik}|$, so it also works for signed
networks.

**$\lambda$, `lam`: decay.** This term shrinks every edge by a fraction $\lambda$ per step. It keeps edge weights
bounded, which the paper deliberately doesn't do. Its default is 0. In legacy it is a module constant read from
`params_fixed["lam"]`, and it appears in the filename as `_lambda{λ:.4f}` only when it is non-zero.

**$\tau$, `tau`: activation memory.** $\partial x_m$ is the average belief change over the last $\tau$ steps. With
$\tau = 1$, only the last step counts. A larger $\tau$ makes the Hebbian term react to slower co-movement and
filters out one-step noise.

**`fixedBNat100` / `learning_until`: freeze the BN.** This gives the paper's adaptive→static condition: the BN
adapts during warm-up and is frozen during and after the pressure.
- Legacy zeroes $\epsilon$ and $\mu$ for $t > 101$, so it still adapts once at $t = 101$, and $\lambda$ keeps
  running.
- `anb` with `learning_until=101` skips the whole eq. 3, $\lambda$ included, from $t = 101$ on.
  `learning_until=102` reproduces legacy's timing. This is a deliberate difference that still needs sign-off,
  see [validation.md](validation.md).

### Initial conditions and structure

**$\omega_0$, `init_w`: initial edge weight.** All 45 edges start at this value. With `eps = 0`, it is the BN
for the whole run, so it decides the static cases: low $\omega_0$ gives full compliance, high $\omega_0$ gives
resistance or resilience. With `eps > 0`, it is only the starting point.

**$p$, `link_prob`: social network density.** This sets the neighbour set $\mathcal K$, which enters the social
term of eq. 1 and the $\bar\omega$ of eq. 3. The network is drawn once per seed and stays fixed. In `anb`, any
`Network` subclass can replace Erdős–Rényi, and its own kwargs then take the place of `link_prob`. That is the
extension point of this project.

**$N$, `n_agents`, and $M$, `M` / `n_beliefs`.** These are the population size and the number of beliefs.
Belief 0 is the focal one. The BN has $M(M-1)/2$ edges. In legacy, both are module constants, and they are
**not** in the filename (see [running-simulations.md](running-simulations.md#sensitivity-runs)).

## Protocol (not model) parameters

These set *when* things happen and what is measured. None of them is in an equation.

| Legacy | `anb` | Default | Meaning |
|--------|-------|---------|---------|
| `ext_time` | `PressureSchedule.events` | `[(101, 150)]` | Pressure steps, inclusive |
| `two_external_events` | `PressureSchedule.two_events(s)` | `False` | Adds a second event at 201–250, $T = 300$ |
| `T` | `TestbedConfig.T` = last event end + `recovery` | `200` | Last time step |
| – | `TestbedConfig.recovery` | `50` | Steps after the event before the "after" window ends |
| `beforeRange` / `duringRange` / `afterRange` | `TestbedConfig.window_len` + `windows()` | 91–100 / 141–150 / 191–200 | Averaging windows for the response classification |
| `detail` (`__main__` only) | `TestbedConfig(detail=)` | `False` | Save every step instead of the ensemble rows |
| `track_times` (`__main__` only) | `TestbedConfig.track_times()` | windows + snapshots | Steps at which metrics are computed |

## Which knob for which question

| If you want to change… | Turn |
|------------------------|------|
| how noisy or greedy belief choices are | $\beta$ |
| how much neighbours matter for the focal belief | $\rho$, $p$ (degree), or the network type |
| how hard the outside world pushes | $s$ |
| whether BNs learn, and how fast | $\epsilon$, $\tau$ |
| whether BNs become alike across neighbours | $\mu$ (and the network) |
| whether edge weights stay bounded | $\lambda$ |
| how tightly beliefs are coupled from the start | $\omega_0$ |
| whether the BN can still rewire under pressure | `fixedBNat100` / `learning_until` |

## Metrics: what is measured and where it is used

The model writes 14 metric columns and 6 response dummies per agent. The analysis scripts derive a few more
quantities from them. This section explains each one and links to where it matters in the analysis. For the
formulas next to the paper's Table 2, see also [model.md §8](model.md#8-metrics-get_energies-get_metrics-fill_metrics)
and [§9](model.md#9-response-classification-get_output). For which figure each script makes, see
[analysis.md](analysis.md).

**When metrics are computed.** Only at the tracked steps (`track_times`, legacy
[L468](../ANBmodel.py#L468), `TestbedConfig.track_times()` in `anb`). In ensemble files, the rows `t = 95.5`,
`145.5` and `195.5` hold the **window means** over t = 91–100, 141–150 and 191–200, so count metrics such as
`tb_foc` are not integers there. The rows `t = 0`, `100` and `200` are snapshots. Almost all analysis uses the
"before" row `t = 95.5`: the paper asks which pre-pressure properties predict the response.

### Per-agent metric columns

Column order as in legacy `metric_cols` ([L83](../ANBmodel.py#L83)) and `anb` `default_metrics()`
([metrics.py L314](../anb/metrics.py#L314)). The label is the one the analysis scripts print
(`metric2titleVerb` / `metric2title`, [analyse_BNmetrics.py L41–72](../analyse_BNmetrics.py#L41)).

| Column | Label in figures/tables | Definition | Range (M = 10) | Legacy | `anb` |
|--------|-------------------------|------------|----------------|--------|-------|
| `n_nbs` | nr social contacts, $\lvert\mathcal K\rvert$ | number of neighbours in the social network | 0…N−1, ≈ 10 | [`fill_metrics` L301](../ANBmodel.py#L301) | [`Degree`](../anb/metrics.py#L40) |
| `x_focal` | focal belief, $x_{foc}$ | the focal belief itself | −1…1 | [L315](../ANBmodel.py#L315) | [`FocalBelief`](../anb/metrics.py#L47) |
| `extr_nonfoc` | extremity non-focal beliefs, $\lvert X_{non\text{-}foc}\rvert$ | $\frac1{M-1}\sum_{m\ne foc}\lvert x_m\rvert$ | 0…1 | [L316](../ANBmodel.py#L316) | [`NonFocalExtremity`](../anb/metrics.py#L205) |
| `absOm_tot` | connectedness BN, BN-$\lvert\Omega\rvert$ | mean $\lvert\omega_{mn}\rvert$ over all 45 edges | ≥ 0, unbounded | [`get_metrics` L258](../ANBmodel.py#L258) | [`MeanAbsEdge`](../anb/metrics.py#L261) |
| `absOm_foc` | connectedness focal BN, BN-$\lvert\Omega_{foc}\rvert$ | mean $\lvert\omega_{foc,m}\rvert$ over the 9 focal edges | ≥ 0 | [L259](../ANBmodel.py#L259) | [`FocalMeanAbsEdge`](../anb/metrics.py#L270) |
| `tb_tot` | balance BN, BN-$\alpha$ | number of balanced triangles, $\omega_{ab}\omega_{ac}\omega_{bc}>0$ | 0…120 | [L245](../ANBmodel.py#L245) | [`TriangleBalance`](../anb/metrics.py#L239) |
| `tb_foc` | balance focal BN, BN-$\alpha_{foc}$ | balanced triangles that contain the focal belief | 0…36 | [L245](../ANBmodel.py#L245) | [`FocalTriangleBalance`](../anb/metrics.py#L249) |
| `clust` | clustering BN, BN-clust | Onnela weighted clustering on $\lvert\omega\rvert$ $\max\lvert\omega\rvert$, mean over nodes | 0…1 | [L262–276](../ANBmodel.py#L262) | [`WeightedClustering`](../anb/metrics.py#L280) |
| `bc_foc` | centrality focal BN, BN-centr$_{foc}$ | unnormalised betweenness of the focal node, path length $1/(\lvert\omega\rvert+10^{-6})$ | 0…36 | [L281](../ANBmodel.py#L281) | [`FocalBetweenness`](../anb/metrics.py#L117) |
| `Hpers` | – (only via `Hpersnonfoc`) | personal dissonance, $-\sum_{m<n}\omega_{mn}x_mx_n$ | any sign | [`get_energies` L200](../ANBmodel.py#L200) | [`PersonalEnergy`](../anb/metrics.py#L64) |
| `Hpersfoc` | dissonance focal BN, $D_\mathrm{BN\text{-}foc}$ | $-x_{foc}\sum_{m\ne foc}\omega_{foc,m}x_m$ | any sign | [L214](../ANBmodel.py#L214) | [`FocalPersonalEnergy`](../anb/metrics.py#L74) |
| `Hsoc` | dissonance focal social, $D_{social}$ | $-\rho\,x_{foc}\cdot\mathrm{mean}_k x^{(k)}_{foc}$, 0 without neighbours | $-\rho…\rho$ | [L230](../ANBmodel.py#L230) | [`SocialEnergy`](../anb/metrics.py#L84) |
| `Hext` | – (not analysed) | $-s\,x_{foc}$ in pressure steps, else 0 | $-s…s$ | [L222](../ANBmodel.py#L222) | [`ExternalEnergy`](../anb/metrics.py#L108) |
| `expI` | – (not analysed) | expected influence, $\sum_m\omega_{foc,m}x_m$ (the BN part of the focal local field) | any sign | [L278](../ANBmodel.py#L278) | [`ExpectedInfluence`](../anb/metrics.py#L305) |

For all dissonance columns, **lower (more negative) means less dissonance**, i.e. a more coherent state.

### What each metric tells you, and where it matters

The analysis scripts use the metrics in a few fixed sets. They are referred to by name below:

- **Table set** `metrics_table_sort` ([analyse_BNmetrics.py L123](../analyse_BNmetrics.py#L123)): the 12 rows of
  Table 3 / A1–A5, the panels of Fig. 5, and the axes of the correlation heatmap Fig. A4.
- **Effect-size set** `selected_metricsl_cols` ([L107](../analyse_BNmetrics.py#L107)): the same 12 metrics, used
  for Cohen's d in Fig. 4.
- **Regression set** `dependent_vars` + `control_vars` ([L448–459](../analyse_BNmetrics.py#L448)): the logit
  predictors `n_nbs, tb_foc, absOm_foc, clust, bc_foc` with control `x_focal`. Despite the name, these are the
  predictors, not the dependent variable.

Usage links, by place in the analysis:

| Where | File and line |
|-------|---------------|
| Table 3 / A1–A4 (by response, before window) | [analyse_BNmetrics.py L217](../analyse_BNmetrics.py#L217) |
| Table A5 (before/during/after, $s = 4$) | [L267](../analyse_BNmetrics.py#L267) |
| Fig. 5 (metric change over time) | [L323](../analyse_BNmetrics.py#L323) |
| Fig. 4 Cohen's d | [L429–437](../analyse_BNmetrics.py#L429) |
| Fig. 4 logit coefficients, Tables A6/A7 | [L448–512](../analyse_BNmetrics.py#L448) |
| Fig. A4 correlation heatmap | [L811](../analyse_BNmetrics.py#L811) |
| greedy uncorrelated subset (predictor choice) | [L849](../analyse_BNmetrics.py#L849) |
| Fig. 6C (metrics vs. $\mu$) | [analyse_socialAdaptation.py L385](../analyse_socialAdaptation.py#L385) |

**Beliefs**

- **`x_focal`** is the quantity the pressure acts on. All responses are defined from its window means. In the
  regression it is the **control variable**: a more negative focal belief needs a larger push to cross 0, so
  any BN effect must be shown on top of it. It is in the table set and the greedy subset. Fig. 4 shows its Cohen's d bar but leaves it out of the
  logit overlay, because it is the control ([L667–670](../analyse_BNmetrics.py#L667)).
- **`extr_nonfoc`** measures how settled the rest of the belief system is. Extreme non-focal beliefs feed a
  strong, stable field into the focal belief through the BN. It is in the table, effect-size and greedy sets,
  but not a regression predictor (commented out at [L457](../analyse_BNmetrics.py#L457)).

**Social embedding**

- **`n_nbs`** is the size of $\mathcal K$. The social field on $x_{foc}$ is a **sum** over neighbours, so more
  contacts mean a stronger pull towards the neighbourhood's focal beliefs. It is constant over
  time, because the network is fixed. It is a regression predictor and is in all sets. This is the metric that
  changing the network type moves most directly, so it is the main link from this project's network extension
  to the response analysis.
- **`Hsoc`** is the agreement between the focal belief and the neighbours' **mean** focal belief. Negative means
  the agent agrees with its neighbourhood. It is in the table and effect-size sets, not in the regression. Because
  it uses the mean, it doesn't scale with `n_nbs` the way the dynamics do ([known-issues.md #1](known-issues.md)).

**Belief-network structure (whole BN)**

- **`absOm_tot`** is how tightly the beliefs are coupled overall. With `eps = 0` it stays at $\omega_0$ and
  separates static-low from static-high. Beyond the table set, it is used in
  [analyse_responseFrequencies.py L452](../analyse_responseFrequencies.py#L452) (mean $\lvert\omega\rvert$ per BN
  configuration, quoted in the paper text). It is not a regression predictor, because it correlates strongly with
  `absOm_foc` (Fig. A4).
- **`tb_tot`** counts the triangles that can be satisfied all at once. A balanced BN has a coherent "camp"
  structure and therefore deep, stable dissonance minima. It is in the table set and printed for the two example
  agents of Fig. 2 ([analyse_timeseries.py L394](../analyse_timeseries.py#L394)). It is excluded from the regression
  because it correlates with `tb_foc`.
- **`clust`** measures how strongly the neighbours of a belief are tied to each other. It is a regression
  predictor and is plotted vs. $\mu$ in Fig. 6C. For an all-zero BN it is NaN.

**Embedding of the focal belief**

- **`absOm_foc`** is how strongly the focal belief is anchored by the other beliefs. Strong focal edges oppose
  any push that would create dissonance with the rest of the BN. It is a regression predictor and is in Fig. 6C.
- **`tb_foc`** is the balance of the focal belief's own triangles. If they are balanced, flipping $x_{foc}$
  breaks many triangles at once, so the BN pulls it back after the pressure (resilience). It is a regression
  predictor, is in Fig. 6C, is the quick-look plot vs. $\mu$ at
  [analyse_socialAdaptation.py L587](../analyse_socialAdaptation.py#L587), and is printed in Fig. 2
  ([analyse_timeseries.py L395](../analyse_timeseries.py#L395)).
- **`bc_foc`** says whether the focal belief lies on the strong paths between other beliefs, i.e. whether it
  is a hub that relays influence in the BN. It is a regression predictor.

**Dissonance**

- **`Hpers`** is not analysed directly. Both scripts replace it by `Hpersnonfoc = Hpers − Hpersfoc`
  ([analyse_BNmetrics.py L194](../analyse_BNmetrics.py#L194),
  [analyse_socialAdaptation.py L245](../analyse_socialAdaptation.py#L245)) and drop it. It is only printed for
  the Fig. 2 agents ([analyse_timeseries.py L438](../analyse_timeseries.py#L438)).
- **`Hpersfoc`** is how well the focal belief fits the rest of the BN. Negative means it fits. A misaligned
  agent (before < 0) whose BN supports the negative value has low `Hpersfoc`, so it has much to lose by
  complying. It is in the table and effect-size sets.
- **`Hpersnonfoc`** (derived) is the dissonance among the non-focal beliefs only. It is in the table and
  effect-size sets.

**Collected but not used in any analysis**

- **`Hext`** is 0 in the before window by definition, so it says nothing about pre-pressure differences. It is
  only loaded ([analyse_BNmetrics.py L95](../analyse_BNmetrics.py#L95)).
- **`expI`** is the BN part of the focal local field, which overlaps with `Hpersfoc` ($\texttt{Hpersfoc} = -x_{foc}\cdot\texttt{expI}$).
  It is only loaded ([L103](../analyse_BNmetrics.py#L103)).

All 14 columns are compared between legacy and `anb` in the validation ([anb/comparison.py L48](../anb/comparison.py#L48),
[validation.md](validation.md)), so keep them even though two are unused.

### Response dummies

0/1 columns, stored only on the `t = 95.5` row of ensemble files (on the last step in detailed files). They are
computed from the focal belief's means over the before (b), during (d) and after (a) windows. Legacy computes
them in [`get_output` L403–413](../ANBmodel.py#L403), `anb` in [`ResponseClassifier`](../anb/metrics.py#L326).

| Column | Condition | Meaning |
|--------|-----------|---------|
| `persistPos` | b ≥ 0, d ≥ 0, a ≥ 0 | already aligned, stays aligned |
| `nonpersistPos` | b ≥ 0, a < 0 | already aligned, ends negative |
| `compliant` | b < 0, d ≥ 0, a ≥ 0 | **compliance**: lasting change |
| `resilient` | b < 0, d ≥ 0, a < 0 | **resilience**: changes during the pressure, reverts afterwards |
| `resistant` | b < 0, d < 0, a < 0 | **resistance**: never changes |
| `latecompliant` | b < 0, d < 0, a > 0 | changes only after the pressure (rare) |

Only the misaligned agents (b < 0) are analysed. The boundary cases at exactly 0 are listed in
[known-issues.md #9](known-issues.md).

Where they matter:

- **Relative response frequency** = count ÷ (compliant + resilient + resistant [+ latecompliant]) per simulation.
  This is the y-axis of Fig. 3B–E ([analyse_responseFrequencies.py L158](../analyse_responseFrequencies.py#L158)),
  Fig. 6D ([analyse_socialAdaptation.py L231](../analyse_socialAdaptation.py#L231), without `latecompliant` in the
  denominator) and Fig. A8 ([analyse_sensitivity.py L161](../analyse_sensitivity.py#L161)).
- **Response label per agent**: [analyse_BNmetrics.py L176–189](../analyse_BNmetrics.py#L176) copies the
  `t = 95.5` label to the during and after rows. It groups Tables 3/A1–A5 and Fig. 5, and defines the two
  contrasts of Fig. 4: `rR-C` (resistant + resilient vs. compliant) and `R-r` (resistant vs. resilient)
  ([L462–475](../analyse_BNmetrics.py#L462)). `latecompliant` counts as compliant in `rR-C`.
- `persistPos` and `nonpersistPos` are loaded everywhere but not plotted.

### Derived population-level quantities

These are computed in the analysis scripts per simulation, mostly from the edge columns at the `t = 100`
snapshot. They are not saved by the model.

| Name | Definition | Where | Used for |
|------|------------|-------|----------|
| `Hpersnonfoc` | `Hpers − Hpersfoc`, per agent | [analyse_BNmetrics.py L194](../analyse_BNmetrics.py#L194), [analyse_socialAdaptation.py L245](../analyse_socialAdaptation.py#L245) | tables, Fig. 4, Fig. 5, Fig. A4 |
| BN heterogeneity (`mean`, z-scored as `meanR`) | mean pairwise Euclidean distance between the agents' 45-edge vectors | [analyse_socialAdaptation.py L187–190](../analyse_socialAdaptation.py#L187), [L369](../analyse_socialAdaptation.py#L369) | Fig. 6C: BNs become alike as $\mu$ grows |
| `groupishness` (social adaptation) | var / mean of those Euclidean distances, 0 if `eps = 0` | [analyse_socialAdaptation.py L188](../analyse_socialAdaptation.py#L188) | stored in `relres`, not plotted |
| `groupishness` (response frequencies) | std / mean of the pairwise **Manhattan** distances, 0 if `eps = 0` | [analyse_responseFrequencies.py L72–73](../analyse_responseFrequencies.py#L72) | computed, not plotted. The two scripts define it differently. |
| `var`, `kurtosis` | std and negative kurtosis of the Euclidean distances | [analyse_socialAdaptation.py L189–191](../analyse_socialAdaptation.py#L189) | stored, not plotted (`var` is actually the std) |
| `nr_negative_edges` | number of edges < 0 across all agents | [analyse_socialAdaptation.py L186](../analyse_socialAdaptation.py#L186) | stored, not plotted |
| `mean_abs_edges` / `mean_absedges` | mean $\lvert\omega\rvert$ (and mean $\omega$) over all agents and edges | [analyse_responseFrequencies.py L96](../analyse_responseFrequencies.py#L96), [analyse_socialAdaptation.py L215](../analyse_socialAdaptation.py#L215) | bar plot per BN config and the summary at [L452](../analyse_responseFrequencies.py#L452) |
| `std_focal` | std of the focal beliefs across agents at `t = 100` | [analyse_responseFrequencies.py L74](../analyse_responseFrequencies.py#L74) | polarisation $\sigma_{foc}$ in the paper text, summary at [L448](../analyse_responseFrequencies.py#L448) |
| `nr_negs` | agents with a before-mean < 0, i.e. the misaligned agents | [analyse_responseFrequencies.py L75](../analyse_responseFrequencies.py#L75) | stored, not plotted |
