# Parameter cheatsheet

Every parameter of the ANB model in one place: what it does, which equation it belongs to, and where to set it,
in legacy `ANBmodel.py` and in the `anb/` package. The equations come from the paper ([ANB.md](ANB.md)). For the
derivations and implementation detail, see [model.md](model.md).

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
