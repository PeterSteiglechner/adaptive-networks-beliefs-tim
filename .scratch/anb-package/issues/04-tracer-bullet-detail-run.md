# 04: Tracer bullet: one detail-mode run end to end

**What to build:** A researcher builds a `ModelSpec` (`ANBAgent`, `ErdosRenyiNetwork` + `link_prob`, `n_agents`, ρ) and a detail-mode `TestbedConfig`, calls `Testbed.run(seed)`, and gets a `SimulationResult` whose frame has `t`, `id`, the belief columns, the edge columns and two metrics (`n_nbs`, `x_focal`) for every time step. This is the thinnest path through every layer:

- `AgentState`: beliefs, symmetric `(N, M, M)` edge weights, `delta_memory`, per-agent `focal`, `edge_vector()` in legacy `w{m}{n}` order
- the `Agent` ABC with scalar-or-`(N,)` parameters, `fixed` and `learning_until`
- `ANBAgent`'s random-sequential sweep in plain Python (no `@njit` yet)
- the `Network` ABC with `neighbours`/`degree`
- `ModelSpec.build`, which validates shapes, and `ANBModel.step`
- `StepContext`, the `Metric` ABC and a minimal `MetricsRecorder`
- the time loop in `Testbed.run` with one `default_rng(seed)` passed everywhere

The sweep semantics are fixed in the spec (`docu/spec-anb-package.md`, "`agents` module" steps 1–5): ρ times the **sum** of the neighbours' focal beliefs, Glauber over the 21-point grid one dimension at a time in random order, and eq. 3 with the neighbours' mean edge matrix (μ = 0 with no neighbours) skipped entirely once `t ≥ learning_until`. There's no global RNG anywhere.

**Blocked by:** 02 (Pressure protocol)

**Status:** done

- [x] `Testbed.run(seed)` returns a detail-mode frame with `t`, `id`, `n_nbs`, `x_focal`, `0`…`9` and `w01`…`w89` for every t in 0…T
- [x] Beliefs always stay on the 21-point grid, and edge matrices stay symmetric with a zero diagonal
- [x] Agents marked `fixed` never change beliefs or edges
- [x] Edge weights stop changing from `learning_until` on (λ decay included)
- [x] The same spec, config and seed give an identical frame. Different seeds give different frames
- [x] Mean degree ≈ p(N − 1) over several seeds
- [x] `ModelSpec.build` rejects per-agent parameters whose length isn't `n_agents`, and agent/network sizes that disagree
- [x] `ModelSpec` round-trips through `pickle`
- [x] The package imports and runs from a plain module, with nothing read from `__main__`
