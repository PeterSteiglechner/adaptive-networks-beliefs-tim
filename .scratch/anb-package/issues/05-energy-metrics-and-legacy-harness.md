# 05: Energy metrics + legacy-state harness (seam 2, part 1)

**What to build:** A researcher can compute the energy-type metric columns on any model state, and they match legacy. This ticket builds the seam-2 harness: take a legacy state array (built through the ticket-01 helper), convert it into an `AgentState` plus adjacency, and compare each `Metric.compute(StepContext)` against legacy `fill_metrics`/`get_output` on that same state. Then it ports the energy metrics, vectorised over agents (one `einsum` over the `(N, M, M)` edges, one product with the adjacency):

- `PersonalEnergy` (`Hpers`)
- `FocalPersonalEnergy` (`Hpersfoc`)
- `SocialEnergy` (`Hsoc`), defaulting to `aggregate="mean"` as legacy does, with `"sum"` giving the eq. 1 value (known issue #1)
- `ExternalEnergy` (`Hext`)
- `NonFocalExtremity` (`extr_nonfoc`)

See the spec `docu/spec-anb-package.md`, "`metrics` module" and seam 2.

**Blocked by:** 04 (Tracer bullet: one detail-mode run end to end)

**Status:** ready-for-agent

- [ ] The harness converts a legacy state into an `AgentState` + adjacency, and is reusable by later metric tickets
- [ ] Each of the five metrics equals legacy to floating-point tolerance on several random legacy states, including states with zero and non-zero pressure
- [ ] `SocialEnergy()` reproduces legacy's mean quirk, and `SocialEnergy(aggregate="sum")` gives the neighbour sum
- [ ] An agent with no neighbours is handled the way legacy handles it
- [ ] The metrics are vectorised over agents, with no Python loop over agents
