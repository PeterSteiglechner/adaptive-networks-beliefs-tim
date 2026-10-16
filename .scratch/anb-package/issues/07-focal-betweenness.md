# 07: `FocalBetweenness` metric

**What to build:** A researcher gets the `bc_foc` column: the betweenness of each agent's focal belief inside its own belief network, matching legacy. It starts on `igraph`, as legacy does, with one graph built per agent at tracked steps only. It's its own ticket because its weighting and edge-length conventions have to match legacy exactly, and because ticket 11 may later swap its internals for an `@njit` Brandes helper without changing the `Metric` interface. See the spec `docu/spec-anb-package.md`, "`metrics` module".

**Blocked by:** 05 (Energy metrics + legacy-state harness)

**Status:** ready-for-agent

- [ ] `FocalBetweenness.compute` equals legacy `bc_foc` to floating-point tolerance on several random legacy states, through the ticket-05 harness
- [ ] Uses the same edge-weight to distance transform, normalisation and treatment of zero/negative edges as legacy
- [ ] Respects each agent's per-agent `focal`
- [ ] Wall time per tracked step is recorded in the test output or a note, as the baseline for ticket 11's profiling
