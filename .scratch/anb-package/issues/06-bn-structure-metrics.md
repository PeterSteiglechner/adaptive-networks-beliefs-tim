# 06: Belief-network structure metrics (seam 2, part 2)

**What to build:** A researcher gets the belief-network structure columns, and they match legacy on the same state. This ticket ports:

- `TriangleBalance` (`tb_tot`) and `FocalTriangleBalance` (`tb_foc`), sharing one private triangle pass
- `MeanAbsEdge` (`absOm_tot`) and `FocalMeanAbsEdge` (`absOm_foc`)
- `WeightedClustering` (`clust`)
- `ExpectedInfluence` (`expI`)

Shared helpers stay module-level, outside the class hierarchy. `FocalBetweenness` is ticket 07. See the spec `docu/spec-anb-package.md`, "`metrics` module".

**Blocked by:** 05 (Energy metrics + legacy-state harness)

**Status:** done

- [x] Each of the six metrics equals legacy `fill_metrics`/`get_output` to floating-point tolerance on several random legacy states, through the ticket-05 harness
- [x] Both triangle-balance metrics come from one shared triangle computation
- [x] The focal variants respect each agent's per-agent `focal` (default 0)
- [x] Edge cases legacy handles specially (for example all-zero edge weights) give the legacy result
