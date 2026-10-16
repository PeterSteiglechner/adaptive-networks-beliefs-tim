# 11: Statistical validation, full benchmark, docs

**What to build:** The supervisor can trust results from the new package and sign off on its deliberate differences. This ticket covers four things:

- **Statistical validation (on demand, excluded from the default test run).** Run the 4 paper configurations × 6 pressures with ≥ 100 seeds each, new vs `ANBmodel.py`. Compare response frequencies per cell with a two-proportion or χ² test, and the BN metric distributions per window with a two-sample KS test or a comparison of means with CIs. Report the failures after multiple-comparison correction.
- **Full benchmark.** Time the full 240-run ensemble, new vs legacy, with metrics profiled separately. If the metrics, mainly `FocalBetweenness`, now dominate, replace its per-agent igraph builds with a private `@njit` Brandes helper and check it against igraph betweenness on random edge matrices.
- **Difference write-up.** Document the different RNG stream and the single `fixedBNat100` meaning (known issues #2, #8) for sign-off.
- **Docs.** Update `docu/README.md`, CLAUDE.md's "planned, not yet implemented" note and the ADR action-item checkboxes.

See the spec `docu/spec-anb-package.md`, "Testing Decisions" seam 1 and "Further Notes".

**Blocked by:** 10 (Numba sweep kernel)

**Status:** done (2026-10-16). Results in `docu/validation.md`; supervisor sign-off still open

- [x] The validation script/test runs on demand and produces a report: per-cell test statistics, corrected p-values and a list of failures
- [x] Validation results are summarised in `docu/`, with any significant deviations explained or turned into a follow-up ticket
- [x] The 240-run benchmark is reported new vs legacy, with metric time shown separately
- [x] If the Brandes swap is made, it matches igraph on random edge matrices and the `Metric` interface is unchanged. If it isn't made, the profiling numbers that justify skipping it are recorded
- [x] The deliberate differences from legacy are documented for supervisor sign-off
- [x] The README, CLAUDE.md and ADR checkboxes reflect the implemented package
