# 09: Drop-in legacy output

**What to build:** An analysis-script user points an unchanged `analyse_*.py` script at CSVs written by the new package, and it runs. The default `Testbed` metric set is all 14 legacy metrics. `SimulationResult.to_frame()` appends the parameter columns from `spec.params()` plus the schedule info:

- uniform parameters are scalar columns, as legacy writes them
- per-agent parameters are per-agent columns of the same name, joined on `id`
- `fixedBNat100` is derived from `learning_until`

`SimulationResult.save(directory, legacy_names=True)` writes the CSV under `legacy_filename`. See the spec `docu/spec-anb-package.md`, "Output compatibility" (stories 21–27).

**Blocked by:** 03 (`legacy_filename`), 06 (Belief-network structure metrics), 07 (`FocalBetweenness` metric), 08 (Response classification, ensemble mode, `run_ensemble`)

**Status:** ready-for-agent

- [ ] The column set **and order** equal legacy's output exactly, in both ensemble and detail mode (compared against a legacy-produced frame)
- [ ] Homogeneous runs produce the scalar legacy parameter columns unchanged, so old and new CSVs concatenate cleanly
- [ ] A heterogeneous run (for example per-agent β) produces a per-agent `beta` column matching each agent's value
- [ ] `save()` writes to the legacy path and filename in both modes
- [ ] Demo: a small batch of new ensemble CSVs, enough for one analysis script to read, is generated, and that script runs unchanged on them (written up in the ticket or PR, not as a default test)
