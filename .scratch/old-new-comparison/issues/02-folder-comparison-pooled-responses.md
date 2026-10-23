# 02: Folder comparison tracer bullet: pairing, cell parsing, pooled response table

**Labels:** ready-for-agent

**What to build:** The main entry point takes an old and a new output folder, plus options (cache directory, permutation seed, n_permutations), and returns a result object. It writes no files. To build that result it:
- pairs the CSVs by legacy filename;
- reports files present on one side only and carries on;
- parses each cell key (the filename tokens without the seed: link_prob, init_w, beta, rho, eps, mu, fixedBNat100, ext_strength) as the exact inverse of `legacy_filename`;
- labels a cell as a paper config (static0.2, static0.8, adaptive-static, adaptive) or by the parameters that differ (e.g. `mu=0.1`);
- reduces each run to a summary row with the response counts and `unlabelled` from the t = 95.5 row.

The result holds the pairing report, the run-summary frames for both sides, and the pooled response table: a χ² run-level permutation test per cell, BH at q = 0.05 within the family, and Cramér's V, ranked by effect size with the q-value alongside.

**Blocked by:** #01

**Status:** done

- [x] Synthetic ensemble-format CSVs in tmp folders (real legacy columns, t = 95.5 response dummies, a few seeds and cells) are the test fixture
- [x] Pairing is by filename, and files on one side only are listed in the pairing report without aborting
- [x] Filename parsing round-trips the paper configs, the μ sweep and the `fixedBNat100` token
- [x] Cells get paper-config labels or differing-parameter labels
- [x] An injected response shift in one cell flags exactly that cell in the pooled table
- [x] The table is ranked by Cramér's V, with BH q-values shown
- [x] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
