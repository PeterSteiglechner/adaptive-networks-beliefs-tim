# 01: Move the validation statistics into an `anb` comparison module

**Labels:** ready-for-agent

**What to build:** A pure prefactor. The run summariser, the χ² permutation test, KS/Welch, Benjamini–Hochberg correction and the table builders move out of the slow-validation helper and into a new comparison module in the `anb` package. They stay plain Python, with no Numba. The slow-validation helper imports them from there, and behaviour doesn't change. This makes the later slices "add to the module" rather than "extract and extend".

**Blocked by:** None (can start immediately)

**Status:** done

- [x] The statistics, summariser and table builders live in the new `anb` comparison module, and the validation helper imports them from there
- [x] The existing validation-stats test cases move with the functions and pass unchanged
- [x] No behaviour change: the slow validation's report content is identical
- [x] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
