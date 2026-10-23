# 04: Summary cache and parallel reading

**Labels:** ready-for-agent

**What to build:** Re-running a comparison over 16,800 CSVs is fast once the summaries exist. Run summaries are cached per (folder, cell) in the given cache directory. The cache key is the sorted file list plus their modification times and sizes, so any change to the folder rebuilds the affected cell. Uncached summaries are read in parallel with joblib, as the validation already does.

**Blocked by:** #02

**Status:** open

- [ ] A second comparison of unchanged folders reuses the cache, which a test checks through observable behaviour (e.g. no CSV reads), not through the cache layout
- [ ] Touching, adding or removing a CSV in a cell invalidates that cell's cache
- [ ] Results with and without the cache are identical
- [ ] `pytest` and `NUMBA_DISABLE_JIT=1 pytest` pass
