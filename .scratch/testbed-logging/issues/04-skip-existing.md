# 04: Resume with `skip_existing=True`

**Labels:** ready-for-agent

**What to build:** A researcher resuming a crashed sweep passes `skip_existing=True` to `save_ensemble`. In the
parent, before anything is sent to workers, every seed whose CSV already exists (found with the path helper from
01) is filtered out. Only the missing seeds run. The skip count is logged at INFO, and progress counts only the
seeds that actually run. If every seed is skipped, the ensemble finishes at once with a log line saying so. The
returned list still has every seed's path (written or skipped), in seed order. Overwriting stays the default. See
`.scratch/testbed-logging/spec.md` (user stories 14–17).

**Blocked by:** 03 (Start, progress and finish logging for both ensemble methods)

**Status:** done

- [x] With `skip_existing=True`, existing files are left untouched (same mtime and content), and only the missing seeds run
- [x] The skip count shows up in the log records, and the finish record's written/skipped counts match
- [x] With all seeds present, nothing runs, the method returns quickly with all paths, and a log record says everything was skipped
- [x] The returned paths cover every seed, in seed order
- [x] With the default `skip_existing=False`, existing files are still overwritten
- [x] Tests pass with `NUMBA_DISABLE_JIT=1`
