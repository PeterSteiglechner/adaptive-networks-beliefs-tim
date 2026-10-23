# 06: Report plots: response shares and metric time series

**Labels:** ready-for-agent

**What to build:** The report script gains plots, each with run-level bootstrap or SE confidence bands:
- old vs new pooled response shares over pressure, one panel per paper config;
- old vs new conditional response shares over μ, one panel per pressure;
- overlaid old/new mean metric time series at the tracked steps, for the top-k differing cells.

**Blocked by:** #05

**Status:** open

- [ ] All three plot families are written to `compare_out/`
- [ ] Every plot shows confidence bands
- [ ] The top-k cells for the time series come from the combined top-differences table
- [ ] The smoke test also checks that the plot files exist
