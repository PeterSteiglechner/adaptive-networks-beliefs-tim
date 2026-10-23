"""On-demand statistical validation, new versus legacy (seam 1): 4 configurations × 6 pressures × 100 seeds.

Takes about 25 min on 12 cores (almost all of it legacy). Run with `pytest -m slow -s tests/test_statistical_validation.py`.
The report and the four tables go to `validation_out/` (`report.md`, `pooled_responses.csv`,
`conditional_responses.csv`, `lean.csv`, `metrics.csv`); run summaries are cached under `validation_out/runs/`,
so delete that folder after changing the model.
"""

from pathlib import Path

import pytest

from validation import OUT_DIR, validate

pytestmark = pytest.mark.slow



def test_new_package_matches_legacy_statistically():
    tables = validate(seeds=range(100), out_dir=OUT_DIR)
    pooled, conditional = tables["pooled_responses"], tables["conditional_responses"]
    lean, metrics = tables["lean"], tables["metrics"]

    print(f"\nReport: {OUT_DIR / 'report.md'}")
    print(pooled[pooled.fail][["config", "pressure", "chi2", "p", "q"]].to_string())
    print(conditional[conditional.fail][["config", "pressure", "chi2", "p", "q"]].to_string())
    print(lean[lean.fail][["config", "pressure", "quantity", "diff", "gap_se", "p", "q"]].to_string())
    print(metrics[metrics.fail][["config", "pressure", "window", "metric", "diff", "p", "q"]].to_string())
    assert not pooled.fail.any(), "response frequencies differ; see the report"
    assert not conditional.fail.any(), "conditional response frequencies differ; see the report"
    assert not lean.fail.any(), "the lean differs; see the report"
    assert not metrics.fail.any(), "metric distributions differ; see the report"
