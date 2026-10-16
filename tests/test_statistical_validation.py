"""On-demand statistical validation, new versus legacy (seam 1): 4 configurations × 6 pressures × 100 seeds.

Takes about 25 min on 12 cores (almost all of it legacy). Run with `pytest -m slow -s tests/test_statistical_validation.py`.
The report and the per-test tables go to `simOut/validation/` (`report.md`, `responses.csv`, `metrics.csv`);
run summaries are cached under `simOut/validation/runs/`, so delete that folder after changing the model.
"""

from pathlib import Path

import pytest

from validation import validate

pytestmark = pytest.mark.slow

OUT = Path(__file__).resolve().parent.parent / "simOut" / "validation"


def test_new_package_matches_legacy_statistically():
    responses, metrics = validate(seeds=range(100), out_dir=OUT)

    print(f"\nReport: {OUT / 'report.md'}")
    print(responses[responses.fail][["config", "pressure", "chi2", "p", "q"]].to_string())
    print(metrics[metrics.fail][["config", "pressure", "window", "metric", "diff", "p", "q"]].to_string())
    assert not responses.fail.any(), "response frequencies differ; see the report"
    assert not metrics.fail.any(), "metric distributions differ; see the report"
