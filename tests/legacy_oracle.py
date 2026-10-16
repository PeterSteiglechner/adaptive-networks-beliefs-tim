"""Load the frozen legacy `ANBmodel.py` as a test oracle.

Legacy defines `detail` and `track_times` only under `if __name__ == "__main__":` but reads them in
`run_simulation` and `get_output` (known issue #7). `load_legacy` injects both into a freshly executed copy
of the module, so the file itself stays untouched.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

LEGACY_PATH = Path(__file__).resolve().parent.parent / "ANBmodel.py"


def legacy_track_times(legacy: ModuleType, detail: bool):
    """`track_times` exactly as legacy's `__main__` block builds it."""
    if detail:
        return np.arange(legacy.T + 1)
    return (
        [0, 1]
        + legacy.beforeRange
        + [100]
        + legacy.duringRange
        + legacy.afterRange
        + [200]
        + ((legacy.during2Range + legacy.after2Range + [300]) if legacy.two_external_events else [])
    )


def load_legacy(detail: bool = False) -> ModuleType:
    """Return a fresh, independent copy of legacy with `detail` and `track_times` set.

    Each call re-executes the file, so module state that legacy mutates (e.g. the shuffled `agentids`)
    doesn't leak between tests.
    """
    spec = importlib.util.spec_from_file_location("ANBmodel_legacy", LEGACY_PATH)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy.detail = detail
    legacy.track_times = legacy_track_times(legacy, detail)
    return legacy
