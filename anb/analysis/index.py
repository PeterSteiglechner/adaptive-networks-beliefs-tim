"""A simulation folder as a table of runs: one row per CSV whose name follows the legacy naming scheme."""

import warnings
from pathlib import Path

import pandas as pd

from anb.io import is_detailed, parse_filename

INDEX_COLUMNS = ("link_prob", "init_w", "beta", "rho", "eps", "mu", "fixedBNat100", "s", "seed", "lam", "n_events",
                 "M", "N", "tau", "detailed", "path")
N_EXAMPLES = 5  # unparseable file names quoted in the warning


def index_folder(sim_dir) -> pd.DataFrame:
    """The runs in `sim_dir/*.csv` and `sim_dir/detailed/*.csv`, one row each (`INDEX_COLUMNS`), sorted by path.

    Parameters are as `parse_filename` reads them from the name (rounded as written, absent suffixes at their
    defaults); `s` is the pressure (`ext_strength`) and `path` the file's `Path`. CSVs whose name doesn't parse
    are left out, with a single warning giving their count and a few examples.
    """
    sim_dir = Path(sim_dir)
    rows, unparsed = [], []
    for path in sorted([*sim_dir.glob("*.csv"), *(sim_dir / "detailed").glob("*.csv")]):
        try:
            cell, seed = parse_filename(path)
        except ValueError:
            unparsed.append(path)
            continue
        rows.append({**cell, "s": cell.pop("ext_strength"), "seed": seed, "detailed": is_detailed(path), "path": path})
    if unparsed:
        examples = ", ".join(str(path.relative_to(sim_dir)) for path in unparsed[:N_EXAMPLES])
        warnings.warn(f"{len(unparsed)} CSV file(s) in {sim_dir} don't follow the legacy naming scheme and are "
                      f"ignored, e.g. {examples}", UserWarning, stacklevel=2)
    return pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
