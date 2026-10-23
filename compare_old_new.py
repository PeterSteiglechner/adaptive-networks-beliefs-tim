# %%
"""Compare two ensemble output folders (legacy vs `anb`, or any two sweeps) and write the tables to `compare_out/`.

Run it as a plain script or cell by cell. The constants below can be overridden by environment variables of the
same name (`COMPARE_OLD`, ...), which the smoke test uses.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from anb.comparison import compare_folders

# %%
OLD_FOLDER = Path(os.environ.get("COMPARE_OLD", "simOut_old_model"))
NEW_FOLDER = Path(os.environ.get("COMPARE_NEW", "simOut_new_model"))
OUT_FOLDER = Path(os.environ.get("COMPARE_OUT", "compare_out"))
TOP_K = int(os.environ.get("COMPARE_TOP_K", 10))
N_PERMUTATIONS = int(os.environ.get("COMPARE_N_PERMUTATIONS", 10_000))
RUN_ABLATIONS = False  # the ablation cell arrives with its own ticket
RUN_SHARED_START = False  # the shared-start cell arrives with its own ticket

# %%
result = compare_folders(OLD_FOLDER, NEW_FOLDER, n_permutations=N_PERMUTATIONS, cache_dir=OUT_FOLDER / "cache")

# %%
TABLES = {
    "pooled_responses": result.pooled_responses,
    "conditional_responses": result.conditional_responses,
    "lean": result.lean,
    "metrics": result.metrics,
    "top_differences": result.top_differences,
}


def markdown_table(table: pd.DataFrame) -> str:
    """A GitHub markdown table; floats to 4 significant digits."""
    cells = table.map(lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "\\|"))
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def file_list(stems: list[str], limit: int = 20) -> str:
    shown = "\n".join(f"- `{stem}`" for stem in stems[:limit])
    return shown + (f"\n- … and {len(stems) - limit} more" if len(stems) > limit else "") if stems else "none"


def summary_markdown() -> str:
    pairing = result.pairing
    families = pd.DataFrame(
        [{"family": name, "tests": len(table), "failing (q < 0.05)": int(table.fail.sum()) if "fail" in table else 0}
         for name, table in TABLES.items() if name != "top_differences"])
    top = result.top_differences.head(TOP_K)
    return "\n".join([
        "# Old vs new comparison", "",
        f"Old: `{OLD_FOLDER}`, new: `{NEW_FOLDER}`. {len(pairing.matched)} files matched.", "",
        "## Files in one folder only", "",
        f"Only in old ({len(pairing.only_old)}):", "", file_list(pairing.only_old), "",
        f"Only in new ({len(pairing.only_new)}):", "", file_list(pairing.only_new), "",
        "## Failing tests per family", "", markdown_table(families), "",
        f"## Top {TOP_K} effects", "", markdown_table(top), ""])


# %%
OUT_FOLDER.mkdir(parents=True, exist_ok=True)
for name, table in TABLES.items():
    table.to_csv(OUT_FOLDER / f"{name}.csv", index=False)
    (OUT_FOLDER / f"{name}.md").write_text(markdown_table(table) + "\n")
(OUT_FOLDER / "summary.md").write_text(summary_markdown())
print((OUT_FOLDER / "summary.md").read_text())
