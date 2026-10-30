"""What a figure or table function returns, and the explicit steps that save them."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

FORMATS = ("png", "pdf")
PNG_DPI = 600  # as the old scripts


def checked_formats(formats) -> tuple[str, ...]:
    """`formats` as a tuple; a `ValueError` if one of them isn't in `FORMATS`."""
    formats = tuple(formats)
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown format(s) {', '.join(unknown)}; choose from {', '.join(FORMATS)}")
    return formats


@dataclass
class FigureResult:
    """A drawn figure, the plotted numbers as a tidy DataFrame and the file stem (old name plus base suffix)."""

    figure: Figure
    data: pd.DataFrame
    name: str

    def save(self, out_dir, formats=("png",)) -> list[Path]:
        """Write `out_dir/<name>.<format>` for each format (PNG at 600 dpi), creating `out_dir`; the paths."""
        formats = checked_formats(formats)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for fmt in formats:
            path = out_dir / f"{self.name}.{fmt}"
            self.figure.savefig(path, **({"dpi": PNG_DPI} if fmt == "png" else {}))
            paths.append(path)
        return paths


TABLES_DIR = "tables"


class LatexTable(str):
    """What a table function returns: its LaTeX (a plain `str` otherwise), carrying the file stem `name`."""

    name: str

    def __new__(cls, latex: str, name: str):
        table = super().__new__(cls, latex)
        table.name = name
        return table


def save_table(latex: str, out_dir, name: str | None = None) -> Path:
    """Write `latex` to `out_dir/tables/<name>.tex`, creating the folder; the path.

    `name` defaults to the table's own (`LatexTable.name`); a plain string needs one.
    """
    name = getattr(latex, "name", None) if name is None else name
    if not name:
        raise ValueError("save_table needs a name for a plain string")
    folder = Path(out_dir) / TABLES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.tex"
    path.write_text(f"{latex}\n")
    return path
