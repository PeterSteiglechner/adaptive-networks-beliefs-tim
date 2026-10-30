"""What a figure function returns, and the explicit step that saves it."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

FORMATS = ("png", "pdf")
PNG_DPI = 600  # as the old scripts


@dataclass
class FigureResult:
    """A drawn figure, the plotted numbers as a tidy DataFrame and the file stem (old name plus base suffix)."""

    figure: Figure
    data: pd.DataFrame
    name: str

    def save(self, out_dir, formats=("png",)) -> list[Path]:
        """Write `out_dir/<name>.<format>` for each format (PNG at 600 dpi), creating `out_dir`; the paths."""
        formats = tuple(formats)
        unknown = [f for f in formats if f not in FORMATS]
        if unknown:
            raise ValueError(f"unknown format(s) {', '.join(unknown)}; choose from {', '.join(FORMATS)}")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for fmt in formats:
            path = out_dir / f"{self.name}.{fmt}"
            self.figure.savefig(path, **({"dpi": PNG_DPI} if fmt == "png" else {}))
            paths.append(path)
        return paths
