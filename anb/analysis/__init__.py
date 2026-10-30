"""Figures and tables of the paper from whatever runs a simulation folder holds (see docu/analysis-package.md).

Plain Python, no Numba. Doesn't import `ANBmodel.py`.
"""

from .index import index_folder
from .load import Layout, Loader, Run
from .output import FigureResult
from .responses import fig3, figA3, polarisation_summary
from .select import (CONFIGS, PAPER, Baseline, Cell, ExampleAgentFallback, FewSeedsWarning, FigureNotApplicable,
                     MissingLevelsWarning, runs, select)
from .timeseries import example_agent_stats, fig2, focal_sign_counts

__all__ = [
    "CONFIGS", "PAPER", "Baseline", "Cell", "ExampleAgentFallback", "FewSeedsWarning", "FigureNotApplicable",
    "FigureResult", "Layout", "Loader", "MissingLevelsWarning", "Run", "example_agent_stats", "fig2", "fig3", "figA3",
    "focal_sign_counts", "index_folder", "polarisation_summary", "runs", "select",
]
