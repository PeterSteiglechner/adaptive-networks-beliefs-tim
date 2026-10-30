"""Figures and tables of the paper from whatever runs a simulation folder holds (see docu/analysis-package.md).

Plain Python, no Numba. Doesn't import `ANBmodel.py`.
"""

from .bnmetrics import LogitFit, cohens_d, fig4, fig5, figA4, greedy_uncorrelated_subset, logit_fit, metric_frame
from .index import index_folder
from .load import Layout, Loader, Run
from .output import FigureResult, LatexTable, save_table
from .responses import fig3, figA3, polarisation_summary
from .select import (CONFIGS, PAPER, Baseline, Cell, ExampleAgentFallback, FewSeedsWarning, FigureNotApplicable,
                     FitWarning, MissingLevelsWarning, runs, select)
from .sensitivity import figA8
from .social import fig6, figA7
from .tables import table3, tableA5, tableA6, tableA7
from .timeseries import example_agent_stats, fig2, focal_sign_counts

__all__ = [
    "CONFIGS", "PAPER", "Baseline", "Cell", "ExampleAgentFallback", "FewSeedsWarning", "FigureNotApplicable",
    "FigureResult", "FitWarning", "LatexTable", "Layout", "Loader", "LogitFit", "MissingLevelsWarning", "Run",
    "cohens_d", "example_agent_stats", "fig2", "fig3", "fig4", "fig5", "fig6", "figA3", "figA4", "figA7", "figA8",
    "focal_sign_counts", "greedy_uncorrelated_subset", "index_folder", "logit_fit", "metric_frame",
    "polarisation_summary", "runs", "save_table", "select", "table3", "tableA5", "tableA6", "tableA7",
]
