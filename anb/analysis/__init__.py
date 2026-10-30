"""Figures and tables of the paper from whatever runs a simulation folder holds (see docu/analysis-package.md).

Plain Python, no Numba. Doesn't import `ANBmodel.py`.
"""

from .bnmetrics import LogitFit, cohens_d, fig4, fig5, figA4, logit_fit, metric_frame
from .index import index_folder
from .load import Layout, Loader, Run
from .output import FigureResult
from .responses import fig3, figA3, polarisation_summary
from .select import (CONFIGS, PAPER, Baseline, Cell, ExampleAgentFallback, FewSeedsWarning, FigureNotApplicable,
                     FitWarning, MissingLevelsWarning, runs, select)
from .social import fig6, figA7
from .timeseries import example_agent_stats, fig2, focal_sign_counts

__all__ = [
    "CONFIGS", "PAPER", "Baseline", "Cell", "ExampleAgentFallback", "FewSeedsWarning", "FigureNotApplicable",
    "FigureResult", "FitWarning", "Layout", "Loader", "LogitFit", "MissingLevelsWarning", "Run", "cohens_d",
    "example_agent_stats", "fig2", "fig3", "fig4", "fig5", "fig6", "figA3", "figA4", "figA7", "focal_sign_counts",
    "index_folder", "logit_fit", "metric_frame", "polarisation_summary", "runs", "select",
]
