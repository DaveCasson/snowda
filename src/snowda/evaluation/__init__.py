"""Streamflow evaluation shared by every run mode.

``evaluate_run`` scores a finished run directory into ``<run_dir>/evaluation``.
Historic output is scored per water year, forecasts per issue date, so runs
can be compared afterwards on the years and issue dates they share.
"""

from snowda.evaluation.run import evaluate_run, evaluate_run_safely

__all__ = ["evaluate_run", "evaluate_run_safely"]
