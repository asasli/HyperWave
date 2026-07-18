"""Posterior-calibration validation: injection campaigns and PP-tests.

A pipeline's credible intervals are only trustworthy if they have correct
coverage. :func:`run_pp_campaign` injects many signals from the prior, recovers
each, and :func:`make_pp_plot` checks that the credible level of the injected
value is Uniform(0, 1) -- the standard PP-test used to validate GW pipelines.
"""

from .campaign import run_pp_campaign
from .pp import credible_levels, make_pp_plot, pp_pvalues

__all__ = ["run_pp_campaign", "credible_levels", "make_pp_plot", "pp_pvalues"]
