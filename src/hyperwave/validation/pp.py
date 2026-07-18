"""Probability-probability (PP) testing for posterior calibration.

A pipeline that produces credible intervals is only trustworthy if those
intervals have the coverage they claim: the X% credible interval should contain
the truth X% of the time. The PP-test verifies this by injecting many signals
drawn from the prior, recovering each, and checking that the *credible level of
the injected value* (see :meth:`hyperwave.Result.credible_level`) is
Uniform(0, 1) across injections.

This module is the pure statistical core -- it operates on a list of
:class:`hyperwave.Result` objects (each carrying its injection) and is fully
testable without running any inference. :mod:`hyperwave.validation.campaign`
drives the (expensive) injection + recovery loop that produces those results.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats

__all__ = ["credible_levels", "pp_pvalues", "make_pp_plot"]


def credible_levels(
    results: Sequence,
    parameters: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack per-injection credible levels into an ``(n_injections, n_param)`` array.

    Parameters
    ----------
    results:
        Sequence of :class:`hyperwave.Result`, each with an ``injection`` set.
    parameters:
        Parameters to include; defaults to those present in every injection.

    Returns
    -------
    levels, names:
        ``levels[i, j]`` is the credible level of the injected value of parameter
        ``names[j]`` in result ``i``.
    """
    per_result = [r.credible_level() for r in results]
    if parameters is None:
        common = set(per_result[0])
        for cl in per_result[1:]:
            common &= set(cl)
        # preserve the parameter order of the first result
        parameters = [n for n in results[0].parameter_names if n in common]
    levels = np.array([[cl[name] for name in parameters] for cl in per_result], dtype=float)
    return levels, list(parameters)


def pp_pvalues(
    levels: np.ndarray,
    names: Sequence[str],
) -> dict[str, float]:
    """Per-parameter and combined p-values that the credible levels are uniform.

    Each parameter gets a one-sample Kolmogorov-Smirnov p-value against
    Uniform(0, 1); the combined p-value pools them via Fisher's method. A small
    combined p-value is evidence of miscalibration (biased or over/under-confident
    posteriors).
    """
    levels = np.atleast_2d(levels)
    out: dict[str, float] = {}
    per_param = []
    for j, name in enumerate(names):
        p = float(stats.kstest(levels[:, j], "uniform").pvalue)
        out[name] = p
        per_param.append(p)
    # Fisher's method for the combined p-value (independent-enough across params).
    out["combined"] = float(stats.combine_pvalues(per_param, method="fisher").pvalue)
    return out


def make_pp_plot(
    results: Sequence,
    parameters: Optional[Sequence[str]] = None,
    *,
    confidence_bands: Sequence[float] = (0.68, 0.95, 0.997),
    title: Optional[str] = None,
    ax=None,
):
    """Draw a PP-plot and return ``(fig, pvalues)``.

    For each parameter the empirical CDF of the credible levels is plotted against
    the diagonal; a calibrated pipeline tracks the diagonal within the grey
    binomial confidence bands. The legend reports each parameter's KS p-value and
    the combined p-value.
    """
    import matplotlib.pyplot as plt

    levels, names = credible_levels(results, parameters)
    pvalues = pp_pvalues(levels, names)
    n_inj = levels.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # grey binomial confidence bands around the diagonal
    x = np.linspace(0, 1, 200)
    for ci in sorted(confidence_bands, reverse=True):
        edge = (1 - ci) / 2
        lo = stats.binom.ppf(edge, n_inj, x) / n_inj
        hi = stats.binom.ppf(1 - edge, n_inj, x) / n_inj
        ax.fill_between(x, lo, hi, color="k", alpha=0.08, linewidth=0)
    ax.plot([0, 1], [0, 1], color="k", lw=1, alpha=0.5)

    grid = np.linspace(0, 1, 101)
    for j, name in enumerate(names):
        ecdf = np.searchsorted(np.sort(levels[:, j]), grid, side="right") / n_inj
        ax.plot(grid, ecdf, lw=1.4, label=f"{name} ({pvalues[name]:.2f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("credible level C")
    ax.set_ylabel("fraction of injections in C")
    ax.set_title(title or f"PP-plot ({n_inj} injections) — combined p={pvalues['combined']:.3f}")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
    ax.set_aspect("equal")
    return fig, pvalues
