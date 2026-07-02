"""PP / percentile-percentile test for posterior calibration.

For a correctly calibrated posterior the credible level of the injected value
(``mean(samples < truth)`` per parameter) is Uniform(0, 1) across many
injections. This module collects those credible levels over a list of
:class:`~hyperwave.result.Result` objects, runs per-parameter KS tests against
the uniform distribution, combines them into a single p-value, and draws the
standard PP plot.

The decisive calibration test for detector calibration is the contrast: with the
truth's calibration error injected, the *base* likelihood (no calibration) must
FAIL (combined p -> 0), while ``*_calmarg`` / ``*_calsample`` must PASS
(combined p above threshold). See ``examples/pp_test/``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["credible_levels", "pp_pvalues", "make_pp_plot"]


def credible_levels(results, parameters=None):
    """Credible levels of the injected truths across a list of results.

    Returns ``(levels, names)`` where ``levels`` has shape
    ``(n_injections, n_parameters)`` and ``names`` are the parameters present
    (with an injected truth) in *every* result, in the order of the first
    result. ``parameters`` optionally restricts/orders the set.
    """
    results = list(results)
    if not results:
        raise ValueError("need at least one Result to compute credible levels")
    per = [r.credible_level() for r in results]
    common = set(per[0])
    for p in per[1:]:
        common &= set(p)
    order = parameters if parameters is not None else results[0].parameter_names
    names = [n for n in order if n in common]
    if not names:
        raise ValueError("no parameter has an injected truth in every result")
    levels = np.array([[p[n] for n in names] for p in per], dtype=float)
    return levels, names


def pp_pvalues(levels, names):
    """Per-parameter and combined PP p-values.

    Each parameter's credible levels are KS-tested against Uniform(0, 1); the
    per-parameter p-values are combined with Fisher's method. Returns a dict
    ``{param: p, ..., "combined": p}``. A well-calibrated set gives a large
    combined p; bias or over/under-confidence drives it toward 0.
    """
    from scipy import stats

    levels = np.atleast_2d(np.asarray(levels, dtype=float))
    pvals, out = [], {}
    for j, name in enumerate(names):
        p = float(stats.kstest(levels[:, j], "uniform").pvalue)
        out[name] = p
        pvals.append(p)
    out["combined"] = float(stats.combine_pvalues(pvals, method="fisher")[1]) if pvals else 1.0
    return out


def make_pp_plot(results, parameters=None, confidence=(0.68, 0.95, 0.997),
                 ax=None, title=True, legend=True):
    """Draw the PP plot and return ``(fig, pvalues)``.

    Each parameter's sorted credible levels are plotted against the empirical
    CDF; a calibrated posterior tracks the diagonal within the binomial
    confidence bands. ``pvalues`` is the :func:`pp_pvalues` dict.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    levels, names = credible_levels(results, parameters=parameters)
    pvals = pp_pvalues(levels, names)
    n = levels.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    x = np.linspace(0, 1, 200)
    for ci in confidence:
        edge = (1 - ci) / 2
        lower = stats.binom.ppf(edge, n, x) / n
        upper = stats.binom.ppf(1 - edge, n, x) / n
        ax.fill_between(x, lower, upper, color="k", alpha=0.1, linewidth=0)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)

    cdf = np.arange(1, n + 1) / n
    for j, name in enumerate(names):
        ax.plot(np.sort(levels[:, j]), cdf, linewidth=1.2,
                label=f"{name} (p={pvals[name]:.2f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("credible level of injected value")
    ax.set_ylabel("fraction of injections")
    if legend and len(names) <= 14:
        ax.legend(fontsize=7, loc="upper left")
    if title:
        ax.set_title(f"combined p = {pvals['combined']:.3f}   (N = {n})")
    return fig, pvals
