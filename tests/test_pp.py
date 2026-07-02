"""Tests for the PP-test statistical core.

The credible level of the injected value must be Uniform(0, 1) for a calibrated
posterior. We construct synthetic results (no inference needed) and check the
combined p-value: it should *pass* calibrated posteriors and *reject* biased or
over-confident ones.
"""

import numpy as np

from hyperwave import Result
from hyperwave.validation import credible_levels, make_pp_plot, pp_pvalues


def _synthetic_results(n_inj, rng, *, sigma=1.0, n_samp=4000, bias=0.0, post_scale=1.0):
    """Calibrated when bias=0, post_scale=1: posterior N(meas, sigma) with
    meas = truth + sigma*z gives credible-level = Phi(-z) ~ Uniform(0, 1)."""
    results = []
    for _ in range(n_inj):
        truth = rng.normal(0.0, 1.0)
        meas = truth + sigma * rng.normal()
        samples = rng.normal(meas + bias, sigma * post_scale, size=(n_samp, 1))
        results.append(Result(samples, ["x"], injection={"x": truth}))
    return results


def test_calibrated_passes():
    rng = np.random.default_rng(0)
    res = _synthetic_results(300, rng)
    pv = pp_pvalues(*credible_levels(res))
    assert pv["combined"] > 0.05  # calibrated -> not rejected


def test_biased_posterior_detected():
    rng = np.random.default_rng(0)
    res = _synthetic_results(300, rng, bias=1.5)  # systematically offset
    pv = pp_pvalues(*credible_levels(res))
    assert pv["combined"] < 0.01


def test_overconfident_posterior_detected():
    rng = np.random.default_rng(0)
    res = _synthetic_results(300, rng, post_scale=0.4)  # too narrow
    pv = pp_pvalues(*credible_levels(res))
    assert pv["combined"] < 0.01


def test_make_pp_plot_smoke():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(3)
    res = _synthetic_results(120, rng)
    fig, pv = make_pp_plot(res)
    assert "combined" in pv and 0.0 <= pv["combined"] <= 1.0
