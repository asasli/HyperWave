"""Smoke tests for ``GWLikelihoods(..., shape_per_detector=True)``.

Uses a stub template (no lalsuite dependency) so the test runs even in minimal
environments. Validates:

  1. Parameter counts and column slicing (per-detector mode adds ``n_ifo``).
  2. The likelihood returns finite values and is shape-correct for batched θ.
  3. ``n_ifo == 1`` is bit-for-bit identical to the default (shared) mode.
  4. Calibration paths raise ``NotImplementedError`` (no silent corruption).
"""

from __future__ import annotations

import numpy as np
import pytest

from hyperwave.likelihoods.gwparallel import GWLikelihoods


# --------------------------------------------------------------------------- #
# Stub template: just enough surface for GWLikelihoods to work.
# --------------------------------------------------------------------------- #
class _ZeroTemplate:
    """Returns a zero signal — the residual is then exactly the data."""

    def __init__(self, ifos, n_freq, n_wf_params=2):
        self.ifos = list(ifos)
        self._n_freq = int(n_freq)
        self.parameters = [f"p{i}" for i in range(int(n_wf_params))]

    def make_injections_to_ifo(self, theta):
        return {ch: np.zeros(self._n_freq, dtype=np.complex128) for ch in self.ifos}

    def make_injections_to_ifo_batch(self, p):
        N = int(np.asarray(p).shape[0])
        return np.zeros((N, len(self.ifos), self._n_freq), dtype=np.complex128)


def _build(ifos, nsegs, shape_per_detector, n_freq=64):
    f = np.linspace(20.0, 80.0, n_freq)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(ifos), n_freq)) + 1j * rng.standard_normal((len(ifos), n_freq))
    noise = np.ones((len(ifos), n_freq)) * 1e-44
    template = _ZeroTemplate(ifos, n_freq, n_wf_params=2)
    return GWLikelihoods(
        data=data, f=f, ifos_list=ifos, noise=noise, template=template,
        ddims=True, nsegs=nsegs, gpu=False, shape_per_detector=shape_per_detector,
    )


# --------------------------------------------------------------------------- #
# 1. Parameter accounting
# --------------------------------------------------------------------------- #
def test_per_detector_dim_accounting():
    nsegs, n_ifo = 3, 2
    lik = _build(["H1", "L1"], nsegs=nsegs, shape_per_detector=True)
    assert lik._shape_per_detector is True
    # 2 wf params + nsegs * n_ifo α + nsegs * n_ifo δ
    assert lik._hdims == 2 + nsegs * n_ifo
    assert lik._ndims == 2 + 2 * nsegs * n_ifo


def test_per_detector_requires_ddims():
    with pytest.raises(NotImplementedError, match="ddims=True"):
        f = np.linspace(20.0, 80.0, 32)
        GWLikelihoods(
            data=np.zeros((2, 32), dtype=complex), f=f,
            ifos_list=["H1", "L1"], noise=np.ones((2, 32)),
            template=_ZeroTemplate(["H1", "L1"], 32, 2),
            ddims=False, nsegs=4, gpu=False, shape_per_detector=True,
        )


# --------------------------------------------------------------------------- #
# 2. End-to-end shape + finite-ness
# --------------------------------------------------------------------------- #
def test_per_detector_hyperbolic_classic_finite():
    nsegs, n_ifo = 2, 2
    lik = _build(["H1", "L1"], nsegs=nsegs, shape_per_detector=True)
    N = 4
    rng = np.random.default_rng(1)
    wf = rng.uniform(-1, 1, size=(N, 2))
    alpha = rng.uniform(1.0, 5.0, size=(N, nsegs * n_ifo))
    delta = rng.uniform(1e-6, 5.0, size=(N, nsegs * n_ifo))
    theta = np.concatenate([wf, alpha, delta], axis=1)
    assert theta.shape[1] == lik._ndims

    ll = lik.hyperbolic_classic(theta)
    assert ll.shape == (N,)
    assert np.all(np.isfinite(ll))


def test_per_detector_hyperbolic_ratio_finite():
    nsegs, n_ifo = 2, 2
    lik = _build(["H1", "L1"], nsegs=nsegs, shape_per_detector=True)
    N = 4
    rng = np.random.default_rng(2)
    wf = rng.uniform(-1, 1, size=(N, 2))
    alpha = rng.uniform(1.0, 5.0, size=(N, nsegs * n_ifo))
    ratio = rng.uniform(1e-3, 1.0, size=(N, nsegs * n_ifo))
    theta = np.concatenate([wf, alpha, ratio], axis=1)
    ll = lik.hyperbolic(theta)
    assert ll.shape == (N,)
    assert np.all(np.isfinite(ll))


# --------------------------------------------------------------------------- #
# 3. Single-detector equivalence
# --------------------------------------------------------------------------- #
def test_single_detector_matches_shared_mode():
    """For n_ifo=1 the per-detector likelihood is mathematically identical to
    the shared (default) likelihood. This pins down the per-detector kernel."""
    nsegs = 2
    lik_pd = _build(["H1"], nsegs=nsegs, shape_per_detector=True)
    lik_sh = _build(["H1"], nsegs=nsegs, shape_per_detector=False)

    N = 3
    rng = np.random.default_rng(3)
    wf = rng.uniform(-1, 1, size=(N, 2))
    alpha = rng.uniform(1.0, 5.0, size=(N, nsegs))   # (N, nsegs * 1)
    delta = rng.uniform(1e-6, 5.0, size=(N, nsegs))
    theta = np.concatenate([wf, alpha, delta], axis=1)

    ll_pd = lik_pd.hyperbolic_classic(theta)
    ll_sh = lik_sh.hyperbolic_classic(theta)
    np.testing.assert_allclose(ll_pd, ll_sh, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. Calibration paths reject per-detector mode (no silent corruption)
# --------------------------------------------------------------------------- #
def test_calmarg_rejects_per_detector():
    nsegs, n_ifo = 2, 2
    cal_bank = np.ones((n_ifo, 4, 64), dtype=complex)  # degenerate bank, n_curves=4
    lik = GWLikelihoods(
        data=np.zeros((n_ifo, 64), dtype=complex),
        f=np.linspace(20.0, 80.0, 64),
        ifos_list=["H1", "L1"],
        noise=np.ones((n_ifo, 64)),
        template=_ZeroTemplate(["H1", "L1"], 64, 2),
        ddims=True, nsegs=nsegs, gpu=False,
        shape_per_detector=True,
        calibration_bank=cal_bank,
    )
    theta = np.zeros((1, lik._ndims))
    theta[:, 2:] = 1.0  # α, δ all 1
    with pytest.raises(NotImplementedError, match="shape_per_detector"):
        lik.hyperbolic_classic_calmarg(theta)
