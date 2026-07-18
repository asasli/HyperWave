"""Priors for waveform-agnostic GW wavelet reconstruction.

These priors are what keep an Eryn RJMCMC wavelet model from soaking up noise:
the dimension prior (number of wavelets) plus the **amplitude/SNR prior** provide
the Occam penalty against spurious low-SNR wavelets.

Distributions follow Eryn's ``eryn.prior`` interface (``rvs``, ``logpdf``,
``pdf``, ``min_val``, ``max_val``, ``copy``) so they drop straight into a
:class:`eryn.prior.ProbDistContainer`.

Per-wavelet (leaf) parameters, in order::

    [t0, f0, Q, amplitude, phi0]

with priors:

* ``t0``        uniform over the segment
* ``f0``        log-uniform over the analysis band
* ``Q``         uniform over ``[Q_min, Q_max]``
* ``amplitude`` :class:`SNRPrior` (induced-SNR prior) when sampling in SNR,
  otherwise uniform in strain amplitude
* ``phi0``      uniform ``[0, 2 pi)`` (periodic)

Shared extrinsic parameters of the elliptical signal, in order::

    [ra, dec, psi, ellipticity]

with ``ra`` uniform ``[0, 2 pi)``, ``dec`` :class:`CosinePrior`
(uniform on the sphere), ``psi`` uniform ``[0, pi)`` and ``ellipticity`` uniform
``[-1, 1]``.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

try:
    from eryn.prior import ProbDistContainer, uniform_dist
except ImportError as exc:  # pragma: no cover - eryn is a core dependency
    raise ImportError("Wavelet priors require Eryn (eryn.prior).") from exc


class _ErynDistribution:
    """Minimal base matching the eryn.prior distribution interface."""

    min_val: float
    max_val: float

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def copy(self):
        return deepcopy(self)


class CosinePrior(_ErynDistribution):
    """Prior uniform on the sphere in declination: ``p(dec) ~ cos(dec)``.

    Default support ``[-pi/2, pi/2]``. Sampling draws ``sin(dec)`` uniformly,
    matching the ``sindelta`` parameterisation while keeping ``dec`` in radians
    for the detector geometry.
    """

    def __init__(self, minimum=-np.pi / 2, maximum=np.pi / 2):
        self.min_val = float(minimum)
        self.max_val = float(maximum)
        self._smin = np.sin(self.min_val)
        self._smax = np.sin(self.max_val)
        self._norm = self._smax - self._smin  # ∫ cos(dec) ddec over support

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.min_val) & (x <= self.max_val)
        out = np.full(x.shape, -np.inf)
        out[inside] = np.log(np.cos(x[inside]) / self._norm)
        return out if out.ndim else float(out)

    def rvs(self, size=1):
        u = np.random.uniform(self._smin, self._smax, size=size)
        return np.arcsin(np.clip(u, -1.0, 1.0))


class LogUniformPrior(_ErynDistribution):
    """Correct log-uniform prior on ``[minimum, maximum]`` (``p(x) ~ 1/x``).

    Eryn's ``log_uniform(min, max)`` mis-passes ``scipy.stats.loguniform(min,
    max-min)`` (scipy's second argument is the upper support, not a scale), so its
    support is silently ``[min, max-min]`` and the top of the band gets ``-inf``.
    This implementation has the correct support and normalisation.
    """

    def __init__(self, minimum, maximum):
        self.min_val = float(minimum)
        self.max_val = float(maximum)
        self._ln_min = np.log(self.min_val)
        self._ln_range = np.log(self.max_val) - self._ln_min

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape, -np.inf)
        inside = (x >= self.min_val) & (x <= self.max_val)
        with np.errstate(divide="ignore"):
            out[inside] = -np.log(x[inside]) - np.log(self._ln_range)
        return out if out.ndim else float(out)

    def rvs(self, size=1):
        u = np.random.uniform(0.0, 1.0, size=size)
        return np.exp(self._ln_min + u * self._ln_range)


class SNRPrior(_ErynDistribution):
    r"""Induced-SNR (amplitude) prior.

    .. math::

        p(\rho) = \frac{3\,\rho}{4\,\rho_*^2}
                  \left(1 + \frac{\rho}{4\rho_*}\right)^{-5}

    normalised on :math:`[0, \infty)`; here truncated/renormalised onto
    ``[snr_min, snr_max]``. It peaks near ``rho_star`` and strongly suppresses
    low-SNR wavelets, which is what prevents the model from fitting noise.

    Sample this when the wavelet ``amplitude`` parameter is the per-wavelet
    optimal SNR (``WaveletTemplate(amplitude_param="snr")``).
    """

    def __init__(self, rho_star=5.0, snr_min=0.0, snr_max=100.0):
        self.rho_star = float(rho_star)
        self.min_val = float(snr_min)
        self.max_val = float(snr_max)
        self._cmin = self._raw_cdf(self.min_val)
        self._cmax = self._raw_cdf(self.max_val)
        self._cnorm = self._cmax - self._cmin

    def _raw_cdf(self, rho):
        # F(rho) = 1 - 4/(1+u)^3 + 3/(1+u)^4,  u = rho / (4 rho_star)
        u = np.asarray(rho, dtype=float) / (4.0 * self.rho_star)
        w = 1.0 + u
        return 1.0 - 4.0 / w**3 + 3.0 / w**4

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.min_val) & (x <= self.max_val)
        out = np.full(x.shape, -np.inf)
        rho = x[inside]
        u = rho / (4.0 * self.rho_star)
        # log[ (3 rho / (4 rho_star^2)) (1+u)^-5 ] - log(norm)
        with np.errstate(divide="ignore"):
            log_p = (
                np.log(3.0 * rho)
                - np.log(4.0 * self.rho_star**2)
                - 5.0 * np.log1p(u)
                - np.log(self._cnorm)
            )
        out[inside] = log_p
        return out if out.ndim else float(out)

    def rvs(self, size=1):
        # inverse-transform sampling via the analytic CDF on a fine grid
        grid = np.linspace(self.min_val, self.max_val, 4096)
        cdf = (self._raw_cdf(grid) - self._cmin) / self._cnorm
        q = np.random.uniform(0.0, 1.0, size=size)
        return np.interp(q, cdf, grid)


def build_wavelet_priors(
    duration,
    *,
    minimum_frequency=20.0,
    maximum_frequency=512.0,
    t0_bounds=None,
    q_bounds=(0.1, 40.0),
    nleaves_max=10,
    nleaves_min=0,
    amplitude_param="snr",
    rho_star=5.0,
    snr_bounds=(0.0, 100.0),
    amplitude_bounds=(0.0, 1e-20),
    branch_name="signal",
):
    """Build Eryn priors for the wavelet (RJ) branch and the extrinsic branch.

    Returns a dict with:

    * ``wavelet`` - :class:`eryn.prior.ProbDistContainer` over the 5 leaf params
    * ``extrinsic`` - :class:`eryn.prior.ProbDistContainer` over ``[ra, dec, psi, ellipticity]``
    * ``priors`` - ``{branch_name: wavelet, "extrinsic": extrinsic}`` for Eryn
    * ``nleaves_max`` / ``nleaves_min`` / ``branch_names`` / ``ndims``
    * ``wavelet_parameters`` / ``extrinsic_parameters`` - the names, in order
    * ``periodic`` - per-branch periodic parameter index maps
    """
    if t0_bounds is None:
        t0_bounds = (0.0, float(duration))

    if amplitude_param == "snr":
        amplitude_prior = SNRPrior(rho_star=rho_star, snr_min=snr_bounds[0], snr_max=snr_bounds[1])
    elif amplitude_param == "amplitude":
        amplitude_prior = uniform_dist(amplitude_bounds[0], amplitude_bounds[1])
    else:
        raise ValueError("amplitude_param must be 'snr' or 'amplitude'.")

    wavelet = ProbDistContainer(
        {
            0: uniform_dist(t0_bounds[0], t0_bounds[1]),          # t0
            1: LogUniformPrior(minimum_frequency, maximum_frequency),  # f0
            2: uniform_dist(q_bounds[0], q_bounds[1]),            # Q
            3: amplitude_prior,                                    # amplitude / snr
            4: uniform_dist(0.0, 2.0 * np.pi),                    # phi0
        }
    )
    extrinsic = ProbDistContainer(
        {
            0: uniform_dist(0.0, 2.0 * np.pi),  # ra
            1: CosinePrior(),                    # dec
            2: uniform_dist(0.0, np.pi),         # psi
            3: uniform_dist(-1.0, 1.0),          # ellipticity
        }
    )

    return {
        "wavelet": wavelet,
        "extrinsic": extrinsic,
        "priors": {branch_name: wavelet, "extrinsic": extrinsic},
        "nleaves_max": {branch_name: int(nleaves_max), "extrinsic": 1},
        "nleaves_min": {branch_name: int(nleaves_min), "extrinsic": 1},
        "branch_names": [branch_name, "extrinsic"],
        "ndims": {branch_name: 5, "extrinsic": 4},
        "wavelet_parameters": ["t0", "f0", "Q", "amplitude", "phi0"],
        "extrinsic_parameters": ["ra", "dec", "psi", "ellipticity"],
        # phi0 (leaf index 4) and ra (extrinsic index 0) are periodic
        "periodic": {branch_name: {4: 2.0 * np.pi}, "extrinsic": {0: 2.0 * np.pi}},
    }


__all__ = ["SNRPrior", "CosinePrior", "LogUniformPrior", "build_wavelet_priors"]
