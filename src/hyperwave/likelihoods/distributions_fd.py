"""Frequency-domain distribution likelihoods with optional GPU support."""

from __future__ import annotations

import numpy as np

from ..backends import gpu_backend_available
from .base import BaseLikelihood


class LogLike(BaseLikelihood):
    """Hyperbolic likelihoods for data-only frequency-domain analyses."""

    def __init__(
        self,
        data,
        f,
        ifos_list,
        ddims=True,
        nsegs=1,
        gpu=False,
        infs=-1e300,
        freq_domain=True,
        detector_dependent_noise=False,
    ):
        self._init_backend(gpu=gpu)

        self._nchannels = len(ifos_list)
        self.ifos = list(ifos_list)
        self._f = np.asarray(f)
        self.df = self._f[1] - self._f[0]
        self.data = self._backend.asarray(data)
        self.freq_domain = bool(freq_domain)
        self.detector_dependent_noise = bool(detector_dependent_noise)

        if self.freq_domain:
            self._full_channels = 2 * self._nchannels
            self._detector_channels = 2
            self.yy_noise = self._get_yy_noise()
        else:
            self._full_channels = self._nchannels
            self._detector_channels = 1
            self.yy_noise = self.data**2
        if self.detector_dependent_noise:
            self.yy_noise_detector = self._get_yy_noise_detector()

        self._d = self._full_channels
        self._lam = (self._d + 1) / 2
        self._C0 = ((1 - self._d) / 2) * np.log(2.0 * np.pi)
        self._detector_lam = (self._detector_channels + 1) / 2
        self._detector_C0 = ((1 - self._detector_channels) / 2) * np.log(2.0 * np.pi)

        self._ddims = bool(ddims)
        self._nsegs = int(nsegs)
        self._inf = infs
        self._logfreq = np.log10(self._f)
        self._ndims = self._nsegs if self._ddims else 1
        if self.detector_dependent_noise:
            self._ndims *= self._nchannels
        self.hyperbolic = self.hyperbolic2D if self._ddims else self.hyperbolic1D

        self._segi, self._Nd, self._fb = self._build_segments(self._f, self._nsegs)

    def _get_yy_noise(self):
        yy = self.data.conj() * self.data
        if self._nchannels == 1:
            syy = 4.0 * self.df * yy
        else:
            syy = 4.0 * self.df * self.xp.sum(yy, axis=0)
        return self.xp.abs(syy)

    def _get_yy_noise_detector(self):
        yy = self.data.conj() * self.data if self.freq_domain else self.data**2
        if self._nchannels == 1 and yy.ndim == 1:
            yy = yy[None, :]
        if self.freq_domain:
            yy = 4.0 * self.df * yy
        return self.xp.abs(yy)

    def _reshape_detector_parameter(self, values, segmented):
        width = self._nsegs if segmented else 1
        return values.reshape(values.shape[0], self._nchannels, width)

    def _hyperbolic_by_detector(self, alpha, delta):
        likelihood = self.xp.zeros((alpha.shape[0], self._nchannels, len(self._segi)))
        for ifo in range(self._nchannels):
            for i, si in enumerate(self._segi):
                alpha_i = alpha[:, ifo, i] if self._ddims else alpha[:, ifo, 0]
                delta_i = delta[:, ifo, i]
                alpha_delta_i = alpha_i * delta_i

                log_kappa = self._backend.log_kv(self._detector_lam, alpha_delta_i)
                term_sqrt = self.xp.sum(
                    self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise_detector[ifo, si]).real,
                    axis=-1,
                )
                term_lambda = self._detector_lam * self.xp.log(alpha_i / delta_i)
                term_rest = self._detector_C0 - self.xp.log(2.0 * alpha_i) - log_kappa

                likelihood[:, ifo, i] = self._Nd[i] * (term_lambda + term_rest) - alpha_i * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=(1, 2)),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic1D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(10.0 ** theta[:, : self._ndims])
        ratio = self._backend.asarray(10.0 ** theta[:, self._ndims :])

        if self.detector_dependent_noise:
            alpha = self._reshape_detector_parameter(alpha, segmented=False)
            ratio = self._reshape_detector_parameter(ratio, segmented=True)
            return self._hyperbolic_by_detector(alpha, alpha * ratio)

        alpha = alpha[:, 0]

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            ratio_i = ratio[:, i]
            delta_i = alpha * ratio_i
            alpha_delta_i = alpha * delta_i

            log_kappa = self._backend.xp.log(self._backend.kve(self._lam, alpha_delta_i)) - alpha_delta_i
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic2D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(10.0 ** theta[:, : self._ndims])
        ratio = self._backend.asarray(10.0 ** theta[:, self._ndims :])

        if self.detector_dependent_noise:
            alpha = self._reshape_detector_parameter(alpha, segmented=True)
            ratio = self._reshape_detector_parameter(ratio, segmented=True)
            return self._hyperbolic_by_detector(alpha, alpha * ratio)

        delta = alpha * ratio

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i]
            delta_i = delta[:, i]
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.xp.log(self._backend.kve(self._lam, alpha_delta_i)) - alpha_delta_i
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha_i / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha_i) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha_i * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic_classic1D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(theta[:, : self._ndims])
        delta = self._backend.asarray(theta[:, self._ndims :])

        if self.detector_dependent_noise:
            alpha = self._reshape_detector_parameter(alpha, segmented=False)
            delta = self._reshape_detector_parameter(delta, segmented=True)
            return self._hyperbolic_by_detector(alpha, delta)

        alpha = alpha[:, 0]

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            delta_i = delta[:, i]
            alpha_delta_i = alpha * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic_classic2D(self, theta):
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(theta[:, : self._ndims])
        delta = self._backend.asarray(theta[:, self._ndims :])

        if self.detector_dependent_noise:
            alpha = self._reshape_detector_parameter(alpha, segmented=True)
            delta = self._reshape_detector_parameter(delta, segmented=True)
            return self._hyperbolic_by_detector(alpha, delta)

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i]
            delta_i = delta[:, i]
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy_noise[si]).real, axis=-1)
            term_lambda = self._lam * self.xp.log(alpha_i / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha_i) - log_kappa

            likelihood[:, i] = self._Nd[i] * (term_lambda + term_rest) - alpha_i * term_sqrt

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()


# Backwards-compatible alias (the class was historically lowercase).
loglike = LogLike


def interpolate_fncn(x, y=None, kind="akima"):
    """
    Interpolation helper kept for backward compatibility.

    ``kind`` is kept for backward compatibility. Only Akima interpolation is
    currently supported.
    """
    from scipy.interpolate import Akima1DInterpolator

    if y is None:
        raise ValueError("interpolate_fncn requires both x and y arrays.")

    if kind.lower() != "akima":
        raise ValueError("Only kind='akima' is currently supported.")

    return Akima1DInterpolator(x, y)


__all__ = ["LogLike", "loglike", "gpu_backend_available", "interpolate_fncn"]
