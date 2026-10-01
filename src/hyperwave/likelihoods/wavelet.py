"""Coherent Gaussian likelihood for wavelet reconstruction.

This is the coherent signal-model noise likelihood: a coherent multi-detector
Gaussian likelihood over a (variable) sum of Morlet-Gabor wavelets, designed to
be driven by Eryn's RJMCMC. The HyperWave hyperbolic likelihoods
(:class:`hyperwave.likelihoods.GWLikelihoods`) can also be used for the
*fixed-dimension* wavelet template via its ``make_injections_to_ifo_batch``
contract; this class adds the variable-dimension (``inds``-masked) path.
"""

from __future__ import annotations

import numpy as np


class WaveletLikelihood:
    """Coherent Gaussian log-likelihood for a wavelet signal model.

    Parameters
    ----------
    data:
        Frequency-domain detector data ``(n_ifo, n_freq)`` on the analysis band.
    psd:
        One-sided PSD ``(n_ifo, n_freq)`` on the same band.
    template:
        A :class:`~hyperwave.detectors.waveforms.wavepackets.WaveletTemplate` whose
        analysis band matches ``data``/``psd``.
    """

    def __init__(self, data, psd, template):
        self.template = template
        self.data = np.asarray(data, dtype=complex)
        self.psd = np.asarray(psd, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data[None, :]
        if self.psd.ndim == 1:
            self.psd = self.psd[None, :]
        self.df = float(template.df)
        self.dd = self._inner(self.data, self.data)
        #: log-likelihood of the empty (zero-wavelet) model
        self.empty_log_likelihood = -0.5 * self.dd

        # device-resident copies for the grouped GPU path (share the template's
        # backend so generation and the inner product run on the same device).
        self.xp = getattr(template, "xp", np)
        self._data_xp = self.xp.asarray(self.data)
        self._psd_xp = self.xp.asarray(self.psd)

    def _inner(self, a, b):
        """Network noise-weighted inner product ``4 Re sum df a* b / Sn``."""
        return float(4.0 * self.df * np.sum((a.conj() * b / self.psd).real))

    def log_likelihood(self, wavelets, ra, dec, psi, ellipticity, inds=None):
        """Vectorised coherent Gaussian log-likelihood.

        ``wavelets`` is ``(N, L, 5)``; ``ra/dec/psi/ellipticity`` are ``(N,)``;
        ``inds`` is the optional Eryn active-leaf mask ``(N, L)``. Returns ``(N,)``
        (the constant noise normalisation is dropped; it cancels in RJMCMC).
        """
        signal = self.template.project_batch(wavelets, ra, dec, psi, ellipticity, inds=inds)
        residual = self.data[None, :, :] - signal
        weighted = (residual.conj() * residual / self.psd[None, :, :]).real
        rr = 4.0 * self.df * np.sum(weighted, axis=(1, 2))
        return -0.5 * rr

    def single(self, wavelets, ra, dec, psi, ellipticity, inds=None):
        """Scalar log-likelihood for one sample's set of wavelets."""
        wavelets = np.asarray(wavelets, dtype=float)
        if wavelets.ndim == 2:
            wavelets = wavelets[None, :, :]
        inds_b = None if inds is None else np.asarray(inds, dtype=bool).reshape(1, -1)
        return float(
            self.log_likelihood(wavelets, [ra], [dec], [psi], [ellipticity], inds=inds_b)[0]
        )

    def grouped_log_like(self, wavelets_flat, groups, ra, dec, psi, ellipticity, n_groups=None):
        """Eryn ``vectorize=True, provide_groups=True`` log-likelihood (fixed sky).

        ``wavelets_flat`` is ``(M, 5)`` (all active leaves across walkers),
        ``groups`` is ``(M,)`` with contiguous values ``[0, G)`` (Eryn's
        group-map ordering). Returns a NumPy ``(G,)`` array of per-walker
        log-likelihoods. All wavelets are generated in one batch and the
        scatter-sum + inner product run on the template's device (GPU when
        ``gpu=True``), making this the fast batched path.
        """
        xp = self.xp
        groups = np.asarray(groups, dtype=np.int64)
        if n_groups is None:
            n_groups = int(groups.max()) + 1 if groups.size else 0
        if n_groups == 0:
            return np.empty(0)

        signal = self.template.project_grouped(
            wavelets_flat, groups, n_groups, ra, dec, psi, ellipticity
        )  # (G, n_ifo, n_freq) on device
        residual = self._data_xp[None, :, :] - signal
        weighted = (residual.conj() * residual / self._psd_xp[None, :, :]).real
        ll = -0.5 * 4.0 * self.df * xp.sum(weighted, axis=(1, 2))  # (G,)
        if self.template.is_gpu:
            return self.template.to_numpy(ll)
        return np.asarray(ll)

    def grouped_log_like_sky(self, params, groups, n_groups=None):
        """Two-branch Eryn log-likelihood with a *sampled* sky.

        ``params`` is ``[wavelets (Mw, 5), extrinsic (Me, 4)]`` and ``groups`` is
        ``[wgroups (Mw,), egroups (Me,)]`` as passed by Eryn for a
        ``signal`` + ``extrinsic`` branch run (``vectorize=True,
        provide_groups=True``). The extrinsic branch carries exactly one leaf per
        walker: ``(ra, dec, psi, ellipticity)``. Returns a NumPy ``(G,)`` array.
        """
        xp = self.xp
        wavelets, extrinsic = params[0], params[1]
        wgroups = np.asarray(groups[0], dtype=np.int64)
        egroups = np.asarray(groups[1], dtype=np.int64)
        extrinsic = np.asarray(extrinsic, dtype=float)
        if n_groups is None:
            n_groups = int(egroups.max()) + 1

        # one extrinsic leaf per walker -> scatter sky into (G, 4) by walker index
        sky = np.zeros((int(n_groups), 4), dtype=float)
        sky[egroups] = extrinsic

        signal = self.template.project_grouped_sky(wavelets, wgroups, sky, n_groups)
        residual = self._data_xp[None, :, :] - signal
        weighted = (residual.conj() * residual / self._psd_xp[None, :, :]).real
        ll = -0.5 * 4.0 * self.df * xp.sum(weighted, axis=(1, 2))  # (G,)
        if self.template.is_gpu:
            return self.template.to_numpy(ll)
        return np.asarray(ll)


__all__ = ["WaveletLikelihood"]
