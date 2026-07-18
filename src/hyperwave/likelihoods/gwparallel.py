"""Likelihoods for GW inference with optional GPU acceleration."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from ..backends import gpu_backend_available
from ..detectors.calibration import read_calibration_file
from .base import BaseLikelihood


class GWLikelihoods(BaseLikelihood):
    """
    Log-likelihoods for GW parameter estimation.

    The waveform generation itself is still driven by the template object, which
    is typically CPU-side. When ``gpu=True`` HyperWave moves the array-heavy
    residual and likelihood algebra to CuPy while keeping a safe NumPy fallback.
    """

    def __init__(
        self,
        data,
        f,
        ifos_list,
        noise,
        template,
        ddims=True,
        nsegs=4,
        gpu=False,
        infs=-1e300,
        cpu_cores=32,
        calibration_marginalization=False,
        calibration_draws=None,
        calibration_lookup_table=None,
        number_of_response_curves=1000,
        starting_index=0,
        calibration_correction_type=None,
        calibration_chunk_size=64,
        calibration_bank=None,
        cal_chunk=64,
        cal_n_nodes=None,
        cal_minimum_frequency=None,
        cal_maximum_frequency=None,
        shape_per_detector=False,
    ):
        """
        Parameters
        ----------
        shape_per_detector : bool, default False
            If True, the hyperbolic shape parameters α and δ (or α and the
            ratio δ/α) are sampled *per detector* rather than shared across
            the detector network. The combined-residual likelihood is replaced
            by a sum of single-detector MVH likelihoods, so each interferometer
            sees its own non-Gaussian noise scale. Requires ``ddims=True`` and
            is currently supported only on the non-calibration paths
            (``hyperbolic`` and ``hyperbolic_classic``); the ``*_calmarg`` and
            ``*_calsample`` variants raise ``NotImplementedError`` when this is
            set. The added parameter count is
            ``2 * nsegs * n_ifo`` instead of ``2 * nsegs``.
        """
        self._init_backend(gpu=gpu)

        self._template = template
        if self._template.parameters is None:
            raise TypeError("Template must define its parameter list.")

        # Prefer the batched template path (one call for all walkers) when the
        # template exposes it; this avoids the per-walker joblib loop entirely
        # and is the fast path for the new lal/ml4gw backends.
        self._batched_template = callable(
            getattr(self._template, "make_injections_to_ifo_batch", None)
        )

        self._wfdims = len(self._template.parameters)
        self._nchannels = len(ifos_list)
        self.ifos = list(ifos_list)
        self._f = np.asarray(f)
        self.df = self._f[1] - self._f[0]
        self.data = self._backend.asarray(data)
        self.psd = self._backend.asarray(noise)

        self._d = 2 * self._nchannels
        self._lam = (self._d + 1) / 2
        self._C0 = ((1 - self._d) / 2) * np.log(2.0 * np.pi)

        self._ddims = bool(ddims)
        self._nsegs = int(nsegs)
        self._inf = infs
        self._logfreq = np.log10(self._f)
        self._shape_per_detector = bool(shape_per_detector)
        if self._shape_per_detector and not self._ddims:
            raise NotImplementedError(
                "shape_per_detector=True currently requires ddims=True "
                "(per-detector × per-segment α, δ)."
            )
        if self._shape_per_detector:
            # α and δ (or ratio) each occupy nsegs * n_ifo columns, stored in
            # segment-major order: (seg0_ifo0, seg0_ifo1, ..., seg1_ifo0, ...).
            self._n_shape_blocks = self._nsegs * self._nchannels
            self._hdims = self._wfdims + self._n_shape_blocks
            self._ndims = self._wfdims + 2 * self._n_shape_blocks
            # Per-detector MVH: each ifo is one complex (= 2 real) channel.
            self._d_pd = 2
            self._lam_pd = (self._d_pd + 1) / 2
            self._C0_pd = ((1 - self._d_pd) / 2) * np.log(2.0 * np.pi)
        else:
            self._ndims = self._wfdims + 2 * self._nsegs if self._ddims else self._wfdims + self._nsegs + 1
            self._hdims = self._wfdims + self._nsegs if self._ddims else self._wfdims + 1
        self.num_jobs = 1 if self._use_gpu else int(cpu_cores)

        self.whiten = None
        self.logwhiten = None
        self.yy = None
        self.yy_noise = -0.5 * self._get_yy_noise()

        self._segi, self._Nd, self._fb = self._build_segments(self._f, self._nsegs)
        self.calibration_marginalization = bool(calibration_marginalization)
        self.calibration_chunk_size = int(calibration_chunk_size)
        self.number_of_response_curves = int(number_of_response_curves)
        self.starting_index = int(starting_index)
        self.calibration_draws = None
        self._cal_draws = None
        self._cal_abs2 = None
        self._n_curves = 0
        self._cal_chunk = int(cal_chunk)
        if calibration_bank is not None:
            draws = self._backend.asarray(calibration_bank)
            if draws.ndim != 3:
                raise ValueError(
                    "calibration_bank must have shape (n_ifo, n_curves, n_freq); "
                    f"got ndim={draws.ndim}."
                )
            if draws.shape[0] != self._nchannels or draws.shape[2] != self.data.shape[1]:
                raise ValueError(
                    "calibration_bank shape "
                    f"{tuple(int(s) for s in draws.shape)} is incompatible with "
                    f"{self._nchannels} detectors x {self.data.shape[1]} frequency bins."
                )
            self._cal_draws = draws
            self._cal_abs2 = (draws.conj() * draws).real
            self._n_curves = int(draws.shape[1])
            self.calibration_draws = draws
            self.number_of_response_curves = self._n_curves

        if self.calibration_marginalization:
            if (
                calibration_draws is None
                and calibration_lookup_table is None
                and calibration_bank is not None
            ):
                calibration_draws = calibration_bank
            self._setup_calibration_marginalization(
                calibration_draws=calibration_draws,
                calibration_lookup_table=calibration_lookup_table,
                correction_type=calibration_correction_type,
            )
            self._cal_draws = self.calibration_draws
            self._cal_abs2 = (self.calibration_draws.conj() * self.calibration_draws).real
            self._n_curves = self.number_of_response_curves

        self._cal_spline = None
        self._caldims = 0
        if cal_n_nodes is not None:
            from ..detectors.calibration import SplineCalibration

            self._cal_spline = SplineCalibration(
                self._f, n_nodes=int(cal_n_nodes),
                minimum_frequency=cal_minimum_frequency,
                maximum_frequency=cal_maximum_frequency,
                gpu=self._use_gpu,
            )
            self._caldims = self._nchannels * 2 * int(cal_n_nodes)

    def _alpha_columns(self, theta):
        """Slice α from ``theta``.

        Shape returned:
          * default: ``(N, nsegs)`` if ``ddims`` else ``(N, 1)``.
          * shape_per_detector: ``(N, nsegs, n_ifo)`` (segment-major).
        """
        alpha = theta[:, self._wfdims:self._hdims]
        if self._shape_per_detector:
            return alpha.reshape(alpha.shape[0], self._nsegs, self._nchannels)
        if self._ddims:
            return alpha
        return alpha[:, :1]

    def _tail_columns(self, theta):
        """Slice the δ / ratio block (everything after ``_hdims`` up to ``_ndims``).

        Shape returned:
          * default: ``(N, nsegs)``.
          * shape_per_detector: ``(N, nsegs, n_ifo)`` (segment-major).
        """
        tail = theta[:, self._hdims:self._ndims]
        if self._shape_per_detector:
            return tail.reshape(tail.shape[0], self._nsegs, self._nchannels)
        return tail

    def _get_yy_noise(self):
        return self.inner_product(self.data, self.data, psd=self.psd)

    def _setup_calibration_marginalization(
        self, calibration_draws=None, calibration_lookup_table=None, correction_type=None
    ):
        if not self._batched_template:
            raise ValueError("Calibration marginalization requires make_injections_to_ifo_batch.")

        low_template = getattr(self._template, "template", self._template)
        if any(str(name).startswith("recalib") for name in self._template.parameters):
            raise ValueError(
                "Calibration marginalization uses response-curve draws, so calibration "
                "parameters should not be part of template.parameters."
            )
        static_parameters = getattr(low_template, "static_parameters", {})
        if any(str(name).startswith("recalib") for name in static_parameters):
            raise ValueError(
                "Calibration marginalization uses response-curve draws, so calibration "
                "parameters should not be fixed in static_parameters."
            )
        for model in getattr(low_template, "calibration_models", []):
            if model is not None and getattr(model, "name", None) != "none":
                raise ValueError(
                    "Calibration marginalization should be used with identity template "
                    "calibration models to avoid double application."
                )
        if calibration_draws is None and calibration_lookup_table is None:
            raise ValueError(
                "calibration_marginalization requires calibration_draws or "
                "calibration_lookup_table."
            )

        if calibration_draws is None:
            calibration_draws = {}
            for name in self.ifos:
                if isinstance(calibration_lookup_table, dict):
                    filename = calibration_lookup_table[name]
                else:
                    filename = calibration_lookup_table
                curves, _ = read_calibration_file(
                    filename=filename,
                    frequency_array=self._f,
                    number_of_response_curves=self.number_of_response_curves,
                    starting_index=self.starting_index,
                    correction_type=correction_type,
                )
                calibration_draws[name] = curves

        arrays = []
        n_curves = None
        for ii, name in enumerate(self.ifos):
            if isinstance(calibration_draws, dict):
                draws = calibration_draws[name]
            else:
                draws = calibration_draws[ii]
            draws = np.asarray(draws, dtype=complex)
            if draws.ndim == 1:
                draws = draws[None, :]
            if draws.ndim != 2 or draws.shape[1] != self._f.size:
                raise ValueError(
                    f"Calibration draws for {name} must have shape (n_curves, {self._f.size}); "
                    f"got {draws.shape}."
                )
            if n_curves is None:
                n_curves = draws.shape[0]
            elif draws.shape[0] != n_curves:
                raise ValueError("All detectors must use the same number of calibration curves.")
            arrays.append(draws)

        if n_curves is None or n_curves < 1:
            raise ValueError("At least one calibration curve is required.")

        self.number_of_response_curves = int(n_curves)
        self.calibration_draws = self._backend.asarray(np.stack(arrays, axis=0))

    def _signal_batch(self, p):
        return self._backend.asarray(self._template.make_injections_to_ifo_batch(p))

    def _calibrated_yy_chunks(self, signal):
        chunk = max(1, self.calibration_chunk_size)
        n_curves = self.number_of_response_curves
        for start in range(0, n_curves, chunk):
            stop = min(start + chunk, n_curves)
            yy = self._backend.zeros((signal.shape[0], stop - start, self._f.size), dtype=float)
            for ifo in range(self._nchannels):
                calibration = self.calibration_draws[ifo, start:stop, :]
                residual = (
                    self.data[ifo][None, None, :]
                    - signal[:, ifo, None, :] * calibration[None, :, :]
                )
                yy += self.xp.real(residual.conj() * residual) / self.psd[ifo][None, None, :]
            yield 4.0 * self.df * yy

    def _combine_calibration_logl(self, chunks):
        running = None
        n_draws = 0
        for logl in chunks:
            n_draws += int(logl.shape[1])
            block = self._logsumexp(logl, axis=1)
            running = block if running is None else self.xp.logaddexp(running, block)
        if running is None or n_draws < 1:
            raise ValueError("At least one calibration likelihood chunk is required.")
        return (running - np.log(n_draws)).real

    def inner_residual(self, theta):
        signal = self._template.make_injections_to_ifo(np.asarray(theta))
        residual = self._backend.zeros(self.data.shape, dtype=self.data.dtype)

        for ifo, channel in enumerate(self.ifos):
            residual[ifo, :] = self.data[ifo, :] - self._backend.asarray(signal[channel])

        yy = residual.conj() * residual
        yy = yy / self.psd
        syy = self.df * self.xp.sum(yy, axis=0)
        return 4.0 * self.xp.real(syy)

    def inner_residual_batch(self, p):
        """Vectorised noise-weighted residual for a batch ``p`` of shape ``(N, wfdims)``.

        Returns ``(N, n_freq)`` = ``4 Re[df * sum_ifo conj(r) r / Sn]`` with one
        batched template call (no per-walker loop).
        """
        signal = self._signal_batch(p)
        residual = self.data[None, :, :] - signal  # (N, n_ifo, n_freq)
        yy = (residual.conj() * residual) / self.psd[None, :, :]
        syy = self.df * self.xp.sum(yy, axis=1)  # sum over detectors -> (N, n_freq)
        return 4.0 * self.xp.real(syy)

    def _get_yy(self, p):
        p = self._ensure_2d(p)
        if self._batched_template:
            return self.inner_residual_batch(p)
        if self._use_gpu or self.num_jobs <= 1:
            residuals = [self.inner_residual(p[walker, :]) for walker in range(p.shape[0])]
        else:
            residuals = Parallel(n_jobs=self.num_jobs)(
                delayed(self.inner_residual)(p[walker, :]) for walker in range(p.shape[0])
            )
        return self._backend.asarray(residuals)

    def inner_residual_per_ifo(self, theta):
        """Single-walker noise-weighted residual, **kept per-detector**.

        Same math as :meth:`inner_residual` but does NOT sum the ``ifo`` axis;
        returns shape ``(n_ifo, n_freq)`` = ``4 Re[df * conj(r) r / Sn]``.
        Used by :meth:`hyperbolic` / :meth:`hyperbolic_classic` when
        ``shape_per_detector=True``.
        """
        signal = self._template.make_injections_to_ifo(np.asarray(theta))
        residual = self._backend.zeros(self.data.shape, dtype=self.data.dtype)
        for ifo, channel in enumerate(self.ifos):
            residual[ifo, :] = self.data[ifo, :] - self._backend.asarray(signal[channel])
        yy = (residual.conj() * residual) / self.psd
        return 4.0 * self.xp.real(self.df * yy)

    def inner_residual_batch_per_ifo(self, p):
        """Batched per-detector residual; returns shape ``(N, n_ifo, n_freq)``.

        Same as :meth:`inner_residual_batch` but with the detector axis kept.
        """
        signal = self._backend.asarray(self._template.make_injections_to_ifo_batch(p))
        residual = self.data[None, :, :] - signal  # (N, n_ifo, n_freq)
        yy = (residual.conj() * residual) / self.psd[None, :, :]
        return 4.0 * self.xp.real(self.df * yy)

    def _get_yy_per_ifo(self, p):
        """Per-detector residual driver (mirrors :meth:`_get_yy`)."""
        p = self._ensure_2d(p)
        if self._batched_template:
            return self.inner_residual_batch_per_ifo(p)
        if self._use_gpu or self.num_jobs <= 1:
            residuals = [self.inner_residual_per_ifo(p[walker, :]) for walker in range(p.shape[0])]
        else:
            residuals = Parallel(n_jobs=self.num_jobs)(
                delayed(self.inner_residual_per_ifo)(p[walker, :]) for walker in range(p.shape[0])
            )
        return self._backend.asarray(residuals)

    def gaussian(self, theta):
        theta = self._ensure_2d(theta)
        if self.calibration_marginalization:
            signal = self._signal_batch(theta[:, : self._wfdims])
            chunks = [
                -0.5 * self.xp.sum(yy, axis=-1).real
                for yy in self._calibrated_yy_chunks(signal)
            ]
            likelihood = self._combine_calibration_logl(chunks)
        else:
            likelihood = -0.5 * self.xp.sum(self._get_yy(p=theta), axis=-1).real
        likelihood = np.nan_to_num(
            self._prepare_outputs(likelihood),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return likelihood.squeeze()

    def gaussian_calmarg(self, theta):
        """Gaussian log-likelihood marginalized over detector calibration.

        Numerically marginalizes over the bank of precomputed calibration
        response curves supplied as ``calibration_bank`` at construction
        (shape ``(n_ifo, n_curves, n_freq)``, e.g. from
        :func:`hyperwave.detectors.make_calibration_bank`). This follows the LVK
        calibration-marginalization scheme (bilby's
        ``calibration_marginalization``) and adds **no** sampled dimensions.

        For curve ``C_k`` the Gaussian residual expands as

            log L_k = -1/2 <d,d> + Re<d, C_k h> - 1/2 <C_k h, C_k h>,

        evaluated for the whole walker batch in two einsums over the curve
        bank, then marginalized as ``logsumexp_k(log L_k) - log(n_curves)``.
        With a degenerate (zero-uncertainty) bank this reduces exactly to
        :meth:`gaussian`.
        """
        if self._cal_draws is None:
            raise RuntimeError(
                "gaussian_calmarg requires a calibration_bank passed to "
                "GWLikelihoods(...). Build one with "
                "hyperwave.detectors.make_calibration_bank(...)."
            )
        theta = self._ensure_2d(theta)
        signal = self._backend.asarray(
            self._template.make_injections_to_ifo_batch(theta[:, : self._wfdims])
        )  # (N, n_ifo, n_freq)
        four_df = 4.0 * self.df
        dh = four_df * (self.data.conj()[None, :, :] * signal) / self.psd[None, :, :]
        hh = four_df * (signal.conj() * signal).real / self.psd[None, :, :]
        # Inner products contracted against every calibration curve at once.
        dh_arr = self.xp.einsum("nif,ikf->nk", dh, self._cal_draws)      # (N, n_curves)
        hh_arr = self.xp.einsum("nif,ikf->nk", hh, self._cal_abs2)       # (N, n_curves)
        const = self.xp.sum(self.yy_noise)  # scalar  -1/2 <d,d>
        logl_k = const + dh_arr.real - 0.5 * hh_arr
        logl = self._logsumexp(logl_k, axis=1) - np.log(self._n_curves)
        logl = np.nan_to_num(
            self._prepare_outputs(logl),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return logl.squeeze()

    # -- calibration-marginalized whittle / hyperbolic --------------------
    #
    # Unlike the Gaussian case, these likelihoods are not quadratic in the
    # residual, so the calibration cannot be contracted away into a pair of
    # scalar inner products. Instead we form the per-curve, per-frequency
    # noise-weighted residual
    #
    #     yy_k[f] = 4 df sum_ifo |d - C_k h|^2 / Sn
    #             = 4 df sum_ifo (|d|^2 - 2 Re(d* h C_k) + |h|^2 |C_k|^2) / Sn,
    #
    # feed it through the usual per-segment kernel for every curve, and
    # marginalize with logsumexp. Curves are processed in chunks (``cal_chunk``)
    # with an online logsumexp so memory stays bounded at ``(N, chunk, n_freq)``.

    def _require_bank(self, name):
        if self._cal_draws is None:
            raise RuntimeError(
                f"{name} requires a calibration_bank passed to GWLikelihoods(...). "
                "Build one with hyperwave.detectors.make_calibration_bank(...)."
            )
        if self._shape_per_detector:
            raise NotImplementedError(
                f"{name} is not yet implemented for shape_per_detector=True. "
                "Use the non-calmarg likelihood (hyperbolic / hyperbolic_classic) "
                "or run with the shared shape parameters."
            )

    def _calmarg(self, theta, kernel):
        """Marginalize ``kernel`` over the calibration curve bank.

        ``kernel(theta2, yy)`` maps a ``(N, K, n_freq)`` per-curve residual to a
        ``(N, K)`` log-likelihood; this returns the ``logsumexp_K - log(n)``
        marginal as a host array.
        """
        theta2 = self._ensure_2d(theta)
        signal = self._backend.asarray(
            self._template.make_injections_to_ifo_batch(theta2[:, : self._wfdims])
        )  # (N, n_ifo, n_freq)
        # Cross / power integrands, computed once and contracted per curve chunk.
        cross = (self.data.conj()[None, :, :] * signal) / self.psd[None, :, :]   # (N, n_ifo, n_freq)
        power = (signal.conj() * signal).real / self.psd[None, :, :]             # (N, n_ifo, n_freq)
        dd = self.xp.sum((self.data.conj() * self.data).real / self.psd, axis=0)  # (n_freq,)
        four_df = 4.0 * self.df

        running = None
        for start in range(0, self._n_curves, self._cal_chunk):
            stop = min(start + self._cal_chunk, self._n_curves)
            cdraws = self._cal_draws[:, start:stop, :]
            cabs2 = self._cal_abs2[:, start:stop, :]
            a1 = self.xp.einsum("nif,ikf->nkf", cross, cdraws)   # (N, K, n_freq) complex
            a2 = self.xp.einsum("nif,ikf->nkf", power, cabs2)    # (N, K, n_freq) real
            yy = four_df * (dd[None, None, :] - 2.0 * a1.real + a2)
            ll = kernel(theta2, yy)                               # (N, K)
            block = self._logsumexp(ll, axis=1)                  # (N,)
            running = block if running is None else self.xp.logaddexp(running, block)

        logl = running - np.log(self._n_curves)
        logl = np.nan_to_num(
            self._prepare_outputs(logl),
            copy=True, nan=self._inf, posinf=self._inf, neginf=self._inf,
        )
        return logl.squeeze()

    def _whittle_kernel(self, theta, yy):
        """Per-segment whittle log-likelihood for ``yy`` of shape ``(N, K, n_freq)``."""
        ll = self.xp.zeros(yy.shape[:2])
        for i, si in enumerate(self._segi):
            level = self._backend.asarray(10.0 ** theta[:, self._wfdims + i])[:, None, None]
            const = 0.0 if self.whiten is None else 2 * self.logwhiten[None, None, si]
            ll = ll + self.xp.sum(
                -0.5 * yy[:, :, si] / level - self._nchannels * self.xp.log(level) + const,
                axis=-1,
            ).real
        return ll

    def _hyperbolic_kernel(self, theta, yy, classic=False):
        """Per-segment hyperbolic log-likelihood for ``yy`` of shape ``(N, K, n_freq)``."""
        alpha = self._backend.asarray(self._alpha_columns(theta))
        tail = self._backend.asarray(theta[:, self._hdims :])  # ratio (classic=False) or delta
        ll = self.xp.zeros(yy.shape[:2])
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i] if self._ddims else alpha[:, 0]
            delta_i = tail[:, i] if classic else alpha_i * tail[:, i]
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(
                self.xp.sqrt(delta_i[:, None, None] ** 2 + yy[:, :, si]).real, axis=-1
            )
            term_lambda = self._lam * self.xp.log(alpha_i / delta_i)
            term_rest = self._C0 - self.xp.log(2.0 * alpha_i) - log_kappa
            ll = ll + self._Nd[i] * (term_lambda + term_rest)[:, None] - alpha_i[:, None] * term_sqrt
        return ll

    def whittle_calmarg(self, theta):
        """Calibration-marginalized per-segment whittle log-likelihood.

        Same parameter layout as :meth:`whittle_level` (calibration is
        marginalized, not sampled). With a degenerate bank reduces to it.
        """
        self._require_bank("whittle_calmarg")
        return self._calmarg(theta, self._whittle_kernel)

    def hyperbolic_calmarg(self, theta):
        """Calibration-marginalized hyperbolic log-likelihood (alpha/ratio form).

        Same parameter layout as :meth:`hyperbolic`. With a degenerate bank
        reduces to it.
        """
        self._require_bank("hyperbolic_calmarg")
        return self._calmarg(theta, lambda th, yy: self._hyperbolic_kernel(th, yy, classic=False))

    def hyperbolic_classic_calmarg(self, theta):
        """Calibration-marginalized hyperbolic log-likelihood (alpha/delta form).

        Same parameter layout as :meth:`hyperbolic_classic`. With a degenerate
        bank reduces to it.
        """
        self._require_bank("hyperbolic_classic_calmarg")
        return self._calmarg(theta, lambda th, yy: self._hyperbolic_kernel(th, yy, classic=True))

    # -- calibration decision diagnostics --------------------------------
    #
    # Two cheap (no-sampling) per-event screens to decide whether running the
    # expensive *_calmarg likelihood is worth it, and if so with how many curves:
    #   * calibration_worth_it : Fisher / Cutler-Vallisneri systematic-bias ratio
    #     sigma_sys / sigma_stat per parameter. >~0.2 -> calibration matters.
    #   * calibration_ess      : effective sample size of the marginalization;
    #     small ESS -> the bank under-samples the prior, raise n_curves.

    def _per_curve_loglike(self, theta, kind):
        """Per-curve log-likelihood ``logL_k`` of shape ``(N, n_curves)`` (no chunking).

        Used by the diagnostics, which need the full per-curve array at one (or a
        few) ``theta`` rather than the marginalized scalar.
        """
        theta = self._ensure_2d(theta)
        signal = self._backend.asarray(
            self._template.make_injections_to_ifo_batch(theta[:, : self._wfdims])
        )  # (N, n_ifo, n_freq)
        four_df = 4.0 * self.df
        if kind == "gaussian":
            dh = four_df * (self.data.conj()[None, :, :] * signal) / self.psd[None, :, :]
            hh = four_df * (signal.conj() * signal).real / self.psd[None, :, :]
            dh_arr = self.xp.einsum("nif,ikf->nk", dh, self._cal_draws)
            hh_arr = self.xp.einsum("nif,ikf->nk", hh, self._cal_abs2)
            return self.xp.sum(self.yy_noise) + dh_arr.real - 0.5 * hh_arr
        cross = (self.data.conj()[None, :, :] * signal) / self.psd[None, :, :]
        power = (signal.conj() * signal).real / self.psd[None, :, :]
        dd = self.xp.sum((self.data.conj() * self.data).real / self.psd, axis=0)
        a1 = self.xp.einsum("nif,ikf->nkf", cross, self._cal_draws)
        a2 = self.xp.einsum("nif,ikf->nkf", power, self._cal_abs2)
        yy = four_df * (dd[None, None, :] - 2.0 * a1.real + a2)
        if kind == "whittle":
            return self._whittle_kernel(theta, yy)
        if kind == "hyperbolic":
            return self._hyperbolic_kernel(theta, yy, classic=False)
        if kind == "hyperbolic_classic":
            return self._hyperbolic_kernel(theta, yy, classic=True)
        raise ValueError(
            f"unknown kind {kind!r}; expected gaussian/whittle/hyperbolic/hyperbolic_classic."
        )

    def calibration_ess(self, theta, kind="gaussian"):
        """Effective sample size of the calibration marginalization at ``theta``.

        ``ESS = 1 / sum_k w_k^2`` with ``w_k = softmax_k(logL_k)`` — how many of
        the ``n_curves`` bank curves effectively carry the marginalization
        weight. A small ESS (rule of thumb: below a few hundred) means the bank
        under-samples the calibration prior and the marginal likelihood is noisy
        / biased low, so ``n_curves`` should be raised. ``ESS`` equals
        ``n_curves`` for a degenerate (zero-uncertainty) bank.

        ``kind`` selects the likelihood; for the non-Gaussian kinds ``theta``
        must include the noise-shape parameters (same layout as the matching
        ``*_calmarg`` method). Returns ESS per walker if ``theta`` is batched.
        """
        self._require_bank("calibration_ess")
        logl_k = self._per_curve_loglike(theta, kind)            # (N, n_curves)
        m = self.xp.max(logl_k, axis=1, keepdims=True)
        w = self.xp.exp(logl_k - m)
        w = w / self.xp.sum(w, axis=1, keepdims=True)
        ess = 1.0 / self.xp.sum(w * w, axis=1)
        return self._prepare_outputs(ess).squeeze()

    def _cv_inner(self, a, b):
        """Real noise-weighted inner product over ``(n_ifo, n_freq)``: ``4 df Re sum conj(a) b / Sn``."""
        val = 4.0 * self.df * self.xp.real(self.xp.sum((a.conj() * b) / self.psd))
        return float(self._backend.to_numpy(val)) if self._use_gpu else float(val)

    def calibration_worth_it(
        self, theta, params_of_interest=None, rel_step=1e-5, abs_step=1e-7, threshold=0.2
    ):
        """Forecast whether calibration marginalization is worth it at ``theta``.

        Linear (Fisher / Cutler-Vallisneri) screen, no sampling. For each science
        parameter it compares the calibration-induced systematic spread to the
        statistical width::

            Gamma_ij      = (d_i h | d_j h)                      # Fisher matrix
            sigma_stat_i  = sqrt((Gamma^-1)_ii)                  # statistical width
            dtheta_k      = Gamma^-1 (d h | (C_k - 1) h)         # bias from ignoring curve C_k
            sigma_sys_i   = std_k(dtheta_k,i)                    # calibration systematic
            R_i           = sigma_sys_i / sigma_stat_i

        ``R >~ threshold`` (default 0.2) for any parameter of interest means the
        calibration error is a non-negligible fraction of the statistical error,
        so the (expensive) ``*_calmarg`` likelihood is worth running; ``R << 0.1``
        everywhere means it only widens posteriors imperceptibly and can be
        skipped. Waveform derivatives use central finite differences with step
        ``rel_step*|theta_i| + abs_step`` — evaluate at a representative best-fit
        point; parameters pinned at a boundary (e.g. ``cos_tilt = 1``) give
        unreliable Fisher entries (regularized via pseudo-inverse).

        Returns a dict with per-parameter ``R``/``sigma_stat``/``sigma_sys``, the
        ``max_ratio`` over ``params_of_interest`` (default: all), and a
        ``worth_it`` boolean.
        """
        self._require_bank("calibration_worth_it")
        theta = np.atleast_1d(np.asarray(theta, dtype=float))[: self._wfdims]
        names = list(self._template.parameters)
        nwf = self._wfdims

        def waveform(th):
            return self._backend.asarray(
                self._template.make_injections_to_ifo_batch(th[None, :])[0]
            )  # (n_ifo, n_freq)

        h0 = waveform(theta)
        derivs = []
        for i in range(nwf):
            step = rel_step * abs(theta[i]) + abs_step
            tp = theta.copy()
            tp[i] += step
            tm = theta.copy()
            tm[i] -= step
            derivs.append((waveform(tp) - waveform(tm)) / (2.0 * step))

        gamma = np.zeros((nwf, nwf))
        for i in range(nwf):
            for j in range(i, nwf):
                gamma[i, j] = gamma[j, i] = self._cv_inner(derivs[i], derivs[j])
        ginv = np.linalg.pinv(gamma)
        sigma_stat = np.sqrt(np.clip(np.diag(ginv), 0.0, None))

        # Systematic shift per curve, vectorised over the bank (chunked).
        D = self.xp.stack(derivs, axis=0)                 # (nwf, n_ifo, n_freq)
        Dw = D.conj() / self.psd[None, :, :]
        b = np.zeros((self._n_curves, nwf))
        for start in range(0, self._n_curves, self._cal_chunk):
            stop = min(start + self._cal_chunk, self._n_curves)
            dH = (self._cal_draws[:, start:stop, :] - 1.0) * h0[:, None, :]  # (n_ifo, K, n_freq)
            bb = 4.0 * self.df * self.xp.real(self.xp.einsum("jif,ikf->kj", Dw, dH))
            b[start:stop] = self._backend.to_numpy(bb) if self._use_gpu else np.asarray(bb)
        dtheta = b @ ginv                                  # (n_curves, nwf), ginv symmetric
        sigma_sys = dtheta.std(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sigma_stat > 0, sigma_sys / sigma_stat, np.inf)

        ratios = {
            names[i]: {
                "R": float(ratio[i]),
                "sigma_stat": float(sigma_stat[i]),
                "sigma_sys": float(sigma_sys[i]),
            }
            for i in range(nwf)
        }
        poi = [p for p in (params_of_interest or names) if p in ratios]
        max_ratio = max((ratios[p]["R"] for p in poi), default=0.0)
        return {
            "ratios": ratios,
            "params_of_interest": poi,
            "max_ratio": float(max_ratio),
            "worth_it": bool(max_ratio >= threshold),
            "threshold": float(threshold),
        }

    # -- calibration SAMPLING (Method A) ---------------------------------
    #
    # Instead of marginalizing over a fixed bank, the spline calibration nodes
    # are sampled as extra dimensions, appended LAST in ``theta``:
    #     theta = [ physics params | (noise-shape params) | calibration nodes ]
    # with ``calibration nodes`` ordered ``recalib_{ifo}_amplitude_{0..n-1}``
    # then ``recalib_{ifo}_phase_{0..n-1}`` per detector (use
    # :func:`hyperwave.inference.calibration_node_priors` to build matching
    # priors in the same order). Each walker gets its own response C(f), applied
    # to the template; the rest of the likelihood is the usual per-segment math.
    # With all nodes at zero (C == 1) this reduces exactly to the base method.

    def _require_spline(self, name):
        if self._cal_spline is None:
            raise RuntimeError(
                f"{name} requires cal_n_nodes to be set on GWLikelihoods(...). "
                "Pass cal_n_nodes=<int> (and matching calibration_node_priors)."
            )
        if self._shape_per_detector:
            raise NotImplementedError(
                f"{name} is not yet implemented for shape_per_detector=True. "
                "Use the non-calsample likelihood (hyperbolic / hyperbolic_classic) "
                "or run with the shared shape parameters."
            )

    def _calibration_factor(self, cal):
        """Per-walker response ``C(f)`` from sampled nodes ``cal`` of shape ``(N, caldims)``.

        Returns ``(N, n_ifo, n_freq)`` complex.
        """
        n = cal.shape[0]
        nn = self._cal_spline.n_nodes
        c4 = np.asarray(cal).reshape(n, self._nchannels, 2, nn)
        return self._backend.asarray(self._cal_spline.factor(c4[:, :, 0, :], c4[:, :, 1, :]))

    def _calsample_yy(self, theta):
        """Split ``theta`` into physics/cal, apply ``C(f)``, return ``(phys, yy)``.

        ``yy = 4 df sum_ifo |d - C h|^2 / Sn`` of shape ``(N, n_freq)``; ``phys``
        is ``theta`` with the calibration columns stripped (the layout the base
        kernels expect).
        """
        phys = theta[:, : -self._caldims]
        cal = theta[:, -self._caldims :]
        signal = self._backend.asarray(
            self._template.make_injections_to_ifo_batch(phys[:, : self._wfdims])
        )
        signal = signal * self._calibration_factor(cal)
        residual = self.data[None, :, :] - signal
        yy = (residual.conj() * residual) / self.psd[None, :, :]
        return phys, 4.0 * self.df * self.xp.real(self.xp.sum(yy, axis=1))

    def _finish(self, logl):
        logl = np.nan_to_num(
            self._prepare_outputs(logl),
            copy=True, nan=self._inf, posinf=self._inf, neginf=self._inf,
        )
        return logl.squeeze()

    def gaussian_calsample(self, theta):
        """Gaussian log-likelihood with sampled calibration nodes (Method A)."""
        self._require_spline("gaussian_calsample")
        _, yy = self._calsample_yy(self._ensure_2d(theta))
        return self._finish(-0.5 * self.xp.sum(yy, axis=-1))

    def whittle_calsample(self, theta):
        """Per-segment whittle log-likelihood with sampled calibration nodes."""
        self._require_spline("whittle_calsample")
        phys, yy = self._calsample_yy(self._ensure_2d(theta))
        return self._finish(self._whittle_kernel(phys, yy[:, None, :])[:, 0])

    def hyperbolic_calsample(self, theta):
        """Hyperbolic (alpha/ratio) log-likelihood with sampled calibration nodes."""
        self._require_spline("hyperbolic_calsample")
        phys, yy = self._calsample_yy(self._ensure_2d(theta))
        return self._finish(self._hyperbolic_kernel(phys, yy[:, None, :], classic=False)[:, 0])

    def hyperbolic_classic_calsample(self, theta):
        """Hyperbolic (alpha/delta) log-likelihood with sampled calibration nodes."""
        self._require_spline("hyperbolic_classic_calsample")
        phys, yy = self._calsample_yy(self._ensure_2d(theta))
        return self._finish(self._hyperbolic_kernel(phys, yy[:, None, :], classic=True)[:, 0])

    def whittle_level(self, theta):
        theta = self._ensure_2d(theta)
        if self.calibration_marginalization:
            return self._whittle_level_calibration_marginalized(theta)

        self.yy = self._get_yy(p=theta[:, : self._wfdims])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            level = self._backend.asarray(10.0 ** theta[:, self._wfdims + i])[:, None]
            const = 0.0 if self.whiten is None else 2 * self.logwhiten[si]
            likelihood[:, i] = self.xp.sum(
                -0.5 * self.yy[:, si] / level - self._nchannels * self.xp.log(level) + const,
                axis=-1,
            ).real

        likelihood = np.nan_to_num(
            self._prepare_outputs(self.xp.sum(likelihood, axis=-1)),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return likelihood.squeeze()

    def _hyperbolic_per_detector(self, theta, classic):
        """Per-detector hyperbolic log-likelihood (sum of independent MVH_2 per ifo).

        Used when ``shape_per_detector=True``. Each interferometer is modeled as
        its own bivariate-MVH (one complex frequency channel), with its own
        ``α``, ``δ``; the total log-likelihood is the sum over detectors and
        segments. Reduces to the standard single-detector hyperbolic likelihood
        when ``n_ifo == 1``.
        """
        theta = self._ensure_2d(theta)
        alpha = self._backend.asarray(self._alpha_columns(theta))  # (N, nsegs, n_ifo)
        tail = self._backend.asarray(self._tail_columns(theta))    # (N, nsegs, n_ifo)
        self.yy_pd = self._get_yy_per_ifo(p=theta[:, : self._wfdims])  # (N, n_ifo, n_freq)

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i, :]                       # (N, n_ifo)
            delta_i = tail[:, i, :] if classic else alpha_i * tail[:, i, :]
            alpha_delta_i = alpha_i * delta_i              # (N, n_ifo)

            log_kappa = self._backend.log_kv(self._lam_pd, alpha_delta_i)  # (N, n_ifo)
            # yy_pd[:, :, si] is (N, n_ifo, len(si)); broadcast δ_i (N, n_ifo, 1).
            term_sqrt = self.xp.sum(
                self.xp.sqrt(delta_i[:, :, None] ** 2 + self.yy_pd[:, :, si]).real,
                axis=-1,
            )                                              # (N, n_ifo)
            term_lambda = self._lam_pd * self.xp.log(alpha_i / delta_i)        # (N, n_ifo)
            term_rest = self._C0_pd - self.xp.log(2.0 * alpha_i) - log_kappa   # (N, n_ifo)

            ll_seg = self._Nd[i] * (term_lambda + term_rest) - alpha_i * term_sqrt
            likelihood[:, i] = self.xp.sum(ll_seg, axis=-1)  # sum over ifo

        likelihood = self.xp.nan_to_num(
            self.xp.sum(likelihood, axis=-1),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()

    def hyperbolic(self, theta):
        if self._shape_per_detector:
            return self._hyperbolic_per_detector(theta, classic=False)
        theta = self._ensure_2d(theta)
        if self.calibration_marginalization:
            return self._hyperbolic_calibration_marginalized(theta, classic=False)

        alpha = self._backend.asarray(self._alpha_columns(theta))
        ratio = self._backend.asarray(theta[:, self._hdims :])
        self.yy = self._get_yy(p=theta[:, : self._wfdims])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i] if self._ddims else alpha[:, 0]
            ratio_i = ratio[:, i]
            delta_i = alpha_i * ratio_i
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy[:, si]).real, axis=-1)
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

    def hyperbolic_classic(self, theta):
        if self._shape_per_detector:
            return self._hyperbolic_per_detector(theta, classic=True)
        theta = self._ensure_2d(theta)
        if self.calibration_marginalization:
            return self._hyperbolic_calibration_marginalized(theta, classic=True)

        alpha = self._backend.asarray(self._alpha_columns(theta))
        delta = self._backend.asarray(theta[:, self._hdims :])
        self.yy = self._get_yy(p=theta[:, : self._wfdims])

        likelihood = self.xp.zeros((theta.shape[0], len(self._segi)))
        for i, si in enumerate(self._segi):
            alpha_i = alpha[:, i] if self._ddims else alpha[:, 0]
            delta_i = delta[:, i]
            alpha_delta_i = alpha_i * delta_i

            log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
            term_sqrt = self.xp.sum(self.xp.sqrt(delta_i[:, None] ** 2 + self.yy[:, si]).real, axis=-1)
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

    def _whittle_level_calibration_marginalized(self, theta):
        signal = self._signal_batch(theta[:, : self._wfdims])
        chunks = []
        for yy in self._calibrated_yy_chunks(signal):
            likelihood = self.xp.zeros((theta.shape[0], yy.shape[1], len(self._segi)))
            for i, si in enumerate(self._segi):
                level = self._backend.asarray(10.0 ** theta[:, self._wfdims + i])
                term = -0.5 * yy[:, :, si] / level[:, None, None]
                term = term - self._nchannels * self.xp.log(level)[:, None, None]
                if self.whiten is not None:
                    term = term + 2 * self.logwhiten[si][None, None, :]
                likelihood[:, :, i] = self.xp.sum(term, axis=-1).real
            chunks.append(self.xp.sum(likelihood, axis=-1))

        likelihood = self._combine_calibration_logl(chunks)
        likelihood = np.nan_to_num(
            self._prepare_outputs(likelihood),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return likelihood.squeeze()

    def _hyperbolic_calibration_marginalized(self, theta, classic):
        alpha = self._backend.asarray(self._alpha_columns(theta))
        tail = self._backend.asarray(theta[:, self._hdims :])
        signal = self._signal_batch(theta[:, : self._wfdims])

        chunks = []
        for yy in self._calibrated_yy_chunks(signal):
            likelihood = self.xp.zeros((theta.shape[0], yy.shape[1], len(self._segi)))
            for i, si in enumerate(self._segi):
                alpha_i = alpha[:, i] if self._ddims else alpha[:, 0]
                delta_i = tail[:, i] if classic else alpha_i * tail[:, i]
                alpha_delta_i = alpha_i * delta_i

                log_kappa = self._backend.log_kv(self._lam, alpha_delta_i)
                term_sqrt = self.xp.sum(
                    self.xp.sqrt(delta_i[:, None, None] ** 2 + yy[:, :, si]).real,
                    axis=-1,
                )
                term_lambda = self._lam * self.xp.log(alpha_i / delta_i)
                term_rest = self._C0 - self.xp.log(2.0 * alpha_i) - log_kappa
                likelihood[:, :, i] = (
                    self._Nd[i] * (term_lambda + term_rest)[:, None]
                    - alpha_i[:, None] * term_sqrt
                )
            chunks.append(self.xp.sum(likelihood, axis=-1))

        likelihood = self._combine_calibration_logl(chunks)
        likelihood = self.xp.nan_to_num(
            likelihood,
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        )
        return self._prepare_outputs(likelihood).squeeze()


__all__ = ["GWLikelihoods", "gpu_backend_available"]
