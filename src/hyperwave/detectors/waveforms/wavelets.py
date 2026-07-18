"""Morlet-Gabor wavelet waveforms for waveform-agnostic GW reconstruction.

This implements the sine-Gaussian (Morlet-Gabor) wavelet frame (Cornish &
Littenberg 2015) as a *GW waveform*: each wavelet is generated analytically in
the frequency domain, a coherent signal is built from a (variable) sum of
wavelets under the elliptical-polarisation constraint, and the result is
projected onto the detector network with HyperWave's lal-backed geometry. It is
designed to be driven by Eryn's RJMCMC over the number of wavelets, with the
induced-SNR amplitude prior keeping the model from latching onto noise (see
:mod:`hyperwave.inference.wavelet_priors`).

Per-wavelet parameters ``(t0, f0, Q, amplitude, phi0)``:

* ``t0``   - central time, **seconds from the segment start** (geocentre)
* ``f0``   - central frequency [Hz]
* ``Q``    - quality factor; sets ``tau = Q / (2 pi f0)``
* ``amplitude`` - either the strain amplitude ``A`` or, with
  ``amplitude_param="snr"``, the per-wavelet network optimal SNR (converted to
  ``A`` using the noise PSD at ``f0``)
* ``phi0`` - phase

Frequency-domain wavelet (continuous-FT convention, matching
:func:`hyperwave.detectors.strain.nfft`)::

    tau   = Q / (2 pi f0)
    Psi(f) = (A tau sqrt(pi) / 2) e^{-2i pi f t0}
             [ e^{+i phi0} e^{-pi^2 tau^2 (f-f0)^2}
             + e^{-i phi0} e^{-pi^2 tau^2 (f+f0)^2} ]

Elliptical signal model (shared ellipticity ``epsilon`` for the whole signal)::

    h_plus  = sum_i Psi_i(phi0_i)
    h_cross = epsilon * sum_i Psi_i(phi0_i + pi/2)
"""

from __future__ import annotations

import numpy as np

from ...backends import get_array_backend
from ..geometry import get_detector


def _scatter_add(xp, out, index, values):
    """Backend-agnostic ``out[index] += values`` with accumulation.

    ``index`` is 1-D (length M); ``values`` is ``(M, ...)`` and ``out`` is
    ``(G, ...)``. Works for NumPy and CuPy.
    """
    if xp is np:
        np.add.at(out, index, values)
        return out
    try:  # CuPy
        import cupyx

        if values.dtype.kind == "c":
            # cupyx.scatter_add supports real dtypes only; scatter the real and
            # imaginary views separately. (Letting the complex call raise into
            # the per-group loop below costs ~250 ms/call instead of ~1 ms — it
            # silently dominated the entire wavelet-RJMCMC runtime.)
            cupyx.scatter_add(out.real, index, values.real)
            cupyx.scatter_add(out.imag, index, values.imag)
        else:
            cupyx.scatter_add(out, index, values)
    except Exception:  # pragma: no cover - fallback if cupyx.scatter_add missing
        for g in range(out.shape[0]):
            sel = index == g
            if bool(sel.any()):
                out[g] = out[g] + values[sel].sum(axis=0)
    return out

#: per-wavelet (leaf) parameter order
WAVELET_PARAMETERS = ("t0", "f0", "Q", "amplitude", "phi0")
#: shared extrinsic parameters of the coherent signal
EXTRINSIC_PARAMETERS = ("ra", "dec", "psi", "ellipticity")

_SQRT_PI = np.sqrt(np.pi)
_SQRT_PI_OVER_2 = np.sqrt(np.pi / 2.0)


def morlet_gabor_fd(frequency_array, t0, f0, Q, amplitude, phi0):
    """Frequency-domain Morlet-Gabor wavelet.

    All wavelet parameters broadcast against each other; the returned array has
    shape ``(*broadcast_shape, n_freq)``.
    """
    f = np.asarray(frequency_array, dtype=float)
    t0, f0, Q, amp, phi = (np.asarray(x, dtype=float) for x in (t0, f0, Q, amplitude, phi0))
    tau = Q / (2.0 * np.pi * f0)

    tau_ = tau[..., None]
    f0_ = f0[..., None]
    t0_ = t0[..., None]
    amp_ = amp[..., None]
    phi_ = phi[..., None]

    prefactor = amp_ * tau_ * _SQRT_PI / 2.0
    lobe_minus = np.exp(-(np.pi**2) * tau_**2 * (f - f0_) ** 2)
    lobe_plus = np.exp(-(np.pi**2) * tau_**2 * (f + f0_) ** 2)
    phase = np.exp(-2j * np.pi * f * t0_)
    return prefactor * phase * (
        np.exp(1j * phi_) * lobe_minus + np.exp(-1j * phi_) * lobe_plus
    )


def amplitude_from_snr(snr, f0, Q, psd_at_f0):
    """Convert a wavelet optimal SNR to its strain amplitude.

    Inverts ``rho^2 = A^2 Q sqrt(pi/2) / (2 pi f0 S(f0))`` (single Morlet-Gabor
    wavelet against a one-sided PSD ``S``).
    """
    snr = np.asarray(snr, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    Q = np.asarray(Q, dtype=float)
    psd_at_f0 = np.asarray(psd_at_f0, dtype=float)
    return snr * np.sqrt(2.0 * np.pi * f0 * psd_at_f0 / (Q * _SQRT_PI_OVER_2))


def snr_from_amplitude(amplitude, f0, Q, psd_at_f0):
    """Inverse of :func:`amplitude_from_snr`."""
    amplitude = np.asarray(amplitude, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    Q = np.asarray(Q, dtype=float)
    psd_at_f0 = np.asarray(psd_at_f0, dtype=float)
    return amplitude * np.sqrt(Q * _SQRT_PI_OVER_2 / (2.0 * np.pi * f0 * psd_at_f0))


def ellipticity_from_ecc(ecc):
    """Ellipticity parameter: ``epsilon = 2 ecc / (1 + ecc^2)``.

    Maps ``ecc in [-1, 1]`` monotonically onto ``epsilon in [-1, 1]``. Sampling
    ``ellipticity`` directly in ``[-1, 1]`` is equivalent and simpler; this
    helper is provided for parity with chains that store ``ecc``.
    """
    ecc = np.asarray(ecc, dtype=float)
    return 2.0 * ecc / (1.0 + ecc * ecc)


def network_optimal_snr(signal, psd, df):
    """Coherent network optimal SNR of a projected signal.

    ``signal`` and ``psd`` are ``(n_ifo, n_freq)`` (one-sided); returns
    ``sqrt(sum_ifo 4 df sum_f |signal|^2 / psd)``.
    """
    signal = np.asarray(signal)
    psd = np.asarray(psd, dtype=float)
    inner = 4.0 * df * np.sum(np.abs(signal) ** 2 / psd)
    return float(np.sqrt(inner.real))


class WaveletTemplate:
    """Coherent GW wavelet template: build + project a sum of Morlet-Gabor wavelets.

    Parameters
    ----------
    detectors:
        Detector prefixes, e.g. ``["H1", "L1"]``.
    frequency_array:
        Full one-sided analysis frequency grid (``df = 1 / duration``).
    duration, start_time:
        Segment duration [s] and GPS start time (``t0`` is measured from here).
    minimum_frequency, maximum_frequency:
        Analysis band; the projected output is restricted to it.
    reference_time:
        GPS time used for the (slowly varying) antenna response and geocentric
        delay. Defaults to the segment centre.
    psd:
        One-sided PSD ``(n_ifo, n_freq_masked)`` on the analysis band, required
        when ``amplitude_param="snr"`` to map SNR -> amplitude. The reference
        curve used is the network harmonic sum ``1 / sum_ifo (1 / S_ifo)``.
    amplitude_param:
        ``"snr"`` (default, induced-SNR prior) or ``"amplitude"``.
    n_wavelets:
        Fixed number of wavelets for the flat ``make_injections_to_ifo_batch``
        path (used for fixed-dimension inference and tests). RJMCMC uses
        :meth:`project_batch` with an explicit ``inds`` mask instead.
    """

    def __init__(
        self,
        detectors,
        frequency_array,
        duration,
        start_time,
        minimum_frequency=20.0,
        maximum_frequency=None,
        reference_time=None,
        psd=None,
        amplitude_param="snr",
        n_wavelets=1,
        gpu=False,
    ):
        self._backend = get_array_backend(gpu=gpu)
        self.xp = self._backend.xp
        self.backend_name = self._backend.name
        # Strings resolve to LVK detectors; duck-typed objects (with
        # ``antenna_response`` + ``time_delay_from_geocenter``) pass through,
        # so the same machinery reconstructs ANY whitened series (LISA TDI
        # channels, ECG, ...) via a unit-response "detector".
        self.detector_names = [getattr(d, "name", str(d)) for d in detectors]
        self.detectors = [d if hasattr(d, "antenna_response") else get_detector(str(d))
                          for d in detectors]
        self.frequency_array = np.asarray(frequency_array, dtype=float)
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = (
            float(self.frequency_array[-1]) if maximum_frequency is None else float(maximum_frequency)
        )
        self.reference_time = (
            float(reference_time) if reference_time is not None
            else self.start_time + 0.5 * self.duration
        )
        self.amplitude_param = str(amplitude_param).lower()
        if self.amplitude_param not in {"snr", "amplitude"}:
            raise ValueError("amplitude_param must be 'snr' or 'amplitude'.")
        self.n_wavelets = int(n_wavelets)

        self.mask = (self.frequency_array >= self.minimum_frequency) & (
            self.frequency_array <= self.maximum_frequency
        )
        self._f = self.frequency_array[self.mask]
        self.df = self.frequency_array[1] - self.frequency_array[0]

        # Reference (network) PSD on the analysis band for SNR <-> amplitude.
        self._reference_psd = None
        if psd is not None:
            psd = np.asarray(psd, dtype=float)
            if psd.ndim == 1:
                psd = psd[None, :]
            if psd.shape[-1] != self._f.size:
                raise ValueError(
                    f"psd last axis ({psd.shape[-1]}) must match the analysis band "
                    f"({self._f.size}); pass the same masked PSD used by the likelihood."
                )
            with np.errstate(divide="ignore"):
                self._reference_psd = 1.0 / np.sum(1.0 / psd, axis=0)

        # flat parameter layout for the GWLikelihoods-compatible path
        self.parameters = [
            f"{name}_{i}" for i in range(self.n_wavelets) for name in WAVELET_PARAMETERS
        ] + list(EXTRINSIC_PARAMETERS)

        # device-resident copies for the batched/grouped GPU path (contiguous;
        # cupy.interp requires C-contiguous inputs)
        self._f_xp = self.xp.ascontiguousarray(self.xp.asarray(self._f))
        self._refpsd_xp = (
            None if self._reference_psd is None
            else self.xp.ascontiguousarray(self.xp.asarray(self._reference_psd))
        )

    # -- public grid / device accessors ----------------------------------
    @property
    def is_gpu(self):
        """True when the array backend runs on the GPU (CuPy)."""
        return bool(getattr(self._backend, "use_gpu", False))

    def to_numpy(self, x):
        """Return ``x`` as a host NumPy array (copies off the device on GPU)."""
        return self._backend.to_numpy(x) if self.is_gpu else np.asarray(x)

    @property
    def band_frequencies(self):
        """Masked analysis-band frequency array ``(n_freq,)``."""
        return self._f

    @property
    def full_frequencies(self):
        """Full one-sided frequency grid ``[0 .. f_Nyquist]``."""
        return self.frequency_array

    @property
    def band_mask(self):
        """Boolean mask selecting the analysis band on the full grid."""
        return self.mask

    @property
    def sampling_frequency(self):
        """Sampling rate implied by the frequency grid (``2 f_Nyquist = N df``)."""
        return 2.0 * float(self.frequency_array[-1])

    @property
    def dt(self):
        """Time-domain sample spacing ``1 / sampling_frequency``."""
        return 1.0 / self.sampling_frequency

    # -- amplitude handling ----------------------------------------------
    def _psd_at(self, f0):
        if self._reference_psd is None:
            raise ValueError(
                "amplitude_param='snr' requires a PSD; pass psd=(n_ifo, n_freq) to WaveletTemplate."
            )
        return np.interp(np.asarray(f0, dtype=float), self._f, self._reference_psd)

    def _strain_amplitude(self, amplitude, f0, Q):
        if self.amplitude_param == "amplitude":
            return np.asarray(amplitude, dtype=float)
        return amplitude_from_snr(amplitude, f0, Q, self._psd_at(f0))

    # -- polarisations ----------------------------------------------------
    def _wavelet_sum(self, wavelets, active):
        """Sum active wavelets -> ``(N, n_freq_masked)`` on the analysis band.

        ``wavelets`` is ``(N, L, 5)`` with columns ``(t0, f0, Q, amplitude, phi0)``;
        ``active`` is a ``(N, L)`` float weight (1 active / 0 inactive).
        """
        t0 = wavelets[..., 0]
        f0 = wavelets[..., 1]
        Q = wavelets[..., 2]
        amp = self._strain_amplitude(wavelets[..., 3], f0, Q)
        phi0 = wavelets[..., 4]

        psi = morlet_gabor_fd(self._f, t0, f0, Q, amp, phi0)  # (N, L, n_freq)
        return np.sum(psi * active[..., None], axis=1)

    def polarizations(self, wavelets, ellipticity, active=None):
        """Return ``(h_plus, h_cross)`` on the analysis band for a batch.

        Uses the elliptical-polarisation convention
        ``h_cross = epsilon * e^{i pi/2} * h_plus = epsilon * 1j * h_plus``.

        ``wavelets`` is ``(N, L, 5)``; ``ellipticity`` is ``(N,)``; ``active`` is
        an optional ``(N, L)`` boolean mask (defaults to all active).
        """
        wavelets = np.asarray(wavelets, dtype=float)
        if wavelets.ndim == 2:
            wavelets = wavelets[None, :, :]
        n, L, _ = wavelets.shape
        if active is None:
            active = np.ones((n, L), dtype=float)
        else:
            active = np.asarray(active, dtype=float).reshape(n, L)
        ellipticity = np.broadcast_to(np.asarray(ellipticity, dtype=float), (n,))

        h_plus = self._wavelet_sum(wavelets, active)
        h_cross = ellipticity[:, None] * 1j * h_plus
        return h_plus, h_cross

    # -- projection -------------------------------------------------------
    def _project(self, h_plus, h_cross, ra, dec, psi):
        n = h_plus.shape[0]
        ra = np.broadcast_to(np.asarray(ra, dtype=float), (n,))
        dec = np.broadcast_to(np.asarray(dec, dtype=float), (n,))
        psi = np.broadcast_to(np.asarray(psi, dtype=float), (n,))
        out = np.zeros((n, len(self.detectors), self._f.size), dtype=complex)
        for j, det in enumerate(self.detectors):
            fp, fc = det.antenna_response(ra, dec, psi, self.reference_time)
            dt = det.time_delay_from_geocenter(ra, dec, self.reference_time)
            signal = fp[:, None] * h_plus + fc[:, None] * h_cross
            signal *= np.exp(-2j * np.pi * self._f[None, :] * dt[:, None])
            out[:, j, :] = signal
        return out

    def project_batch(self, wavelets, ra, dec, psi, ellipticity, inds=None):
        """RJMCMC-facing projection -> ``(N, n_ifo, n_freq_masked)``.

        ``wavelets`` is ``(N, L, 5)``; ``inds`` is the Eryn active-leaf boolean
        mask ``(N, L)`` (defaults to all active).
        """
        h_plus, h_cross = self.polarizations(wavelets, ellipticity, active=inds)
        return self._project(h_plus, h_cross, ra, dec, psi)

    # -- batched / grouped GPU path (fixed extrinsic) ---------------------
    def wavelet_hplus(self, wavelets_flat):
        """Per-wavelet plus-polarisation FD on the analysis band (device array).

        ``wavelets_flat`` is ``(M, 5)`` with columns ``(t0, f0, Q, amplitude,
        phi0)``. Returns an ``xp`` array of shape ``(M, n_freq)``. This is the
        batched generation kernel that runs on GPU when ``gpu=True``.
        """
        xp = self.xp
        w = xp.asarray(np.asarray(wavelets_flat, dtype=float))
        # contiguous 1-D columns (cupy.interp rejects non-contiguous inputs)
        t0 = xp.ascontiguousarray(w[:, 0])[:, None]
        f0 = xp.ascontiguousarray(w[:, 1])[:, None]
        Q = xp.ascontiguousarray(w[:, 2])[:, None]
        amp_in = xp.ascontiguousarray(w[:, 3])[:, None]
        phi0 = xp.ascontiguousarray(w[:, 4])[:, None]

        if self.amplitude_param == "amplitude":
            amp = amp_in
        else:
            if self._refpsd_xp is None:
                raise ValueError(
                    "amplitude_param='snr' requires a PSD; pass psd=(n_ifo, n_freq)."
                )
            psd_at_f0 = xp.interp(f0[:, 0], self._f_xp, self._refpsd_xp)[:, None]
            amp = amp_in * xp.sqrt(2.0 * np.pi * f0 * psd_at_f0 / (Q * _SQRT_PI_OVER_2))

        tau = Q / (2.0 * np.pi * f0)
        f = self._f_xp[None, :]
        prefactor = amp * tau * _SQRT_PI / 2.0
        lobe_minus = xp.exp(-(np.pi**2) * tau**2 * (f - f0) ** 2)
        lobe_plus = xp.exp(-(np.pi**2) * tau**2 * (f + f0) ** 2)
        phase = xp.exp(-2j * np.pi * f * t0)
        return prefactor * phase * (
            xp.exp(1j * phi0) * lobe_minus + xp.exp(-1j * phi0) * lobe_plus
        )

    def projection_factors(self, ra, dec, psi, ellipticity):
        """Fixed-sky per-detector projection factors ``(c_ifo, phase_ifo)``.

        With a fixed sky/ellipticity each detector response is
        ``signal_ifo = c_ifo * phase_ifo * h_plus`` where
        ``c_ifo = F+_ifo + 1j * epsilon * Fx_ifo`` and
        ``phase_ifo = exp(-2j pi f dt_ifo)``. Returns ``xp`` arrays of shape
        ``(n_ifo,)`` and ``(n_ifo, n_freq)``.
        """
        xp = self.xp
        nifo = len(self.detectors)
        c = np.zeros(nifo, dtype=complex)
        phase = np.zeros((nifo, self._f.size), dtype=complex)
        for j, det in enumerate(self.detectors):
            fp, fc = det.antenna_response(ra, dec, psi, self.reference_time)
            dt = det.time_delay_from_geocenter(ra, dec, self.reference_time)
            c[j] = float(fp[0]) + 1j * float(ellipticity) * float(fc[0])
            phase[j] = np.exp(-2j * np.pi * self._f * float(dt[0]))
        return xp.asarray(c), xp.asarray(phase)

    def project_grouped(self, wavelets_flat, groups, n_groups, ra, dec, psi, ellipticity):
        """Scatter-sum active wavelets by walker -> ``(G, n_ifo, n_freq)`` (xp).

        ``wavelets_flat`` is ``(M, 5)`` (all active leaves across walkers),
        ``groups`` is ``(M,)`` with values in ``[0, n_groups)``. The sky and
        ellipticity are fixed scalars, so the per-detector response factorises
        and only the per-walker ``h_plus`` sum is scattered.
        """
        xp = self.xp
        hpsi = self.wavelet_hplus(wavelets_flat)  # (M, n_freq)
        idx = xp.asarray(np.asarray(groups, dtype=np.int64))
        hsum = xp.zeros((int(n_groups), self._f.size), dtype=hpsi.dtype)
        _scatter_add(xp, hsum, idx, hpsi)  # (G, n_freq)

        c, phase = self.projection_factors(ra, dec, psi, ellipticity)  # (nifo,), (nifo,nfreq)
        return c[None, :, None] * phase[None, :, :] * hsum[:, None, :]  # (G, nifo, nfreq)

    def project_grouped_sky(self, wavelets_flat, wgroups, sky, n_groups):
        """Grouped projection with a *per-walker* sky -> ``(G, n_ifo, n_freq)``.

        ``wavelets_flat`` is ``(M, 5)`` with ``wgroups`` mapping each wavelet to
        a walker in ``[0, n_groups)``. ``sky`` is ``(G, 4)`` =
        ``(ra, dec, psi, ellipticity)`` per walker. The per-wavelet ``h_plus`` is
        scatter-summed per walker, then projected with that walker's antenna
        response and geocentric delay (each detector response factorises as
        ``c_g * phase_g * h_plus_g`` with ``c_g = F+_g + 1j*eps_g*Fx_g``).
        """
        xp = self.xp
        G = int(n_groups)
        hpsi = self.wavelet_hplus(wavelets_flat)  # (M, n_freq)
        idx = xp.asarray(np.asarray(wgroups, dtype=np.int64))
        hsum = xp.zeros((G, self._f.size), dtype=hpsi.dtype)
        _scatter_add(xp, hsum, idx, hpsi)  # (G, n_freq)

        ra = np.asarray(sky[:, 0], dtype=float)
        dec = np.asarray(sky[:, 1], dtype=float)
        psi = np.asarray(sky[:, 2], dtype=float)
        ell = np.asarray(sky[:, 3], dtype=float)

        out = xp.zeros((G, len(self.detectors), self._f.size), dtype=hpsi.dtype)
        for j, det in enumerate(self.detectors):
            # antenna pattern + geocentric delay are cheap on CPU (lal); the
            # large exp(-2j*pi*f*dt) over (G, n_freq) runs on the device.
            fp, fc = det.antenna_response(ra, dec, psi, self.reference_time)  # (G,)
            dt = det.time_delay_from_geocenter(ra, dec, self.reference_time)  # (G,)
            c = xp.asarray(fp + 1j * ell * fc)                                # (G,)
            dt_xp = xp.asarray(dt)                                            # (G,)
            phase = xp.exp(-2j * np.pi * self._f_xp[None, :] * dt_xp[:, None])  # (G, n_freq)
            out[:, j, :] = c[:, None] * phase * hsum
        return out

    # -- GWLikelihoods-compatible (fixed-D) path --------------------------
    def make_injections_to_ifo_batch(self, thetas, masked=True):
        """Fixed-dimension flat path -> ``(N, n_ifo, n_freq_masked)``.

        ``thetas`` is ``(N, 5 * n_wavelets + 4)`` laid out as
        ``[t0,f0,Q,amplitude,phi0] * n_wavelets + [ra, dec, psi, ellipticity]``.
        This makes :class:`WaveletTemplate` a drop-in template for
        :class:`~hyperwave.likelihoods.GWLikelihoods` (Gaussian or hyperbolic).
        """
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        n = thetas.shape[0]
        n_w = self.n_wavelets
        wavelets = thetas[:, : 5 * n_w].reshape(n, n_w, 5)
        ra, dec, psi, ellipticity = thetas[:, 5 * n_w : 5 * n_w + 4].T
        return self.project_batch(wavelets, ra, dec, psi, ellipticity)

    def make_injections_to_ifo(self, gw_params):
        signals = self.make_injections_to_ifo_batch(np.atleast_2d(gw_params))
        return {name: signals[0, j, :] for j, name in enumerate(self.detector_names)}

    def frequency_array_masked(self):
        return self._f


__all__ = [
    "WaveletTemplate",
    "morlet_gabor_fd",
    "amplitude_from_snr",
    "snr_from_amplitude",
    "ellipticity_from_ecc",
    "network_optimal_snr",
    "WAVELET_PARAMETERS",
    "EXTRINSIC_PARAMETERS",
]
