"""Random-wavelet synthetic data for source-agnostic ML training.

For each sample:

  D            ~ p(D)                  # number of wavelets
  for n < D:
      t0   ~ Uniform(0, T)
      f0   ~ LogUniform(f_min, f_max)
      Q    ~ Uniform(Q_min, Q_max)
      SNR  ~ SNRPrior(rho_star)
      phi0 ~ Uniform(0, 2 pi)
  (ra, dec, psi, ellipticity)          # random sky
  target_network_SNR   ~ Uniform(...)  # rescale amplitudes
  d_k(f) = c_k * exp(-2pi i f tau_k) * sum_n psi_n(f)
           + sqrt(S_k(f)) * (n_re + i n_im) / (2 sqrt(df))

producing whitened, network-SNR-controlled data with a known wavelet
decomposition. The same generator drives both the per-source-agnostic flow
births (Path A) and the amortized predictor (Path B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from ..detectors.geometry import get_detector
from ..detectors.waveforms.wavelets import (
    amplitude_from_snr,
    morlet_gabor_fd,
)

_DEFAULT_DETECTORS = ("H1", "L1")


@dataclass
class WaveletSample:
    """One synthetic (data, wavelets) realisation.

    Attributes
    ----------
    data : (n_ifo, n_freq) complex
        Whitened frequency-domain strain (per detector).
    psd : (n_ifo, n_freq)
        One-sided PSD on the analysis grid.
    wavelets : (D, 5)
        ``(t0, f0, Q, snr, phi0)`` per wavelet (sampling-space parameters).
    sky : (4,)
        ``(ra, dec, psi, ellipticity)``.
    network_snr : float
        Optimal network SNR of the noiseless signal.
    n_wavelets : int
        ``D`` (redundant with ``wavelets.shape[0]``, kept for fast batching).
    """

    data: np.ndarray
    psd: np.ndarray
    wavelets: np.ndarray
    sky: np.ndarray
    network_snr: float
    n_wavelets: int


def _sample_wavelets(rng, D, *, duration, f_min, f_max, q_bounds, snr_prior_rho_star):
    t0 = rng.uniform(0.0, duration, size=D)
    f0 = np.exp(rng.uniform(np.log(f_min), np.log(f_max), size=D))
    Q = rng.uniform(q_bounds[0], q_bounds[1], size=D)
    # SNRPrior inverse CDF: p(rho) ~ 3 rho / (4 rho_*^2) (1 + rho/(4 rho_*))^{-5}
    u = rng.uniform(size=D)
    rho_star = snr_prior_rho_star
    # closed-form inverse: rho/(4 rho_*) = (1-u)^{-1/4} - 1
    snr = 4.0 * rho_star * ((1.0 - u) ** (-0.25) - 1.0)
    phi0 = rng.uniform(0.0, 2 * np.pi, size=D)
    return np.column_stack([t0, f0, Q, snr, phi0])


def _antenna_and_delay(name, ra, dec, psi, geocent_time):
    det = get_detector(name)
    fp, fc = det.antenna_response(ra, dec, psi, geocent_time)
    dt = det.time_delay_from_geocenter(ra, dec, geocent_time)
    fp = float(np.asarray(fp).reshape(-1)[0])
    fc = float(np.asarray(fc).reshape(-1)[0])
    dt = float(np.asarray(dt).reshape(-1)[0])
    return fp, fc, dt


def generate_synthetic_signal(
    rng,
    *,
    duration=4.0,
    sampling_rate=2048.0,
    f_min=20.0,
    f_max=512.0,
    detectors=_DEFAULT_DETECTORS,
    psd_amp=1e-46,
    target_network_snr=None,
    d_distribution=("poisson", 8.0),
    d_max=80,
    q_bounds=(0.1, 40.0),
    snr_prior_rho_star=5.0,
    reference_time=0.0,
    whiten=True,
) -> WaveletSample:
    """Generate one whitened synthetic (data, wavelets) pair.

    Parameters
    ----------
    rng : np.random.Generator
    duration, sampling_rate, f_min, f_max : float
        Analysis-band geometry (matches ``examples/bbh_wavelet_reconstruction.py``).
    detectors : sequence of str
        Detector names (e.g. ``("H1", "L1")``).
    psd_amp : float
        Flat one-sided PSD level (per detector). Replace with an analytic ASD
        for realistic colouring.
    target_network_snr : float or None
        If set, rescale every wavelet amplitude so the noiseless network SNR
        matches this value. Otherwise the per-wavelet SNRs are kept as drawn.
    d_distribution : ("poisson", mean) or ("uniform", (low, high))
        Distribution over the number of wavelets ``D``.
    d_max : int
        Hard cap on ``D`` (training fixed-output-size architectures).
    q_bounds : (Q_min, Q_max)
    snr_prior_rho_star : float
        Truncated SNR-prior peak (Zackay+18-style).
    reference_time : float
        Geocentric reference epoch (sets antenna pattern). Defaults to 0.
    whiten : bool
        If True, return ``d / sqrt(S/2 df)`` (white-noise complex Gaussian).
    """
    n = int(round(duration * sampling_rate))
    df = 1.0 / duration
    f_full = np.arange(n // 2 + 1) * df
    band = (f_full >= f_min) & (f_full <= f_max)
    f = f_full[band]

    if d_distribution[0] == "poisson":
        D = int(min(rng.poisson(d_distribution[1]), d_max))
    elif d_distribution[0] == "uniform":
        D = int(rng.integers(d_distribution[1][0], d_distribution[1][1] + 1))
    else:
        raise ValueError(f"unknown d_distribution: {d_distribution}")

    psd = np.full((len(detectors), f.size), psd_amp)
    if D == 0:
        signal_fd = np.zeros((len(detectors), f.size), dtype=complex)
        net_snr = 0.0
        wavelets = np.zeros((0, 5))
    else:
        wavelets = _sample_wavelets(
            rng, D, duration=duration, f_min=f_min, f_max=f_max,
            q_bounds=q_bounds, snr_prior_rho_star=snr_prior_rho_star,
        )

        ra = rng.uniform(0.0, 2 * np.pi)
        dec = np.arcsin(rng.uniform(-1.0, 1.0))
        psi = rng.uniform(0.0, np.pi)
        ell = rng.uniform(-1.0, 1.0)

        # Sum the wavelets in h_plus (per-wavelet SNR is detector-frame-ish here;
        # we use the network optimal SNR as the calibration knob below).
        s_ref = np.interp(wavelets[:, 1], f, psd[0])
        amp = amplitude_from_snr(wavelets[:, 3], wavelets[:, 1], wavelets[:, 2], s_ref)
        h_p = np.sum(
            morlet_gabor_fd(f, wavelets[:, 0], wavelets[:, 1], wavelets[:, 2],
                            amp, wavelets[:, 4]),
            axis=0,
        )
        h_c = 1j * ell * h_p

        signal_fd = np.zeros((len(detectors), f.size), dtype=complex)
        for k, name in enumerate(detectors):
            fp, fc, dt = _antenna_and_delay(name, ra, dec, psi, reference_time)
            delay = np.exp(-2j * np.pi * f * dt)
            signal_fd[k] = (fp * h_p + fc * h_c) * delay

        net_snr2 = np.sum(4 * df * (np.abs(signal_fd) ** 2 / psd).real)
        net_snr = float(np.sqrt(net_snr2)) if net_snr2 > 0 else 0.0
        if target_network_snr is not None and net_snr > 0:
            scale = target_network_snr / net_snr
            signal_fd *= scale
            wavelets[:, 3] *= scale  # SNR column scales linearly
            net_snr = float(target_network_snr)

    # Gaussian noise (real + imag) with the correct one-sided PSD normalisation.
    sigma = np.sqrt(psd / (4.0 * df))
    noise = sigma * (rng.standard_normal(signal_fd.shape)
                     + 1j * rng.standard_normal(signal_fd.shape))
    data = signal_fd + noise

    if whiten:
        data = data / np.sqrt(psd / (4.0 * df))

    sky = (
        np.array([ra, dec, psi, ell])
        if D > 0
        else np.array([0.0, 0.0, 0.0, 0.0])
    )
    return WaveletSample(
        data=data, psd=psd, wavelets=wavelets, sky=sky,
        network_snr=net_snr, n_wavelets=int(D),
    )


class RandomWaveletDataset:
    """Iterable producing :class:`WaveletSample` batches on demand.

    Designed for streaming (no fixed corpus on disk): each ``__iter__`` re-seeds
    from a base RNG so training runs are reproducible. Use ``to_torch_batch`` to
    stack a batch of samples into padded tensors for a fixed-output-size
    architecture.
    """

    def __init__(self, *, seed=0, **generator_kwargs):
        self._seed = int(seed)
        self._gen_kwargs = generator_kwargs

    def __iter__(self) -> Iterator[WaveletSample]:
        rng = np.random.default_rng(self._seed)
        while True:
            yield generate_synthetic_signal(rng, **self._gen_kwargs)

    def batch(self, n: int, rng: Optional[np.random.Generator] = None):
        rng = rng if rng is not None else np.random.default_rng(self._seed)
        return [generate_synthetic_signal(rng, **self._gen_kwargs) for _ in range(n)]
