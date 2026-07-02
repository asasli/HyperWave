"""Batched frequency-domain CBC waveform generation.

* :class:`Template` - CBC parameter adapter + detector projection (talk to this).
* :class:`WaveformBackend` - backend interface.
* :class:`LALWaveform` - default, direct-lalsimulation, bit-exact vs bilby.
* :class:`ML4GWWaveform` - optional Torch/ml4gw batched backend (experimental).
* :class:`WaveletTemplate` - Morlet-Gabor wavelet signal model (waveform-agnostic).
"""

from __future__ import annotations

from .base import WaveformBackend, normalize_intrinsic_batch
from .lal_backend import LALWaveform
from .template import DEFAULT_BBH_PARAMETERS, Template, component_masses

try:
    from .wavelets import (
        EXTRINSIC_PARAMETERS,
        WAVELET_PARAMETERS,
        WaveletTemplate,
        amplitude_from_snr,
        ellipticity_from_ecc,
        morlet_gabor_fd,
        network_optimal_snr,
        snr_from_amplitude,
    )
except ImportError:
    EXTRINSIC_PARAMETERS = WAVELET_PARAMETERS = None
    WaveletTemplate = None
    amplitude_from_snr = ellipticity_from_ecc = morlet_gabor_fd = None
    network_optimal_snr = snr_from_amplitude = None

__all__ = [
    "Template",
    "WaveformBackend",
    "LALWaveform",
    "DEFAULT_BBH_PARAMETERS",
    "component_masses",
    "normalize_intrinsic_batch",
    "ML4GWWaveform",
    # wavelets
    "WaveletTemplate",
    "morlet_gabor_fd",
    "amplitude_from_snr",
    "snr_from_amplitude",
    "ellipticity_from_ecc",
    "network_optimal_snr",
    "WAVELET_PARAMETERS",
    "EXTRINSIC_PARAMETERS",
]


def __getattr__(name):
    # Lazy import so the optional ml4gw/torch deps are only needed on use.
    if name == "ML4GWWaveform":
        from .ml4gw_backend import ML4GWWaveform

        return ML4GWWaveform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
