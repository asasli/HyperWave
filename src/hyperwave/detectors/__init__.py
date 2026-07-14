"""Detector interfaces for HyperWave.

Generic, bilby-free building blocks:

* :mod:`~hyperwave.detectors.geometry` - :class:`Detector` (lal-backed antenna
  response and time delays)
* :mod:`~hyperwave.detectors.strain` - :class:`StrainData` and FFT helpers
* :mod:`~hyperwave.detectors.psd` - :class:`PowerSpectralDensity`
* :mod:`~hyperwave.detectors.data` - :class:`Interferometer` /
  :class:`InterferometerList` containers
* :mod:`~hyperwave.detectors.waveforms` - batched waveform backends and
  :class:`Template`

Instrument-specific layers: :mod:`~hyperwave.detectors.lvk`,
:mod:`~hyperwave.detectors.lisa`.
"""

from . import conditioning, data, geometry, lisa, lvk, psd, strain, waveforms
from .conditioning import condition_band, find_spectral_lines, whiteness_report
from .calibration import SplineCalibration, make_calibration_bank
from .data import Interferometer, InterferometerList
from .geometry import Detector
from .psd import PowerSpectralDensity
from .strain import StrainData
from .waveforms import Template, WaveletTemplate

__all__ = [
    "Detector",
    "PowerSpectralDensity",
    "StrainData",
    "Interferometer",
    "InterferometerList",
    "Template",
    "WaveletTemplate",
    "SplineCalibration",
    "make_calibration_bank",
    "geometry",
    "psd",
    "strain",
    "data",
    "waveforms",
    "lvk",
    "lisa",
]
