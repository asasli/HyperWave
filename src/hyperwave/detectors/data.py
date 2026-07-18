"""Lean interferometer containers.

These replace ``bilby.gw.detector.Interferometer`` / ``InterferometerList`` with
small, transparent objects that bundle a detector's geometry
(:class:`~hyperwave.detectors.geometry.Detector`), its strain
(:class:`~hyperwave.detectors.strain.StrainData`) and its noise model
(:class:`~hyperwave.detectors.psd.PowerSpectralDensity`).

They expose just the attributes the HyperWave detector/waveform layer used from
bilby (``name``, ``strain_data``, ``power_spectral_density``,
``power_spectral_density_array``, ``frequency_array``, ``frequency_mask``,
``minimum_frequency``, ``maximum_frequency``) plus vectorised antenna-response
and time-delay helpers.
"""

from __future__ import annotations

from .calibration import Recalibrate
from .geometry import get_detector
from .psd import PowerSpectralDensity
from .strain import DEFAULT_ROLL_OFF, StrainData


class Interferometer:
    def __init__(
        self,
        name,
        sampling_frequency,
        duration,
        start_time=0.0,
        minimum_frequency=20.0,
        maximum_frequency=2048.0,
        roll_off=DEFAULT_ROLL_OFF,
        calibration_model=None,
    ):
        self.name = str(name)
        self.detector = get_detector(self.name)
        self.calibration_model = (
            calibration_model if calibration_model is not None else Recalibrate()
        )
        self.strain_data = StrainData(sampling_frequency, duration, start_time, roll_off)
        self.power_spectral_density: PowerSpectralDensity | None = None
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = float(maximum_frequency)

    # -- grids ------------------------------------------------------------
    @property
    def frequency_array(self):
        return self.strain_data.frequency_array

    @property
    def frequency_mask(self):
        f = self.frequency_array
        return (f >= self.minimum_frequency) & (f <= self.maximum_frequency)

    @property
    def sampling_frequency(self):
        return self.strain_data.sampling_frequency

    @property
    def duration(self):
        return self.strain_data.duration

    @property
    def start_time(self):
        return self.strain_data.start_time

    # -- noise model ------------------------------------------------------
    @property
    def power_spectral_density_array(self):
        if self.power_spectral_density is None:
            raise ValueError(f"No PSD set for detector {self.name}.")
        return self.power_spectral_density.power_spectral_density_interpolated(self.frequency_array)

    @property
    def amplitude_spectral_density_array(self):
        return self.power_spectral_density_array**0.5

    # -- geometry passthrough --------------------------------------------
    def antenna_response(self, ra, dec, psi, gps_time):
        return self.detector.antenna_response(ra, dec, psi, gps_time)

    def time_delay_from_geocenter(self, ra, dec, gps_time):
        return self.detector.time_delay_from_geocenter(ra, dec, gps_time)

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"Interferometer({self.name!r})"


class InterferometerList(list):
    """A list of :class:`Interferometer` with a couple of bilby-like helpers."""

    @property
    def number_of_interferometers(self):
        return len(self)

    @property
    def frequency_array(self):
        if not self:
            raise ValueError("Empty InterferometerList.")
        return self[0].frequency_array

    @property
    def names(self):
        return [ifo.name for ifo in self]


#: Alias matching the package-level vocabulary in the plan/docs.
DetectorNetwork = InterferometerList


__all__ = ["Interferometer", "InterferometerList", "DetectorNetwork"]
