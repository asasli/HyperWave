"""LISA noise model — analytic AET sensitivity on a frequency grid.

The LISA counterpart of :class:`hyperwave.detectors.lvk.noise.DetectorNoise`:
it builds the analytic instrument noise PSD (SciRD acceleration + optical-metrology
terms) on a chosen frequency band, so a LISA pipeline can construct its likelihood
the same way an LVK one does — ``noise`` carries the PSD and the analysis grid,
the template carries the waveform.

Example::

    noise = LISANoise.narrowband(f0_hz=2.613e-3, tobs_years=1.0, buffer_bins=2000)
    f, psd = noise.frequency_array, noise.psd      # (nfreq,), (nchannels, nfreq)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLIGHT = 299_792_458.0
LISA_ARM = 2.5e9                       # LISA arm length [m]
LISA_LT = LISA_ARM / CLIGHT            # light travel time along one arm [s]
SECS_PER_YEAR = 365.25 * 86400.0

# SciRD defaults (log10 of acceleration / optical-metrology variances), matching
# the paper's noise_model: 3e-15 m/s^2/sqrt(Hz) and 15e-12 m/sqrt(Hz).
SA_DEFAULT = float(np.log10((3.0e-15) ** 2))
SI_DEFAULT = float(np.log10((15.0e-12) ** 2))

AET_CHANNELS = ("A", "E")


def lisa_aet_psd(fvec, Sa=SA_DEFAULT, Si=SI_DEFAULT, nchannels=2):
    """Analytic LISA A/E noise PSD on ``fvec`` (paper ``statutils.noise_model``).

    Returns ``(nchannels, len(fvec))``; the A and E channels share the same
    diagonal sensitivity in this model.
    """
    f = np.asarray(fvec, dtype=float)
    x = 2.0 * np.pi * LISA_LT * f
    Spm = (10.0 ** Sa) * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8e-3) ** 4) \
        * (2 * np.pi * f) ** -4 * (2 * np.pi * f / CLIGHT) ** 2
    Sop = (10.0 ** Si) * (1.0 + (2e-3 / f) ** 4) * (2 * np.pi * f / CLIGHT) ** 2
    Sn = 8.0 * np.sin(x) ** 2 * (2 * Spm * (3 + 2 * np.cos(x) + np.cos(2 * x))
                                 + Sop * (2 + np.cos(x)))
    return np.array([Sn] * nchannels)


@dataclass
class LISANoise:
    """LISA analysis band + analytic AET noise PSD.

    Parameters
    ----------
    frequency_array : ndarray
        The analysis frequency grid [Hz].
    tobs_years, dt : float
        Observation time [yr] and cadence [s] (carried through to the template).
    Sa, Si : float
        log10 acceleration / optical-metrology noise variances.
    channels : sequence of str
        AET channel labels (default ``("A", "E")``).
    """

    frequency_array: np.ndarray
    tobs_years: float = 1.0
    dt: float = 15.0
    Sa: float = SA_DEFAULT
    Si: float = SI_DEFAULT
    channels: tuple = AET_CHANNELS

    @property
    def df(self) -> float:
        return float(self.frequency_array[1] - self.frequency_array[0])

    @property
    def tobs_seconds(self) -> float:
        return self.tobs_years * SECS_PER_YEAR

    @property
    def psd(self) -> np.ndarray:
        """``(nchannels, nfreq)`` analytic AET PSD on the analysis grid."""
        return lisa_aet_psd(self.frequency_array, self.Sa, self.Si,
                            nchannels=len(self.channels))

    @classmethod
    def narrowband(cls, f0_hz, tobs_years=1.0, dt=15.0, buffer_bins=2000,
                   Sa=SA_DEFAULT, Si=SI_DEFAULT, half_window_hz=1e-6):
        """Build a dense narrow band around ``f0_hz`` (a near-monochromatic UCB).

        ``df = 1 / Tobs``; the band spans ``f0 ± (half_window + buffer_bins·df)``,
        matching the paper's per-source window.
        """
        tobs_s = tobs_years * SECS_PER_YEAR
        df = 1.0 / tobs_s
        lo = f0_hz - half_window_hz
        hi = f0_hz + half_window_hz
        start = max(int((lo - buffer_bins * df) / df), 1)
        end = int((hi + buffer_bins * df) / df)
        fvec = (start + np.arange(end - start + 1)) * df
        return cls(frequency_array=fvec, tobs_years=tobs_years, dt=dt, Sa=Sa, Si=Si)
