"""LVK detector noise handling (bilby-free).

``DetectorNoise`` downloads/loads strain, estimates PSDs and generates synthetic
coloured noise. It keeps the public API the HyperWave examples use, but the
bilby ``Interferometer`` objects are replaced by the lean
:class:`~hyperwave.detectors.data.Interferometer` containers and the
``lal``-backed geometry/strain/PSD helpers. Real-data fetching uses ``gwpy``
directly (as before); synthetic noise uses ``lalsimulation`` analytic design
sensitivity curves by default.
"""

from __future__ import annotations

import concurrent.futures
import logging
from functools import lru_cache

import numpy as np
from gwpy.timeseries import TimeSeries

from ...ml4gw import ml4gw_available, require_ml4gw_modules, resolve_torch_device
from ..data import Interferometer, InterferometerList
from ..psd import PowerSpectralDensity

logger = logging.getLogger("hyperwave.detectors.lvk")

# Per-detector colour scheme retained for plotting helpers.
color = {
    "L1": {"asd": "#455A64", "signal": "#ca0147", "noise": "lightgray"},
    "H1": {"asd": "#455A64", "signal": "#0f9b8e", "noise": "lightgray"},
    "V1": {"asd": "#455A64", "signal": "#f2ab15", "noise": "lightgray"},
}

# Default analytic design-sensitivity PSDs (lalsimulation) for synthetic noise.
_DESIGN_PSD = {
    "H1": "SimNoisePSDaLIGODesignSensitivityP1200087",
    "L1": "SimNoisePSDaLIGODesignSensitivityP1200087",
    "V1": "SimNoisePSDAdVDesignSensitivityP1200087",
    "K1": "SimNoisePSDKAGRADesignSensitivityT1600593",
}


def analytic_design_psd(detector, frequency_array, flow=10.0):
    """Analytic design-sensitivity PSD on ``frequency_array`` via lalsimulation.

    Returns a :class:`PowerSpectralDensity` restricted to the band where the
    analytic curve is finite and positive (lalsimulation returns 0 below
    ``flow``), so downstream interpolation stays finite.
    """
    import lal
    import lalsimulation as lalsim

    name = _DESIGN_PSD.get(detector, _DESIGN_PSD["H1"])
    frequency_array = np.asarray(frequency_array, dtype=float)
    df = frequency_array[1] - frequency_array[0]
    series = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0.0, df, lal.SecondUnit, len(frequency_array)
    )
    getattr(lalsim, name)(series, float(flow))
    psd = np.array(series.data.data, dtype=float)
    good = np.isfinite(psd) & (psd > 0)
    return PowerSpectralDensity(frequency_array[good], psd[good])


class DetectorNoise:
    """Manage GW detector noise: real (gwpy) or synthetic (analytic PSD)."""

    def __init__(
        self,
        duration,
        sampling_rate,
        trigger_time,
        detectors,
        minimum_frequency=20,
        maximum_frequency=800,
        post_trigger_duration=2,
        spectral_backend="numpy",
        gpu=False,
        torch_device=None,
    ):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.trigger_time = trigger_time
        self.post_trigger_duration = post_trigger_duration
        self._end_time = None
        self._start_time = None
        self.detectors = detectors
        self._ifos = None
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.spectral_backend = spectral_backend
        self.gpu = gpu
        self.torch_device = torch_device
        self.psd_duration = 32 * self.duration

    # -- segment timing ---------------------------------------------------
    @property
    @lru_cache
    def end_time(self):
        if self._end_time is None:
            self._end_time = self.trigger_time + self.post_trigger_duration
        return self._end_time

    @property
    @lru_cache
    def start_time(self):
        if self._start_time is None:
            self._start_time = self.end_time - self.duration
        return self._start_time

    @property
    def ifos(self):
        """Lazy :class:`InterferometerList` of lean detector containers."""
        if self._ifos is None:
            self._ifos = InterferometerList(
                Interferometer(
                    det,
                    self.sampling_rate,
                    self.duration,
                    start_time=self.start_time,
                    minimum_frequency=self.minimum_frequency,
                    maximum_frequency=self.maximum_frequency,
                )
                for det in self.detectors
            )
        return self._ifos

    # -- PSD estimation (bilby-free; unchanged numerics) ------------------
    def _compute_numpy_psd(self, ts):
        psd_alpha = 0.1
        from scipy.signal.windows import tukey

        window = tukey(int(self.duration * self.sampling_rate), psd_alpha)
        psd_duration_2 = (self.psd_end_time - self.psd_start_time - self.duration) / 2

        first_segment = ts.crop(self.psd_start_time, self.psd_start_time + psd_duration_2)
        second_segment = ts.crop(
            self.psd_start_time + self.duration + psd_duration_2, self.psd_end_time
        )
        first = first_segment.value.reshape(-1, len(window)) * window
        second = second_segment.value.reshape(-1, len(window)) * window
        first_blocked_psd = abs(np.fft.rfft(first, axis=-1) / self.sampling_rate) ** 2
        second_blocked_psd = abs(np.fft.rfft(second, axis=-1) / self.sampling_rate) ** 2

        simply_psd = 0.5 * (
            np.mean(first_blocked_psd, axis=0) * 2 / self.duration
            + np.mean(second_blocked_psd, axis=0) * 2 / self.duration
        )
        freqs = np.fft.rfftfreq(len(window), 1 / self.sampling_rate)
        return freqs, simply_psd

    def _compute_ml4gw_psd(self, ts):
        modules = require_ml4gw_modules()
        torch = modules.torch
        device = resolve_torch_device(gpu=self.gpu, device=self.torch_device)

        psd_alpha = 0.1
        from scipy.signal.windows import tukey

        window_values = tukey(int(self.duration * self.sampling_rate), psd_alpha)
        window = torch.as_tensor(window_values, dtype=torch.float64, device=device)

        estimator = modules.SpectralDensity(
            sample_rate=self.sampling_rate,
            fftlength=self.duration,
            overlap=0.0,
            average="mean",
            window=window,
            fast=False,
        ).to(device)

        psd_duration_2 = (self.psd_end_time - self.psd_start_time - self.duration) / 2
        first_segment = ts.crop(self.psd_start_time, self.psd_start_time + psd_duration_2)
        second_segment = ts.crop(
            self.psd_start_time + self.duration + psd_duration_2, self.psd_end_time
        )

        def _segment_psd(segment):
            values = torch.as_tensor(
                np.asarray(segment.value, dtype=np.float64), dtype=torch.float64, device=device
            )
            return estimator(values).detach().cpu().numpy()

        first_psd = _segment_psd(first_segment)
        second_psd = _segment_psd(second_segment)
        simply_psd = 0.5 * (first_psd + second_psd)
        freqs = np.fft.rfftfreq(len(window_values), 1 / self.sampling_rate)
        return freqs, simply_psd

    def compute_simply_psd(self, ts, backend=None):
        backend = (backend or self.spectral_backend).lower()
        if backend == "ml4gw":
            if not ml4gw_available():
                logger.warning("ml4gw PSD estimation requested but unavailable; using NumPy.")
            else:
                return self._compute_ml4gw_psd(ts)
        if backend != "numpy":
            raise ValueError(f"Unsupported spectral backend {backend!r}. Expected 'numpy' or 'ml4gw'.")
        return self._compute_numpy_psd(ts)

    def compute_psd(self, ts, backend=None):
        return self.compute_simply_psd(ts, backend=backend)

    # -- whitening (ml4gw) ------------------------------------------------
    def whiten_timeseries(self, data, psd, fduration=2.0, crop=True, highpass=None,
                          lowpass=None, backend="ml4gw"):
        backend = backend.lower()
        if backend != "ml4gw":
            raise ValueError("Whitening currently supports only backend='ml4gw'.")

        modules = require_ml4gw_modules()
        torch = modules.torch
        device = resolve_torch_device(gpu=self.gpu, device=self.torch_device)

        data_array = np.asarray(data, dtype=np.float64)
        psd_array = np.asarray(psd, dtype=np.float64)
        original_ndim = data_array.ndim
        if original_ndim == 1:
            data_array = data_array[None, None, :]
        elif original_ndim == 2:
            data_array = data_array[None, :, :]
        elif original_ndim != 3:
            raise ValueError(f"Expected 1D/2D/3D data, got shape {data_array.shape}.")

        whitener = modules.Whiten(
            fduration=fduration, sample_rate=self.sampling_rate, highpass=highpass, lowpass=lowpass
        ).to(device)
        whitened = whitener(
            torch.as_tensor(data_array, dtype=torch.float64, device=device),
            torch.as_tensor(psd_array, dtype=torch.float64, device=device),
            crop=crop,
        ).detach().cpu().numpy()

        if original_ndim == 1:
            return whitened[0, 0]
        if original_ndim == 2:
            return whitened[0]
        return whitened

    def whiten_ifo_data(self, ifo, fduration=2.0, crop=True, highpass=None, lowpass=None,
                        backend="ml4gw"):
        if isinstance(ifo, str):
            ifo = self.detectors.index(ifo)
        strain = np.asarray(self.ifos[ifo].strain_data.time_domain_strain)
        psd = np.asarray(self.ifos[ifo].power_spectral_density_array)
        return self.whiten_timeseries(strain, psd, fduration=fduration, crop=crop,
                                      highpass=highpass, lowpass=lowpass, backend=backend)

    def whiten_all_ifos(self, fduration=2.0, crop=True, highpass=None, lowpass=None, backend="ml4gw"):
        return {
            det: self.whiten_ifo_data(det, fduration=fduration, crop=crop, highpass=highpass,
                                      lowpass=lowpass, backend=backend)
            for det in self.detectors
        }

    # -- noise generation -------------------------------------------------
    def generate_noise(self, real_noise=True, gwf_files=None, gwf_channel=None,
                       gwf_start_time=None, gwf_end_time=None, gwf_time_series_start=None,
                       psd=None, seed=None):
        """Generate detector noise.

        Parameters
        ----------
        real_noise:
            If True, download real open data around the trigger. If False,
            generate synthetic coloured Gaussian noise.
        psd:
            Optional noise model for the synthetic path: a
            :class:`PowerSpectralDensity`, or a ``{detector: PowerSpectralDensity}``
            mapping. Defaults to lalsimulation analytic design curves.
        seed:
            Seed for reproducible synthetic noise.
        """
        if gwf_files is not None:
            self._ifos = self.load_gwf_data(
                gwf_files=gwf_files, channel=gwf_channel, start_time=gwf_start_time,
                end_time=gwf_end_time, series_start=gwf_time_series_start
            )
            return

        if real_noise:
            self._ifos = self.download_data()
        else:
            self._generate_synthetic_noise(psd=psd, seed=seed)

    def _generate_synthetic_noise(self, psd=None, seed=None):
        rng = np.random.default_rng(seed)
        for ifo in self.ifos:
            if isinstance(psd, dict):
                model = psd[ifo.name]
            elif psd is not None:
                model = psd
            else:
                model = analytic_design_psd(ifo.name, ifo.frequency_array)
            ifo.power_spectral_density = model
            fd_noise, _ = model.get_noise_realisation(self.sampling_rate, self.duration, rng=rng)
            ifo.strain_data.set_from_frequency_domain_strain(fd_noise)

    def _download_detector_data(self, det):
        try:
            self.psd_start_time = self.start_time - self.psd_duration
            self.psd_end_time = self.start_time + self.duration

            logger.info("Downloading analysis data for ifo %s", det)
            ifo = Interferometer(
                det, self.sampling_rate, self.duration, start_time=self.start_time,
                minimum_frequency=self.minimum_frequency, maximum_frequency=self.maximum_frequency,
            )
            data = TimeSeries.fetch_open_data(
                det, self.start_time, self.end_time, sample_rate=self.sampling_rate
            )
            ifo.strain_data.set_from_gwpy_timeseries(data)

            logger.info("Downloading psd data for ifo %s", det)
            self.psd_data = TimeSeries.fetch_open_data(
                det, self.psd_start_time, self.psd_end_time, sample_rate=self.sampling_rate
            )
            freqs, psd_array = self.compute_simply_psd(self.psd_data)
            ifo.power_spectral_density = PowerSpectralDensity(freqs, psd_array)
            return ifo
        except Exception as exc:
            logger.error("Error downloading data for detector %s: %s", det, exc)
            raise

    def download_data(self):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._download_detector_data, det) for det in self.detectors]
                ifos = [f.result() for f in futures]
            return InterferometerList(ifos)
        except Exception as exc:
            logger.error("Error in parallel data download: %s", exc)
            raise

    def _load_detector_from_gwf(self, det, filepath, channel, start_time=None, end_time=None,
                                series_start=None):
        try:
            ifo = Interferometer(
                det, self.sampling_rate, self.duration, start_time=self.start_time,
                minimum_frequency=self.minimum_frequency, maximum_frequency=self.maximum_frequency,
            )
            read_start = start_time if start_time is not None else 0
            read_end = end_time if end_time is not None else self.duration
            self.psd_start_time = series_start if series_start is not None else 0
            self.psd_end_time = self.psd_start_time + self.duration + self.psd_duration

            data = TimeSeries.read(filepath, channel, start=read_start, end=read_end)
            if self.sampling_rate is not None and hasattr(data, "sample_rate"):
                current_fs = float(data.sample_rate.value)
                if abs(current_fs - float(self.sampling_rate)) > 1e-6:
                    data = data.resample(self.sampling_rate)
            ifo.strain_data.set_from_gwpy_timeseries(data)

            psd_data = TimeSeries.read(filepath, channel, start=self.psd_start_time, end=self.psd_end_time)
            freqs, psd_array = self.compute_simply_psd(psd_data)
            ifo.power_spectral_density = PowerSpectralDensity(freqs, psd_array)
            return ifo
        except Exception as exc:
            logger.error("Error loading gwf for detector %s from %s: %s", det, filepath, exc)
            raise

    def load_gwf_data(self, gwf_files, channel=None, start_time=None, end_time=None, series_start=None):
        try:
            if isinstance(gwf_files, str):
                file_map = {det: gwf_files for det in self.detectors}
            else:
                file_map = dict(gwf_files)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._load_detector_from_gwf, det, file_map[det], channel,
                                    start_time, end_time, series_start)
                    for det in self.detectors
                ]
                ifos = [f.result() for f in futures]
            return InterferometerList(ifos)
        except Exception as exc:
            logger.error("Error loading gwf data: %s", exc)
            raise


__all__ = ["DetectorNoise", "analytic_design_psd", "color"]
