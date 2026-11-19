"""Batched GW template: parameter adapter + detector projection.

:class:`Template` is what the likelihood talks to. It

1. maps HyperWave's sampling parameters (``chirp_mass``, ``mass_ratio``,
   ``chi_1``, ``cos_tilt_1``, ``cos_theta_jn`` ...) onto the bilby/lalsimulation
   intrinsic convention used by the waveform backends,
2. asks the backend for a batch of plus/cross polarisations, and
3. projects them onto each detector with vectorised antenna patterns and a
   continuous-phase geocentric time delay,

returning ``(N, n_ifo, n_freq)`` on the masked analysis grid in one call. The
per-parameter spin/inclination interpretation matches HyperWave's existing
ml4gw path (``a_1 = |chi_1|``, ``tilt_1 = arccos(cos_tilt_1)`` flipped for
``chi_1 < 0``; ``theta_jn = arccos(cos_theta_jn)``) so the two backends agree.
"""

from __future__ import annotations

import numpy as np

from ..calibration import Recalibrate, _batch_calibration_factor, calibration_parameter_names
from ..geometry import get_detector
from .base import INTRINSIC_PARAMETERS
from .lal_backend import LALWaveform

# Default sampled parameters (geocent_time is usually supplied via
# ``static_parameters``, matching HyperWave's existing examples).
DEFAULT_BBH_PARAMETERS = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
    "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1", "cos_tilt_2",
    "phi_12", "phi_jl",
]


def component_masses(chirp_mass, mass_ratio):
    """HyperWave's chirp-mass/mass-ratio -> component masses (unchanged formula)."""
    total_mass = chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio**0.6
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2


def _spin_amplitude_and_tilt(chi, cos_tilt):
    """Signed aligned-ish spin (chi) + cos_tilt -> (a, tilt), matching ml4gw path."""
    chi = np.asarray(chi, dtype=float)
    tilt = np.arccos(np.clip(cos_tilt, -1.0, 1.0))
    amplitude = np.abs(chi)
    tilt = np.where(chi < 0, np.pi - tilt, tilt)
    return amplitude, tilt


class Template:
    def __init__(
        self,
        detectors,
        frequency_array,
        sampling_rate,
        duration,
        start_time,
        minimum_frequency=20.0,
        maximum_frequency=None,
        reference_frequency=50.0,
        approximant="IMRPhenomPv2",
        parameters=None,
        static_parameters=None,
        backend="lal",
        trigger_time=None,
        n_jobs=1,
        gpu=False,
        torch_device=None,
        sequence=False,
        calibration_models=None,
    ):
        # sequence=True (lal backend only): evaluate exactly at frequency_array,
        # which may be sparse/non-uniform — used by the heterodyne likelihood.
        self.sequence = bool(sequence)
        self.detector_names = [str(d) for d in detectors]
        self.detectors = [get_detector(name) for name in self.detector_names]
        self.frequency_array = np.asarray(frequency_array, dtype=float)
        self.sampling_rate = float(sampling_rate)
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = (
            float(self.frequency_array[-1]) if maximum_frequency is None else float(maximum_frequency)
        )
        self.reference_frequency = float(reference_frequency)
        self.approximant = str(approximant).strip("'\"")
        self.parameters = list(parameters) if parameters is not None else list(DEFAULT_BBH_PARAMETERS)
        self.static_parameters = dict(static_parameters or {})
        self.trigger_time = trigger_time if trigger_time is not None else self.start_time
        self.n_jobs = int(n_jobs)
        self.calibration_models = self._normalize_calibration_models(calibration_models)
        self.parameters = self._append_calibration_parameters(self.parameters)

        self.mask = (self.frequency_array >= self.minimum_frequency) & (
            self.frequency_array <= self.maximum_frequency
        )
        self._f_masked = self.frequency_array[self.mask]

        self.backend_name = backend
        self.backend = self._build_backend(backend, gpu, torch_device)

    # -- backend ----------------------------------------------------------
    def _build_backend(self, backend, gpu, torch_device):
        backend = str(backend).lower()
        if backend == "lal":
            return LALWaveform(
                self.frequency_array,
                approximant=self.approximant,
                reference_frequency=self.reference_frequency,
                minimum_frequency=self.minimum_frequency,
                maximum_frequency=self.maximum_frequency,
                n_jobs=self.n_jobs,
                sequence=self.sequence,
            )
        if backend == "ml4gw":
            if self.sequence:
                raise ValueError("sequence=True is only supported by the 'lal' backend.")
            from .ml4gw_backend import ML4GWWaveform  # local import: optional dep

            right_pad = float(self.start_time + self.duration - self.trigger_time)
            return ML4GWWaveform(
                self.frequency_array,
                approximant=self.approximant,
                reference_frequency=self.reference_frequency,
                minimum_frequency=self.minimum_frequency,
                duration=self.duration,
                sampling_rate=self.sampling_rate,
                right_pad=max(0.0, min(right_pad, self.duration)),
                gpu=gpu,
                torch_device=torch_device,
            )
        raise ValueError(f"Unknown waveform backend {backend!r}. Expected 'lal' or 'ml4gw'.")

    def _normalize_calibration_models(self, calibration_models):
        if calibration_models is None:
            return [Recalibrate() for _ in self.detector_names]
        if hasattr(calibration_models, "get_calibration_factor"):
            if len(self.detector_names) != 1:
                raise ValueError("A single calibration model can only be used with one detector.")
            return [calibration_models]
        if isinstance(calibration_models, dict):
            models = []
            for name in self.detector_names:
                model = calibration_models.get(name)
                models.append(model if model is not None else Recalibrate())
            return models

        models = list(calibration_models)
        if len(models) != len(self.detector_names):
            raise ValueError(
                "calibration_models must have one entry per detector "
                f"({len(self.detector_names)} expected, got {len(models)})."
            )
        return [model if model is not None else Recalibrate() for model in models]

    def _append_calibration_parameters(self, parameters):
        ordered = list(parameters)
        known = set(ordered) | set(self.static_parameters)
        for model in self.calibration_models:
            for name in calibration_parameter_names(model):
                if name not in known:
                    ordered.append(name)
                    known.add(name)
        return ordered

    # -- parameter adapter ------------------------------------------------
    def _to_intrinsic(self, named):
        """Map a dict of (N,) HyperWave arrays to a bilby-convention intrinsic dict."""
        get = named.get

        if "mass_1" in named and "mass_2" in named:
            mass_1 = np.asarray(named["mass_1"], float)
            mass_2 = np.asarray(named["mass_2"], float)
        else:
            mass_1, mass_2 = component_masses(
                np.asarray(named["chirp_mass"], float), np.asarray(named["mass_ratio"], float)
            )

        if "theta_jn" in named:
            theta_jn = np.asarray(named["theta_jn"], float)
        else:
            theta_jn = np.arccos(np.clip(np.asarray(get("cos_theta_jn", 1.0), float), -1.0, 1.0))

        if "a_1" in named:
            a_1 = np.abs(np.asarray(named["a_1"], float))
            tilt_1 = np.asarray(get("tilt_1", np.arccos(np.clip(get("cos_tilt_1", 1.0), -1.0, 1.0))), float)
        else:
            a_1, tilt_1 = _spin_amplitude_and_tilt(get("chi_1", 0.0), get("cos_tilt_1", 1.0))

        if "a_2" in named:
            a_2 = np.abs(np.asarray(named["a_2"], float))
            tilt_2 = np.asarray(get("tilt_2", np.arccos(np.clip(get("cos_tilt_2", 1.0), -1.0, 1.0))), float)
        else:
            a_2, tilt_2 = _spin_amplitude_and_tilt(get("chi_2", 0.0), get("cos_tilt_2", 1.0))

        n = mass_1.shape[0] if mass_1.ndim else 1
        ones = np.ones(n)
        intrinsic = {
            "mass_1": mass_1 * ones,
            "mass_2": mass_2 * ones,
            "luminosity_distance": np.asarray(named["luminosity_distance"], float) * ones,
            "theta_jn": theta_jn * ones,
            "phase": np.asarray(named["phase"], float) * ones,
            "a_1": a_1 * ones,
            "a_2": a_2 * ones,
            "tilt_1": tilt_1 * ones,
            "tilt_2": tilt_2 * ones,
            "phi_12": np.asarray(get("phi_12", 0.0), float) * ones,
            "phi_jl": np.asarray(get("phi_jl", 0.0), float) * ones,
            "lambda_1": np.asarray(get("lambda_1", 0.0), float) * ones,
            "lambda_2": np.asarray(get("lambda_2", 0.0), float) * ones,
            "eccentricity": np.asarray(get("eccentricity", 0.0), float) * ones,
        }
        return {k: intrinsic[k] for k in INTRINSIC_PARAMETERS}

    def _named_from_theta(self, thetas):
        """Turn a ``(N, ndim)`` sampling array into a dict of ``(N,)`` arrays."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        named = {name: thetas[:, i] for i, name in enumerate(self.parameters)}
        n = thetas.shape[0]
        for key, value in self.static_parameters.items():
            named[key] = np.full(n, float(value))
        return named, n

    # -- projection -------------------------------------------------------
    def _project(self, hp, hc, named, masked=True):
        """Project ``(N, n_freq)`` polarisations onto detectors.

        Returns ``(N, n_ifo, n_freq)`` on the masked analysis grid (``masked=True``)
        or the full frequency grid (``masked=False``, used for injections).
        """
        if masked:
            hp_m = hp[:, self.mask]
            hc_m = hc[:, self.mask]
            f = self._f_masked
        else:
            hp_m = hp
            hc_m = hc
            f = self.frequency_array

        ra = np.asarray(named["ra"], float)
        dec = np.asarray(named["dec"], float)
        psi = np.asarray(named["psi"], float)
        if "geocent_time" in named:
            gps = np.asarray(named["geocent_time"], float)
        else:
            gps = np.full(hp.shape[0], float(self.trigger_time))

        n = hp.shape[0]
        out = np.zeros((n, len(self.detectors), len(f)), dtype=complex)
        for j, det in enumerate(self.detectors):
            fp, fc = det.antenna_response(ra, dec, psi, gps)          # (N,)
            dt = (gps - self.start_time) + det.time_delay_from_geocenter(ra, dec, gps)
            signal = fp[:, None] * hp_m + fc[:, None] * hc_m          # (N, n_freq)
            signal *= np.exp(-2j * np.pi * f[None, :] * dt[:, None])
            cal_params = dict(named)
            cal_params.setdefault("prefix", f"recalib_{self.detector_names[j]}_")
            if masked:
                signal *= _batch_calibration_factor(self.calibration_models[j], f, cal_params, n)
            else:
                signal[:, self.mask] *= _batch_calibration_factor(
                    self.calibration_models[j], f[self.mask], cal_params, n
                )
            out[:, j, :] = signal
        return out

    # -- public API -------------------------------------------------------
    def make_injections_to_ifo_batch(self, thetas, masked=True):
        """Batched projected waveforms, shape ``(N, n_ifo, n_freq[_masked])``."""
        named, _ = self._named_from_theta(thetas)
        intrinsic = self._to_intrinsic(named)
        hp, hc = self.backend.polarizations(intrinsic)
        return self._project(hp, hc, named, masked=masked)

    def make_injections_to_ifo(self, gw_params):
        """Legacy single-vector path: returns ``{ifo_name: masked complex array}``."""
        signals = self.make_injections_to_ifo_batch(np.atleast_2d(gw_params))
        return {name: signals[0, j, :] for j, name in enumerate(self.detector_names)}

    def frequency_array_masked(self):
        return self._f_masked


__all__ = ["Template", "DEFAULT_BBH_PARAMETERS", "component_masses"]
