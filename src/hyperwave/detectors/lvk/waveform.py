"""LVK CBC waveform facade (bilby-free, batched).

``GW`` keeps the public surface the HyperWave examples and likelihood rely on
(``make_injections_to_ifo``, ``frequency_array``, ``detector_data_fd``,
``detector_asd_masked`` ...), but delegates generation to the batched
:class:`~hyperwave.detectors.waveforms.template.Template` (LAL backend by
default, optional ml4gw) and projects with the ``lal``-backed detector geometry.
A new :meth:`make_injections_to_ifo_batch` exposes the fast all-walkers path used
by the batched likelihood.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from ..waveforms.template import Template
from .noise import color


class GW:
    def __init__(
        self,
        noise,
        approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        parameters=(
            "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
            "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
            "cos_tilt_2", "phi_12", "phi_jl", "geocent_time",
        ),
        static_parameters=None,
        waveform_backend="lal",
        gpu=False,
        torch_device=None,
        n_jobs=1,
        calibration_models=None,
    ):
        self.noise = noise
        self.reference_frequency = reference_frequency
        self.duration = noise.duration
        self.start_time = noise.start_time
        self.end_time = noise.end_time
        self.sampling_rate = noise.sampling_rate
        self.minimum_frequency = noise.minimum_frequency
        self.maximum_frequency = noise.maximum_frequency
        self.ifos = noise.ifos
        self.Nifo = self.ifos.number_of_interferometers
        self.ifos_list = noise.detectors
        self.approximant = approximant.strip("'\"")
        self.parameters = list(parameters)
        self.static_parameters = {} if static_parameters is None else dict(static_parameters)
        self.injection_parameters = self.parameters + list(self.static_parameters.keys())
        self.color_scheme = color
        if calibration_models is None:
            calibration_models = {
                ifo.name: ifo.calibration_model
                for ifo in self.ifos
                if getattr(ifo, "calibration_model", None) is not None
            }
        self.calibration_models = calibration_models

        self._frequency_array = self.ifos.frequency_array
        self.df = self._frequency_array[1] - self._frequency_array[0]
        self.mask = (self._frequency_array >= self.minimum_frequency) & (
            self._frequency_array <= self.maximum_frequency
        )

        # Map legacy waveform_backend names ("bilby" -> "lal") for compatibility.
        backend = {"bilby": "lal"}.get(str(waveform_backend).lower(), str(waveform_backend).lower())
        self.template = Template(
            detectors=self.ifos_list,
            frequency_array=self._frequency_array,
            sampling_rate=self.sampling_rate,
            duration=self.duration,
            start_time=self.start_time,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            reference_frequency=self.reference_frequency,
            approximant=self.approximant,
            parameters=self.parameters,
            static_parameters=self.static_parameters,
            backend=backend,
            trigger_time=noise.trigger_time,
            n_jobs=n_jobs,
            gpu=gpu,
            torch_device=torch_device,
            calibration_models=self.calibration_models,
        )
        self.waveform_backend = self.template.backend_name
        self.parameters = list(self.template.parameters)
        self.injection_parameters = self.parameters + list(self.static_parameters.keys())

        # Snapshot the noise-only spectrum so repeated injections don't accumulate.
        self._base_fd = [
            None if ifo.strain_data._frequency_domain_strain is None and ifo.strain_data._time_domain_strain is None
            else np.array(ifo.strain_data.frequency_domain_strain, copy=True)
            for ifo in self.ifos
        ]

    # -- frequency grids --------------------------------------------------
    def frequency_array(self):
        return self._frequency_array[self.mask]

    def frequency_array_unmasked(self):
        return self._frequency_array

    # -- detector noise model --------------------------------------------
    def detector_asd(self, ifo):
        psd = self.ifos[ifo].power_spectral_density
        return np.array(psd.frequency_array), np.sqrt(psd.psd_array)

    def detector_asd_masked(self, ifo):
        f, y = self.detector_asd(ifo)
        f_masked = self.frequency_array()
        y_interpolated = Akima1DInterpolator(f, y)(f_masked)
        return f_masked, y_interpolated

    def detector_psd(self, ifo):
        return np.array(self.ifos[ifo].frequency_array), self.ifos[ifo].power_spectral_density_array

    # -- detector data ----------------------------------------------------
    def _zero_outside_mask(self, data):
        result = np.zeros_like(data)
        result[self.mask] = data[self.mask]
        return result

    def detector_data_asd(self, ifo):
        return abs(self.ifos[ifo].strain_data.frequency_domain_strain)[self.mask]

    def detector_data_fd(self, ifo):
        return self.ifos[ifo].strain_data.frequency_domain_strain[self.mask]

    def detector_data_fd_padding(self, ifo):
        return self._zero_outside_mask(self.ifos[ifo].strain_data.frequency_domain_strain)

    def detector_data_td(self, ifo):
        return self.ifos[ifo].strain_data.time_domain_strain

    def time_array(self, ifo):
        return self.ifos[ifo].strain_data.time_array

    def signal_td(self, signal):
        from ..strain import infft

        return infft(signal, self.sampling_rate)

    # -- parameters -------------------------------------------------------
    def init_infer_params(self, gw_params):
        params = {key: t for key, t in zip(self.parameters, gw_params)}
        params.update(self.static_parameters)
        return params

    def ifo_object(self):
        return self.ifos

    # -- waveform generation ---------------------------------------------
    def make_injections_to_ifo_batch(self, thetas):
        """Batched, pure generation for the likelihood: ``(N, n_ifo, n_freq_masked)``."""
        return self.template.make_injections_to_ifo_batch(thetas, masked=True)

    def _inject_into_data(self, gw_params):
        """Add the projected signal to the stored strain (base + signal)."""
        full = self.template.make_injections_to_ifo_batch(np.atleast_2d(gw_params), masked=False)[0]
        for j, ifo in enumerate(self.ifos):
            base = self._base_fd[j]
            ifo.strain_data.set_from_frequency_domain_strain(
                full[j] if base is None else base + full[j]
            )

    def make_injections_to_ifo(self, gw_params, raise_error=False):
        """Inject the signal into the stored data and return the masked response dict.

        Mirrors the original behaviour: an injection mutates the detector data
        (without accumulating across calls) and returns ``{ifo_name: masked}``.
        """
        self._inject_into_data(gw_params)
        signals = self.template.make_injections_to_ifo_batch(np.atleast_2d(gw_params), masked=True)[0]
        return {str(name): signals[j] for j, name in enumerate(self.ifos_list)}

    def make_injections_to_ifo_without_mask(self, ifo, gw_params, raise_error=False):
        self._inject_into_data(gw_params)
        full = self.template.make_injections_to_ifo_batch(np.atleast_2d(gw_params), masked=False)[0]
        idx = ifo if isinstance(ifo, int) else self.ifos_list.index(ifo)
        return full[idx]

    def waveform_ifo(self, gw_params, ifoID, raise_error=False):
        signals = self.template.make_injections_to_ifo_batch(np.atleast_2d(gw_params), masked=True)[0]
        return signals[ifoID]

    def waveform_ifo_padding(self, gw_params, ifoID, raise_error=False):
        full = self.template.make_injections_to_ifo_batch(np.atleast_2d(gw_params), masked=False)[0]
        return self._zero_outside_mask(full[ifoID])

    # -- plotting ---------------------------------------------------------
    def plot_detector_response(self, ifoID, fd_waveform, save_plot=False, outdir=None, label=None):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        name = self.ifos_list[ifoID]
        freqs = self.frequency_array()
        plot_waveform = np.abs(fd_waveform)
        plot_strain = np.abs(self.detector_data_fd(ifoID))
        plt.loglog(freqs, np.sqrt(2 * self.df) * plot_waveform, lw=0.8,
                   color=self.color_scheme[str(name)]["signal"], label="Waveform", alpha=0.8)
        plt.loglog(freqs, np.sqrt(2 * self.df) * plot_strain,
                   label="Data", color=self.color_scheme[str(name)]["noise"], lw=0.5)
        psd = self.ifos[ifoID].power_spectral_density
        plt.loglog(psd.frequency_array, np.sqrt(psd.psd_array), "-.", lw=0.8,
                   label="ASD", color=self.color_scheme[str(name)]["asd"])
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel(r"$|\tilde{h}(f)|~\sqrt{2\Delta f}$ & $\sqrt{S(f)}$", fontsize=12)
        plt.legend(loc="best")
        plt.xlim(20, 500)
        plt.ylim(1e-25, 1e-19)
        if save_plot and outdir and label:
            plt.savefig(f"{outdir}/{name}_{label}.pdf")
        return plt.gcf()


__all__ = ["GW"]
