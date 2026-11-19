from pathlib import Path

import numpy as np

from hyperwave.detectors.lvk import GW, DetectorNoise
from hyperwave.likelihoods import GWLikelihoods

trigger_time = 1268189526.951953
params = ["chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
          "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
          "cos_tilt_2", "phi_12", "phi_jl"]

# Synthetic design-sensitivity noise; pass real_noise=True for open data.
noise = DetectorNoise(4, 4096, trigger_time, ["H1", "L1"], maximum_frequency=800)
noise.generate_noise(real_noise=False, seed=42)

template = GW(noise, approximant="IMRPhenomPv2", reference_frequency=50.0,
              parameters=params, static_parameters={"geocent_time": trigger_time})

theta = [28.1, 0.806, 1000.0, 1.2, 0.64, 1.375, 0.21,
         0.0, 0.0, np.cos(0.4), 1.0, 1.0, 0.0, 0.0]
template.make_injections_to_ifo(theta)  # inject a signal into the data

f, asd0 = template.detector_asd_masked(0)
psd = np.array([asd0 ** 2, template.detector_asd_masked(1)[1] ** 2])
data = np.array([template.detector_data_fd(0), template.detector_data_fd(1)])

likelihood = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"],
                           noise=psd, template=template, nsegs=4, gpu=False)

# One template call evaluates the whole population of samples at once.
samples = np.array([theta, theta])
print(likelihood.gaussian(samples))

# Optional diagnostic plots.  These are not posterior samples from an inference
# run; they are a small local cloud around the injected parameters.
outdir = Path(__file__).resolve().with_name("readme_diagnostics")
outdir.mkdir(exist_ok=True)

rng = np.random.default_rng(1234)
plot_samples = np.tile(theta, (128, 1)).astype(float)
plot_samples += rng.normal(
    scale=[0.2, 0.01, 40.0, 0.05, 0.05, 0.05, 0.03,
           0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05],
    size=plot_samples.shape,
)
plot_samples[:, 0] = np.clip(plot_samples[:, 0], 5.0, None)
plot_samples[:, 1] = np.clip(plot_samples[:, 1], 0.05, 1.0)
plot_samples[:, 2] = np.clip(plot_samples[:, 2], 1.0, None)
plot_samples[:, 3] = np.mod(plot_samples[:, 3], np.pi)
plot_samples[:, 4] = np.mod(plot_samples[:, 4], 2 * np.pi)
plot_samples[:, 5] = np.mod(plot_samples[:, 5], 2 * np.pi)
plot_samples[:, 6] = np.clip(plot_samples[:, 6], -np.pi / 2, np.pi / 2)
plot_samples[:, 7:9] = np.clip(plot_samples[:, 7:9], -0.99, 0.99)
plot_samples[:, 9:12] = np.clip(plot_samples[:, 9:12], -1.0, 1.0)
plot_samples[:, 12:14] = np.mod(plot_samples[:, 12:14], 2 * np.pi)

try:
    from hyperwave.plots.corners import plot_posterior

    corner_params = [r"$\mathcal{M}$", "$q$", "$d_L$"]
    plot_posterior(
        plot_samples[:, :3],
        corner_params,
        case="gaussian",
        package="corner",
        truths=theta[:3],
        save_dir=str(outdir / "corner.pdf"),
        show=False,
    )
    print(f"corner plot -> {outdir / 'corner.pdf'}")
except Exception as exc:
    print(f"corner plot skipped: {exc}")

try:
    from hyperwave.plots.fd_reconstruction import (
        compute_credible_region_fd,
        plot_fd_reconstruction,
        reconstruct_fd_waveforms,
    )
    from hyperwave.plots.td_reconstruction import (
        compute_credible_region,
        plot_td_reconstruction,
        reconstruct_td_waveforms,
    )

    detector_idx = 0
    signal_fd = np.abs(template.waveform_ifo(theta, detector_idx))
    signal_td = template.signal_td(template.waveform_ifo_padding(theta, detector_idx))
    data_td = np.array(template.detector_data_td(detector_idx), copy=True)
    times = template.time_array(detector_idx) - trigger_time

    fd_waveforms = reconstruct_fd_waveforms(
        template, plot_samples, detector_idx=detector_idx,
        num_samples=48, n_jobs=1, random_seed=1234,
    )
    fd_region = compute_credible_region_fd(fd_waveforms)
    plot_fd_reconstruction(
        f, fd_region, signal_fd=signal_fd, case="gaussian",
        title="H1 frequency-domain reconstruction", xlim=(20, 800),
        outpath=outdir / "fd_reconstruction.png", show=False,
    )
    print(f"FD reconstruction -> {outdir / 'fd_reconstruction.png'}")

    td_waveforms = reconstruct_td_waveforms(
        template, plot_samples, detector_idx=detector_idx,
        num_samples=48, n_jobs=1, random_seed=1234,
    )
    td_region = compute_credible_region(td_waveforms)
    plot_td_reconstruction(
        times, td_region, signal_td=signal_td, data_td=data_td,
        case="gaussian", title="H1 time-domain reconstruction",
        xlim=(-0.1, 0.05), outpath=outdir / "td_reconstruction.png",
        show=False,
    )
    print(f"TD reconstruction -> {outdir / 'td_reconstruction.png'}")
except Exception as exc:
    print(f"waveform reconstruction plots skipped: {exc}")
