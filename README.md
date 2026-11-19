<p align="center">
  <img src="static/logo.png" alt="HyperWave logo" height="260" />
</p>

# HyperWave

HyperWave is a Python package for robust gravitational-wave inference. It pairs
hyperbolic (heavy-tailed) likelihoods for non-Gaussian, glitch-prone data with
batched, GPU-ready waveform generation. It provides detector and noise
utilities, frequency-domain likelihoods, sampler drivers, and plotting helpers,
and runs on top of `lalsuite`, `gwpy`, `pocoMC` and `eryn`.

## Highlights

- Hyperbolic and Gaussian likelihoods for parameter estimation that tolerate
  non-Gaussian noise and glitches.
- Batched waveform generation: a whole population of parameter sets is evaluated per call. The default backend calls `lalsimulation` directly and reproduces `bilby` waveforms to machine precision; an optional `ml4gw` backend generates batches with PyTorch.
- GPU acceleration through CuPy for the likelihood algebra, with a NumPy fallback when no GPU is present.
- Lean, `lal`-backed detector classes (geometry, antenna response, PSDs, strain FFTs); `bilby` is used for prior distributions and for reading LVK/bilby calibration response-curve files.

## Installation

```bash
git clone https://github.com/asasli/HyperWave.git
cd HyperWave
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install .
```

`lalsuite` and `gwpy` are most reliably installed from conda-forge:

```bash
conda install -c conda-forge lalsuite gwpy
pip install .
```

Optional extras:

| Extra | Installs | Use |
| --- | --- | --- |
| `.[plot]` | `corner`, `chainconsumer` | corner/posterior plots |
| `.[sampling]` | `pocomc` | preconditioned Monte Carlo sampler |
| `.[gpu]` | `cupy-cuda12x` | CuPy likelihood algebra |
| `.[ml4gw]` | `ml4gw` | PyTorch waveform backend (Python < 3.13) |
| `.[lisa]` | `lisaanalysistools` | LISA A/E/T helpers |
| `.[dev]` | build/test/lint tooling | development |

## Quick start: CBC parameter estimation

```python
import numpy as np
from hyperwave.detectors.lvk import DetectorNoise, GW
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
```

Drive a full run with the `LVKinference` helper (Eryn or pocoMC) using `bilby` priors; see `examples/bbh_noise_inference_eryn.py`.

## Waveform backends

```python
template = GW(noise, approximant="IMRPhenomPv2", waveform_backend="lal")    # default
template = GW(noise, approximant="IMRPhenomD",   waveform_backend="ml4gw")  # optional
```

## GPU acceleration

The likelihood algebra (`GWLikelihoods(..., gpu=True)`) runs on CuPy when a CUDA
device is available and falls back to NumPy otherwise. Waveform and detector
generation stay on CPU; the array-heavy residual and inner-product computations
move to the GPU.

```python
from hyperwave import gpu_backend_available, torch_cuda_available
print(gpu_backend_available(), torch_cuda_available())
```

## Public API

```python
from hyperwave import (
    GWLikelihoods, loglike,                      # likelihoods
    DetectorNoise, GW,                           # LVK data and waveform template
    Detector, PowerSpectralDensity, StrainData,  # detector building blocks
    Interferometer, InterferometerList,
    Template,                                    # batched waveform model
    LVKinference, DataInference,                 # sampler drivers
    gpu_backend_available, torch_cuda_available,
)
from hyperwave.detectors.waveforms import LALWaveform, ML4GWWaveform
```

## Package layout

```
src/hyperwave/
  likelihoods/   hyperbolic and Gaussian likelihoods
  detectors/     geometry, psd, strain, data  (lal-backed building blocks)
    waveforms/   waveform backends, batched CBC template
    lvk/         DetectorNoise / GW (LVK-facing classes)
    lisa/        LISA A/E/T helpers
  inference/     Eryn / pocoMC drivers, priors, flow proposals
  plots/         plotting helpers
examples/        runnable scripts and notebooks
tests/           validation against bilby and internal consistency checks
```

## Testing

```bash
pip install .[dev]
pytest
```

The suite validates the `lal` backend bit-for-bit against `bilby.gw.source.lal_binary_black_hole`, checks the antenna response and time delays against `bilby`, and confirms the batched likelihood matches the per-sample reference. Tests that need `lalsuite` or `ml4gw` are skipped automatically when those packages are unavailable.

## Citing

If HyperWave contributes to your work, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

## Authors

A. Sasli, N. Karnesis, M. Karamanis, M. W. Coughlin, V. Mandic, N. Stergioulas.

## Funding

- U.S. National Science Foundation HDR Institute for Accelerating AI Algorithms
  for Data Driven Discovery (A3D3), Cooperative Agreement PHY-2117997
- Bodossaki Foundation

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and please open
an issue or pull request.
