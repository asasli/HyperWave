# HyperWave

**Fast, robust Bayesian inference for gravitational-wave data.**

HyperWave is a parameter-estimation pipeline built around a heavy-tailed
**hyperbolic likelihood** that is robust to non-Gaussian noise (glitches,
confusion foregrounds), with a fully **vectorized** architecture: every
likelihood evaluates whole walker populations in single batched calls, and both
supported samplers (Eryn parallel-tempered MCMC and pocoMC preconditioned SMC)
run with `vectorize=True` end to end.

## Features

| | |
|---|---|
| **Four likelihoods** | Gaussian, hyperbolic (heavy-tailed), Whittle (per-segment levels), and heterodyne/relative-binning — one shared template/PSD interface |
| **Heterodyne speed** | per-evaluation cost independent of signal duration (~0.3 ms); measured 4.9× (4 s BBH) to 68.6× (64 s) over the full Gaussian likelihood |
| **Benchmark** | 64-s signal, same data/priors: eryn+heterodyne 659 s vs bilby+dynesty 4409 s (**6.7×**), max JS 0.006 |
| **GPU waveforms** | ml4gw (Torch) batched CBC generation; bbhx (SMBHB) and GBGPU (galactic binaries) for LISA; CuPy likelihood algebra with automatic CPU fallback |
| **Wavelet reconstruction** | BayesWave-style Morlet–Gabor RJMCMC with Fisher, half-cycle, sky-ring and matched-filter birth proposals |
| **LISA** | A/E/T bridge to lisatools-style data products; end-to-end SMBHB and UCB examples |
| **Validation** | `Result` objects, PP-test machinery, calibrated/biased/overconfident detection tests |

## Install

```bash
pip install hyperwave              # core (CPU, LVK)
pip install "hyperwave[plot,sampling]"   # + corner plots, pocoMC
pip install "hyperwave[gpu]"       # + CuPy likelihoods
pip install "hyperwave[ml4gw]"     # + Torch GPU waveforms (python < 3.13)
pip install "hyperwave[flows]"     # + normalizing-flow proposals
```

LISA waveforms (bbhx/GBGPU) require source builds at present — see
[LISA](lisa.md) for the verified recipe.

## Thirty seconds of HyperWave

```python
import numpy as np
from hyperwave.detectors.lvk import GW, DetectorNoise
from hyperwave.likelihoods import GWLikelihoods
from hyperwave.inference import LVKinference

noise = DetectorNoise(4.0, 2048.0, trigger_time, ["H1", "L1"],
                      minimum_frequency=20.0, maximum_frequency=512.0)
noise.generate_noise(real_noise=False, seed=42)
template = GW(noise, approximant="IMRPhenomPv2", parameters=names,
              static_parameters={"geocent_time": trigger_time})
template.make_injections_to_ifo(theta_true)

f, asd = template.detector_asd_masked(0)
likelihood = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"], noise=psd,
                           template=template, ddims=False, nsegs=4)

inf = LVKinference(likelihood.hyperbolic_classic, sampler_name="eryn",
                   priors=priors, noise_priors=noise_priors,
                   common_params={"save_dir": "out", "TAG": "bbh", "like": "hyperbolic"},
                   sampler_kwargs=dict(nwalkers=50, ntemps=10, burn=5000, nsteps=20000))
inf.run()
result = inf.get_result(injection=theta_true)
result.corner()
```

See the [Quickstart](quickstart.md) for the full runnable version.

## Citing

If you use HyperWave, please cite the code (see `CITATION.cff` in the
repository).