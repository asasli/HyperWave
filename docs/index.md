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
| **Bilby boundary** | bilby priors plus LVK/bilby calibration response-curve file readers; waveform projection and likelihood evaluation stay HyperWave-native and batched |

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
repository) and the hyperbolic-likelihood paper:

> [arXiv:2602.22074](https://arxiv.org/abs/2602.22074).
> [PhysRevD.111.022005](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.022005).
```
@article{PhysRevD.111.022005,
  title = {Characterization of non-Gaussian stochastic signals with heavier-tailed likelihoods},
  author = {Karnesis, N. and Sasli, A. and Buscicchio, R. and Stergioulas, N.},
  journal = {Phys. Rev. D},
  volume = {111},
  issue = {2},
  pages = {022005},
  numpages = {16},
  year = {2025},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevD.111.022005},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.111.022005}
}
```
> [PhysRevD.108.103005](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.103005).
```
@article{PhysRevD.108.103005,
  title = {Heavy-tailed likelihoods for robustness against data outliers: Applications to the analysis of gravitational wave data},
  author = {Sasli, Argyro and Karnesis, Nikolaos and Stergioulas, Nikolaos},
  journal = {Phys. Rev. D},
  volume = {108},
  issue = {10},
  pages = {103005},
  numpages = {17},
  year = {2023},
  month = {Nov},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevD.108.103005},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.108.103005}
}
```

In addition, please cite any external software used in your analysis,
including:

- Eryn (when using the PTMCMC sampler),
```
@article{Karnesis_2023,
   title={Eryn: a multipurpose sampler for Bayesian inference},
   volume={526},
   ISSN={1365-2966},
   url={http://dx.doi.org/10.1093/mnras/stad2939},
   DOI={10.1093/mnras/stad2939},
   number={4},
   journal={Monthly Notices of the Royal Astronomical Society},
   publisher={Oxford University Press (OUP)},
   author={Karnesis, Nikolaos and Katz, Michael L and Korsakova, Natalia and Gair, Jonathan R and Stergioulas, Nikolaos},
   year={2023},
   month=Sept, pages={4814–4830} }
```
- pocoMC (when using the SMC sampler),
```
@article{karamanis2022accelerating,
    title={Accelerating astronomical and cosmological inference with preconditioned Monte Carlo},
    author={Karamanis, Minas and Beutler, Florian and Peacock, John A and Nabergoj, David and Seljak, Uro{\v{s}}},
    journal={Monthly Notices of the Royal Astronomical Society},
    volume={516},
    number={2},
    pages={1644--1653},
    year={2022},
    publisher={Oxford University Press}
}

@article{karamanis2022pocomc,
    title={pocoMC: A Python package for accelerated Bayesian inference in astronomy and cosmology},
    author={Karamanis, Minas and Nabergoj, David and Beutler, Florian and Peacock, John A and Seljak, Uros},
    journal={arXiv preprint arXiv:2207.05660},
    year={2022}
}
```
- [BBHx](https://github.com/lisa-analysis-tools/BBHx) (for LISA SMBHB waveform generation),
- [GBGPU](https://github.com/lisa-analysis-tools/GBGPU) (for LISA UCB waveform generation),
- ml4gw (when using GPU-based CBC waveform generation).

Please also cite any waveform models and detector-specific software products
used in your analysis as appropriate.
