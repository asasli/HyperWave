# Likelihoods

All four likelihoods share one interface: construct with `(data, f, ifos_list, noise, template)`, call with a **batch** of parameter vectors `(N, ndim)`, get back `(N,)` log-likelihoods. The waveform for the whole batch is generated in a single backend call (`make_injections_to_ifo_batch`).

## Gaussian

The standard matched-filter likelihood
\( \ln L = -\tfrac12 \langle d-h \,|\, d-h \rangle \).
Use when the noise is well-behaved and the PSD is trusted.

```python
like = GWLikelihoods(..., template=template)
logl = like.gaussian(thetas)          # (N,)
```

## Time-Domain Covariance Likelihoods

`TimeDomainGWLikelihoods` evaluates real time-domain residuals with a Toeplitz
covariance. The first covariance row is built from the one-sided PSD as

\[
  \mathrm{ACF} = \frac{1}{2}\,\mathrm{irfft}(S_n \Delta f)\,N .
\]

The detector waveform is generated on the full frequency grid and inverse-FFT'd
after the same continuous phase time delay used by the frequency-domain LVK
template. PSD values outside the active band are patched to large finite values
before the ACF is built, following the `bilby_greg/TD` rule.

```python
from hyperwave.likelihoods import TimeDomainGWLikelihoods

like = TimeDomainGWLikelihoods(
    data=data_td, sampling_rate=4096, ifos_list=["H1", "L1"],
    noise=full_psd, template=template,
    likelihood_method="toeplitz-inversion",
    time_bands=[0.05, 0.10],
)

logl_g = like.gaussian(thetas[:, :like._wfdims])
logl_t = like.student_t(thetas)             # theta = [signal, nu_bands]
logl_h = like.hyperbolic_classic(thetas)    # theta = [signal, alpha, delta]
```

`likelihood_method` accepts `direct-inversion`, `cholesky-solve-triangular`,
`toeplitz-inversion`, and `gohberg-semencul` (`gs`). Set
`split_inner_products=True` to compute the pyRing diagnostic form
\(\langle d|d\rangle - 2\langle d|h\rangle + \langle h|h\rangle\).

Time bands can be an integer number of uniform sample-count bands or a list of
cut times in seconds. With `detector_dependent_noise=True`, Student-t and
hyperbolic parameters are detector-major, then band-major. Mixed detector
families are supported:

```python
like = TimeDomainGWLikelihoods(
    data=data_td, sampling_rate=4096, ifos_list=["H1", "L1"],
    acf=acf, template=template, detector_dependent_noise=True,
    detector_likelihoods={"H1": "gaussian", "L1": "hyperbolic"},
)

logl = like.mixed(thetas)   # theta = [signal, alpha_L1, delta_L1]
```

## Hyperbolic (heavy-tailed) — the HyperWave default

Replaces the Gaussian residual penalty with a hyperbolic one,
\( \sum_f \sqrt{\delta^2 + |r(f)|^2} \), governed by per-segment shape
parameters \((\alpha, \delta_i)\) that are sampled alongside the signal. Large outliers (glitches, mis-modelled noise) are penalised *linearly* rather than quadratically, so they do not drag the fit. The shape parameters themselves diagnose non-Gaussianity (the \(\xi\)–\(\chi\) "shape triangle").

```python
like = GWLikelihoods(..., ddims=False, nsegs=4)
logl = like.hyperbolic_classic(thetas)   # thetas = [signal params, alpha, delta_0..3]
```

Set `detector_dependent_noise=True` to give each detector its own hyperbolic
shape parameters. The detector-specific noise parameters are positional and
detector-major, following `ifos_list`.

For `ddims=False` and `ifos_list=["H1", "L1"]`:

```python
like = GWLikelihoods(..., ifos_list=["H1", "L1"], ddims=False, nsegs=2,
                     detector_dependent_noise=True)
# thetas = [signal params, alpha_H1, alpha_L1,
#           delta_H1_0, delta_H1_1, delta_L1_0, delta_L1_1]
```

For `ddims=True`, each detector has one `alpha` and one `delta` per segment:

```python
# thetas = [signal params,
#           alpha_H1_0, alpha_H1_1, alpha_L1_0, alpha_L1_1,
#           delta_H1_0, delta_H1_1, delta_L1_0, delta_L1_1]
```

You can also mix Gaussian and hyperbolic detectors by passing
`detector_noise_models` alongside `detector_dependent_noise=True`. Only
detectors marked `"hyperbolic"` receive shape parameters.

```python
like = GWLikelihoods(..., ifos_list=["H1", "L1"], ddims=False, nsegs=2,
                     detector_dependent_noise=True,
                     detector_noise_models=["gaussian", "hyperbolic"])
# thetas = [signal params, alpha_L1, delta_L1_0, delta_L1_1]
```

## Whittle (per-segment noise levels)

The Whittle likelihood with a free log-level per frequency segment — use when the PSD normalisation is uncertain but Gaussianity is acceptable.

```python
logl = like.whittle_level(thetas)
```

## Heterodyne (relative binning)

The Gaussian likelihood accelerated with the Zackay–Dai–Venumadhav scheme: a reference waveform \(h_0\) is computed once, the smooth ratio \(h/h_0\) is piecewise-linear over PN-spaced bins, and each evaluation needs the waveform only at the **bin edges** (a few hundred frequencies, via LAL's sequence API). Per-evaluation cost is independent of signal duration.

```python
from hyperwave.likelihoods import HeterodyneLikelihood

het = HeterodyneLikelihood.from_lvk_template(
    template, data=data, f=f, psd=psd, ifos_list=["H1", "L1"],
    theta_ref=theta_ref,    # injection or trigger point
    eps=0.1,                # max per-bin differential phase [rad]
)
logl = het.logl(thetas)
```

Measured against the full Gaussian likelihood (IMRPhenomPv2, 2 detectors):

| configuration | full grid | edges | full ms/eval | het ms/eval | speedup |
|---|---|---|---|---|---|
| BBH, 4 s @ 2048 Hz | 1 969 | 296 | 1.20 | 0.24 | **4.9×** |
| BNS-like, 64 s | 31 489 | 305 | 19.06 | 0.28 | **68.6×** |

!!! warning "Validity"
    The linear-ratio approximation holds in the posterior bulk around the
    reference point; logL-difference errors scale as `eps**2`. Validate against
    the full likelihood when changing `eps` (the test suite includes this
    check).

## Heterodyned hyperbolic (relative binning for the robust likelihood)

The hyperbolic square root cannot be binned exactly, but
`HeterodynedHyperbolicLikelihood` implements a **first+second-order heterodyne around the reference residual**: per-bin summaries with reference-fixed weights \(w(f;\delta)\), and the sampled \(\delta\) handled by tabulating summaries on a \(\delta\)-grid (splines for the dominant term). Measured: **47–53×** at 64 s (18.9 → 0.4 ms/eval), ~3× at 4 s; logL accuracy ~0.2 in the posterior bulk (same trust region as Gaussian relative binning). The exact hyperbolic remains available in `GWLikelihoods` — use it for final checks.

```python
from hyperwave.likelihoods import HeterodynedHyperbolicLikelihood
het_hyp = HeterodynedHyperbolicLikelihood.from_lvk_template(
    template, data=data, f=f, psd=psd, ifos_list=["H1", "L1"],
    theta_ref=theta_ref, nsegs=2, eps=0.1)
logl = het_hyp.logl(thetas)   # theta = [waveform, alpha, delta_0..delta_n]
```

A middle option, `InterpolatedWaveformTemplate`, bins only the *waveform*
(edge evaluation + ratio interpolation) and keeps the likelihood exact —
2.4–3.8× with <0.5% error, also valid for Whittle.

## Choosing

| situation | likelihood |
|---|---|
| clean data, trusted PSD | `gaussian` |
| glitches / non-Gaussian noise / unknown tails | `hyperbolic_classic` |
| uncertain PSD level | `whittle_level` |
| long signals (BNS), production PE throughput | `HeterodyneLikelihood` |
| ringdown / TD covariance checks | `TimeDomainGWLikelihoods` |
| robust likelihood at production throughput | `HeterodynedHyperbolicLikelihood` (exact hyperbolic for final checks) |
