# LISA

HyperWave runs LISA parameter estimation through an A/E(/T) bridge that adapts
any TDI waveform generator to the standard likelihood stack:

```
generator (bbhx / GBGPU)  →  LISAAETTemplate  →  build_lisa_aet_likelihood  →  sampler
```

The bridge accepts raw NumPy arrays or lisatools containers
(`AnalysisContainer` / `DataResidualArray`; imported lazily), and exposes the
**vectorized** path: pass `batch_signal_model` and the whole walker population
is generated in one bbhx/GBGPU call per likelihood evaluation.

## Massive black-hole binaries (SMBHB, bbhx)

```bash
python examples/lisa/smbhb_bbhx_pe.py --sampler both --quick   # eryn + pocomc timing
python examples/lisa/smbhb_bbhx_pe.py --sampler eryn --steps 40000
```

11 source parameters + hyperbolic shape parameters, A/E channels, PhenomD
(2,2). The example calibrates the analytic PSD stub to a target SNR
(`--target-snr`).

## Galactic binaries (UCB, GBGPU)

```bash
python examples/lisa/ucb_gbgpu_pe.py --sampler eryn --steps 40000
```

GBGPU returns each source's narrow band; the example scatters evaluations onto
a fixed global grid (window centred on the injected `f0`, `df = 1/Tobs`) so the
likelihood residual is well defined. 8 sampled parameters; `fddot` static.

## Building the LISA stack

!!! danger "Do not `pip install bbhx`"
    The PyPI wheel (1.2.3) crashes on every call (an orbits-pointer bug fixed
    upstream only in unreleased v1.2.5) **and** is AVX-512-compiled, which
    SIGILLs on AMD nodes. Build from source — the verified recipe (four
    packages, ~15 min) is in `ENVIRONMENT.md` at the repository root.

`gbgpu` 1.1.3 from PyPI works where the CPU has AVX-512 (Intel Skylake) or via
its CUDA path; its modern (`master`) API requires a migration tracked in
`TODO.md`.

## Custom generators

Any callable returning `(nchannels, nfreq)` — or `(N, nchannels, nfreq)` for
the batched path — plugs in directly:

```python
from hyperwave.detectors.lisa import LISAAETTemplate, build_lisa_aet_likelihood

template = LISAAETTemplate(
    parameters=["m1", "m2", ...],
    signal_model=my_generator,            # kwargs -> (2, nfreq)
    batch_signal_model=my_batch_generator,  # arrays -> (N, 2, nfreq)
    channels=("A", "E"),
)
like = build_lisa_aet_likelihood(data=data, template=template,
                                 sensitivity=psd, freqs=freqs,
                                 channels=("A", "E"), ddims=False, nsegs=2)
```

## Sky maps and credible areas

`hyperwave.plots.SkyPosterior` turns the `(lam, sin_beta)` posterior columns into
a gridded sky map using the canonical HyperWave estimator (equal-area
`(lon, sin lat)` grid at 0.5 degree, one-cell Gaussian smoothing, greedy HPD),
so the numbers it reports match `sky_credible_area`.

```python
from hyperwave.plots import SkyPosterior, plot_skymap, compare_skymaps

sky = result.skymap()                       # Result with lam / sin_beta columns
sky.credible_area(0.9), sky.credible_area(0.5)   # deg^2
sky.modes(0.9)          # distinct modes: mass, area, centre, peak, antipode_of
sky.dominant_mode_fraction(0.9)
sky.truth_credible_level(lam0, beta0)       # searched probability of the injection
sky.write_fits("alert.fits", nside=64)      # HEALPix PROB map for telescope teams (healpy)

fig, axes = plot_skymap(sky, truth=(lam0, beta0), fov={"Rubin": 9.6, "Athena": 0.4})
fig, axes = plot_skymap(sky, dark=True)      # talk version (dark page)
fig, axes = compare_skymaps([sky_sub, sky_fg, sky_clean],
                            ["HERON-subtracted", "foreground", "clean"],
                            truth=(lam0, beta0), reference=sky_unseeded)
```

The figures follow the HERON pointing figure:

- `plot_skymap` draws a Mollweide all-sky map (posterior density on a dark
  sky, thin white 50%/90% contours, the strongest mode labelled "dominant" and
  its antipodal image "antipodal") and a zoom on the dominant mode with filled
  50%/90% regions, the injection star and equal-area circles for the requested
  fields of view. `sky.plot()` and `sky.plot_zoom()` give the two panels
  separately.
- `compare_skymaps` overlays several runs: the first one filled, the others as
  contour lines, an optional `reference` run (e.g. unseeded) as a dotted 90%
  outline. The legend quotes `A90 / A50` and `f_dom` for every run.
  `allsky=False` returns the zoom panel alone and
  `window=(lon_min, lon_max, lat_min, lat_max)` fixes its limits in degrees.

Conventions:

- **Areas** always come from the canonical estimator. The drawn density and
  contours use three cells of smoothing for display (`smooth=`), as stated in
  the paper caption.
- **Modes**: islands of the credible region closer than `merge_deg=45` degrees
  form one mode, so a fragmented island is counted once. `f_dom` is the
  probability of the part of the sky nearest to the dominant mode. In
  `compare_skymaps` it is evaluated for every run on the sky region of the
  first run's dominant mode.
- **Style**: the functions apply their own matplotlib style (serif, inward
  ticks) and leave the global `rcParams` unchanged; `usetex=True` renders the
  text with LaTeX, and `sky_style()` is the same style as a context manager.
- Longitude increases to the right; `sky.plot(astro=True)` puts east on the
  left.
