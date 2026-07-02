"""BBH parameter estimation with per-detector hyperbolic shape parameters.

Demonstrates ``GWLikelihoods(..., shape_per_detector=True)``: each detector
fits its OWN α/δ per segment, so a detector whose residual is non-Gaussian
(glitches, line contamination) gets its own heavier-tailed model **without**
polluting the noise inference for the other detector.

To make the difference visible, we deliberately add a synthetic non-Gaussian
outlier to L1 only (one high-amplitude frequency bin). With shared α/δ both
detectors must agree on a single tail weight, biasing the recovery. With
per-detector α/δ, L1's δ inflates locally while H1's stays tight — sharper
posteriors on the science parameters.

Run::

    python examples/pe_per_detector/bbh_per_detector_pe.py            # default
    python examples/pe_per_detector/bbh_per_detector_pe.py --quick    # smoke test
    python examples/pe_per_detector/bbh_per_detector_pe.py --sampler pocomc

For the shared-α/δ baseline (no per-detector flag) see
``examples/pe_fast/bbh_fast_pe.py``.
"""

from __future__ import annotations

import argparse
import os
import time

import bilby
import numpy as np

from hyperwave.detectors.lvk import GW, DetectorNoise
from hyperwave.inference import LVKinference, per_detector_noise_priors
from hyperwave.likelihoods import GWLikelihoods


BBH_PARAMETER_NAMES = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
    "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
    "cos_tilt_2", "phi_12", "phi_jl",
]


def injected_bbh():
    """A GW150914-like BBH (component masses ~36/29 Msun)."""
    m1, m2 = 36.0, 29.0
    mc = (m1 + m2) * (m1 * m2 / (m1 + m2) ** 2) ** 0.6
    theta = dict(
        chirp_mass=mc, mass_ratio=m2 / m1, luminosity_distance=600.0, psi=1.1,
        phase=0.9, ra=1.375, dec=-0.2108, chi_1=0.0, chi_2=0.0,
        cos_theta_jn=np.cos(0.4), cos_tilt_1=1.0, cos_tilt_2=1.0,
        phi_12=0.0, phi_jl=0.0,
    )
    return [theta[k] for k in BBH_PARAMETER_NAMES]


def contaminate_l1(data, f, f0=60.0, factor=40.0):
    """Add a synthetic non-Gaussian outlier at ``f0`` Hz in **L1 only**.

    Multiplies the frequency bin nearest ``f0`` by ``factor`` — a one-bin
    glitch surrogate that the Gaussian likelihood cannot accommodate without
    over-inflating the noise level, but that the hyperbolic likelihood
    absorbs through a wider per-segment δ.
    """
    out = data.copy()
    idx = int(np.argmin(np.abs(f - f0)))
    out[1, idx] *= factor
    return out


def build_problem(duration=4.0, fs=2048.0, fmin=20.0, fmax=512.0, nsegs=2,
                  seed=42, glitch_factor=40.0):
    """Inject a BBH + L1-only outlier and build the per-detector hyperbolic likelihood."""
    trigger_time = 1268189526.951953
    detectors = ["H1", "L1"]
    theta_true = injected_bbh()

    noise = DetectorNoise(duration, fs, trigger_time, detectors,
                          minimum_frequency=fmin, maximum_frequency=fmax)
    noise.generate_noise(real_noise=False, seed=seed)

    template = GW(noise, approximant="IMRPhenomPv2", reference_frequency=50.0,
                  parameters=BBH_PARAMETER_NAMES,
                  static_parameters={"geocent_time": trigger_time})
    template.make_injections_to_ifo(theta_true)

    f, asd0 = template.detector_asd_masked(0)
    psd = np.array([asd0 ** 2, template.detector_asd_masked(1)[1] ** 2])
    data = np.array([template.detector_data_fd(0), template.detector_data_fd(1)])
    data = contaminate_l1(data, f, f0=60.0, factor=glitch_factor)

    likelihood = GWLikelihoods(
        data=data, f=f, ifos_list=detectors, noise=psd, template=template,
        ddims=True, nsegs=nsegs, gpu=False,
        shape_per_detector=True,                 # <-- the new flag
    )

    # Per-detector layout (ddims=True): [14 wf params,
    #                                    nsegs * n_ifo α (segment-major),
    #                                    nsegs * n_ifo δ (segment-major)].
    n_noise = 2 * nsegs * len(detectors)
    fixed = np.asarray(theta_true[2:], dtype=float)            # 12 held-fixed CBC params

    def loglike_2d(sampled):
        sampled = np.atleast_2d(np.asarray(sampled, dtype=float))
        mc_q, noise_shape = sampled[:, :2], sampled[:, 2:]
        wf = np.column_stack([mc_q, np.tile(fixed, (sampled.shape[0], 1))])  # (N, 14)
        full = np.column_stack([wf, noise_shape])                            # (N, 14 + n_noise)
        return likelihood.hyperbolic_classic(full)

    return loglike_2d, theta_true, n_noise, detectors


def make_priors(detectors, nsegs):
    """Bilby priors: 2 science params + per-detector (α, δ) shape priors."""
    priors = bilby.core.prior.PriorDict()
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
        minimum=25.0, maximum=31.0, name="chirp_mass", latex_label=r"$\mathcal{M}$")
    priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(
        minimum=0.5, maximum=1.0, name="mass_ratio", latex_label="$q$")
    # Per-detector helper: segment-major key order matches the per-detector
    # _alpha_columns / _tail_columns reshape inside GWLikelihoods.
    noise_priors = per_detector_noise_priors(
        ifo_names=detectors, nsegs=nsegs,
        alpha_range=(0.0, 30.0),
        delta_range=(0.0, 30.0),
        classic=True,                          # matches hyperbolic_classic
    )
    return priors, noise_priors


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampler", choices=["eryn", "pocomc"], default="eryn")
    p.add_argument("--nsegs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--glitch-factor", type=float, default=40.0,
                   help="amplitude scale for the L1-only synthetic outlier (1 = no glitch)")
    p.add_argument("--outdir", default="results/pe_per_detector")
    p.add_argument("--quick", action="store_true", help="tiny settings, smoke test")
    args = p.parse_args()

    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    loglike_2d, theta_true, n_noise, detectors = build_problem(
        nsegs=args.nsegs, seed=args.seed, glitch_factor=args.glitch_factor,
    )
    priors, noise_priors = make_priors(detectors, args.nsegs)

    print(f"sampled dims: 2 science + {n_noise} per-detector noise = {2 + n_noise} total")
    print(f"noise-prior order: {list(noise_priors.keys())}")

    tag = f"bbh_per_det_{args.sampler}"
    if args.sampler == "eryn":
        kw = dict(nwalkers=24, ntemps=4, burn=100, nsteps=200) if args.quick \
            else dict(nwalkers=60, ntemps=10, burn=3000, nsteps=8000)
    else:
        kw = dict(n_total=2000, n_effective=512, n_active=256) if args.quick \
            else dict(n_total=30000, n_effective=4000, n_active=1000)

    t0 = time.perf_counter()
    inf = LVKinference(
        loglike_2d, sampler_name=args.sampler,
        priors=priors, noise_priors=noise_priors,
        common_params={"save_dir": args.outdir, "TAG": tag, "like": "hyperbolic"},
        sampler_kwargs=kw,
    )
    inf.run()
    wall = time.perf_counter() - t0
    samples = inf.get_samples()

    mc_med, q_med = np.median(samples[:, 0]), np.median(samples[:, 1])
    print(f"\n[{args.sampler}] wall-clock {wall:.1f} s | {samples.shape[0]} samples")
    print(f"  chirp_mass: inj {theta_true[0]:.4f}  median {mc_med:.4f}")
    print(f"  mass_ratio: inj {theta_true[1]:.4f}  median {q_med:.4f}")

    # Per-detector noise-shape summary.
    noise_names = list(noise_priors.keys())
    noise_samples = samples[:, 2:]
    print("\nPer-detector noise-shape posteriors:")
    for name, col in zip(noise_names, noise_samples.T):
        lo, med, hi = np.percentile(col, [5, 50, 95])
        print(f"  {name:<22} median={med:6.3f}  90% CI=({lo:6.3f}, {hi:6.3f})")
    # Look for the L1-vs-H1 contrast on δ: with a real glitch we expect
    # δ_L1 > δ_H1 in the segment that contains the outlier (60 Hz → first
    # segment when the band starts at 20 Hz).
    deltas = {k: v for k, v in zip(noise_names, noise_samples.T) if k.startswith("delta_")}
    if deltas:
        h1 = [v for k, v in deltas.items() if "_H1_" in k]
        l1 = [v for k, v in deltas.items() if "_L1_" in k]
        if h1 and l1:
            print(f"\n  median δ(H1) = {np.median(np.concatenate(h1)):.3f}")
            print(f"  median δ(L1) = {np.median(np.concatenate(l1)):.3f}  "
                  f"(should exceed H1 when glitch-factor > 1)")

    try:
        import corner
        import matplotlib
        matplotlib.use("Agg")
        fig = corner.corner(samples[:, :2], labels=[r"$\mathcal{M}$", "$q$"],
                            truths=theta_true[:2], show_titles=True)
        path = os.path.join(args.outdir, f"{tag}_corner.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"\ncorner -> {path}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"  (corner skipped: {exc})")

    return wall, samples


if __name__ == "__main__":
    main()
