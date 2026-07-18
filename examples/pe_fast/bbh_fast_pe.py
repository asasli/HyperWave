"""Fast 2-parameter BBH parameter estimation with the hyperbolic likelihood.

Gaussian-noise BBH injection, then PE over **only the two intrinsic mass
parameters** (chirp mass + mass ratio) while the heavy-tailed *hyperbolic*
likelihood marginalises the per-segment noise shape. The other 12 CBC
parameters are held at their injected values, which makes this a fast,
low-dimensional demonstration of the pipeline that runs on a laptop.

Run the same problem through either sampler and compare::

    python examples/pe_fast/bbh_fast_pe.py --sampler eryn
    python examples/pe_fast/bbh_fast_pe.py --sampler pocomc

A ``--quick`` flag uses tiny sampler settings for a smoke test.
"""

from __future__ import annotations

import argparse
import os
import time

import bilby
import numpy as np

from hyperwave.detectors.lvk import GW, DetectorNoise
from hyperwave.inference import LVKinference
from hyperwave.likelihoods import GWLikelihoods

# Full intrinsic+extrinsic parameter vector the waveform generator expects.
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
        cos_theta_jn=np.cos(0.4), cos_tilt_1=1.0, cos_tilt_2=1.0, phi_12=0.0, phi_jl=0.0,
    )
    return [theta[k] for k in BBH_PARAMETER_NAMES]


def build_problem(duration=4.0, fs=2048.0, fmin=20.0, fmax=512.0, nsegs=2, seed=42):
    """Inject a BBH into Gaussian noise and build the hyperbolic likelihood.

    Returns ``(loglike_2d, theta_true, n_noise, info)`` where ``loglike_2d`` maps
    the sampled vector ``[chirp_mass, mass_ratio, <noise shape>]`` to the network
    log-likelihood (the 12 fixed CBC parameters are spliced in internally).
    """
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

    likelihood = GWLikelihoods(data=data, f=f, ifos_list=detectors, noise=psd,
                               template=template, ddims=False, nsegs=nsegs, gpu=False)

    # ddims=False hyperbolic_classic layout: [14 waveform, 1 alpha, nsegs delta].
    n_noise = 1 + nsegs
    fixed = np.asarray(theta_true[2:], dtype=float)            # the 12 held-fixed CBC params

    def loglike_2d(sampled):
        sampled = np.atleast_2d(np.asarray(sampled, dtype=float))
        mc_q, noise_shape = sampled[:, :2], sampled[:, 2:]
        wf = np.column_stack([mc_q, np.tile(fixed, (sampled.shape[0], 1))])  # (N, 14)
        full = np.column_stack([wf, noise_shape])                            # (N, 14 + n_noise)
        return likelihood.hyperbolic_classic(full)

    info = dict(f=f, psd=psd, data=data, template=template, nsegs=nsegs)
    return loglike_2d, theta_true, n_noise, info


def make_priors(nsegs):
    """Bilby priors: 2 science params + (1 alpha + nsegs delta) noise-shape params."""
    priors = bilby.core.prior.PriorDict()
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
        minimum=25.0, maximum=31.0, name="chirp_mass", latex_label=r"$\mathcal{M}$")
    priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(
        minimum=0.5, maximum=1.0, name="mass_ratio", latex_label="$q$")
    noise_priors = {r"$\alpha$": bilby.core.prior.Uniform(minimum=0.0, maximum=30.0)}
    for i in range(nsegs):
        noise_priors[r"$\delta_{}$".format(i)] = bilby.core.prior.Uniform(minimum=0.0, maximum=30.0)
    return priors, noise_priors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampler", choices=["eryn", "pocomc", "dynesty"], default="eryn")
    p.add_argument("--nsegs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results/pe_fast")
    p.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = p.parse_args()

    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    loglike_2d, theta_true, n_noise, info = build_problem(nsegs=args.nsegs, seed=args.seed)
    priors, noise_priors = make_priors(args.nsegs)
    tag = f"bbh_fast_{args.sampler}"

    if args.sampler == "eryn":
        kw = dict(nwalkers=20, ntemps=4, burn=50, nsteps=100) if args.quick \
            else dict(nwalkers=40, ntemps=10, burn=3000, nsteps=8000)
    elif args.sampler == "dynesty":
        kw = dict(nlive=200, dlogz=1.0) if args.quick \
            else dict(nlive=500, walks=100, dlogz=0.1)
    else:
        kw = dict(n_total=2000, n_effective=512, n_active=256) if args.quick \
            else dict(n_total=30000, n_effective=4000, n_active=1000)

    t0 = time.perf_counter()
    inf = LVKinference(
        loglike_2d, sampler_name=args.sampler, priors=priors, noise_priors=noise_priors,
        common_params={"save_dir": args.outdir, "TAG": tag, "like": "hyperbolic"},
        sampler_kwargs=kw,
    )
    inf.run()
    wall = time.perf_counter() - t0
    samples = inf.get_samples()

    mc_med, q_med = np.median(samples[:, 0]), np.median(samples[:, 1])
    print(f"\n[{args.sampler}] wall-clock {wall:.1f} s | {samples.shape[0]} samples")
    print(f"  chirp_mass: inj {theta_true[0]:.3f}  median {mc_med:.3f}")
    print(f"  mass_ratio: inj {theta_true[1]:.3f}  median {q_med:.3f}")

    try:
        import corner
        import matplotlib
        matplotlib.use("Agg")
        fig = corner.corner(samples[:, :2], labels=[r"$\mathcal{M}$", "$q$"],
                            truths=theta_true[:2], show_titles=True)
        path = os.path.join(args.outdir, f"{tag}_corner.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"  corner -> {path}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"  (corner skipped: {exc})")

    return wall, samples


if __name__ == "__main__":
    main()