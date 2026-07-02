"""Real-event PE on GW150914 — Eryn vs pocoMC.

Downloads ~4 s of open H1+L1 strain around GW150914 (GWOSC), conditions it, and
runs full CBC parameter estimation with the selected likelihood under **both**
samplers, reporting wall-clock times for a head-to-head comparison. There is no
injected truth — this is the real signal.

    python examples/real_event/gw150914_pe.py --sampler eryn
    python examples/real_event/gw150914_pe.py --sampler pocomc
    python examples/real_event/gw150914_pe.py --sampler both     # run both, report times
    python examples/real_event/gw150914_pe.py --setup-only       # download + 1 likelihood eval

See ``examples/clusters/gw150914_pe.slurm`` for the cluster submission.
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

# GW150914
GW150914_TIME = 1126259462.4
HYPERBOLIC_SHAPE_MIN = 1e-6
BBH_PARAMETER_NAMES = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
    "ra", "dec", "a_1", "a_2", "cos_theta_jn", "cos_tilt_1",
    "cos_tilt_2", "phi_12", "phi_jl", "geocent_time",
]
PERIODIC = ["psi", "phase", "ra", "phi_12", "phi_jl"]


def build_problem(duration=4.0, fs=4096.0, fmin=20.0, fmax=512.0, nsegs=4, workers=1):
    """Download GW150914 open data and build the selected likelihood object."""
    detectors = ["H1", "L1"]
    noise = DetectorNoise(duration, fs, GW150914_TIME, detectors,
                          minimum_frequency=fmin, maximum_frequency=fmax)
    print("> downloading GW150914 open data (GWOSC)...")
    noise.generate_noise(real_noise=True)

    # geocent_time is SAMPLED (not pinned): the exact merger time of real data is
    # unknown a priori, and fixing it to a rounded value biases the masses (the
    # template can't time-align, so chirp_mass/q distort to compensate).
    template = GW(noise, approximant="IMRPhenomPv2", reference_frequency=50.0,
                  parameters=BBH_PARAMETER_NAMES, static_parameters={},
                  n_jobs=workers)

    f, asd0 = template.detector_asd_masked(0)
    psd = np.array([asd0 ** 2, template.detector_asd_masked(1)[1] ** 2])
    data = np.array([template.detector_data_fd(0), template.detector_data_fd(1)])
    likelihood = GWLikelihoods(data=data, f=f, ifos_list=detectors, noise=psd,
                               template=template, ddims=False, nsegs=nsegs,
                               gpu=False, cpu_cores=workers)
    return likelihood


def make_priors(nsegs, like="hyperbolic"):
    pr = bilby.core.prior.PriorDict()
    pr["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(23.0, 35.0, name="chirp_mass", latex_label=r"$\mathcal{M}$")
    pr["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(0.4, 1.0, name="mass_ratio", latex_label="$q$")
    pr["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(name="luminosity_distance", minimum=100.0, maximum=2000.0, latex_label="$d_L$", unit="Mpc")
    pr["psi"] = bilby.core.prior.Uniform(0, np.pi, name="psi", latex_label=r"$\psi$")
    pr["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phase", latex_label=r"$\phi$")
    pr["ra"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="ra", latex_label=r"$\alpha$")
    pr["dec"] = bilby.core.prior.Cosine(name="dec", latex_label=r"$\delta$")
    pr["a_1"] = bilby.core.prior.Uniform(0, 0.99, name="a_1", latex_label=r"$a_1$")
    pr["a_2"] = bilby.core.prior.Uniform(0, 0.99, name="a_2", latex_label=r"$a_2$")
    pr["cos_theta_jn"] = bilby.core.prior.Uniform(-1, 1, name="cos_theta_jn", latex_label=r"$\cos\theta_{JN}$")
    pr["cos_tilt_1"] = bilby.core.prior.Uniform(-1, 1, name="cos_tilt_1", latex_label=r"$\cos t_1$")
    pr["cos_tilt_2"] = bilby.core.prior.Uniform(-1, 1, name="cos_tilt_2", latex_label=r"$\cos t_2$")
    pr["phi_12"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phi_12", latex_label=r"$\phi_{12}$")
    pr["phi_jl"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phi_jl", latex_label=r"$\phi_{JL}$")
    pr["geocent_time"] = bilby.core.prior.Uniform(
        GW150914_TIME - 0.1, GW150914_TIME + 0.1, name="geocent_time", latex_label=r"$t_c$")
    # Noise priors depend on the likelihood family:
    #   * gaussian: none (fixed-PSD Gaussian likelihood)
    #   * hyperbolic: scale alpha + per-segment delta (heavy-tailed shape)
    #   * whittle:    per-segment log10 noise level (data-driven Whittle)
    if like == "gaussian":
        noise_priors = {}
    elif like == "whittle":
        noise_priors = {f"log_level_{i}": bilby.core.prior.Uniform(-1.0, 1.0)
                        for i in range(nsegs)}
    else:  # hyperbolic
        noise_priors = {r"$\alpha$": bilby.core.prior.Uniform(HYPERBOLIC_SHAPE_MIN, 30.0)}
        for i in range(nsegs):
            noise_priors[r"$\delta_{}$".format(i)] = bilby.core.prior.Uniform(
                HYPERBOLIC_SHAPE_MIN, 30.0)
    return pr, noise_priors


def run_one(sampler, likelihood, priors, noise_priors, outdir, quick, like="hyperbolic"):
    if sampler == "eryn":
        kw = dict(nwalkers=44, ntemps=4, burn=50, nsteps=100) if quick \
            else dict(nwalkers=50, ntemps=20, burn=20000, nsteps=70000)
    else:
        kw = dict(n_total=2000, n_effective=512, n_active=256) if quick \
            else dict(n_total=50000, n_effective=8000, n_active=2000)
    like_fn = {
        "gaussian":   likelihood.gaussian,
        "hyperbolic": likelihood.hyperbolic_classic,
        "whittle":    likelihood.whittle_level,
    }[like]
    tag = f"gw150914_{like}_{sampler}"
    t0 = time.perf_counter()
    inf = LVKinference(
        like_fn, sampler_name=sampler, priors=priors,
        noise_priors=noise_priors,
        common_params={"save_dir": outdir, "TAG": tag, "like": like},
        sampler_kwargs=kw, periodic=PERIODIC,
    )
    inf.run()
    wall = time.perf_counter() - t0
    samples = inf.get_samples()
    mc = np.median(samples[:, 0])
    print(f"[{sampler}] wall-clock {wall:.1f} s ({wall/60:.1f} min) | "
          f"{samples.shape[0]} samples | chirp_mass median {mc:.2f} Msun")
    return sampler, wall, samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampler", choices=["eryn", "pocomc", "both"], default="both")
    p.add_argument("--likelihood", choices=["gaussian", "hyperbolic", "whittle"],
                   default="hyperbolic",
                   help="GW likelihood family. 'gaussian' for direct GWTC comparison, "
                        "'hyperbolic' for the heavy-tailed robustness story, 'whittle' "
                        "for data-driven per-segment noise level.")
    p.add_argument("--nsegs", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--outdir", default="results/gw150914")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--setup-only", action="store_true",
                   help="download + one likelihood evaluation, then exit")
    args = p.parse_args()

    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    likelihood = build_problem(nsegs=args.nsegs, workers=args.workers)
    priors, noise_priors = make_priors(args.nsegs, like=args.likelihood)

    if args.setup_only:
        # One vectorised likelihood eval at two actual prior draws.
        theta = np.column_stack([prior.sample(2) for prior in priors.values()])
        if noise_priors:
            noise_theta = np.column_stack([prior.sample(2) for prior in noise_priors.values()])
            theta = np.column_stack([theta, noise_theta])
        like_fn = {"gaussian": likelihood.gaussian,
                   "hyperbolic": likelihood.hyperbolic_classic,
                   "whittle": likelihood.whittle_level}[args.likelihood]
        ll = like_fn(theta)
        print(f"> setup OK ({args.likelihood}): likelihood eval -> {np.asarray(ll).ravel()}")
        return

    samplers = ["eryn", "pocomc"] if args.sampler == "both" else [args.sampler]
    results = [run_one(s, likelihood, priors, noise_priors, args.outdir,
                       args.quick, like=args.likelihood)
               for s in samplers]

    print("\n========== TIMING SUMMARY (GW150914) ==========")
    for sampler, wall, samples in results:
        print(f"  {sampler:8s}: {wall:8.1f} s  ({wall/60:5.1f} min)  [{samples.shape[0]} samples]")


if __name__ == "__main__":
    main()
