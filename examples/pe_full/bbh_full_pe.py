"""Full 14-parameter BBH parameter estimation with the hyperbolic likelihood.

Gaussian-noise BBH injection, then PE over the **full** CBC parameter vector
(masses, distance, spins, orientation, sky) plus the per-segment noise-shape
nuisance parameters of the heavy-tailed hyperbolic likelihood. Runs with either
Eryn (parallel-tempered RJ-free MCMC) or pocoMC (preconditioned nested-style
SMC); defaults to pocoMC. This is a cluster-scale run — see the companion
``examples/clusters/bbh_full_pe.slurm``.

    python examples/pe_full/bbh_full_pe.py --sampler pocomc
    python examples/pe_full/bbh_full_pe.py --sampler eryn --quick   # smoke test
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

BBH_PARAMETER_NAMES = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
    "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
    "cos_tilt_2", "phi_12", "phi_jl",
]
PERIODIC = ["psi", "phase", "ra", "phi_12", "phi_jl"]


def injected_bbh():
    m1, m2 = 36.0, 29.0
    mc = (m1 + m2) * (m1 * m2 / (m1 + m2) ** 2) ** 0.6
    theta = dict(
        chirp_mass=mc, mass_ratio=m2 / m1, luminosity_distance=600.0, psi=1.1,
        phase=0.9, ra=1.375, dec=-0.2108, chi_1=0.0, chi_2=0.0,
        cos_theta_jn=np.cos(0.4), cos_tilt_1=1.0, cos_tilt_2=1.0, phi_12=0.0, phi_jl=0.0,
    )
    return [theta[k] for k in BBH_PARAMETER_NAMES]


def build_problem(duration=4.0, fs=2048.0, fmin=20.0, fmax=512.0, nsegs=4, seed=42):
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
    return likelihood, theta_true


def make_priors(nsegs):
    """Full CBC priors + (1 alpha + nsegs delta) hyperbolic noise-shape priors."""
    pr = bilby.core.prior.PriorDict()
    pr["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(25.0, 31.0, name="chirp_mass", latex_label=r"$\mathcal{M}$")
    pr["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(0.5, 1.0, name="mass_ratio", latex_label="$q$")
    pr["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(name="luminosity_distance", minimum=200.0, maximum=2000.0, latex_label="$d_L$", unit="Mpc")
    pr["psi"] = bilby.core.prior.Uniform(0, np.pi, name="psi", latex_label=r"$\psi$")
    pr["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phase", latex_label=r"$\phi$")
    pr["ra"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="ra", latex_label=r"$\alpha$")
    pr["dec"] = bilby.core.prior.Cosine(name="dec", latex_label=r"$\delta$")
    pr["chi_1"] = bilby.core.prior.Uniform(-1, 1, name="chi_1", latex_label=r"$\chi_1$")
    pr["chi_2"] = bilby.core.prior.Uniform(-1, 1, name="chi_2", latex_label=r"$\chi_2$")
    pr["cos_theta_jn"] = bilby.core.prior.Uniform(-1, 1, name="cos_theta_jn", latex_label=r"$\cos\theta_{JN}$")
    pr["cos_tilt_1"] = bilby.core.prior.Uniform(-1, 1, name="cos_tilt_1", latex_label=r"$\cos t_1$")
    pr["cos_tilt_2"] = bilby.core.prior.Uniform(-1, 1, name="cos_tilt_2", latex_label=r"$\cos t_2$")
    pr["phi_12"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phi_12", latex_label=r"$\phi_{12}$")
    pr["phi_jl"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phi_jl", latex_label=r"$\phi_{JL}$")

    noise_priors = {r"$\alpha$": bilby.core.prior.Uniform(0.0, 30.0)}
    for i in range(nsegs):
        noise_priors[r"$\delta_{}$".format(i)] = bilby.core.prior.Uniform(0.0, 30.0)
    return pr, noise_priors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampler", choices=["eryn", "pocomc"], default="pocomc")
    p.add_argument("--nsegs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--outdir", default="results/pe_full")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-total", type=int, default=None, help="pocoMC n_total override")
    p.add_argument("--flow", action="store_true",
                   help="eryn only: mix a trained normalizing-flow independence proposal "
                        "(30%%) with the stretch move — global jumps through the "
                        "correlated/degenerate directions of the full BBH posterior")
    p.add_argument("--flow-frac", type=float, default=0.3)
    p.add_argument("--flow-train-every", type=int, default=200)
    args = p.parse_args()

    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    likelihood, theta_true = build_problem(nsegs=args.nsegs, seed=args.seed)
    priors, noise_priors = make_priors(args.nsegs)
    tag = f"bbh_full_{args.sampler}" + ("_flow" if args.flow else "")

    if args.sampler == "eryn":
        kw = dict(nwalkers=44, ntemps=4, burn=50, nsteps=100) if args.quick \
            else dict(nwalkers=50, ntemps=20, burn=20000, nsteps=70000)
        if args.flow:
            # exact-MH flow proposal over the FULL sampled vector (science +
            # noise-shape); prior fallback until the callback has trained it.
            from eryn.moves import StretchMove
            from hyperwave.inference.flow_proposals import (
                FlowTrainingCallback,
                build_pe_flow_proposal,
                make_flow_distribution_move,
            )
            ordered = [priors[k] for k in priors] + list(noise_priors.values())
            name_list = list(priors.keys())
            periodic_idx = [name_list.index(n) for n in PERIODIC]
            flow = build_pe_flow_proposal(ordered, periodic_indices=periodic_idx,
                                          min_training_samples=512)
            flow_move = make_flow_distribution_move({"model_0": flow})
            callback = FlowTrainingCallback({"model_0": flow},
                                            every=args.flow_train_every, verbose=True)
            kw.update(moves=[(StretchMove(), 1.0 - args.flow_frac),
                             (flow_move, args.flow_frac)],
                      update_fn=callback, update_iterations=args.flow_train_every)
            print(f"[flow] {args.flow_frac:.0%} flow independence proposal, "
                  f"retrain every {args.flow_train_every} iters")
    else:
        kw = dict(n_total=2000, n_effective=512, n_active=256) if args.quick \
            else dict(n_total=50000, n_effective=8000, n_active=2000)
        if args.n_total is not None:
            kw["n_total"] = args.n_total

    print(f"[setup] sampler={args.sampler} seed={args.seed} kw={kw}")
    t0 = time.perf_counter()
    inf = LVKinference(
        likelihood.hyperbolic_classic, sampler_name=args.sampler, priors=priors,
        noise_priors=noise_priors,
        common_params={"save_dir": args.outdir, "TAG": tag, "like": "hyperbolic"},
        sampler_kwargs=kw, periodic=PERIODIC,
    )
    inf.run()
    wall = time.perf_counter() - t0
    samples = inf.get_samples()
    print(f"\n[{args.sampler}] full PE wall-clock {wall:.1f} s "
          f"({wall/60:.1f} min) | {samples.shape[0]} samples, {samples.shape[1]} dims")

    try:
        from hyperwave.plots.corners import plot_posterior
        labels = [r"$\mathcal{M}$", "$q$", "$d_L$", r"$\psi$", r"$\phi$", r"$\alpha$",
                  r"$\delta$", r"$\chi_1$", r"$\chi_2$", r"$\cos\theta_{JN}$",
                  r"$\cos t_1$", r"$\cos t_2$", r"$\phi_{12}$", r"$\phi_{JL}$"]
        plot_posterior(samples[:, :14], param_names=labels, case="hyperbolic",
                       package="corner", truths=theta_true, save_dir=args.outdir + "/",
                       TAG=tag, show=False)
        print(f"  posterior -> {args.outdir}/{tag}*")
    except Exception as exc:  # pragma: no cover
        print(f"  (corner skipped: {exc})")


if __name__ == "__main__":
    main()
