"""Paper-faithful BBH PE with pocoMC (real LVK noise) — timing harness.

Faithful to the PI's working pattern: real H1/L1 noise around the trigger, a
GW150914-like BBH injection, the heavy-tailed **hyperbolic** likelihood with
per-segment α/δ, broad astrophysical priors, and pocoMC. The pocoMC config
(n_total / n_effective / n_active / n_steps) and ``cpu_cores`` are exposed so the
two PI timing points can be run:

    # Run A
    python examples/pe_timing/bbh_paper_pocomc.py \\
        --n-total 10000 --n-effective 2048 --n-active 1024 --n-steps 20 --cpu-cores 32

    # Run B
    python examples/pe_timing/bbh_paper_pocomc.py \\
        --n-total 10000 --n-effective 4096 --n-active 2048 --n-steps 40 --cpu-cores 64

Notes
-----
* α and δ are sampled **linearly** Uniform(0, 30) in this HyperWave version (the
  old paper code used log10). The science (CBC) posteriors are what must match.
* The waveform batch is generated through the *batched* template path (≈6 ms per
  waveform). On this build that path is serial — joblib (``GW(n_jobs>1)``)
  deadlocks here, and ``cpu_cores`` on the batched path is a no-op — so the run
  is single-core. It is still cheap because the per-waveform cost is tiny.
"""

from __future__ import annotations

import argparse
import os
import time

import bilby
import numpy as np

from hyperwave.detectors.lvk import noise as noisemod
from hyperwave.detectors.lvk import waveform as waveformmod
from hyperwave.likelihoods import gwparallel
from hyperwave.inference import sampling

# silence bilby chatter
import logging, io  # noqa: E402
_bl = logging.getLogger("bilby"); _bl.propagate = False
_bl.addHandler(logging.StreamHandler(stream=io.StringIO())); _bl.setLevel(logging.WARNING)
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

# geocent_time (t_c) is SAMPLED (last position) — not held static — so the
# merger time is inferred, as in a real analysis.
PARAMS = ["chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
          "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
          "cos_tilt_2", "phi_12", "phi_jl", "geocent_time"]
PERIODIC = ["psi", "phase", "ra", "phi_12", "phi_jl"]
DT_PRIOR = 0.1   # geocent_time prior half-width [s] around the trigger

TRIGGER_TIME = 1165578732.45 - 0.025
M1, M2 = 36.0, 29.0


def injected_bbh():
    eta = M1 * M2 / (M1 + M2) ** 2
    mc = (M1 + M2) * eta ** 0.6
    q = M2 / M1
    return [mc, q, 1000.0, 1.228444, 0.641716, 1.375, 0.2108,
            0.0, 0.0, np.cos(0.4), np.cos(0.0), np.cos(0.0), 0.0, 0.0,
            TRIGGER_TIME], mc


def build_problem(nsegs, cpu_cores, real_noise=True, fmax=800.0, seed=42):
    detectors = ["H1", "L1"]
    duration, sampling_rate = 4, 4096
    static_parameters = {}   # nothing held fixed — geocent_time is sampled now
    bbh_params, mc = injected_bbh()

    # signal: real noise + injected BBH
    ng = noisemod.DetectorNoise(duration, sampling_rate, TRIGGER_TIME, detectors,
                                maximum_frequency=fmax)
    ng.generate_noise(real_noise=real_noise, seed=seed)
    BBH = waveformmod.GW(ng, reference_frequency=50.0, parameters=PARAMS,
                         static_parameters=static_parameters)
    BBH.make_injections_to_ifo(bbh_params)

    # template: independent real-noise realisation (its PSD is what the
    # likelihood whitens with), same waveform model
    tmpl_noise = noisemod.DetectorNoise(duration, sampling_rate, TRIGGER_TIME, detectors)
    tmpl_noise.generate_noise(real_noise=real_noise, seed=seed + 1)
    template = waveformmod.GW(tmpl_noise, approximant="IMRPhenomPv2",
                              reference_frequency=50.0, parameters=PARAMS,
                              static_parameters=static_parameters)

    f, Sn0 = BBH.detector_asd_masked(0)
    Sn = np.array([Sn0 ** 2, BBH.detector_asd_masked(1)[1] ** 2])
    data = np.array([BBH.detector_data_fd(0), BBH.detector_data_fd(1)])

    likels = gwparallel.GWLikelihoods(
        data=data, f=f, ifos_list=detectors, noise=Sn, template=template,
        ddims=False, nsegs=nsegs, gpu=False, infs=-1e300, cpu_cores=cpu_cores)
    return likels, bbh_params, mc


def make_priors(nsegs, mc):
    pr = bilby.core.prior.PriorDict()
    pr["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(25.0, 30.0, name="chirp_mass", latex_label=r"$\mathcal{M}$")
    pr["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(0.5, 0.9, name="mass_ratio", latex_label="$q$")
    pr["luminosity_distance"] = bilby.gw.prior.UniformComovingVolume(name="luminosity_distance", minimum=500.0, maximum=2000.0, latex_label="$d_L$", unit="Mpc")
    # NOTE: dict order MUST match PARAMS (the template reads the vector
    # positionally) -> psi before phase.
    pr["psi"] = bilby.gw.prior.Uniform(name="psi", minimum=0, maximum=np.pi, latex_label=r"$\psi$")
    pr["phase"] = bilby.gw.prior.Uniform(name="phase", minimum=0, maximum=2 * np.pi, latex_label=r"$\phi$")
    pr["ra"] = bilby.gw.prior.Uniform(name="ra", minimum=0, maximum=2 * np.pi, latex_label=r"$\mathrm{RA}$")
    pr["dec"] = bilby.core.prior.Cosine(name="dec", minimum=-np.pi / 2, maximum=np.pi / 2, latex_label=r"$\mathrm{Dec}$")
    pr["chi_1"] = bilby.gw.prior.Uniform(name="chi_1", minimum=-1, maximum=1)
    pr["chi_2"] = bilby.gw.prior.Uniform(name="chi_2", minimum=-1, maximum=1)
    pr["cos_theta_jn"] = bilby.gw.prior.Uniform(name="cos_theta_jn", minimum=-1, maximum=1)
    pr["cos_tilt_1"] = bilby.gw.prior.Uniform(name="cos_tilt_1", minimum=-1, maximum=1)
    pr["cos_tilt_2"] = bilby.gw.prior.Uniform(name="cos_tilt_2", minimum=-1, maximum=1)
    pr["phi_12"] = bilby.gw.prior.Uniform(name="phi_12", minimum=0, maximum=2 * np.pi)
    pr["phi_jl"] = bilby.gw.prior.Uniform(name="phi_jl", minimum=0, maximum=2 * np.pi)
    pr["geocent_time"] = bilby.core.prior.Uniform(
        name="geocent_time", minimum=TRIGGER_TIME - DT_PRIOR,
        maximum=TRIGGER_TIME + DT_PRIOR, latex_label="$t_c$")

    # hyperbolic noise shape: one shared α + one δ per segment, linear (0, 30)
    names = [r"$\alpha$"] + [r"$\delta_{{{}}}$".format(i) for i in range(nsegs)]
    noise_priors = {n: bilby.gw.prior.Uniform(minimum=0, maximum=30) for n in names}
    return pr, noise_priors


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nsegs", type=int, default=4)
    p.add_argument("--cpu-cores", type=int, default=32)
    p.add_argument("--n-total", type=int, default=10000)
    p.add_argument("--n-effective", type=int, default=2048)
    p.add_argument("--n-active", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--gaussian", action="store_true", help="use Gaussian noise instead of real LVK data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results/pe_timing")
    p.add_argument("--quick", action="store_true", help="tiny pocoMC settings, smoke test")
    args = p.parse_args()

    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    print("Generating data + injecting signal...")
    likels, bbh_params, mc = build_problem(
        args.nsegs, args.cpu_cores, real_noise=not args.gaussian, seed=args.seed)
    priors, noise_priors = make_priors(args.nsegs, mc)

    kw = dict(n_total=2000, n_effective=512, n_active=256, n_steps=10) if args.quick else \
        dict(n_total=args.n_total, n_effective=args.n_effective,
             n_active=args.n_active, n_steps=args.n_steps)
    tag = f"{TRIGGER_TIME}_hyper_e{kw['n_effective']}_a{kw['n_active']}_s{kw['n_steps']}_c{args.cpu_cores}"
    print(f"[setup] nsegs={args.nsegs} cpu_cores={args.cpu_cores} "
          f"dims={len(priors)+len(noise_priors)}  noise={'gauss' if args.gaussian else 'real'}")
    print(f"[pocomc] {kw}")

    t0 = time.perf_counter()
    inf = sampling.LVKinference(
        likels.hyperbolic_classic, priors=priors, noise_priors=noise_priors,
        sampler_name="pocomc",
        common_params={"save_dir": args.outdir, "TAG": tag, "like": "hyperbolic"},
        sampler_kwargs=kw, periodic=PERIODIC)
    inf.run()
    wall = time.perf_counter() - t0
    samples = inf.get_samples()

    print(f"\n[RESULT] cpu_cores={args.cpu_cores} n_eff={kw['n_effective']} "
          f"n_act={kw['n_active']} n_steps={kw['n_steps']}")
    print(f"  wall-clock = {wall:.1f} s ({wall/60:.2f} min) | {samples.shape[0]} samples, {samples.shape[1]} dims")
    labels = ["Mc", "q", "dL", "psi", "phase", "ra", "dec", "chi1", "chi2",
              "cosThJN", "cosT1", "cosT2", "phi12", "phiJL", "tc"]
    print("  CBC recovery (median / truth):")
    for i, lab in enumerate(labels):
        col = samples[:, i]
        med, std = np.median(col), np.std(col)
        pull = (med - bbh_params[i]) / std if std > 0 else np.nan
        print(f"    {lab:8s} med={med:+.4g}  truth={bbh_params[i]:+.4g}  pull={pull:+.2f}")

    np.savez(os.path.join(args.outdir, f"{tag}_samples.npz"),
             samples=samples, truths=np.array(bbh_params), wall=wall,
             config=dict(n_total=kw["n_total"], n_effective=kw["n_effective"],
                         n_active=kw["n_active"], n_steps=kw["n_steps"],
                         cpu_cores=args.cpu_cores))
    print(f"  saved -> {args.outdir}/{tag}_samples.npz")
    return wall, samples


if __name__ == "__main__":
    main()
