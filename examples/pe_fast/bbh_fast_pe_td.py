"""Fast 2-parameter BBH parameter estimation with time-domain likelihoods.

Gaussian-noise BBH injection, then PE over **only the two intrinsic mass
parameters** (chirp mass + mass ratio). The other 12 CBC parameters are held at
their injected values, which makes this a fast, low-dimensional demonstration of
the time-domain covariance likelihoods.

Run the same problem through either sampler and compare::

    python examples/pe_fast/bbh_fast_pe_td.py --sampler eryn
    python examples/pe_fast/bbh_fast_pe_td.py --sampler pocomc

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
from hyperwave.likelihoods import TimeDomainGWLikelihoods

# Full intrinsic+extrinsic parameter vector the waveform generator expects.
BBH_PARAMETER_NAMES = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
    "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
    "cos_tilt_2", "phi_12", "phi_jl",
]

HYPERBOLIC_SHAPE_MIN = 1e-6


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


def build_problem(
    duration=2.0,
    fs=1024.0,
    fmin=20.0,
    fmax=50.0,
    time_bands=2,
    likelihood_type="hyperbolic",
    likelihood_method="gohberg-semencul",
    seed=42,
):
    """Inject a BBH into Gaussian noise and build a TD covariance likelihood.

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

    f, psd0 = template.detector_psd(0)
    psd = np.array([psd0, template.detector_psd(1)[1]])
    data_td = np.array([template.detector_data_td(0), template.detector_data_td(1)])

    likelihood = TimeDomainGWLikelihoods(
        data=data_td, sampling_rate=fs, ifos_list=detectors,
        f=f, noise=psd, template=template,
        likelihood_method=likelihood_method,
        minimum_frequency=fmin, maximum_frequency=fmax,
        time_bands=time_bands,
        detector_likelihoods=likelihood_type,
    )

    fixed = np.asarray(theta_true[2:], dtype=float)            # the 12 held-fixed CBC params

    def loglike_2d(sampled):
        sampled = np.atleast_2d(np.asarray(sampled, dtype=float))
        mc_q, noise_shape = sampled[:, :2], sampled[:, 2:]
        wf = np.column_stack([mc_q, np.tile(fixed, (sampled.shape[0], 1))])  # (N, 14)
        full = np.column_stack([wf, noise_shape])                            # (N, 14 + n_noise)
        return likelihood.log_likelihood(full)

    n_noise = {
        "gaussian": 0,
        "student-t": time_bands,
        "hyperbolic": 2 * time_bands,
    }[likelihood_type]
    info = dict(
        f=f, psd=psd, data=data_td, template=template, likelihood=likelihood,
        time_bands=time_bands,
    )
    return loglike_2d, theta_true, n_noise, info


def make_priors(time_bands, like):
    """Bilby priors: 2 science params + optional TD noise-shape params."""
    priors = bilby.core.prior.PriorDict()
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
        minimum=25.0, maximum=31.0, name="chirp_mass", latex_label=r"$\mathcal{M}$")
    priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(
        minimum=0.5, maximum=1.0, name="mass_ratio", latex_label="$q$")

    if like == "gaussian":
        return priors, {}
    if like == "student-t":
        noise_priors = {}
        for i in range(time_bands):
            noise_priors[r"$\nu_{}$".format(i)] = bilby.core.prior.Uniform(
                minimum=2.0, maximum=30.0)
        return priors, noise_priors

    noise_priors = {}
    for i in range(time_bands):
        noise_priors[r"$\alpha_{}$".format(i)] = bilby.core.prior.Uniform(
            minimum=HYPERBOLIC_SHAPE_MIN, maximum=30.0)
    for i in range(time_bands):
        noise_priors[r"$\delta_{}$".format(i)] = bilby.core.prior.Uniform(
            minimum=HYPERBOLIC_SHAPE_MIN, maximum=30.0)
    return priors, noise_priors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampler", choices=["eryn", "pocomc"], default="eryn")
    p.add_argument("--likelihood", choices=["gaussian", "student-t", "hyperbolic"],
                   default="gaussian")
    p.add_argument("--time-bands", type=int, default=2,
                   help="number of uniform time-domain sample-count bands")
    p.add_argument("--likelihood-method", default="gohberg-semencul",
                   choices=["direct-inversion", "cholesky-solve-triangular",
                            "toeplitz-inversion", "gohberg-semencul", "gs"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results/pe_fast_td")
    p.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = p.parse_args()

    if args.time_bands < 1:
        raise ValueError("--time-bands must be positive")

    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    loglike_2d, theta_true, n_noise, info = build_problem(
        time_bands=args.time_bands,
        likelihood_type=args.likelihood,
        likelihood_method=args.likelihood_method,
        seed=args.seed,
    )
    priors, noise_priors = make_priors(args.time_bands, args.likelihood)
    if len(noise_priors) != n_noise and args.likelihood != "gaussian":
        raise RuntimeError("noise prior count does not match the TD likelihood layout")
    tag = f"bbh_fast_td_{args.likelihood}_{args.sampler}"

    if args.sampler == "eryn":
        kw = dict(nwalkers=20, ntemps=4, burn=50, nsteps=100) if args.quick \
            else dict(nwalkers=40, ntemps=10, burn=3000, nsteps=8000)
    else:
        kw = dict(n_total=2000, n_effective=512, n_active=256) if args.quick \
            else dict(n_total=30000, n_effective=4000, n_active=1000)

    t0 = time.perf_counter()
    inf = LVKinference(
        loglike_2d, sampler_name=args.sampler, priors=priors, noise_priors=noise_priors,
        common_params={"save_dir": args.outdir, "TAG": tag, "like": args.likelihood},
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
