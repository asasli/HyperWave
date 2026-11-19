"""Recover a signal with Gaussian noise in one detector and hyperbolic noise in another.

This is a small deterministic example of the detector-level noise model mask:

    H1 -> Gaussian likelihood
    L1 -> hyperbolic likelihood with its own alpha/delta parameters

Run from the repository root with:

    PYTHONPATH=src python examples/noise/mixed_detector_noise_recovery.py --no-plot
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.optimize import minimize

from hyperwave.likelihoods import GWLikelihoods


class AmplitudeTemplate:
    parameters = ["amplitude"]

    def __init__(self, ifos, f):
        self.ifos = list(ifos)
        self.f = np.asarray(f)
        self.detector_scale = {"H1": 1.0, "L1": 1.15}
        profile = (self.f / self.f[0]) ** -0.4 * np.exp(0.04j * self.f)
        self.profile = profile / np.sqrt(np.vdot(profile, profile).real / profile.size)
        self.batch_calls = 0

    def make_injections_to_ifo(self, theta):
        theta = np.asarray(theta)
        return {
            ifo: self.detector_scale[ifo] * theta[0] * self.profile
            for ifo in self.ifos
        }

    def make_injections_to_ifo_batch(self, theta):
        self.batch_calls += 1
        theta = np.atleast_2d(np.asarray(theta))
        signal = np.zeros((theta.shape[0], len(self.ifos), self.f.size), dtype=complex)
        for i, ifo in enumerate(self.ifos):
            signal[:, i, :] = self.detector_scale[ifo] * theta[:, :1] * self.profile[None, :]
        return signal


def _complex_normal(rng, n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)


def _complex_student_t(rng, n, df):
    scale = np.sqrt(2.0 * df / (df - 2.0))
    return (rng.standard_t(df, n) + 1j * rng.standard_t(df, n)) / scale


def build_problem(seed=1234, nfreq=96, nsegs=2):
    rng = np.random.default_rng(seed)
    f = np.linspace(20.0, 180.0, nfreq)
    ifos = ["H1", "L1"]
    true_amplitude = 1.0
    template = AmplitudeTemplate(ifos, f)
    clean = template.make_injections_to_ifo([true_amplitude])

    sigma = 0.18
    h1_noise = sigma * _complex_normal(rng, nfreq)
    l1_noise = sigma * _complex_student_t(rng, nfreq, df=3.0)
    glitch = np.zeros(nfreq, dtype=complex)
    glitch[36:44] = 1.2 * template.profile[36:44]

    data = np.array([
        clean["H1"] + h1_noise,
        clean["L1"] + l1_noise + glitch,
    ])
    psd = np.ones((2, nfreq))
    return dict(
        f=f,
        data=data,
        psd=psd,
        ifos=ifos,
        nsegs=nsegs,
        true_amplitude=true_amplitude,
    )


def fit_all_gaussian(problem):
    template = AmplitudeTemplate(problem["ifos"], problem["f"])
    likelihood = GWLikelihoods(
        data=problem["data"],
        f=problem["f"],
        ifos_list=problem["ifos"],
        noise=problem["psd"],
        template=template,
        ddims=False,
        nsegs=problem["nsegs"],
        cpu_cores=1,
    )

    amp = np.linspace(0.4, 1.8, 600)
    theta = amp[:, None]
    logl = np.atleast_1d(likelihood.gaussian(theta))
    best = int(np.argmax(logl))
    return dict(amplitude=float(amp[best]), logl=float(logl[best]), batch_calls=template.batch_calls)


def fit_mixed_gaussian_hyperbolic(problem):
    template = AmplitudeTemplate(problem["ifos"], problem["f"])
    likelihood = GWLikelihoods(
        data=problem["data"],
        f=problem["f"],
        ifos_list=problem["ifos"],
        noise=problem["psd"],
        template=template,
        ddims=False,
        nsegs=problem["nsegs"],
        cpu_cores=1,
        detector_dependent_noise=True,
        detector_noise_models=["gaussian", "hyperbolic"],
    )

    def objective(x):
        amp = x[0]
        alpha = np.exp(x[1])
        delta = np.exp(x[2:])
        theta = np.concatenate([[amp, alpha], delta])
        return -float(likelihood.hyperbolic_classic(theta))

    starts = [
        [1.0, np.log(1.0), np.log(0.2), np.log(0.2)],
        [1.0, np.log(3.0), np.log(0.5), np.log(0.5)],
        [1.2, np.log(1.0), np.log(1.0), np.log(1.0)],
    ]
    bounds = [(0.4, 1.8), (np.log(0.05), np.log(30.0))]
    bounds += [(np.log(0.03), np.log(30.0))] * problem["nsegs"]

    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        for start in starts
    ]
    fit = min(fits, key=lambda out: out.fun)
    amp = float(fit.x[0])
    alpha = float(np.exp(fit.x[1]))
    delta = np.exp(fit.x[2:]).astype(float)
    return dict(
        amplitude=amp,
        alpha_L1=alpha,
        delta_L1=delta,
        logl=float(-fit.fun),
        success=bool(fit.success),
        batch_calls=template.batch_calls,
    )


def maybe_plot(problem, gaussian, mixed, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"plot skipped: {exc}")
        return None

    os.makedirs(outdir, exist_ok=True)
    f = problem["f"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(f, np.abs(problem["data"][0]), label="H1 data (Gaussian)", lw=1.2)
    ax.plot(f, np.abs(problem["data"][1]), label="L1 data (heavy-tailed)", lw=1.2)
    ax.axhline(problem["true_amplitude"], color="k", ls="--", lw=1.0, label="true amplitude")
    ax.axhline(gaussian["amplitude"], color="C2", ls=":", lw=1.3, label="all-Gaussian fit")
    ax.axhline(mixed["amplitude"], color="C3", ls="-.", lw=1.3, label="mixed fit")
    ax.set_xlabel("frequency")
    ax.set_ylabel("data amplitude")
    ax.legend(loc="best", fontsize=8)
    path = os.path.join(outdir, "mixed_detector_noise_recovery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def run_demo(seed=1234, make_plot=True, outdir="results/noise"):
    problem = build_problem(seed=seed)
    gaussian = fit_all_gaussian(problem)
    mixed = fit_mixed_gaussian_hyperbolic(problem)
    plot_path = maybe_plot(problem, gaussian, mixed, outdir) if make_plot else None
    return dict(problem=problem, gaussian=gaussian, mixed=mixed, plot_path=plot_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--outdir", default="results/noise")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    result = run_demo(seed=args.seed, make_plot=not args.no_plot, outdir=args.outdir)
    true = result["problem"]["true_amplitude"]
    gaussian = result["gaussian"]
    mixed = result["mixed"]

    print(f"true amplitude:        {true:.4f}")
    print(f"all-Gaussian recovery: {gaussian['amplitude']:.4f}")
    print(f"mixed recovery:        {mixed['amplitude']:.4f}")
    print(f"L1 alpha:              {mixed['alpha_L1']:.4f}")
    print(f"L1 delta:              {mixed['delta_L1']}")
    print(f"mixed optimizer ok:    {mixed['success']}")
    print(f"mixed batch calls:     {mixed['batch_calls']}")
    if result["plot_path"] is not None:
        print(f"plot:                  {result['plot_path']}")


if __name__ == "__main__":
    main()
