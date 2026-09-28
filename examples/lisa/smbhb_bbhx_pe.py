"""LISA SMBHB parameter estimation with bbhx + the hyperbolic likelihood.

Massive black-hole binary (SMBHB) PE for LISA. **bbhx** (``BBHWaveformFD``)
generates the frequency-domain A/E/T TDI waveform (LISA response included), and
HyperWave's heavy-tailed *hyperbolic* likelihood models the noise. The wiring
goes through HyperWave's LISA A/E bridge:

    bbhx generator  ->  LISAAETTemplate  ->  build_lisa_aet_likelihood  ->  sampler

i.e. the *same* hyperbolic ``GWLikelihoods`` / ``LVKinference`` stack as the
ground-based ``examples/pe_full/bbh_full_pe.py``, with a LISA front end. Runs
with **Eryn** (parallel-tempered MCMC), **pocoMC** (preconditioned SMC), or
``--sampler both`` to time them head-to-head.

**Vectorised.** bbhx is natively batched: one generator call evaluates the whole
walker population. The bridge exposes that as ``make_injections_to_ifo_batch`` so
``GWLikelihoods`` generates all walkers' waveforms in a single bbhx call (both
samplers evaluate the likelihood vectorised) — far faster than a per-walker loop.

Dependencies (the ``hyperwave-dev`` env; NOT the default ``hyperwave-gpu``)::

    pip install bbhx lisaanalysistools     # see ENVIRONMENT.md

Run (Intel-Skylake/AVX-512 node with the LISA stack; the AMD A100 nodes SIGILL
the prebuilt wheels — see examples/clusters/lisa_pe.slurm)::

    python examples/lisa/smbhb_bbhx_pe.py --sampler both --quick
    python examples/lisa/smbhb_bbhx_pe.py --sampler eryn --steps 40000
    python examples/lisa/smbhb_bbhx_pe.py --snr-only
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    from bbhx.waveformbuild import BBHWaveformFD
    from bbhx.utils.constants import YRSID_SI
except ImportError as exc:  # pragma: no cover - LISA env only
    raise SystemExit(
        "This example needs bbhx (`pip install bbhx`). It is intentionally kept "
        "out of the default HyperWave env; run it in a LISA environment."
    ) from exc

MPC = 3.0856775815e22  # metres per Mpc (bbhx distance is in metres)

SMBHB_PARAMETER_NAMES = [
    "m1", "m2", "chi1z", "chi2z", "distance", "phi_ref",
    "inc", "lam", "beta", "psi", "t_ref",
]
PERIODIC = ["phi_ref", "lam", "psi"]

# Paper-faithful sampling space (UCB convention): isotropic inclination/sky are
# sampled as cos(inc), sin(beta) ~ Uniform(-1,1) -- the *same* prior as Sine(inc)
# / Cosine(beta) but in well-conditioned coordinates, which removes the bad
# geometry near inc=0/pi that biased the inclination block (inc was +5.1 sigma
# in the Sine-prior run 10813656).
# Paper-faithful sampling space: chirp mass + mass ratio (Mc, q) instead of the
# strongly-correlated (m1, m2). Mc and q are nearly orthogonal, so the eryn
# stretch move mixes them cleanly -- the m1/m2 banana is what left the old corner
# scattered/unconverged. (asasli/hyperbolic_likelihood_filter sampling order.)
SMBHB_SAMPLING_NAMES = [
    "Mc", "q", "chi1z", "chi2z", "log10_distance", "phi_ref",
    "cos_inc", "lam", "sin_beta", "psi", "t_ref",
]


def mc_q_to_m1_m2(Mc, q):
    """Chirp mass + mass ratio (q = m1/m2 >= 1) -> component masses.

    Paper ``smbhb.get_m1_m2_from_chirp_and_eta``: eta = q/(1+q)^2.
    """
    Mc = np.asarray(Mc, dtype=float); q = np.asarray(q, dtype=float)
    eta = q / (1.0 + q) ** 2
    Mtot = Mc * eta ** -0.6
    disc = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
    m1 = 0.5 * Mtot * (1.0 + disc)
    m2 = 0.5 * Mtot * (1.0 - disc)
    return m1, m2


def sampling_to_physical(p):
    """Sampling dict -> physical dict bbhx expects.

    Paper-faithful coordinates (asasli/hyperbolic_likelihood_filter): masses as
    (Mc, q), isotropic inclination/sky as cos(inc), sin(beta), distance as
    log10(D) -- all better-conditioned than the raw bbhx parameters.
    """
    q = dict(p)
    m1, m2 = mc_q_to_m1_m2(q.pop("Mc"), q.pop("q"))
    q["m1"], q["m2"] = m1, m2
    q["inc"] = np.arccos(np.clip(np.asarray(q.pop("cos_inc"), dtype=float), -1.0, 1.0))
    q["beta"] = np.arcsin(np.clip(np.asarray(q.pop("sin_beta"), dtype=float), -1.0, 1.0))
    q["distance"] = 10.0 ** np.asarray(q.pop("log10_distance"), dtype=float)
    return q


class SMBHBbbhxTemplate:
    """Batched frequency-domain LISA SMBHB A/E/T template backed by bbhx.

    ``batch_eval`` takes arrays (length ``N``) and returns ``(N, 3, nfreq)``;
    bbhx requires array inputs (it derives the batch size from ``len(m1)``), so
    this is both the correct call *and* the vectorised fast path.
    """

    def __init__(self, freqs, f_ref=0.0, t_obs_years=1.0, modes=None, run_phenomd=True,
                 length=1024, force_backend=None):
        self.freqs = np.asarray(freqs, dtype=float)
        self.f_ref = float(f_ref)
        self.t_obs = float(t_obs_years) * YRSID_SI
        self.modes = modes  # None -> dominant (2,2) for PhenomD
        self.length = int(length)
        kw = dict(amp_phase_kwargs=dict(run_phenomd=run_phenomd),
                  response_kwargs=dict(TDItag="AET"))
        if force_backend is not None:
            # bbhx asserts orbits.backend == response.backend, and the orbits
            # default auto-picks the best available backend — pair explicitly.
            from lisatools.detector import EqualArmlengthOrbits
            kw["response_kwargs"]["orbits"] = EqualArmlengthOrbits(force_backend=force_backend)
            self.wave_gen = BBHWaveformFD(**kw, force_backend=force_backend)
        else:
            try:
                self.wave_gen = BBHWaveformFD(**kw)
            except ValueError:
                # gpubackendtools-main asks for backends (e.g. bbhx_cuda13x) that a
                # CPU-only source build never registered; fall back explicitly.
                from lisatools.detector import EqualArmlengthOrbits
                kw["response_kwargs"]["orbits"] = EqualArmlengthOrbits(force_backend="cpu")
                self.wave_gen = BBHWaveformFD(**kw, force_backend="cpu")
        # GPU backends require device (CuPy) frequency arrays and return device
        # output; keep a device copy + converter so callers stay NumPy-facing.
        self._xp = np
        if "cuda" in getattr(self.wave_gen.backend, "name", "cpu"):
            import cupy
            self._xp = cupy
        self._freqs_dev = self._xp.asarray(self.freqs)

    def batch_eval(self, p):
        """dict of arrays -> (N, 3, nfreq) complex A/E/T."""
        def arr(name):
            return np.atleast_1d(np.asarray(p[name], dtype=float))
        aet = self.wave_gen(
            arr("m1"), arr("m2"), arr("chi1z"), arr("chi2z"), arr("distance"),
            arr("phi_ref"), self.f_ref, arr("inc"), arr("lam"), arr("beta"),
            arr("psi"), arr("t_ref"), freqs=self._freqs_dev, modes=self.modes,
            direct=False, fill=True, squeeze=False, length=self.length,
        )
        aet = aet.get() if hasattr(aet, "get") else np.asarray(aet)
        return aet.reshape(-1, 3, self.freqs.shape[0])

    def signal_model(self, **p):           # single source -> (2, nfreq)
        return self.batch_eval(p)[0, :2, :]

    def batch_signal_model(self, **p):     # batch -> (N, 2, nfreq)
        return self.batch_eval(p)[:, :2, :]


def lisa_aet_psd(freqs, channels=("A", "E")):
    """Analytic LISA A/E/T noise PSD (SciRD-like stub). Returns ``(nch, nfreq)``."""
    f = np.clip(np.asarray(freqs, dtype=float), 1e-5, None)
    Sacc = (3.0e-15) ** 2 * (1 + (4e-4 / f) ** 2) / (2 * np.pi * f) ** 4
    Soms = (15.0e-12) ** 2 * (1 + (2e-3 / f) ** 4)
    Sn = 20.0 / 3.0 * (Soms + 2 * (1 + np.cos(f / 0.019) ** 2) * Sacc)
    return np.array([Sn for _ in channels])


def injected_smbhb(t_obs_years):
    return dict(
        m1=1.0e6, m2=5.0e5, chi1z=0.3, chi2z=0.2, distance=20.0e3 * MPC,
        phi_ref=1.2, inc=0.7, lam=2.1, beta=0.4, psi=0.9,
        t_ref=0.5 * t_obs_years * YRSID_SI,
    )


def make_priors(t_obs_years, nsegs):
    """SMBHB priors (bbhx order) + (1 alpha + nsegs delta) hyperbolic shape priors."""
    import bilby

    Tobs = t_obs_years * YRSID_SI
    # truth Mc, q for the prior window (paper: Mc in [0.1,2]*Mc_true, q in [0.001, 2*q])
    th = injected_smbhb(t_obs_years)
    eta_t = th["m1"] * th["m2"] / (th["m1"] + th["m2"]) ** 2
    Mc_t = (th["m1"] + th["m2"]) * eta_t ** 0.6
    q_t = th["m1"] / th["m2"]
    pr = bilby.core.prior.PriorDict()
    pr["Mc"] = bilby.core.prior.Uniform(0.1 * Mc_t, 2.0 * Mc_t, name="Mc", latex_label=r"$\mathcal{M}$")
    pr["q"] = bilby.core.prior.Uniform(1.0, 2.0 * q_t, name="q", latex_label=r"$q$")
    pr["chi1z"] = bilby.core.prior.Uniform(-0.99, 0.99, name="chi1z", latex_label=r"$\chi_{1z}$")
    pr["chi2z"] = bilby.core.prior.Uniform(-0.99, 0.99, name="chi2z", latex_label=r"$\chi_{2z}$")
    # log10(distance) in metres: better-conditioned for the inc-distance degeneracy
    pr["log10_distance"] = bilby.core.prior.Uniform(
        np.log10(5.0e3 * MPC), np.log10(80.0e3 * MPC),
        name="log10_distance", latex_label=r"$\log_{10}D_L$")
    pr["phi_ref"] = bilby.core.prior.Uniform(0.0, 2 * np.pi, name="phi_ref", latex_label=r"$\phi_{\rm ref}$")
    # Isotropic inclination/sky in well-conditioned coordinates (== Sine/Cosine
    # prior, better geometry): cos(inc), sin(beta) ~ Uniform(-1,1).
    pr["cos_inc"] = bilby.core.prior.Uniform(-1.0, 1.0, name="cos_inc", latex_label=r"$\cos\iota$")
    pr["lam"] = bilby.core.prior.Uniform(0.0, 2 * np.pi, name="lam", latex_label=r"$\lambda$")
    pr["sin_beta"] = bilby.core.prior.Uniform(-1.0, 1.0, name="sin_beta", latex_label=r"$\sin\beta$")
    pr["psi"] = bilby.core.prior.Uniform(0.0, np.pi, name="psi", latex_label=r"$\psi$")
    pr["t_ref"] = bilby.core.prior.Uniform(0.3 * Tobs, 0.7 * Tobs, name="t_ref", latex_label=r"$t_{\rm ref}$")

    noise_priors = {r"$\alpha$": bilby.core.prior.Uniform(0.0, 30.0)}
    for i in range(nsegs):
        noise_priors[r"$\delta_{}$".format(i)] = bilby.core.prior.Uniform(0.0, 30.0)
    return pr, noise_priors


def sampler_kwargs(sampler, args):
    if sampler == "eryn":
        if args.quick:
            # eryn red-blue needs nwalkers >= 2*ndims (11 source + 1 alpha + nsegs)
            return dict(nwalkers=32, ntemps=4, burn=50, nsteps=200)
        return dict(nwalkers=args.nwalkers, ntemps=args.ntemps, burn=args.burn, nsteps=args.steps)
    if args.quick:
        return dict(n_total=2000, n_effective=512, n_active=256, n_steps=10)
    # Robust-but-walltime-aware SMC for the (phi_ref, psi) orientation
    # multimodality: larger ensemble + more MCMC steps per SMC iteration than the
    # old n_active=2000 (which collapsed), moderated so it fits the budget
    # (n_active=5000/n_steps=50 timed out for UCB at 6h).
    return dict(n_total=8000, n_effective=12000, n_active=3000, n_steps=25)


def run_pe(sampler, likelihood, priors, noise_priors, kw, outdir):
    from hyperwave.inference import LVKinference
    t0 = time.perf_counter()
    inf = LVKinference(
        likelihood.hyperbolic_classic, sampler_name=sampler, priors=priors,
        noise_priors=noise_priors,
        common_params={"save_dir": outdir, "TAG": f"lisa_smbhb_{sampler}", "like": "hyperbolic"},
        sampler_kwargs=kw, periodic=PERIODIC,
    )
    inf.run()
    wall = time.perf_counter() - t0
    samples = np.asarray(inf.get_samples())
    return samples, wall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampler", choices=["eryn", "pocomc", "both"], default="both")
    p.add_argument("--t-obs", type=float, default=1.0, help="observation time [yr]")
    p.add_argument("--fmax", type=float, default=1.5e-2, help="max analysis frequency [Hz]")
    p.add_argument("--df", type=float, default=1e-6, help="frequency resolution [Hz]")
    p.add_argument("--force-backend", default=None,
                   help="bbhx compute backend: cpu | cuda11x | cuda12x. Default: auto "
                        "(tries bbhx's pick, falls back to cpu). On GPU nodes pass the "
                        "cuda backend matching the toolkit the build found.")
    p.add_argument("--target-snr", type=float, default=100.0,
                   help="calibrate the noise level so the injection has this SNR (the analytic A/E PSD "
                        "stub's units do not match bbhx's TDI output); <=0 leaves the PSD unscaled")
    p.add_argument("--nsegs", type=int, default=2)
    # paper defaults: nwalkers=40, ntemps=30, burn=10000, nsteps=60000
    p.add_argument("--steps", type=int, default=60000, help="eryn nsteps / pocomc n_total")
    p.add_argument("--burn", type=int, default=10000)
    p.add_argument("--nwalkers", type=int, default=40)
    p.add_argument("--ntemps", type=int, default=30)
    p.add_argument("--noise", action="store_true", help="add a Gaussian noise draw (default: zero-noise)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results/lisa_smbhb")
    p.add_argument("--quick", action="store_true", help="tiny smoke run")
    p.add_argument("--snr-only", action="store_true", help="just generate the waveform and report SNR")
    args = p.parse_args()

    channels = ("A", "E")  # T is signal-poor in the LISA band; analyse A, E
    freqs = np.arange(args.df, args.fmax, args.df)
    template = SMBHBbbhxTemplate(freqs, t_obs_years=args.t_obs, force_backend=args.force_backend)
    theta_true = injected_smbhb(args.t_obs)

    sig = template.signal_model(**theta_true)    # (2, nfreq)
    psd = lisa_aet_psd(freqs, channels)          # (2, nfreq)
    raw_snr = float(np.sqrt(np.sum(4 * args.df * (np.abs(sig) ** 2 / psd).real)))
    if args.target_snr > 0 and raw_snr > 0:
        psd = psd * (raw_snr / args.target_snr) ** 2   # calibrate noise -> target SNR
    snr = float(np.sqrt(np.sum(4 * args.df * (np.abs(sig) ** 2 / psd).real)))
    print(f"> injected SMBHB A/E SNR ~ {snr:.1f} (raw {raw_snr:.2e}; target {args.target_snr:g})"
          f"  ({sig.shape[1]} bins, {len(channels)} channels)")
    if args.snr_only:
        return

    data = np.asarray(sig, dtype=complex)
    if args.noise:
        rng = np.random.default_rng(args.seed)
        sigma = np.sqrt(psd / (4.0 * args.df))
        data = data + sigma * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))

    # Sample in the paper-faithful space (cos_inc, sin_beta); transform to the
    # physical (inc, beta) bbhx expects inside the signal model.
    def _sig(**p):
        return template.signal_model(**sampling_to_physical(p))

    def _batch_sig(**p):
        return template.batch_signal_model(**sampling_to_physical(p))

    from hyperwave.detectors.lisa import LISAAETTemplate, build_lisa_aet_likelihood
    lisa_template = LISAAETTemplate(
        parameters=SMBHB_SAMPLING_NAMES, signal_model=_sig,
        batch_signal_model=_batch_sig, channels=channels, call_mode="kwargs",
    )
    likelihood = build_lisa_aet_likelihood(
        data=data, template=lisa_template, sensitivity=psd, freqs=freqs,
        channels=channels, ddims=False, nsegs=args.nsegs, gpu=False,
    )
    priors, noise_priors = make_priors(args.t_obs, args.nsegs)
    samplers = ["eryn", "pocomc"] if args.sampler == "both" else [args.sampler]

    os.makedirs(os.path.join(args.outdir, "chains"), exist_ok=True)
    wfdims = len(SMBHB_SAMPLING_NAMES)
    theta_true_samp = dict(theta_true)
    _eta = theta_true["m1"] * theta_true["m2"] / (theta_true["m1"] + theta_true["m2"]) ** 2
    theta_true_samp["Mc"] = float((theta_true["m1"] + theta_true["m2"]) * _eta ** 0.6)
    theta_true_samp["q"] = float(theta_true["m1"] / theta_true["m2"])
    theta_true_samp["cos_inc"] = float(np.cos(theta_true["inc"]))
    theta_true_samp["sin_beta"] = float(np.sin(theta_true["beta"]))
    theta_true_samp["log10_distance"] = float(np.log10(theta_true["distance"]))
    truths = [theta_true_samp[k] for k in SMBHB_SAMPLING_NAMES]

    # Warm-start vector: source params at truth + a near-Gaussian noise start
    # (alpha, delta per segment). The sharply-peaked SNR~100 SMBHB posterior is
    # never found from prior-draw init -- the chain sticks edge-on -- so we seed
    # the walkers at the truth, exactly the PI's gen_data_points_close_to_true.
    noise_start = [10.0] + [10.0] * args.nsegs          # alpha, delta_0..
    truth_full = np.array(truths + noise_start, dtype=float)
    all_priors = dict(priors); all_priors.update(noise_priors)
    lo = np.array([all_priors[k].minimum for k in all_priors])
    hi = np.array([all_priors[k].maximum for k in all_priors])

    def warm_coords(ntemps, nwalkers, scatter=3e-4, seed=0):
        rng = np.random.default_rng(seed)
        w = truth_full[None, None, :] + rng.normal(
            0.0, 1.0, (ntemps, nwalkers, truth_full.size)) * (scatter * (hi - lo))
        return np.clip(w, lo, hi)

    timings = {}
    for sampler in samplers:
        kw = sampler_kwargs(sampler, args)
        if sampler == "eryn":
            kw = dict(kw, init_coords=warm_coords(kw["ntemps"], kw["nwalkers"],
                                                  seed=args.seed))
            print("[warm-start] eryn walkers seeded near truth (init_coords)")
        print(f"\n[setup] LISA SMBHB {sampler}  nsegs={args.nsegs}  vectorised(batch)  kw={ {k:v for k,v in kw.items() if k!='init_coords'} }")
        samples, wall = run_pe(sampler, likelihood, priors, noise_priors, kw, args.outdir)
        timings[sampler] = (wall, samples.shape[0])
        print(f"[{sampler}] wall {wall:.1f}s ({wall/60:.1f} min) | {samples.shape[0]} samples, {samples.shape[1]} dims")
        np.savez(os.path.join(args.outdir, f"lisa_smbhb_{sampler}_samples.npz"),
                 samples=samples, parameter_names=SMBHB_SAMPLING_NAMES, truths=truths, snr=snr, wall=wall)
        try:
            import corner, matplotlib
            matplotlib.use("Agg")
            fig = corner.corner(samples[:, :wfdims], labels=SMBHB_SAMPLING_NAMES, truths=truths)
            fig.savefig(os.path.join(args.outdir, f"lisa_smbhb_{sampler}_corner.png"), dpi=120)
        except Exception as exc:  # pragma: no cover
            print(f"  (corner skipped: {exc})")

    if len(timings) > 1:
        print("\n===== sampler speed (LISA SMBHB, vectorised) =====")
        for s, (w, n) in timings.items():
            print(f"  {s:7s}: {w:8.1f} s  for {n} samples  ({1e3*w/max(n,1):.2f} ms/sample)")


if __name__ == "__main__":
    main()
