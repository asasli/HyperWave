"""Sangria MBHB PE — verbatim reproduction of Sect. III.B / Appendix A (Gaussian,
known noise) of Sasli et al. (arXiv:2602.22074, "Beyond Gaussian Assumptions").

Injection = LDC Sangria training MBHB (paper Table I), Tobs = 30.4368 days with
coalescence one day before the end. PhenomD via BBHx, A/E TDI channels (T
neglected), SciRD noise (paper Eqs. 12-13). Priors verbatim from Table II;
sampling in (Mc, q, chi1, chi2, log10 D[Mpc], tc, cos i, sin beta, lambda, psi,
phi0) -> 11 params. Eryn PTMCMC: 20 temperatures, 80 walkers, default
affine-invariant moves, walkers initialised from small perturbations around the
injection (paper Sect. III.A). Reference posteriors: paper Table V.

    python examples/lisa/smbhb_sangria_pe.py --snr-only --force-backend cuda12x
    python examples/lisa/smbhb_sangria_pe.py --force-backend cuda12x --steps 20000 --thin 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smbhb_bbhx_pe import SMBHBbbhxTemplate, mc_q_to_m1_m2  # noqa: E402

MPC_M = 3.0856775814913674e22   # Mpc in metres
DAY = 86400.0
C_SI = 299792458.0
L_ARM = 2.5e9                    # LISA arm length [m]

# ---- paper Table I (Sangria training MBHB, SNR=357 @ 1yr) ------------------
TRUTH_PHYS = dict(
    m1=4956676.2876, m2=4067166.60352,
    chi1z=-0.523732715, chi2z=-0.117412144,
    distance_mpc=61097.116076,
    inc=1.420048341, beta=-1.081082148, lam=4.052962883,
    psi=1.22844350008, phi_ref=0.6417162631,
)
TOBS_S = 30.4368 * DAY                    # paper: 30.4368 days
TREF_S = TOBS_S - 1.0 * DAY               # coalescence 1 day before the end

SAMPLING_NAMES = ["Mc", "q", "chi1z", "chi2z", "log10_D", "tc_h",
                  "cos_inc", "sin_beta", "lam", "psi", "phi_ref"]
LABELS = [r"$\mathcal{M}$", r"$q$", r"$\chi_1$", r"$\chi_2$", r"$\log_{10}D_l$",
          r"$t_c$[h]", r"$\cos\iota$", r"$\sin\beta$", r"$\lambda$", r"$\psi$",
          r"$\phi_0$"]


def truth_sampling():
    m1, m2 = TRUTH_PHYS["m1"], TRUTH_PHYS["m2"]
    eta = m1 * m2 / (m1 + m2) ** 2
    return np.array([
        (m1 + m2) * eta ** 0.6,                 # Mc
        m1 / m2,                                # q
        TRUTH_PHYS["chi1z"], TRUTH_PHYS["chi2z"],
        np.log10(TRUTH_PHYS["distance_mpc"]),   # log10 D [Mpc]
        TREF_S / 3600.0,                        # tc [hours]
        np.cos(TRUTH_PHYS["inc"]), np.sin(TRUTH_PHYS["beta"]),
        TRUTH_PHYS["lam"], TRUTH_PHYS["psi"], TRUTH_PHYS["phi_ref"],
    ])


# ---- paper Table II priors (verbatim) ---------------------------------------
def prior_bounds():
    log10_dt = np.log10(TRUTH_PHYS["distance_mpc"])
    lo = np.array([0.39e6, 0.99999, -1.0, -1.0, 0.1 * log10_dt,
                   365.2422, -1.0, -1.0, 0.0, 0.0, 0.0])
    hi = np.array([7.8e6, 2.0, 1.0, 1.0, 2.0 * log10_dt,
                   8765.8128, 1.0, 1.0, 2 * np.pi, np.pi, 2 * np.pi])
    return lo, hi


def sampling_to_physical(x):
    """(N, 11) sampling-space -> dict of physical bbhx arrays."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    m1, m2 = mc_q_to_m1_m2(x[:, 0], x[:, 1])
    return dict(
        m1=m1, m2=m2, chi1z=x[:, 2], chi2z=x[:, 3],
        distance=10.0 ** x[:, 4] * MPC_M,
        phi_ref=x[:, 10], inc=np.arccos(np.clip(x[:, 6], -1, 1)),
        lam=x[:, 8], beta=np.arcsin(np.clip(x[:, 7], -1, 1)),
        psi=x[:, 9], t_ref=x[:, 5] * 3600.0,
    )


# ---- SciRD A/E analytic PSD (paper Eqs. 12-13) -------------------------------
def scird_psd_ae(f):
    """Average A/E TDI-1 noise PSD, SciRD levels, relative frequency units."""
    f = np.asarray(f, dtype=float)
    omega = 2 * np.pi * f * L_ARM / C_SI
    # SciRD single-link noises  (Sa in m^2/Hz/s^4, Si in m^2/Hz)
    Sa = (3.0e-15) ** 2
    Si = (15.0e-12) ** 2
    # acceleration -> relative frequency units (with SciRD shape factors)
    S_pm = Sa * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8e-3) ** 4) \
        * (1.0 / (2 * np.pi * f) ** 4) * (2 * np.pi * f / C_SI) ** 2
    # OMS -> relative frequency units
    S_op = Si * (1.0 + (2e-3 / f) ** 4) * (2 * np.pi * f / C_SI) ** 2
    return (8 * np.sin(omega) ** 2
            * (2 * S_pm * (3 + 2 * np.cos(omega) + np.cos(2 * omega))
               + S_op * (2 + np.cos(omega))))


def psd_ae(f):
    """Paper Eq.(12) explicit SciRD A/E PSD.

    NOTE: lisatools' A1TDISens is 0.3-0.6x this (different TDI conventions) and
    would give SNR ~470; the explicit Eq.(12) reproduces the paper's SNR=357
    scale, so the explicit form is the faithful choice here.
    """
    return scird_psd_ae(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force-backend", default=None, help="cpu | cuda12x")
    p.add_argument("--fmax", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=20000, help="stored steps")
    p.add_argument("--thin", type=int, default=5,
                   help="iterations per stored step (paper ran 3e5+1e5 iters)")
    p.add_argument("--burn", type=int, default=20000, help="burn-in iterations")
    p.add_argument("--nwalkers", type=int, default=80)   # paper
    p.add_argument("--ntemps", type=int, default=20)     # paper
    p.add_argument("--noise", dest="noise", action="store_true", default=True)
    p.add_argument("--zero-noise", dest="noise", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results/lisa_smbhb_sangria")
    p.add_argument("--snr-only", action="store_true")
    args = p.parse_args()

    df = 1.0 / TOBS_S
    freqs = np.arange(df, args.fmax, df)
    print(f"[setup] Tobs={TOBS_S/DAY:.4f} d  df={df:.3e} Hz  {freqs.size} bins  "
          f"fmax={args.fmax}", flush=True)

    template = SMBHBbbhxTemplate(freqs, t_obs_years=TOBS_S / (365.25 * DAY),
                                 force_backend=args.force_backend)
    truth = truth_sampling()
    print(f"[setup] backend={getattr(template.wave_gen.backend, 'name', '?')}")
    print(f"[setup] truth (sampling space): "
          + "  ".join(f"{n}={v:.6g}" for n, v in zip(SAMPLING_NAMES, truth)))

    sig = template.batch_eval(sampling_to_physical(truth[None]))[0, :2, :]  # A,E
    psd = np.array([psd_ae(freqs), psd_ae(freqs)])
    snr = float(np.sqrt(np.sum(4 * df * (np.abs(sig) ** 2 / psd).real)))
    print(f"[setup] injected A/E SNR (30.44 d) = {snr:.1f}   "
          f"(paper: 357 @ 1 yr)", flush=True)
    if args.snr_only:
        return

    rng = np.random.default_rng(args.seed)
    data = np.asarray(sig, dtype=complex)
    if args.noise:
        sigma = np.sqrt(psd / (4.0 * df))
        data = data + sigma * (rng.standard_normal(sig.shape)
                               + 1j * rng.standard_normal(sig.shape))

    lo, hi = prior_bounds()

    def log_like(x):
        x = np.atleast_2d(x)
        h = template.batch_eval(sampling_to_physical(x))[:, :2, :]
        r = data[None] - h
        ll = -0.5 * np.sum(4 * df * (np.abs(r) ** 2 / psd[None]).real, axis=(1, 2))
        ll[~np.isfinite(ll)] = -1e300
        return ll

    # timing probe: one full ensemble batch = one eryn iteration
    nb = args.ntemps * args.nwalkers
    probe = np.clip(truth[None] * (1 + 1e-4 * rng.standard_normal((nb, 11))),
                    lo[None], hi[None])
    log_like(probe[:8])
    t0 = time.perf_counter()
    log_like(probe)
    step_s = time.perf_counter() - t0
    total_iters = args.burn + args.steps * args.thin
    print(f"[timing] {nb} waveforms/iter -> {step_s:.2f} s/iter; "
          f"{total_iters} iters ~ {step_s*total_iters/3600:.1f} h", flush=True)

    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.ensemble import EnsembleSampler
    from eryn.state import State
    from eryn.backends import HDFBackend

    priors = {"model_0": ProbDistContainer(
        {i: uniform_dist(lo[i], hi[i]) for i in range(11)})}

    # warm start: small perturbations around the injection (paper Sect. III.A)
    coords = np.clip(truth[None, None] *
                     (1 + 1e-4 * rng.standard_normal((args.ntemps, args.nwalkers, 11))),
                     lo[None, None], hi[None, None])
    ll0 = log_like(coords.reshape(-1, 11)).reshape(args.ntemps, args.nwalkers)
    print(f"[init] warm-start logL: max={ll0.max():.1f} med={np.median(ll0):.1f}")

    os.makedirs(args.outdir, exist_ok=True)
    backend = HDFBackend(os.path.join(args.outdir, "sangria_eryn.h5"))

    periodic = {"model_0": {8: 2 * np.pi, 9: np.pi, 10: 2 * np.pi}}
    sampler = EnsembleSampler(
        args.nwalkers, 11, log_like, priors,
        tempering_kwargs=dict(ntemps=args.ntemps),
        vectorize=True, periodic=periodic, backend=backend)

    state = State(coords[:, :, None, :], log_like=ll0)
    t0 = time.perf_counter()
    sampler.run_mcmc(state, args.steps, burn=args.burn, thin_by=args.thin,
                     progress=True)
    wall = time.perf_counter() - t0
    print(f"[done] wall {wall/3600:.2f} h", flush=True)

    chain = sampler.get_chain()["model_0"][:, 0, :, 0, :]     # cold chain
    ns = chain.shape[0]
    post = chain[ns // 2:].reshape(-1, 11)
    np.savez(os.path.join(args.outdir, "sangria_eryn_samples.npz"),
             samples=post, parameter_names=SAMPLING_NAMES, truths=truth,
             snr=snr, wall=wall)
    print(f"\npulls (last half, {post.shape[0]} samples):")
    for i, n in enumerate(SAMPLING_NAMES):
        med, std = np.median(post[:, i]), np.std(post[:, i])
        pull = (med - truth[i]) / std if std > 0 else 0.0
        print(f"  {n:9s} truth={truth[i]:+.6g}  med={med:+.6g}  pull={pull:+.2f}"
              + ("  <-- >2sig" if abs(pull) > 2 else ""))
    # paper Table V reference: M=3.882e6 +2.3e4/-1.1e4, q=1.228+0.107/-0.058,
    # chi1~-0.31, chi2~-0.41+0.59/-0.32, log10Dl=4.78+0.16/-0.12,
    # tc=706.504+0.021/-0.024 h, cosi=0.01+0.41/-0.33, sinb~-0.855, lam=4.01+0.18/-0.21


if __name__ == "__main__":
    main()
