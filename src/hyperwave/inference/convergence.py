"""Convergence-based stopping for Eryn runs.

Instead of a fixed number of steps, stop when the chain has produced *enough
independent samples* and the science marginal of interest (here the posterior on
the number of wavelets) has stopped moving. This is the Goodman-Weare /
Foreman-Mackey autocorrelation criterion, combined with a split-half
stationarity check on the model-order posterior, and is always used under a hard
maximum-steps cap (the ``nsteps`` passed to ``run_mcmc``).

The criterion plugs into Eryn via::

    sampler = EnsembleSampler(..., stopping_fn=stopper, stopping_iterations=K)

and is checked every ``K`` sampler iterations during the sampling phase.

Why these signals (and not evidence):

* Autocorrelation time tau on the cold-chain **log-likelihood** and **nleaves**
  is robust under trans-dimensional moves (both are well-defined scalars even as
  the parameter dimension changes), unlike per-parameter tau.
* The **nleaves posterior** is the headline output, so a split-half
  total-variation test on p(D) directly certifies the result has converged and
  catches pathologies such as railing into ``nleaves_max``.
* Evidence (thermodynamic integration / stepping stone) converges more slowly
  and noisily than the posterior; it is best computed once at the end for model
  comparison, not used as a stopping signal.
"""

from __future__ import annotations

import numpy as np

try:
    from eryn.utils import get_integrated_act
    from eryn.utils.stopping import Stopping as _ErynStopping
except Exception:  # pragma: no cover - eryn optional at import time
    get_integrated_act = None
    _ErynStopping = object


def _integrated_act(chain_2d):
    """Mean integrated autocorrelation time of a ``(n_steps, n_walkers)`` chain."""
    if get_integrated_act is None or chain_2d.shape[0] < 60:
        return np.nan
    try:
        tau = get_integrated_act(np.asarray(chain_2d, dtype=float)[:, :, None])
    except Exception:
        return np.nan
    tau = np.asarray(tau, dtype=float).ravel()
    tau = tau[np.isfinite(tau) & (tau > 0)]
    return float(np.mean(tau)) if tau.size else np.nan


def _pD(counts, n_max):
    """Normalised histogram of integer wavelet counts over ``0..n_max``."""
    hist = np.bincount(np.asarray(counts, dtype=int).ravel(), minlength=n_max + 1).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _total_variation(p, q):
    n = max(p.size, q.size)
    p = np.pad(p, (0, n - p.size))
    q = np.pad(q, (0, n - q.size))
    return 0.5 * float(np.abs(p - q).sum())


class WaveletConvergenceStopping(_ErynStopping):
    """Autocorrelation + model-order stationarity stopping criterion.

    Parameters
    ----------
    nleaves_branch:
        Branch whose leaf count is the model order (default ``"signal"``), or a
        list of branches whose counts are summed (multi-family wave packets).
    nleaves_max:
        Upper bound on the leaf count (for the p(D) histogram).
    autocorr_mult:
        Require ``chain_length > autocorr_mult * tau`` (emcee rule, default 50).
    target_ess:
        Also require an effective sample size ``n_steps * n_walkers / tau`` above
        this (default 2000).
    tau_rtol:
        Require the tau estimate to be stable between checks to within this
        relative tolerance (default 0.05).
    pd_tol:
        Require the split-half total variation of p(D) below this (default 0.02).
    n_consecutive:
        Number of consecutive passing checks required before stopping (default 2).
    discard_frac:
        Fraction of the stored chain discarded as burn-in for the estimates
        (default 0.3).
    verbose:
        Print a diagnostics line at each check.
    """

    def __init__(self, nleaves_branch="signal", nleaves_max=40, autocorr_mult=50,
                 target_ess=2000, tau_rtol=0.05, pd_tol=0.02, n_consecutive=2,
                 discard_frac=0.3, verbose=True):
        self.nleaves_branch = nleaves_branch
        self.nleaves_max = int(nleaves_max)
        self.autocorr_mult = float(autocorr_mult)
        self.target_ess = float(target_ess)
        self.tau_rtol = float(tau_rtol)
        self.pd_tol = float(pd_tol)
        self.n_consecutive = int(n_consecutive)
        self.discard_frac = float(discard_frac)
        self.verbose = verbose
        self._old_tau = None
        self._consec = 0
        #: filled in at each check, so the driver can report the final state
        self.last = {}

    def __call__(self, iteration, sample, sampler):
        logl = np.asarray(sampler.get_log_like())            # (n, ntemps, nwalkers)
        n = logl.shape[0]
        if n < 60:
            return False
        discard = int(self.discard_frac * n)
        cold_logl = logl[discard:, 0, :]                     # (n', nwalkers)
        branches = ([self.nleaves_branch] if isinstance(self.nleaves_branch, str)
                    else list(self.nleaves_branch))
        all_nleaves = sampler.get_nleaves()
        nleaves = sum(np.asarray(all_nleaves[b]) for b in branches)[discard:, 0, :]

        tau_l = _integrated_act(cold_logl)
        tau_n = _integrated_act(nleaves.astype(float))
        taus = [t for t in (tau_l, tau_n) if np.isfinite(t)]
        if not taus:
            return False
        tau = max(taus)
        n_eff_steps = cold_logl.shape[0]
        ess = n_eff_steps * cold_logl.shape[1] / tau

        half = nleaves.shape[0] // 2
        tv = (_total_variation(_pD(nleaves[:half], self.nleaves_max),
                               _pD(nleaves[half:], self.nleaves_max))
              if half > 0 else np.inf)

        tau_stable = self._old_tau is not None and abs(tau - self._old_tau) / tau < self.tau_rtol
        self._old_tau = tau

        passed = (n_eff_steps > self.autocorr_mult * tau and ess > self.target_ess
                  and tau_stable and tv < self.pd_tol)
        self._consec = self._consec + 1 if passed else 0

        self.last = dict(iteration=int(iteration), tau=float(tau), ess=float(ess),
                         pD_tv=float(tv), tau_stable=bool(tau_stable),
                         consecutive=int(self._consec))
        if self.verbose:
            print(f"[converge] it={iteration} tau={tau:.1f} "
                  f"ESS={ess:.0f}/{self.target_ess:.0f} "
                  f"n/tau={n_eff_steps/tau:.1f}/{self.autocorr_mult:.0f} "
                  f"pD_tv={tv:.3f}/{self.pd_tol} stable={tau_stable} "
                  f"pass={self._consec}/{self.n_consecutive}", flush=True)

        return self._consec >= self.n_consecutive


__all__ = ["WaveletConvergenceStopping"]
