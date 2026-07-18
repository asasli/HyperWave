"""Data-informed (time-frequency) birth proposal for wavelet RJMCMC.

The default Eryn reversible-jump births draw a new wavelet from the *prior* over
the whole ``(t0, f0, Q, A, phi)`` volume, so a new wavelet almost never lands on
the signal and the SNR prior cancels out of the acceptance ratio. That is why a
naive run neither locks onto the signal nor resists adding noise wavelets.

This module builds a data-informed fix: a birth proposal whose ``(t0, f0)``
are drawn from the **time-frequency power of the (whitened) data**, so new
wavelets are proposed where there is unexplained power. Because the proposal
differs from the prior, Eryn's :class:`~eryn.moves.DistributionGenerateRJ` keeps
``+logpdf(old)`` / ``-logpdf(new)`` of this proposal in the Hastings ratio, which
both steers births onto the signal and re-instates the SNR-prior Occam penalty.

The proposal is separable (independent ``t0`` and ``f0`` marginals of the
whitened data power) — a robust, correctly-normalised approximation to the full
2-D spectrogram proposal.
"""

from __future__ import annotations

import numpy as np

try:
    from eryn.moves import GroupStretchMove
    from eryn.prior import ProbDistContainer, uniform_dist
except ImportError as exc:  # pragma: no cover - eryn is a core dependency
    raise ImportError("Wavelet proposals require Eryn (eryn.prior).") from exc

from .wavelet_priors import SNRPrior, _ErynDistribution


class DataInformedMarginal(_ErynDistribution):
    """1-D distribution defined by a (positive) weight curve on a grid.

    The weights are floored (so the density is positive across the whole support,
    keeping ``logpdf`` finite for death moves), normalised to a probability
    density, and sampled by inverse-CDF interpolation.
    """

    def __init__(self, grid, weights, floor_frac=0.1):
        grid = np.asarray(grid, dtype=float)
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        w = w + floor_frac * (np.mean(w) if np.mean(w) > 0 else 1.0)
        area = np.trapz(w, grid)
        self._grid = grid
        self._pdf = w / area
        self._logpdf = np.log(self._pdf)
        dcdf = 0.5 * (self._pdf[1:] + self._pdf[:-1]) * np.diff(grid)
        cdf = np.concatenate([[0.0], np.cumsum(dcdf)])
        self._cdf = cdf / cdf[-1]
        self.min_val = float(grid[0])
        self.max_val = float(grid[-1])

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape, -np.inf)
        inside = (x >= self.min_val) & (x <= self.max_val)
        out[inside] = np.interp(x[inside], self._grid, self._logpdf)
        return out if out.ndim else float(out)

    def rvs(self, size=1):
        u = np.random.uniform(0.0, 1.0, size=size)
        return np.interp(u, self._cdf, self._grid)


def _whitened_time_envelope(template, data, psd, n=512):
    """Whitened-data time-domain power envelope on ``[0, duration]``."""
    n_full = template.frequency_array.size
    wd = np.zeros((data.shape[0], n_full), dtype=complex)
    wd[:, template.mask] = np.asarray(data) / np.sqrt(np.asarray(psd))
    td = np.fft.irfft(wd, axis=-1)                       # (n_ifo, n_time)
    env = np.sum(td**2, axis=0)
    t_axis = np.linspace(0.0, template.duration, td.shape[-1], endpoint=False)
    grid = np.linspace(0.0, template.duration, n)
    return grid, np.interp(grid, t_axis, env)


def build_guided_birth(template, data, psd, spec, *, q_bounds=(0.1, 40.0),
                       rho_star=5.0, snr_bounds=(0.0, 100.0), floor_frac=0.1, n=512):
    """Build the ``generate_dist`` for a data-informed wavelet birth move.

    Returns ``{branch_name: ProbDistContainer, "extrinsic": ProbDistContainer}``
    suitable for :class:`eryn.moves.DistributionGenerateRJ`. ``t0`` and ``f0``
    follow the whitened-data time/frequency power; ``Q``, SNR and ``phi0`` follow
    their priors (so only the placement is data-informed).
    """
    f = np.asarray(template.frequency_array_masked())
    power_f = np.sum(np.abs(np.asarray(data)) ** 2 / np.asarray(psd), axis=0)  # network whitened |d|^2
    fgrid = np.linspace(f[0], f[-1], n)
    f0_dist = DataInformedMarginal(fgrid, np.interp(fgrid, f, power_f), floor_frac)

    tgrid, env = _whitened_time_envelope(template, data, psd, n=n)
    t0_dist = DataInformedMarginal(tgrid, env, floor_frac)

    branch = next(k for k in spec["priors"] if k != "extrinsic")
    birth = ProbDistContainer({
        0: t0_dist,                                   # time   (data-informed)
        1: f0_dist,                                   # freq   (data-informed)
        2: uniform_dist(q_bounds[0], q_bounds[1]),    # Q      (prior)
        3: SNRPrior(rho_star=rho_star, snr_min=snr_bounds[0], snr_max=snr_bounds[1]),  # SNR (prior)
        4: uniform_dist(0.0, 2.0 * np.pi),            # phi0   (prior)
    })
    return {branch: birth, "extrinsic": spec["extrinsic"]}


class MatchedFilterBirth(_ErynDistribution):
    """Joint 5-parameter birth proposal with data-fitted SNR and phase.

    BayesWave's birth move owes its acceptance rate to fitting the *linear*
    wavelet parameters to the data at the proposed location. This proposal does
    the same exactly: ``(t0, f0)`` are drawn from the whitened-data
    time/frequency maps and ``Q`` from its prior range, then the matched-filter
    fit of the data against the unit wavelet at ``(t0, f0, Q)``,

    .. math:: z_k = 4\\,\\Delta f \\sum_f d_k(f)\\,\\bar w_0(f) / S_k(f),

    gives the proposal centre: ``SNR ~ TruncNormal(rho_hat, sigma_snr)`` and
    ``phi0 ~ VonMises(arg z, kappa_phi)``, each mixed with the prior (weight
    ``mix``) so deaths of odd wavelets keep a finite reverse density. Both
    ``rvs`` and ``logpdf`` evaluate the same deterministic fit, so the Hastings
    ratio in :class:`eryn.moves.DistributionGenerateRJ` stays exact.

    Use as a joint entry in ``ProbDistContainer({(0,1,2,3,4): this})``.
    """

    def __init__(self, f_masked, data, psd, reference_psd, t0_dist, f0_dist,
                 q_bounds=(0.1, 40.0), snr_prior=None, snr_bounds=(0.0, 100.0),
                 sigma_snr=1.5, kappa_phi=8.0, mix=0.2, snr_shrink=0.9,
                 chunk=2048):
        self._f = np.asarray(f_masked, dtype=float)
        self._df = float(self._f[1] - self._f[0])
        data = np.asarray(data)
        psd = np.asarray(psd, dtype=float)
        self._d_over_S = data / psd                  # (n_ifo, nf)
        self._inv_S = 1.0 / psd                      # (n_ifo, nf)
        self._Sref = np.asarray(reference_psd, dtype=float)
        self._t0 = t0_dist
        self._f0 = f0_dist
        self._qmin, self._qmax = float(q_bounds[0]), float(q_bounds[1])
        self._logq_q = -np.log(self._qmax - self._qmin)
        self._snr_prior = snr_prior if snr_prior is not None else SNRPrior(
            snr_min=snr_bounds[0], snr_max=snr_bounds[1])
        self._smin, self._smax = float(snr_bounds[0]), float(snr_bounds[1])
        self._sig = float(sigma_snr)
        self._kap = float(kappa_phi)
        self._mix = float(mix)
        self._shrink = float(snr_shrink)
        self._chunk = int(chunk)

    # -- the deterministic matched-filter fit ---------------------------------
    def _fit(self, t0, f0, Q):
        """(B,) params -> (mu_snr, phi_hat), chunked to bound memory."""
        from ..detectors.waveforms.wavelets import morlet_gabor_fd, snr_from_amplitude

        t0 = np.atleast_1d(np.asarray(t0, dtype=float))
        f0 = np.atleast_1d(np.asarray(f0, dtype=float))
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        B = t0.shape[0]
        mu = np.empty(B)
        ph = np.empty(B)
        for lo in range(0, B, self._chunk):
            sl = slice(lo, min(lo + self._chunk, B))
            w0 = morlet_gabor_fd(self._f, t0[sl], f0[sl], Q[sl],
                                 np.ones_like(t0[sl]), np.zeros_like(t0[sl]))  # (b, nf)
            z = 4.0 * self._df * np.einsum("kf,bf->kb", self._d_over_S, np.conj(w0))
            n2 = 4.0 * self._df * np.einsum("kf,bf->kb", self._inv_S, np.abs(w0) ** 2).real
            # modulus-sum amplitude: exact for a coherent network signal and
            # robust to per-detector phase offsets (we ignore sky delays here)
            a_hat = np.sum(np.abs(z), axis=0) / np.maximum(np.sum(n2, axis=0), 1e-300)
            k_best = np.argmax(np.abs(z), axis=0)
            ph[sl] = np.angle(z[k_best, np.arange(z.shape[1])])
            sref = np.interp(f0[sl], self._f, self._Sref)
            mu[sl] = snr_from_amplitude(a_hat, f0[sl], Q[sl], sref) * self._shrink
        return np.clip(mu, self._smin, self._smax), ph

    # -- mixture component densities ------------------------------------------
    def _log_truncnorm(self, x, mu):
        from scipy.special import log_ndtr

        a = (self._smin - mu) / self._sig
        b = (self._smax - mu) / self._sig
        zed = (x - mu) / self._sig
        log_norm = np.log(np.maximum(
            np.exp(log_ndtr(b)) - np.exp(log_ndtr(a)), 1e-300))
        out = -0.5 * zed**2 - 0.5 * np.log(2 * np.pi) - np.log(self._sig) - log_norm
        return np.where((x >= self._smin) & (x <= self._smax), out, -np.inf)

    def _log_vonmises(self, x, mu):
        from scipy.special import i0e

        # log p = kappa*(cos(x-mu)-1) - log(2 pi I0e(kappa))
        return self._kap * (np.cos(x - mu) - 1.0) - np.log(2 * np.pi * i0e(self._kap))

    # -- eryn distribution API --------------------------------------------------
    def rvs(self, size=1):
        size = (size,) if isinstance(size, int) else tuple(size)
        B = int(np.prod(size))
        t0 = np.asarray(self._t0.rvs(size=B), dtype=float).reshape(B)
        f0 = np.asarray(self._f0.rvs(size=B), dtype=float).reshape(B)
        Q = np.random.uniform(self._qmin, self._qmax, size=B)
        mu, ph = self._fit(t0, f0, Q)

        from scipy.special import ndtr, ndtri

        # inverse-CDF truncated-normal draw (clipping would put point mass at
        # the bounds that logpdf does not represent, breaking detailed balance)
        a = ndtr((self._smin - mu) / self._sig)
        b = ndtr((self._smax - mu) / self._sig)
        u = np.random.uniform(a, np.maximum(b, a + 1e-12))
        snr_mf = mu + self._sig * ndtri(np.clip(u, 1e-15, 1 - 1e-15))
        use_prior = np.random.uniform(size=B) < self._mix
        snr = np.where(
            use_prior,
            np.asarray(self._snr_prior.rvs(size=B), dtype=float).reshape(B),
            np.clip(snr_mf, self._smin, self._smax),
        )
        phi = np.where(
            np.random.uniform(size=B) < self._mix,
            np.random.uniform(0.0, 2 * np.pi, size=B),
            np.mod(ph + np.random.vonmises(0.0, self._kap, size=B), 2 * np.pi),
        )
        out = np.stack([t0, f0, Q, snr, phi], axis=-1)
        return out.reshape(*size, 5)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        flat = np.atleast_2d(x.reshape(-1, 5))
        t0, f0, Q, snr, phi = (flat[:, i] for i in range(5))
        mu, ph = self._fit(t0, f0, Q)

        lp = np.asarray(self._t0.logpdf(t0), dtype=float)
        lp = lp + np.asarray(self._f0.logpdf(f0), dtype=float)
        lp = lp + np.where((Q >= self._qmin) & (Q <= self._qmax), self._logq_q, -np.inf)

        l_snr_mf = self._log_truncnorm(snr, mu)
        l_snr_pr = np.asarray(self._snr_prior.logpdf(snr), dtype=float)
        lp = lp + np.logaddexp(np.log(self._mix) + l_snr_pr,
                               np.log1p(-self._mix) + l_snr_mf)

        dphi = np.mod(phi - ph + np.pi, 2 * np.pi) - np.pi
        l_phi_mf = self._log_vonmises(dphi, 0.0)
        lp = lp + np.logaddexp(np.log(self._mix) - np.log(2 * np.pi),
                               np.log1p(-self._mix) + l_phi_mf)
        out = lp.reshape(x.shape[:-1])
        return out if out.ndim else float(out)


def build_mf_birth(template, data, psd, spec, *, q_bounds=(0.1, 40.0),
                   rho_star=5.0, snr_bounds=(0.0, 100.0), floor_frac=0.1, n=512,
                   sigma_snr=1.5, kappa_phi=8.0, mix=0.2, snr_shrink=0.9):
    """Matched-filter birth ``generate_dist`` (data-fitted SNR + phase).

    Same placement maps as :func:`build_guided_birth`, but the linear wavelet
    parameters are proposed around the data's matched-filter fit at the
    proposed ``(t0, f0, Q)`` — the missing BayesWave ingredient that lifts birth
    acceptance from "lands on the signal" to "lands on the signal with the
    right amplitude and phase".
    """
    f = np.asarray(template.frequency_array_masked())
    power_f = np.sum(np.abs(np.asarray(data)) ** 2 / np.asarray(psd), axis=0)
    fgrid = np.linspace(f[0], f[-1], n)
    f0_dist = DataInformedMarginal(fgrid, np.interp(fgrid, f, power_f), floor_frac)
    tgrid, env = _whitened_time_envelope(template, data, psd, n=n)
    t0_dist = DataInformedMarginal(tgrid, env, floor_frac)

    reference_psd = 1.0 / np.sum(1.0 / np.asarray(psd, dtype=float), axis=0)
    joint = MatchedFilterBirth(
        f, data, psd, reference_psd, t0_dist, f0_dist,
        q_bounds=q_bounds,
        snr_prior=SNRPrior(rho_star=rho_star, snr_min=snr_bounds[0], snr_max=snr_bounds[1]),
        snr_bounds=snr_bounds, sigma_snr=sigma_snr, kappa_phi=kappa_phi,
        mix=mix, snr_shrink=snr_shrink,
    )
    branch = next(k for k in spec["priors"] if k != "extrinsic")
    return {branch: ProbDistContainer({(0, 1, 2, 3, 4): joint}),
            "extrinsic": spec["extrinsic"]}


class _FlowDistAdapter:
    """Wrap an Eryn 1-D distribution for AdaptiveFlowProposal (needs bounds)."""

    def __init__(self, dist, minimum, maximum):
        self._d = dist
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def rvs(self, size=1, random_state=None):
        return np.asarray(self._d.rvs(size=size), dtype=float).reshape(size)

    def logpdf(self, x):
        return np.asarray(self._d.logpdf(np.asarray(x, dtype=float)), dtype=float)


def build_flow_proposal(duration, *, minimum_frequency=20.0, maximum_frequency=512.0,
                        q_bounds=(0.1, 40.0), rho_star=5.0, snr_bounds=(0.0, 100.0),
                        device=None, min_training_samples=256, **flow_kwargs):
    """Adaptive normalizing-flow proposal over the 5 wavelet parameters.

    Learns a joint distribution of single-wavelet ``(t0, f0, Q, SNR, phi0)`` from
    the chain and proposes from it for both in-model moves (parameters) and RJ
    births (number of wavelets). Falls back to the prior until trained.
    """
    from eryn.prior import uniform_dist

    from .flow_proposals import AdaptiveFlowProposal
    from .wavelet_priors import LogUniformPrior

    dists = {
        0: _FlowDistAdapter(uniform_dist(0.0, duration), 0.0, duration),
        1: _FlowDistAdapter(LogUniformPrior(minimum_frequency, maximum_frequency),
                            minimum_frequency, maximum_frequency),
        2: _FlowDistAdapter(uniform_dist(q_bounds[0], q_bounds[1]), q_bounds[0], q_bounds[1]),
        3: _FlowDistAdapter(SNRPrior(rho_star=rho_star, snr_min=snr_bounds[0], snr_max=snr_bounds[1]),
                            snr_bounds[0], snr_bounds[1]),
        4: _FlowDistAdapter(uniform_dist(0.0, 2.0 * np.pi), 0.0, 2.0 * np.pi),
    }
    return AdaptiveFlowProposal(
        dists, periodic_parameters={4: 2.0 * np.pi}, device=device,
        min_training_samples=min_training_samples, **flow_kwargs,
    )


def guided_initial_wavelets(template, data, psd, spec, shape, rng):
    """Draw initial wavelets from the data-informed birth distribution.

    ``shape`` is ``(ntemps, nwalkers, nleaves_max)``. Returns a
    ``(*shape, 5)`` array. Warm-starts the chain near the signal so the flow
    trains on useful samples from the outset.
    """
    birth = build_guided_birth(template, data, psd, spec)
    branch = next(k for k in spec["priors"] if k != "extrinsic")
    total = int(np.prod(shape))
    draws = birth[branch].rvs(size=total)
    return np.asarray(draws, dtype=float).reshape(*shape, 5)


try:
    from eryn.moves import MHMove
except ImportError as exc:  # pragma: no cover - eryn is a core dependency
    raise ImportError("Wavelet moves require Eryn (eryn.moves).") from exc


def _wavelet_fisher_sigma(p, snr_floor):
    """Analytic per-parameter sine-Gaussian Fisher widths for ``[t0,f0,Q,snr,phi]``.

    Follows BayesWave's ``intrinsic_fisher_update``. Because HyperWave's wavelet
    amplitude parameter *is* the per-wavelet SNR, no PSD is needed: the wavelet's
    own ``snr`` (floored) sets the scale. ``p`` is ``(n, 5)``; returns ``(n, 5)``.
    """
    f0, Q, snr = p[:, 1], p[:, 2], p[:, 3]
    SNR = np.maximum(np.abs(snr), snr_floor)
    inv = 1.0 / SNR
    Qsq = Q * Q
    sig = np.empty_like(p)
    sig[:, 0] = (Q / (2.0 * np.pi * f0)) / np.sqrt(Qsq + 1.0) * inv   # t0
    sig[:, 1] = 2.0 * f0 / np.sqrt(Qsq + 3.0) * inv                  # f0
    sig[:, 2] = 2.0 * Q / np.sqrt(3.0) * inv                         # Q
    sig[:, 3] = np.abs(snr) * inv                                    # snr (amplitude)
    sig[:, 4] = inv                                                  # phi
    return np.clip(sig, 1e-12, None)


class WaveletFisherMove(MHMove):
    """Local Fisher-matrix in-model proposal for Morlet-Gabor wavelets.

    BayesWave's dominant in-model move (used ~80% of the time there): for each
    active wavelet the step size in each of ``[t0, f0, Q, snr, phi]`` is the
    analytic sine-Gaussian Fisher width, which depends on the wavelet's own
    parameters and SNR -- narrow/high-frequency wavelets get small t/f steps,
    broad ones get large steps. The widths are recomputed at the proposed point
    for the asymmetric Metropolis-Hastings factor. Only ``branch_name`` is
    updated (other branches are passed through unchanged).
    """

    def __init__(self, branch_name="signal", scale=None, snr_floor=5.0, **kwargs):
        self.branch_name = branch_name
        self.snr_floor = float(snr_floor)
        self._scale = scale
        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        name = self.branch_name
        coords = branches_coords[name]
        ntemps, nwalkers, nleaves_max, ndim = coords.shape
        scale = self._scale if self._scale is not None else 1.0 / np.sqrt(ndim)
        inds = (np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
                if branches_inds is None else branches_inds[name])
        q = {n: c.copy() for n, c in branches_coords.items()}
        factors = np.zeros((ntemps, nwalkers))

        ti, wi, li = np.where(inds)
        if ti.size:
            x = coords[ti, wi, li]                                   # (n_active, ndim)
            sx = _wavelet_fisher_sigma(x, self.snr_floor) * scale
            y = x + sx * random.standard_normal(x.shape)
            y[:, 4] = np.mod(y[:, 4], 2.0 * np.pi)                   # wrap phase
            sy = _wavelet_fisher_sigma(y, self.snr_floor) * scale
            logqyx = np.sum(-0.5 * ((y - x) / sx) ** 2 - np.log(sx), axis=1)
            logqxy = np.sum(-0.5 * ((x - y) / sy) ** 2 - np.log(sy), axis=1)
            q[name][ti, wi, li] = y
            np.add.at(factors, (ti, wi), logqxy - logqyx)
        return q, factors


class WaveletHalfCycleMove(MHMove):
    """Phase<->time half-cycle degeneracy proposal (BayesWave-style).

    Shifts ``t0`` by *n* half-periods and ``phi`` by pi when *n* is odd (small
    jitter added). ``n`` is drawn symmetrically about 0, so the proposal is
    (approximately) symmetric and the MH factor is zero. Resolves the
    phase/time degeneracy that traps fixed-step Gaussian moves.
    """

    def __init__(self, branch_name="signal", **kwargs):
        self.branch_name = branch_name
        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        name = self.branch_name
        coords = branches_coords[name]
        ntemps, nwalkers, nleaves_max, ndim = coords.shape
        inds = (np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
                if branches_inds is None else branches_inds[name])
        q = {n: c.copy() for n, c in branches_coords.items()}
        ti, wi, li = np.where(inds)
        if ti.size:
            x = coords[ti, wi, li].copy()
            n = np.floor(5.0 * random.uniform(size=ti.size)).astype(int) - 2  # {-2..2}
            f0 = x[:, 1]
            dt = (n / 2.0) / f0 * (1.0 + 0.1 * random.standard_normal(ti.size))
            dp = np.where(n % 2 != 0, np.pi * (1.0 + 0.1 * random.standard_normal(ti.size)), 0.0)
            x[:, 0] = x[:, 0] + dt
            x[:, 4] = np.mod(x[:, 4] + dp, 2.0 * np.pi)
            q[name][ti, wi, li] = x
        return q, np.zeros((ntemps, nwalkers))


def _rotate_about_axis(k, axis, omega):
    """Rodrigues rotation of vectors ``k`` (3, n) about unit ``axis`` (3,) by ``omega`` (n,)."""
    c, s = np.cos(omega), np.sin(omega)
    axdotk = axis[0] * k[0] + axis[1] * k[1] + axis[2] * k[2]              # (n,)
    cross = np.array([axis[1] * k[2] - axis[2] * k[1],
                      axis[2] * k[0] - axis[0] * k[2],
                      axis[0] * k[1] - axis[1] * k[0]])                    # (3, n)
    return k * c + cross * s + axis[:, None] * axdotk * (1.0 - c)


class WaveletSkyRingMove(MHMove):
    """Sky-ring proposal: rotate (ra, dec) about the detector baseline (BayesWave).

    Rotates the line-of-sight vector by a random angle about the axis joining two
    detectors, moving along the constant-time-delay ring where the likelihood is
    nearly degenerate. A rigid sphere rotation is symmetric in the sphere measure
    ``d(ra) d(sin dec)``; since HyperWave samples ``dec`` (with a ``CosinePrior``),
    the MH factor ``log cos(dec_x) - log cos(dec_y)`` cancels that prior's
    Jacobian so the net move is the intended sphere-uniform jump. Operates on the
    ``extrinsic`` branch ``[ra, dec, psi, ellipticity]``.
    """

    _DET_INDEX = {
        "H1": "LALDetectorIndexLHODIFF", "L1": "LALDetectorIndexLLODIFF",
        "V1": "LALDetectorIndexVIRGODIFF", "K1": "LALDetectorIndexKAGRADIFF",
        "G1": "LALDetectorIndexGEO600DIFF",
    }

    def __init__(self, detector_names, reference_time, branch_name="extrinsic", **kwargs):
        import lal
        locs = [np.asarray(lal.CachedDetectors[getattr(lal, self._DET_INDEX[d])].location,
                           dtype=float) for d in detector_names[:2]]
        axis = locs[0] - locs[1]
        self.axis = axis / np.linalg.norm(axis)
        gmst = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(reference_time))))
        self.gmst = gmst % (2.0 * np.pi)
        self.branch_name = branch_name
        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        name = self.branch_name
        coords = branches_coords[name]
        ntemps, nwalkers, nleaves_max, ndim = coords.shape
        inds = (np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
                if branches_inds is None else branches_inds[name])
        q = {n: c.copy() for n, c in branches_coords.items()}
        factors = np.zeros((ntemps, nwalkers))

        ti, wi, li = np.where(inds)
        if ti.size:
            x = coords[ti, wi, li]                                   # (n, 4) [ra,dec,psi,ellip]
            ra, dec = x[:, 0], x[:, 1]
            cosd = np.cos(dec)
            k = np.array([np.cos(self.gmst - ra) * cosd,
                          -np.sin(self.gmst - ra) * cosd,
                          np.sin(dec)])                              # (3, n)
            omega = 2.0 * np.pi * random.uniform(size=ti.size)
            kp = _rotate_about_axis(k, self.axis, omega)
            new_dec = np.arcsin(np.clip(kp[2], -1.0, 1.0))
            new_ra = np.mod(np.arctan2(kp[1], kp[0]) + self.gmst, 2.0 * np.pi)
            y = x.copy()
            y[:, 0], y[:, 1] = new_ra, new_dec
            leaf_fac = (np.log(np.clip(cosd, 1e-12, None))
                        - np.log(np.clip(np.cos(new_dec), 1e-12, None)))
            q[name][ti, wi, li] = y
            np.add.at(factors, (ti, wi), leaf_fac)
        return q, factors


class WaveletGroupStretchMove(GroupStretchMove):
    """Group-stretch (affine-invariant) move for the variable-D wavelet branch.

    eryn-compatible analogue of BayesWave's Differential-Evolution proposal. A
    plain red-blue :class:`~eryn.moves.StretchMove` would demand
    ``2 * nleaves_max * ndim`` (~400) walkers because it sees the whole leaf
    space; the group variant draws the complementary point from a *stationary*
    pool of reference leaves instead, so it runs at the production walker count.

    The pool ("friends") is the set of currently-active wavelet leaves, gathered
    **per temperature** across the whole walker ensemble and refreshed every
    ``n_iter_update`` iterations (eryn stores the pre-update state to keep
    detailed balance). For every leaf slot we draw one friend uniformly from
    that temperature's pool and apply the standard stretch; eryn supplies the
    ``(ndim - 1) log z`` factor.
    """

    def setup_friends(self, branches):
        self._pools = {}
        for name, branch in branches.items():
            coords = np.asarray(branch.coords)   # (ntemps, nwalkers, nleaves, ndim)
            inds = np.asarray(branch.inds)       # (ntemps, nwalkers, nleaves) bool
            ntemps, _, _, ndim = coords.shape
            pools = []
            for t in range(ntemps):
                on = coords[t][inds[t]]          # (Non, ndim)
                if on.shape[0] == 0:             # no active leaf yet -> use all slots
                    on = coords[t].reshape(-1, ndim)
                pools.append(on)
            self._pools[name] = pools

    def find_friends(self, name, s, s_inds=None, branch_supps=None):
        s = np.asarray(s)
        ntemps, nwalkers, nleaves_max, ndim = s.shape
        pools = self._pools[name]
        c = np.empty_like(s)
        for t in range(ntemps):
            pool = pools[t]
            idx = np.random.randint(0, pool.shape[0], size=nwalkers * nleaves_max)
            c[t] = pool[idx].reshape(nwalkers, nleaves_max, ndim)
        return c


__all__ = ["DataInformedMarginal", "build_guided_birth", "build_flow_proposal",
           "guided_initial_wavelets", "WaveletFisherMove", "WaveletHalfCycleMove",
           "WaveletGroupStretchMove",
           "WaveletSkyRingMove"]
