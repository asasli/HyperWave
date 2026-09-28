"""SMBHB (massive black-hole binary) layer for LISA: bbhx template + the
paper's Table-I Sangria conventions, as an importable module.

Everything here is the validated code path of the HyperWave Sangria analyses
(arXiv paper Table I/II): the ``SMBHBbbhxTemplate`` wraps ``bbhx``
(BBHWaveformFD, PhenomD (2,2), TDI-1 AET) on an arbitrary frequency grid; the
sampling space is (Mc, q, chi1z, chi2z, log10 D, tc[h], cos i, sin beta,
lambda, psi, phi_ref) with the priors of Table II. ``sky_credible_area`` is the
canonical estimator used for all quoted sky areas (0.5-degree equal-area grid,
1-cell Gaussian smoothing, greedy HPD).

Downstream projects (e.g. HERON hand-offs) should import from here so that both
sides share one set of conventions:

    from hyperwave.detectors.lisa.smbhb import (
        SMBHBbbhxTemplate, sampling_to_physical, prior_bounds,
        SAMPLING_NAMES, PERIODIC_IDX, TRUTH_PHYS, sky_credible_area)
"""

from __future__ import annotations

import numpy as np

YRSID_SI = 31558149.763545603      # sidereal year [s] (bbhx convention)
MPC_M = 3.0856775814913674e22      # Mpc in metres
DAY = 86400.0
C_SI = 299792458.0
L_ARM = 2.5e9                      # LISA arm length [m]

TOBS_S = 30.4368 * DAY             # paper: 30.4368 days
TREF_S = TOBS_S - 1.0 * DAY        # coalescence 1 day before the end

#: periodic sampling dimensions: lambda, psi, phi_ref
PERIODIC_IDX = {8: 2 * np.pi, 9: np.pi, 10: 2 * np.pi}

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


def mc_q_to_m1_m2(Mc, q):
    """Chirp mass + mass ratio (q = m1/m2 >= 1) -> component masses.

    Paper ``smbhb.get_m1_m2_from_chirp_and_eta``: eta = q/(1+q)^2.
    """
    Mc = np.asarray(Mc, dtype=float)
    q = np.asarray(q, dtype=float)
    eta = q / (1.0 + q) ** 2
    Mtot = Mc * eta ** -0.6
    disc = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
    m1 = 0.5 * Mtot * (1.0 + disc)
    m2 = 0.5 * Mtot * (1.0 - disc)
    return m1, m2


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


class SMBHBbbhxTemplate:
    """Batched frequency-domain LISA SMBHB A/E/T template backed by bbhx.

    ``batch_eval`` takes arrays (length ``N``) and returns ``(N, 3, nfreq)``;
    bbhx requires array inputs (it derives the batch size from ``len(m1)``), so
    this is both the correct call *and* the vectorised fast path.
    """

    def __init__(self, freqs, f_ref=0.0, t_obs_years=1.0, modes=None, run_phenomd=True,
                 length=1024, force_backend=None):
        from bbhx.waveformbuild import BBHWaveformFD  # lazy: keep import cheap
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


def sky_credible_area(lon, sinlat, p, nbins=(720, 360), smooth=1.0):
    """Credible sky area [deg^2] -- the canonical HyperWave estimator.

    Equal-area (lon, sin lat) histogram at 0.5-degree resolution, 1-cell
    Gaussian smoothing (balances raw-histogram noise-chasing against
    over-smoothing at ESS ~ few x 10^3), greedy HPD cell count. All quoted
    HyperWave sky areas use these defaults; state them when reporting.
    """
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(np.asarray(lon) % (2 * np.pi),
                               np.clip(sinlat, -1, 1), bins=nbins,
                               range=[[0, 2 * np.pi], [-1, 1]])
    if smooth:
        H = gaussian_filter(H, smooth)
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h) / h.sum()
    nsel = int(np.searchsorted(c, p) + 1)
    return nsel * (xe[1] - xe[0]) * (ye[1] - ye[0]) * (180 / np.pi) ** 2
