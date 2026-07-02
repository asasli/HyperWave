"""Heterodyned (relative-binning) Gaussian likelihood.

Implements the relative-binning scheme of Zackay, Dai & Venumadhav (2018):
the waveform is written as ``h(f) = r(f) h0(f)`` against a fixed reference
waveform ``h0`` (evaluated once on the full grid), and the smooth ratio ``r(f)``
is approximated as piecewise-linear over a small number of frequency bins whose
edges follow the post-Newtonian phase evolution. The inner products
``<d|h>`` and ``<h|h>`` then reduce to per-bin summary data (``A0/A1/B0/B1``)
contracted with the ratio at the bin edges — so each likelihood call needs the
waveform only at ``O(100)`` edge frequencies instead of the full ``O(10^5-10^6)``
grid. This is the big speed lever for long-duration (BNS-like) signals.

The class is **template-agnostic and batch-native**: it takes a callable
``waveform_edges(thetas) -> (N, n_ifo, n_edges)`` evaluating the
detector-projected waveform for a *batch* of parameter vectors at the bin-edge
frequencies (build e.g. a second LVK ``Template``/``GW`` or a LISA
``LISAAETTemplate`` on the edge grid), plus the reference waveform ``h0`` on the
full grid. GPU-capable through the standard array backend; falls back to CPU
automatically.

Validity: the linear-ratio approximation holds while ``h`` stays phase-coherent
with ``h0`` within each bin — i.e. in the bulk of the posterior around the
reference point. Use ``eps`` (max per-bin differential phase, radians) to trade
bins for accuracy; logL errors scale ~``eps**2``.
"""

from __future__ import annotations

import numpy as np

from .base import BaseLikelihood

# PN phase-evolution exponents used for bin placement (Zackay+18 eq. 2.6).
_GAMMAS = np.array([-5.0 / 3.0, -2.0 / 3.0, 1.0, 5.0 / 3.0, 7.0 / 3.0])


def heterodyne_bin_edges(f, chi=1.0, eps=0.5):
    """Choose relative-binning bin-edge *indices* into the frequency grid ``f``.

    Edges are placed so the maximal differential phase a PN-like waveform can
    accumulate within one bin is ``<= eps`` radians (smaller ``eps`` = more
    bins = more accurate).

    Returns an integer index array ``edges`` (monotone, first=0,
    last=len(f)-1); bin ``b`` spans ``f[edges[b]] .. f[edges[b+1]]``.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or f.size < 3:
        raise ValueError("need a 1-D frequency grid with >= 3 points")
    f_star = np.where(_GAMMAS < 0, f[0], f[-1])
    dphi = 2.0 * np.pi * chi * np.sum(
        np.sign(_GAMMAS) * (f[:, None] / f_star[None, :]) ** _GAMMAS[None, :], axis=1
    )
    dphi = dphi - dphi[0]  # monotone increasing from 0
    nbins = max(1, int(np.ceil(dphi[-1] / float(eps))))
    targets = np.linspace(0.0, dphi[-1], nbins + 1)
    edges = np.unique(np.searchsorted(dphi, targets))
    edges[0] = 0
    edges[-1] = f.size - 1
    return np.unique(edges)


class HeterodyneLikelihood(BaseLikelihood):
    """Relative-binning Gaussian log-likelihood.

    Parameters
    ----------
    data : array (n_ifo, n_freq)
        Frequency-domain data on the full grid.
    f : array (n_freq,)
        Full frequency grid (uniform spacing).
    psd : array (n_ifo, n_freq)
        One-sided noise PSDs.
    ifos_list : sequence of str
        Channel names (sets ``n_ifo``).
    h0 : array (n_ifo, n_freq)
        Reference waveform on the full grid (detector-projected), evaluated at
        ``theta_ref`` — ideally near the likelihood maximum (the injection or a
        search trigger point works).
    waveform_edges : callable
        ``waveform_edges(thetas) -> (N, n_ifo, n_edges)`` giving the
        detector-projected waveform of a parameter batch at ``self.f_edges``.
    chi, eps : float
        Bin-placement tuning (see :func:`heterodyne_bin_edges`).
    gpu : bool
        Run the per-call contraction on the GPU backend (CuPy) when available.

    Notes
    -----
    ``logl`` returns ``<d|h> - <h|h>/2`` (the ``-<d|d>/2`` constant is dropped;
    add ``self.dd_const`` if an absolute Gaussian logL is needed).
    """

    def __init__(self, data, f, psd, ifos_list, h0, waveform_edges,
                 chi=1.0, eps=0.5, gpu=False, infs=-1e300):
        self._init_backend(gpu=gpu)
        xp = self.xp

        f = np.asarray(f, dtype=float)
        data = np.asarray(data, dtype=complex)
        psd = np.asarray(psd, dtype=float)
        h0 = np.asarray(h0, dtype=complex)
        self.ifos = list(ifos_list)
        n_ifo = len(self.ifos)
        if data.shape != (n_ifo, f.size) or h0.shape != data.shape or psd.shape != data.shape:
            raise ValueError(
                f"shape mismatch: data {data.shape}, h0 {h0.shape}, psd {psd.shape}, "
                f"expected ({n_ifo}, {f.size})"
            )
        self.df = float(f[1] - f[0])
        self._inf = float(infs)
        self._waveform_edges = waveform_edges

        # --- bins & edge grid (host-side, done once) ---
        self.edge_idx = heterodyne_bin_edges(f, chi=chi, eps=eps)
        self.f_edges = f[self.edge_idx]
        self.n_bins = self.f_edges.size - 1
        # reference waveform at the edges; bins where h0 ~ 0 at an edge cannot
        # support the ratio expansion and are masked out of the sums entirely.
        h0_edge = h0[:, self.edge_idx]
        tiny = 1e-3 * np.max(np.abs(h0), axis=1, keepdims=True)
        self._bin_ok = (np.abs(h0_edge[:, :-1]) > tiny) & (np.abs(h0_edge[:, 1:]) > tiny)
        safe = np.where(np.abs(h0_edge) > 0, h0_edge, 1.0)
        self._h0_edge_safe = safe

        # --- per-bin summary data A0/A1/B0/B1 (host-side, done once) ---
        w_dh = 4.0 * self.df * data * np.conj(h0) / psd          # (n_ifo, n_freq)
        w_hh = 4.0 * self.df * (np.abs(h0) ** 2) / psd
        A0 = np.zeros((n_ifo, self.n_bins), dtype=complex)
        A1 = np.zeros_like(A0)
        B0 = np.zeros((n_ifo, self.n_bins))
        B1 = np.zeros_like(B0)
        for b in range(self.n_bins):
            lo, hi = self.edge_idx[b], self.edge_idx[b + 1]
            sl = slice(lo, hi + 1 if b == self.n_bins - 1 else hi)
            fm = 0.5 * (f[lo] + f[self.edge_idx[b + 1]])
            dfm = f[sl] - fm
            A0[:, b] = np.sum(w_dh[:, sl], axis=1)
            A1[:, b] = np.sum(w_dh[:, sl] * dfm, axis=1)
            B0[:, b] = np.sum(w_hh[:, sl], axis=1).real
            B1[:, b] = np.sum(w_hh[:, sl] * dfm, axis=1).real
        mask = self._bin_ok.astype(float)
        # device copies for the hot loop
        self._A0 = xp.asarray(A0 * mask)
        self._A1 = xp.asarray(A1 * mask)
        self._B0 = xp.asarray(B0 * mask)
        self._B1 = xp.asarray(B1 * mask)
        self._fL = xp.asarray(self.f_edges[:-1])
        self._dfbin = xp.asarray(np.diff(self.f_edges))
        self._h0e = xp.asarray(safe)

        # absolute-normalisation constant, if a true Gaussian logL is wanted
        self.dd_const = -0.5 * float(np.sum((4.0 * self.df * np.abs(data) ** 2 / psd).real))

    # ------------------------------------------------------------------
    def logl(self, theta):
        """Vectorised heterodyned logL for a batch ``theta`` of shape (N, ndim)."""
        xp = self.xp
        theta = self._ensure_2d(theta)
        h_edge = self._waveform_edges(theta)            # (N, n_ifo, n_edges)
        h_edge = xp.asarray(h_edge)
        r = h_edge / self._h0e[None, :, :]
        rL, rR = r[..., :-1], r[..., 1:]
        r0 = 0.5 * (rL + rR)
        r1 = (rR - rL) / self._dfbin[None, None, :]
        dh = xp.sum(
            (self._A0[None] * xp.conj(r0) + self._A1[None] * xp.conj(r1)).real,
            axis=(1, 2),
        )
        hh = xp.sum(
            self._B0[None] * xp.abs(r0) ** 2
            + 2.0 * self._B1[None] * (r0 * xp.conj(r1)).real,
            axis=(1, 2),
        )
        out = dh - 0.5 * hh
        out = xp.nan_to_num(out, copy=False, nan=self._inf, posinf=self._inf, neginf=self._inf)
        return self._prepare_outputs(out).squeeze()

    __call__ = logl
    gaussian = logl  # API parity with GWLikelihoods

    # ------------------------------------------------------------------
    @classmethod
    def from_lvk_template(cls, template, data, f, psd, ifos_list, theta_ref,
                          chi=1.0, eps=0.5, gpu=False):
        """Build a heterodyne likelihood from an LVK template in one call.

        ``template`` is a :class:`hyperwave.detectors.lvk.GW` (or its low-level
        ``Template``) on the **full masked grid** ``f``; ``theta_ref`` is the
        reference point (injection / trigger). Internally this evaluates ``h0``
        once on the full grid and constructs a second lal ``Template`` in
        *sequence mode* on the sparse bin-edge grid for the per-call batch
        evaluations.
        """
        from ..detectors.waveforms.template import Template

        low = getattr(template, "template", template)  # GW wraps Template
        theta_ref = np.atleast_2d(np.asarray(theta_ref, dtype=float))
        h0 = np.asarray(template.make_injections_to_ifo_batch(theta_ref))[0]

        f = np.asarray(f, dtype=float)
        edge_idx = heterodyne_bin_edges(f, chi=chi, eps=eps)
        f_edges = f[edge_idx]
        edge_template = Template(
            detectors=low.detector_names,
            frequency_array=f_edges,
            sampling_rate=low.sampling_rate,
            duration=low.duration,
            start_time=low.start_time,
            minimum_frequency=f_edges[0],
            maximum_frequency=f_edges[-1],
            reference_frequency=low.reference_frequency,
            approximant=low.approximant,
            parameters=low.parameters,
            static_parameters=low.static_parameters,
            backend="lal",
            trigger_time=low.trigger_time,
            sequence=True,
            calibration_models=getattr(low, "calibration_models", None),
        )

        def waveform_edges(thetas):
            return np.asarray(edge_template.make_injections_to_ifo_batch(thetas))

        like = cls(data=data, f=f, psd=psd, ifos_list=ifos_list, h0=h0,
                   waveform_edges=waveform_edges, chi=chi, eps=eps, gpu=gpu)
        like.edge_template = edge_template
        return like




class InterpolatedWaveformTemplate:
    """Edge-evaluated waveform reconstructed onto the full grid.

    "Heterodyne the waveform, not the likelihood": the waveform is generated
    only at the relative-binning edge frequencies (the expensive part — LAL
    evaluation dominates likelihood cost), the smooth ratio ``h/h0`` is
    interpolated linearly onto the full grid, and ``h = ratio * h0`` is handed
    to ANY exact likelihood. This accelerates likelihoods that cannot be
    binned analytically — the **hyperbolic** (the sqrt does not expand into
    per-bin moments, and its delta weights are sampled) and the Whittle — at
    the same piecewise-linear-ratio accuracy as standard relative binning.

    Drop-in template for :class:`~hyperwave.likelihoods.GWLikelihoods`
    (exposes ``parameters`` + ``make_injections_to_ifo_batch``).
    """

    def __init__(self, template, f, theta_ref, chi=1.0, eps=0.1):
        from ..detectors.waveforms.template import Template

        low = getattr(template, "template", template)  # GW wraps Template
        self.parameters = list(low.parameters)
        theta_ref = np.atleast_2d(np.asarray(theta_ref, dtype=float))
        h0 = np.asarray(template.make_injections_to_ifo_batch(theta_ref))[0]

        f = np.asarray(f, dtype=float)
        edge_idx = heterodyne_bin_edges(f, chi=chi, eps=eps)
        self.f_edges = f[edge_idx]
        self._edge_template = Template(
            detectors=low.detector_names, frequency_array=self.f_edges,
            sampling_rate=low.sampling_rate, duration=low.duration,
            start_time=low.start_time, minimum_frequency=self.f_edges[0],
            maximum_frequency=self.f_edges[-1],
            reference_frequency=low.reference_frequency,
            approximant=low.approximant, parameters=low.parameters,
            static_parameters=low.static_parameters, backend="lal",
            trigger_time=low.trigger_time, sequence=True,
            calibration_models=getattr(low, "calibration_models", None),
        )
        h0_edge = h0[:, edge_idx]
        tiny = 1e-3 * np.max(np.abs(h0), axis=1, keepdims=True)
        self._h0_edge_safe = np.where(np.abs(h0_edge) > tiny, h0_edge, np.inf)
        self._h0 = h0

        # linear-interpolation stencil from the edge grid onto the full grid
        lo = np.clip(np.searchsorted(self.f_edges, f, side="right") - 1,
                     0, self.f_edges.size - 2)
        df_bin = self.f_edges[lo + 1] - self.f_edges[lo]
        self._lo = lo
        self._w = ((f - self.f_edges[lo]) / df_bin).astype(float)

    def make_injections_to_ifo_batch(self, thetas, masked=True):
        h_edge = np.asarray(self._edge_template.make_injections_to_ifo_batch(thetas))
        r_edge = h_edge / self._h0_edge_safe[None, :, :]
        r_full = (r_edge[..., self._lo] * (1.0 - self._w)
                  + r_edge[..., self._lo + 1] * self._w)
        return r_full * self._h0[None, :, :]




class HeterodynedHyperbolicLikelihood(BaseLikelihood):
    r"""Heterodyned (relative-binning) **hyperbolic** log-likelihood.

    The hyperbolic frequency sum :math:`T_s(\delta)=\sum_{f\in s}
    \sqrt{\delta^2+yy(f)}` cannot be binned exactly (the square root has no
    finite per-bin moment expansion and :math:`\delta` is sampled). This class
    implements the **first-order heterodyne around the reference residual**:
    with :math:`yy = yy_0 + \Delta` and reference-fixed weights
    :math:`w_1(f;\delta) = 1/(2\sqrt{\delta^2+yy_0(f)})`,

    .. math:: T_s \approx T^0_s(\delta) + \sum_f w_1(f;\delta)\,\Delta(f),

    where :math:`\Delta(f)` is linear+quadratic in the piecewise-linear ratio
    :math:`\tilde r = h/h_0`, so the sum collapses into per-bin summaries
    :math:`U_0,U_1,U_2` (real) and :math:`V_0,V_1` (complex). The sampled
    :math:`\delta_s` is handled by tabulating :math:`T^0_s` and the summaries
    on a :math:`\delta`-grid and interpolating linearly per walker.

    Exact at the reference waveform for every :math:`(\alpha,\delta)`;
    first-order accurate in the waveform perturbation (the neglected term is
    :math:`-\sum_f \Delta^2/(8(\delta^2+yy_0)^{3/2})`, second order in the
    residual change). Bins are forced to respect the segment boundaries.

    The parameter layout matches ``GWLikelihoods.hyperbolic_classic`` with
    ``ddims=False``: ``theta = [waveform params, alpha, delta_0..delta_{nsegs-1}]``.
    The exact hyperbolic remains available in :class:`GWLikelihoods` — this is
    an additional, faster likelihood, not a replacement.
    """

    def __init__(self, data, f, psd, ifos_list, h0, waveform_edges,
                 nsegs=2, chi=1.0, eps=0.1, delta_max=30.0, n_delta_grid=193,
                 gpu=False, infs=-1e300):
        self._init_backend(gpu=gpu)

        f = np.asarray(f, dtype=float)
        data = np.asarray(data, dtype=complex)
        psd = np.asarray(psd, dtype=float)
        h0 = np.asarray(h0, dtype=complex)
        self.ifos = list(ifos_list)
        n_ifo = len(self.ifos)
        if data.shape != (n_ifo, f.size) or h0.shape != data.shape or psd.shape != data.shape:
            raise ValueError("data/h0/psd must all be (n_ifo, n_freq)")
        self.df = float(f[1] - f[0])
        self._inf = float(infs)
        self._waveform_edges = waveform_edges
        self._nsegs = int(nsegs)

        # hyperbolic constants (mirror GWLikelihoods)
        d = 2 * n_ifo
        self._lam = (d + 1) / 2.0
        self._C0 = ((1 - d) / 2.0) * np.log(2.0 * np.pi)

        # --- segments and segment-respecting bins ---
        segi, Nd, _fb = self._build_segments(f, self._nsegs)
        self._Nd = np.asarray(Nd, dtype=float)
        seg_starts = np.array([si[0] for si in segi], dtype=int)
        edge_idx = heterodyne_bin_edges(f, chi=chi, eps=eps)
        edge_idx = np.unique(np.concatenate([edge_idx, seg_starts]))
        self.edge_idx = edge_idx
        self.f_edges = f[edge_idx]
        B = self.f_edges.size - 1
        self.n_bins = B
        # bin -> segment assignment by bin start index
        self._seg_of_bin = (np.searchsorted(seg_starts, edge_idx[:-1], side="right") - 1).astype(int)

        # reference quantities on the full grid
        r0f = data - h0
        yy0 = 4.0 * self.df * np.sum((r0f.conj() * r0f).real / psd, axis=0)  # (nf,)
        u = 4.0 * self.df * (np.abs(h0) ** 2) / psd                          # (k, nf)
        v = 4.0 * self.df * np.conj(data) * h0 / psd                         # (k, nf) complex

        h0_edge = h0[:, edge_idx]
        tiny = 1e-3 * np.max(np.abs(h0), axis=1, keepdims=True)
        self._h0e = np.where(np.abs(h0_edge) > tiny, h0_edge, np.inf)
        fm = 0.5 * (self.f_edges[:-1] + self.f_edges[1:])
        dfm = f - fm[np.clip(np.searchsorted(self.f_edges, f, side="right") - 1, 0, B - 1)]
        self._dfbin = np.diff(self.f_edges)

        # --- delta-grid tabulation (the sampled delta enters the weights) ---
        dgrid = np.concatenate([[0.0], np.geomspace(1e-2, float(delta_max), n_delta_grid - 1)])
        self._dgrid = dgrid
        G = dgrid.size
        starts = edge_idx[:-1]
        U0 = np.zeros((G, n_ifo, B))
        U1 = np.zeros_like(U0)
        U2 = np.zeros_like(U0)
        V0 = np.zeros((G, n_ifo, B), dtype=complex)
        V1 = np.zeros_like(V0)
        # second-order (w3) summaries: same structure, weight 1/(8(d^2+yy0)^{3/2})
        W3 = np.zeros((G, B))
        P0 = np.zeros((G, n_ifo, B))
        P1 = np.zeros_like(P0)
        P2 = np.zeros_like(P0)
        Q0 = np.zeros((G, n_ifo, B), dtype=complex)
        Q1 = np.zeros_like(Q0)
        seg_id_f = (np.searchsorted(seg_starts, np.arange(f.size), side="right") - 1).astype(int)
        for g, dg in enumerate(dgrid):
            root = np.sqrt(dg * dg + yy0)               # (nf,)
            w1 = 0.5 / root
            w3 = 0.125 / root**3
            U0[g] = np.add.reduceat(u * w1, starts, axis=1)
            U1[g] = np.add.reduceat(u * w1 * dfm, starts, axis=1)
            U2[g] = np.add.reduceat(u * w1 * dfm * dfm, starts, axis=1)
            V0[g] = np.add.reduceat(v * w1, starts, axis=1)
            V1[g] = np.add.reduceat(v * w1 * dfm, starts, axis=1)
            W3[g] = np.add.reduceat(w3, starts)
            P0[g] = np.add.reduceat(u * w3, starts, axis=1)
            P1[g] = np.add.reduceat(u * w3 * dfm, starts, axis=1)
            P2[g] = np.add.reduceat(u * w3 * dfm * dfm, starts, axis=1)
            Q0[g] = np.add.reduceat(v * w3, starts, axis=1)
            Q1[g] = np.add.reduceat(v * w3 * dfm, starts, axis=1)
        self._U0, self._U1, self._U2, self._V0, self._V1 = U0, U1, U2, V0, V1
        self._W3, self._P0, self._P1, self._P2, self._Q0, self._Q1 = W3, P0, P1, P2, Q0, Q1
        # T0 enters logL multiplied by alpha — tabulate it on a much denser grid
        # (cheap: one scalar per segment) and interpolate with cubic splines.
        from scipy.interpolate import CubicSpline
        dgrid_T = np.concatenate([[0.0], np.geomspace(1e-3, float(delta_max), 768)])
        T0_T = np.zeros((dgrid_T.size, self._nsegs))
        for g, dg in enumerate(dgrid_T):
            np.add.at(T0_T[g], seg_id_f, np.sqrt(dg * dg + yy0))
        self._T0_spline = [CubicSpline(dgrid_T, T0_T[:, s_]) for s_ in range(self._nsegs)]

    # -- delta interpolation helpers ------------------------------------------
    def _grid_weights(self, deltas):
        """(N, S) deltas -> (g0, frac) linear-interp coordinates on the grid."""
        dg = self._dgrid
        idx = np.clip(np.searchsorted(dg, deltas, side="right") - 1, 0, dg.size - 2)
        frac = (deltas - dg[idx]) / (dg[idx + 1] - dg[idx])
        return idx, np.clip(frac, 0.0, 1.0)

    def logl(self, theta):
        """theta = (N, wf + 1 + nsegs): [waveform, alpha, delta_0..]."""
        theta = self._ensure_2d(np.asarray(theta, dtype=float))
        nseg = self._nsegs
        wf = theta[:, : theta.shape[1] - 1 - nseg]
        alpha = theta[:, -1 - nseg]
        deltas = theta[:, -nseg:]                       # (N, S)
        N = theta.shape[0]

        # ratio at bin edges -> per-bin (r0, r1) per detector
        h_edge = np.asarray(self._waveform_edges(wf))    # (N, k, E)
        r = h_edge / self._h0e[None, :, :]
        rL, rR = r[..., :-1], r[..., 1:]
        r0 = 0.5 * (rL + rR)                             # (N, k, B)
        r1 = (rR - rL) / self._dfbin[None, None, :]

        # interp coordinates per walker/segment -> per bin
        g0, fr = self._grid_weights(deltas)              # (N, S)
        gb = g0[:, self._seg_of_bin]                     # (N, B)
        fb = fr[:, self._seg_of_bin]

        def gather(S):                                   # S: (G, k, B) -> (N, k, B)
            lo = S[gb, :, np.arange(self.n_bins)[None, :]]      # (N, B, k)
            hi = S[gb + 1, :, np.arange(self.n_bins)[None, :]]
            out = lo * (1.0 - fb[..., None]) + hi * fb[..., None]
            return np.moveaxis(out, -1, 1)               # (N, k, B)

        U0, U1, U2 = gather(self._U0), gather(self._U1), gather(self._U2)
        V0, V1 = gather(self._V0), gather(self._V1)

        def delta_contraction(A0_, A1_, A2_, C0_, C1_):
            """Sum_f w * Delta(f) per (walker, bin), for weight tables w."""
            b = ((np.abs(r0) ** 2 - 1.0) * A0_
                 + 2.0 * (r0 * np.conj(r1)).real * A1_
                 + (np.abs(r1) ** 2) * A2_
                 - 2.0 * ((r0 - 1.0) * C0_).real
                 - 2.0 * (r1 * C1_).real)               # (N, k, B)
            return np.sum(b, axis=1)                     # (N, B)

        # first order: + Sum w1*Delta
        corr_kb = delta_contraction(U0, U1, U2, V0, V1)
        # second order: - Sum w3*Delta^2 ~ -(Sum w3*Delta)^2 / Sum w3 per bin
        # (exact up to the in-bin variation of Delta, which the PN bin spacing
        # keeps small by construction)
        P0, P1, P2 = gather(self._P0), gather(self._P1), gather(self._P2)
        Q0, Q1 = gather(self._Q0), gather(self._Q1)
        s3 = delta_contraction(P0, P1, P2, Q0, Q1)       # (N, B)
        W3g = self._W3[gb, np.arange(self.n_bins)[None, :]] * (1.0 - fb) \
              + self._W3[gb + 1, np.arange(self.n_bins)[None, :]] * fb
        corr_kb = corr_kb - np.where(W3g > 0, s3 * s3 / np.maximum(W3g, 1e-300), 0.0)
        # accumulate bins into segments
        corr_s = np.zeros((N, nseg))
        for s in range(nseg):
            corr_s[:, s] = np.sum(corr_kb[:, self._seg_of_bin == s], axis=1)

        # T0 per walker/segment via cubic splines (accuracy: T0 multiplies alpha)
        T0 = np.column_stack([self._T0_spline[s_](deltas[:, s_]) for s_ in range(nseg)])
        T = T0 + corr_s

        # analytic hyperbolic terms (alpha shared, delta per segment)
        from scipy.special import kve
        with np.errstate(divide="ignore", invalid="ignore"):
            a_d = alpha[:, None] * deltas
            log_kv = np.log(kve(self._lam, a_d)) - a_d   # log K_lambda
            term = self._Nd[None, :] * (
                self._lam * np.log(alpha[:, None] / deltas)
                + self._C0 - np.log(2.0 * alpha[:, None]) - log_kv
            ) - alpha[:, None] * T
        out = np.nan_to_num(np.sum(term, axis=1), nan=self._inf,
                            posinf=self._inf, neginf=self._inf)
        return out.squeeze()

    __call__ = logl

    @classmethod
    def from_lvk_template(cls, template, data, f, psd, ifos_list, theta_ref,
                          nsegs=2, chi=1.0, eps=0.1, **kw):
        """One-call setup from an LVK template (mirrors
        :meth:`HeterodyneLikelihood.from_lvk_template`)."""
        from ..detectors.waveforms.template import Template

        low = getattr(template, "template", template)
        theta_ref = np.atleast_2d(np.asarray(theta_ref, dtype=float))
        h0 = np.asarray(template.make_injections_to_ifo_batch(theta_ref))[0]

        f = np.asarray(f, dtype=float)
        like = cls(data=data, f=f, psd=psd, ifos_list=ifos_list, h0=h0,
                   waveform_edges=None, nsegs=nsegs, chi=chi, eps=eps, **kw)
        edge_template = Template(
            detectors=low.detector_names, frequency_array=like.f_edges,
            sampling_rate=low.sampling_rate, duration=low.duration,
            start_time=low.start_time, minimum_frequency=like.f_edges[0],
            maximum_frequency=like.f_edges[-1],
            reference_frequency=low.reference_frequency,
            approximant=low.approximant, parameters=low.parameters,
            static_parameters=low.static_parameters, backend="lal",
            trigger_time=low.trigger_time, sequence=True,
            calibration_models=getattr(low, "calibration_models", None),
        )
        like._waveform_edges = lambda th: np.asarray(
            edge_template.make_injections_to_ifo_batch(th))
        like.edge_template = edge_template
        return like


__all__ = ["HeterodyneLikelihood", "HeterodynedHyperbolicLikelihood", "InterpolatedWaveformTemplate", "heterodyne_bin_edges"]
