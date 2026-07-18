"""Wavelet (RJMCMC) reconstruction plots for HyperWave.

Frequency- and time-domain reconstruction figures for the Morlet-Gabor wavelet
model, in the shared HyperWave per-detector colours and typography.

This module owns only the *wavelet-specific* pieces:

* turning a reversible-jump (variable-dimension) chain into per-draw network
  waveforms, ``h(f)`` and whitened ``h(t)``;
* the per-detector colour scheme.

Everything generic is reused from the shared plot helpers (no duplication):

* credible intervals  -> :func:`hyperwave.plots.fd_reconstruction.compute_credible_region_fd`
                         and :func:`hyperwave.plots.td_reconstruction.compute_credible_region`
* global style/fonts  -> :func:`hyperwave.plots.style.apply_style` / ``style_axes``
* legend handler      -> :class:`hyperwave.plots.fd_reconstruction.HandlerPatchLine`

Reconstruction summaries are taken on the **amplitude** ``|h(f)|`` (frequency
domain) and on the **real strain** ``h(t)`` (time domain) -- never on the median
of the complex parts, which cancels under per-sample phase scatter and produces
a spurious high-frequency collapse.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .corners import rcparams1, rcparams2
from .fd_reconstruction import HandlerPatchLine, compute_credible_region_fd
from .style import IFO_COLORS, apply_style, ifo_palette, style_axes
from .td_reconstruction import compute_credible_region

_FD_YLABEL = (
    r"$|\tilde{h}(f)|\,\sqrt{2\Delta f}$ & $\sqrt{S(f)}$"
    r"$\quad\left[\sqrt{\rm{Hz}^{-1}}\right]$"
)


# ======================================================================
# Wavelet-specific reconstruction (variable-dimension RJ chain -> waveforms)
# ======================================================================

def select_wavelet_draws(sampler=None, *, coords=None, inds=None, sample_sky=True,
                         discard_frac=0.3, n_draws=1000, seed=0):
    """Pick cold-chain posterior draws from an Eryn wavelet run.

    Pass either an Eryn ``sampler`` or pre-extracted ``coords``/``inds`` dicts.
    Returns ``(wav, msk, sky)`` where ``wav`` is ``(D, lmax, 5)`` wavelet
    parameters, ``msk`` is the ``(D, lmax)`` active-leaf mask, and ``sky`` is
    ``(D, 4)`` extrinsic parameters (``None`` when ``sample_sky`` is False).
    """
    if coords is None or inds is None:
        coords, inds = sampler.get_chain(), sampler.get_inds()
    wav = np.asarray(coords["signal"])[:, 0]          # (nsteps, nwalkers, lmax, 5)
    msk = np.asarray(inds["signal"])[:, 0]            # (nsteps, nwalkers, lmax)
    ns, nw, lmax, _ = wav.shape
    start = int(discard_frac * ns)
    wav = wav[start:].reshape(-1, lmax, 5)
    msk = msk[start:].reshape(-1, lmax)
    sky = None
    if sample_sky:
        sky = np.asarray(coords["extrinsic"])[:, 0][start:].reshape(-1, 4)
    rng = np.random.default_rng(seed)
    pick = rng.choice(wav.shape[0], size=min(n_draws, wav.shape[0]), replace=False)
    return wav[pick], msk[pick], (sky[pick] if sample_sky else None)


def reconstruct_fd(template, wav, msk, *, sky=None, fixed_sky=None, chunk=200):
    """Per-draw complex network waveform ``h(f)`` of shape ``(D, n_ifo, n_freq)``.

    ``sky`` (``(D, 4)``) uses the sky-sampled projection; otherwise pass
    ``fixed_sky=(ra, dec, psi, ellipticity)``.
    """
    to_np = template.to_numpy
    D = wav.shape[0]
    hrec = None
    for s in range(0, D, chunk):
        e = min(s + chunk, D)
        flat, grp = [], []
        for d in range(s, e):
            active = wav[d][msk[d]]
            if active.shape[0]:
                flat.append(active)
                grp += [d - s] * active.shape[0]
        n_g = e - s
        flat = np.vstack(flat) if flat else np.zeros((0, 5))
        grp = np.asarray(grp, dtype=int)
        if sky is not None:
            sig = template.project_grouped_sky(flat, grp, sky[s:e], n_g)
        else:
            ra, dec, psi, ell = fixed_sky
            sig = template.project_grouped(flat, grp, n_g, ra, dec, psi, ell)
        sig = to_np(sig)
        if hrec is None:
            hrec = np.zeros((D, sig.shape[1], sig.shape[2]), dtype=complex)
        hrec[s:e] = sig
    return hrec


def whiten_to_td(h_fd, full_frequency_array, mask, *, asd=None):
    """Inverse-FFT band-limited ``h(f)`` to real strain ``h(t)``.

    Everything (duration, ``df``, ``dt``, the rfft length ``N``) is derived from
    the *actual* one-sided frequency grid -- nothing is fixed to a 4 s / 2048 Hz
    segment. ``full_frequency_array`` is the template's unmasked grid
    ``[0 .. f_Nyquist]`` (use ``template.full_frequencies``); ``mask`` selects the
    analysis band (``template.band_mask``); ``h_fd`` is ``(..., n_band)`` on that band.

    For a one-sided grid of length ``M`` the time series has ``N = 2 (M - 1)``
    samples with ``dt = 1 / (N df)`` and ``df = full[1] - full[0]``. Pass ``asd``
    ``(..., n_band)`` to whiten; omit it for the coloured waveform. Returns
    ``(t, h_t)`` with ``h_t`` of shape ``(..., N)``.
    """
    h_fd = np.asarray(h_fd)
    full = np.asarray(full_frequency_array, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n_full = full.size
    N = 2 * (n_full - 1)                          # rfft length -> # time samples
    df = float(full[1] - full[0])
    band = h_fd if asd is None else h_fd / asd
    spec = np.zeros(h_fd.shape[:-1] + (n_full,), dtype=complex)
    spec[..., mask] = band
    h_t = np.fft.irfft(spec, n=N, axis=-1)
    dt = 1.0 / (N * df)                           # = duration / N
    return np.arange(N) * dt, h_t


def fd_credible(hrec, ifo_idx, credibility=0.9):
    """Amplitude median + credible band for one detector (reuses FD helper)."""
    return compute_credible_region_fd(np.abs(hrec[:, ifo_idx, :]), credibility)


def td_credible(h_t, ifo_idx, credibility=0.9):
    """Strain median + percentile band for one detector (reuses TD helper)."""
    return compute_credible_region(h_t[:, ifo_idx, :], credibility, method="percentile")


# ======================================================================
# Plotting (per detector, shared HyperWave palette)
# ======================================================================

def plot_wavelet_fd(freqs, recon, ifo, *, signal_fd=None, asd=None, df=None,
                    title=None, xlabel=r"$f\ \rm{[Hz]}$", ylabel=_FD_YLABEL,
                    xlim=None, ylim=None, outpath=None, show=False,
                    black_background=False, panel_scale=0.9, preset="prd",
                    grid=True, legend_fontsize=14, dpi=300, also_pdf=True):
    """Frequency-domain wavelet reconstruction for one detector.

    ``recon`` is a dict with ``median``/``lower``/``upper`` amplitude arrays
    (e.g. from :func:`fd_credible`). ``signal_fd`` is the injected ``h(f)``
    (complex); ``asd`` overplots the sensitivity ``sqrt(S(f))``.
    """
    pal = ifo_palette(ifo)
    rc_col, inj_col = pal["reconstructed"], pal["injected"]
    matplotlib.rcParams.update(rcparams1 if black_background else rcparams2)
    apply_style(preset=preset, black_background=black_background,
                transparent=black_background, panel_scale=panel_scale)

    scale = np.sqrt(2.0 * df) if df is not None else 1.0
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(freqs, np.asarray(recon["lower"]) * scale,
                    np.asarray(recon["upper"]) * scale,
                    color=rc_col, alpha=0.3, edgecolor="none", zorder=3)
    ax.plot(freqs, np.asarray(recon["median"]) * scale, color=rc_col, lw=1.1, zorder=4)
    if signal_fd is not None:
        ax.plot(freqs, np.abs(signal_fd) * scale, color=inj_col, lw=0.9, zorder=5)
    if asd is not None:
        ax.plot(freqs, np.asarray(asd), "--", color="k", lw=0.6, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_xlim(xlim if xlim is not None else (float(freqs[0]), float(freqs[-1])))
    if ylim is None and signal_fd is not None:
        peak = float(np.nanmax(np.abs(signal_fd) * scale))
        if peak > 0:
            ylim = (peak * 1e-3, peak * 3.0)
    if ylim is not None:
        ax.set_ylim(ylim)

    # legend: patch+line for median&CI, then injected, then ASD
    patch = Patch(facecolor=rc_col, alpha=0.3, edgecolor="none")
    line = Line2D([0], [0], color=rc_col, lw=1.1)
    handles = [(patch, line)]
    labels = [f"{ifo}: median & 90% CI"]
    if signal_fd is not None:
        handles.append(Line2D([0], [0], color=inj_col, lw=0.9))
        labels.append("injected")
    if asd is not None:
        handles.append(Line2D([0], [0], color="k", lw=0.6, ls="--"))
        labels.append(r"$\sqrt{S(f)}$")
    ax.legend(handles, labels, handler_map={tuple: HandlerPatchLine()},
              handlelength=2.2, handletextpad=0.35, fontsize=legend_fontsize,
              frameon=not black_background)
    ax.grid(grid, which="both", alpha=0.25 if grid else 0.0)
    plt.tight_layout()
    _save(fig, outpath, black_background, dpi, also_pdf)
    return _finish(fig, ax, show)


def plot_wavelet_td(times, recon, ifo, *, signal_td=None, t_ref=0.0,
                    title=None, xlabel=r"$t-t_{\rm ref}$ [s]",
                    ylabel="whitened strain (arb. units)", xlim=None, zoom_window=None,
                    outpath=None, show=False, black_background=False,
                    panel_scale=0.88, preset="prd", legend_fontsize=14,
                    dpi=300, also_pdf=True):
    """Whitened time-domain reconstruction for one detector.

    ``recon`` is a dict with ``median``/``lower``/``upper`` strain arrays (e.g.
    from :func:`td_credible`). The view auto-centres on the injected-signal peak
    unless ``xlim`` is given; ``zoom_window`` (seconds) sets the half-width.
    """
    pal = ifo_palette(ifo)
    rc_col, inj_col = pal["reconstructed"], pal["injected"]
    matplotlib.rcParams.update(rcparams1 if black_background else rcparams2)
    apply_style(preset=preset, black_background=black_background,
                transparent=black_background, panel_scale=panel_scale)

    t = np.asarray(times) - t_ref
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(t, np.asarray(recon["lower"]), np.asarray(recon["upper"]),
                    color=rc_col, alpha=0.3, edgecolor="none", zorder=3,
                    label=f"{ifo}: 90% CI")
    ax.plot(t, np.asarray(recon["median"]), color=rc_col, lw=0.9, zorder=4,
            label=f"{ifo}: median")
    if signal_td is not None:
        ax.plot(t, np.asarray(signal_td), color=inj_col, lw=0.7, zorder=5,
                label="injected")
        if xlim is None:
            tc = float(t[int(np.argmax(np.abs(signal_td)))])
            w = zoom_window if zoom_window is not None else 0.3
            xlim = (tc - w, tc + 0.5 * w)

    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(loc="best", frameon=not black_background, fontsize=legend_fontsize)
    ax.grid(False)
    plt.tight_layout()
    _save(fig, outpath, black_background, dpi, also_pdf)
    return _finish(fig, ax, show)


# ======================================================================
# High-level convenience: full network from a sampler
# ======================================================================

def plot_network_reconstruction(template, true_signal, psd, *,
                                detectors, sampler=None, coords=None, inds=None,
                                sample_sky=True, fixed_sky=None, time_domain=True,
                                n_draws=1000, discard_frac=0.3, seed=0,
                                outdir=".", prefix="wavelet_recon", show=False,
                                black_background=False, **plot_kw):
    """Build wavelet reconstructions and render FD (+ whitened TD) per detector.

    Returns ``{f"{ifo}_fd": (fig, ax), f"{ifo}_td": (fig, ax), ...}``. The band
    ``freqs``, ``df`` and the full time grid are taken from ``template`` (the
    actual data resolution) -- nothing is assumed about the segment length.
    Provide a sampler (or ``coords``/``inds``), the injected ``true_signal``
    ``(n_ifo, n_band)``, the band PSD, and the detector names. ``time_domain``
    toggles the whitened TD panels.
    """
    freqs = np.asarray(template.full_frequencies)[np.asarray(template.band_mask, bool)]
    df = float(template.df)
    wav, msk, sky = select_wavelet_draws(
        sampler, coords=coords, inds=inds, sample_sky=sample_sky,
        discard_frac=discard_frac, n_draws=n_draws, seed=seed)
    hrec = reconstruct_fd(template, wav, msk, sky=sky, fixed_sky=fixed_sky)
    asd = np.sqrt(np.asarray(psd))
    outdir = Path(outdir)
    figs = {}

    for j, ifo in enumerate(detectors):
        figs[f"{ifo}_fd"] = plot_wavelet_fd(
            freqs, fd_credible(hrec, j), ifo, signal_fd=true_signal[j],
            asd=asd[j], df=df, outpath=outdir / f"{prefix}_{ifo}_fd.png",
            show=show, black_background=black_background, **plot_kw)

    if time_domain:
        _, h_t = whiten_to_td(hrec, template.full_frequencies, template.band_mask, asd=asd)
        t, inj_t = whiten_to_td(np.asarray(true_signal), template.full_frequencies,
                                template.band_mask, asd=asd)
        for j, ifo in enumerate(detectors):
            figs[f"{ifo}_td"] = plot_wavelet_td(
                t, td_credible(h_t, j), ifo, signal_td=inj_t[j],
                outpath=outdir / f"{prefix}_{ifo}_td.png",
                show=show, black_background=black_background, **plot_kw)
    return figs


# ----------------------------------------------------------------------
# small shared save/show helpers
# ----------------------------------------------------------------------

def _save(fig, outpath, black_background, dpi, also_pdf):
    if outpath is None:
        return
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=black_background)
    if also_pdf:
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight",
                    transparent=black_background)


def _finish(fig, ax, show):
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


__all__ = [
    "IFO_COLORS",
    "ifo_palette",
    "select_wavelet_draws",
    "reconstruct_fd",
    "whiten_to_td",
    "fd_credible",
    "td_credible",
    "plot_wavelet_fd",
    "plot_wavelet_td",
    "plot_network_reconstruction",
]
