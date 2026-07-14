"""Automatic data conditioning for real detector strain.

Real interferometer data carry instrumental spectral lines (mains harmonics,
calibration lines, suspension/violin modes) that a flexible signal model will
happily fit as "signal", inflating the recovered SNR. This module makes line
identification and whiteness validation a reusable, data-driven part of the
pipeline instead of a per-analysis hand edit:

  * ``find_spectral_lines`` flags narrow outliers above the *local* continuum
    of the whitened spectrum (running median + MAD), so it catches lines on top
    of both flat noise and broadband signal, with no hard-coded frequency list.
  * ``whiteness_report`` measures the whitened power per band; the noise-only
    region should sit at ~2 (2 dof/bin). It is the self-check that warns when the
    PSD or window normalization is wrong (e.g. a stray sqrt(2)).
  * ``condition_band`` combines an ``[fmin, fmax]`` cut, the auto line notches,
    and an optional protected band that is never notched, returning the analysis
    mask plus a human-readable report.

Whitened power convention: ``w_k = |d_k|^2 / (T * S_k / 4)`` has expectation 2
per bin for stationary Gaussian noise (real+imag, unit variance each).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


def whitened_power(data_fd, psd, duration):
    """``|d|^2 / (T S / 4)`` per bin, expectation ~2 for noise. Shapes broadcast;
    a leading detector axis is averaged over."""
    w = np.abs(np.asarray(data_fd)) ** 2 / (float(duration) * np.asarray(psd) / 4.0)
    return w.mean(0) if w.ndim > 1 else w


def _running_stat(x, window, fn):
    n = x.size
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out[i] = fn(x[lo:hi])
    return out


def find_spectral_lines(data_fd, psd, freqs, duration, *, window=12,
                        n_sigma=6.0, max_width=4, protect=None):
    """Boolean mask of line bins: narrow spikes above the local continuum.

    A bin is a line if its whitened power exceeds ``median + n_sigma * 1.4826 MAD``
    over a ``+/-window`` neighborhood (robust to the line itself) AND belongs to
    a contiguous run no wider than ``max_width`` bins (lines are narrow; broadband
    signal excursions are wider and are left alone). ``protect`` = list of
    ``(f_lo, f_hi)`` never flagged.
    """
    freqs = np.asarray(freqs, float)
    w = whitened_power(data_fd, psd, duration)
    med = _running_stat(w, window, np.median)
    mad = _running_stat(w, window, lambda a: np.median(np.abs(a - np.median(a))))
    thresh = med + n_sigma * 1.4826 * np.maximum(mad, 1e-12)
    hot = w > thresh
    # keep only narrow runs (drop broadband excursions = real signal)
    lines = np.zeros_like(hot)
    i = 0
    while i < hot.size:
        if hot[i]:
            j = i
            while j < hot.size and hot[j]:
                j += 1
            if (j - i) <= max_width:
                lines[i:j] = True
            i = j
        else:
            i += 1
    if protect:
        for lo, hi in protect:
            lines &= ~((freqs >= lo) & (freqs <= hi))
    return lines


def whiteness_report(data_fd, psd, freqs, duration, bands=None):
    """Mean whitened power in each ``(f_lo, f_hi)`` band (default a few probes).
    A noise-only band far from any signal should read ~2."""
    w = whitened_power(data_fd, psd, duration)
    freqs = np.asarray(freqs)
    if bands is None:
        fmax = freqs.max()
        bands = [(20, 40), (40, 120), (120, 300), (0.6 * fmax, 0.95 * fmax)]
    return {b: float(w[(freqs >= b[0]) & (freqs < b[1])].mean())
            for b in bands if np.any((freqs >= b[0]) & (freqs < b[1]))}


@dataclass
class ConditioningReport:
    band: np.ndarray
    n_bins: int
    n_notched: int
    line_freqs: np.ndarray
    whiteness: dict
    noise_floor: float                      # whitened power in the highest probe band
    warnings: list = field(default_factory=list)

    def __str__(self):
        lines = "".join(f" {f:.1f}" for f in self.line_freqs[:20])
        s = (f"[conditioning] {self.n_bins} bins, notched {self.n_notched} line bins"
             f" ({int(round(self.line_freqs.size))} freqs):{lines}"
             f"{' ...' if self.line_freqs.size > 20 else ''}\n"
             f"  whiteness (expect ~2 in noise): "
             + "  ".join(f"{lo:.0f}-{hi:.0f}Hz={v:.2f}" for (lo, hi), v in self.whiteness.items())
             + f"\n  noise floor = {self.noise_floor:.2f}")
        for w in self.warnings:
            s += f"\n  !! {w}"
        return s


def auto_fmin(data_fd, psd, freqs, duration, fmin, fmax, *,
              floor_target=4.0, smooth=40):
    """Raise ``fmin`` past a broad low-frequency wall (seismic + dense
    suspension/violin clusters that a narrow-line finder cannot notch). Returns
    the lowest frequency at or above ``fmin`` where the *smoothed* whitened floor
    first drops below ``floor_target`` and stays there for ``smooth`` bins."""
    freqs = np.asarray(freqs, float)
    w = whitened_power(data_fd, psd, duration)
    # running MEAN (not median): a dense low-frequency line cluster raises the
    # mean of its neighbourhood even though each line is a minority of bins, so
    # the mean-based floor stays high until the band clears the cluster.
    sm = _running_stat(w, smooth, np.mean)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    for k in idx:
        if sm[k] < floor_target and np.all(sm[k:min(k + smooth, sm.size)] < 2 * floor_target):
            return float(freqs[k])
    return float(fmin)


def condition_band(data_fd, psd, freqs, duration, fmin, fmax, *,
                   protect=None, notch_extra=None, auto_lowcut=True,
                   floor_target=4.0, soft_notch_factor=1e20, **line_kwargs):
    """Condition real strain for a *contiguous*-band analysis.

    Returns ``(band, psd_conditioned, report)``:
      * ``band`` = a contiguous ``[fmin_eff, fmax]`` mask (``fmin_eff`` raised
        past a broad low-frequency wall when ``auto_lowcut``), so it matches a
        ``WaveletTemplate``'s ``[minimum_frequency, maximum_frequency]`` mask.
      * ``psd_conditioned`` = ``psd`` with detected line bins (and any
        ``notch_extra``) inflated by ``soft_notch_factor`` -- a *soft* notch that
        drives their likelihood/SNR weight to zero without removing bins, so no
        template/data shape mismatch. Slice it by ``band`` for the analysis.
      * ``report`` warns if the noise floor departs from ~2 (mis-normalized
        PSD/window; a stray sqrt(2) lands it at 1 or 4) or if lines remain.
    Set the template/prior ``minimum_frequency`` to ``report.fmin``.
    """
    freqs = np.asarray(freqs, float)
    psd = np.asarray(psd, float)
    if auto_lowcut:
        fmin = max(fmin, auto_fmin(data_fd, psd, freqs, duration, fmin, fmax,
                                   floor_target=floor_target))
    band = (freqs >= fmin) & (freqs <= fmax)
    lines = find_spectral_lines(data_fd, psd, freqs, duration,
                                protect=protect, **line_kwargs) & band
    for fc, hw in (notch_extra or []):
        lines |= (freqs > fc - hw) & (freqs < fc + hw) & band

    psd_cond = psd.copy()
    psd_cond[..., lines] = psd[..., lines] * float(soft_notch_factor)

    whit = whiteness_report(data_fd, psd, freqs, duration)
    floor = list(whit.values())[-1] if whit else float("nan")
    warns = []
    if np.isfinite(floor) and not (1.5 < floor < 2.6):
        warns.append(f"noise floor {floor:.2f} != ~2: check PSD / window "
                     f"normalization (a stray sqrt(2) shifts it to 1 or 4)")
    kept = band & ~lines
    resid = whitened_power(data_fd, psd, duration)[kept]
    if resid.size and np.median(resid) > 4.0:
        warns.append(f"median in-band whitened power {np.median(resid):.1f} high; "
                     f"lines may remain -> raise n_sigma or floor_target")
    rep = ConditioningReport(band=band, n_bins=int(band.sum()),
                             n_notched=int(lines.sum()),
                             line_freqs=freqs[lines], whiteness=whit,
                             noise_floor=floor, warnings=warns)
    rep.fmin = float(freqs[band][0]) if band.any() else float(fmin)
    return band, psd_cond, rep
