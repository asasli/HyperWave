"""
Time-Domain (TD) Reconstruction plotting utilities.

Provides functions for reconstructing and visualizing time-domain waveforms 
from MCMC posterior samples, with confidence bands and comparison to injected signals.

Settings are inherited from corners.py for consistency, but overridden by HyperWave style.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import t

# Import rcParams and colors from corners
from .corners import default_colors, rcparams1, rcparams2

# Import HyperWave plotting style
from .style import DATA_COLOR, SIGNAL_COLOR, apply_style, style_axes

# ============================================================
# Helper: waveform generation
# ============================================================

def _perform_injection_td(template_obj, params, detector_idx=0):
    """Generate time-domain signal for a given detector and parameter set."""
    return template_obj.signal_td(
        template_obj.make_injections_to_ifo_without_mask(detector_idx, params)
    )


# ============================================================
# Reconstruction from posterior samples
# ============================================================

def reconstruct_td_waveforms(
    template_obj,
    samples,
    detector_idx=0,
    num_samples=20000,
    n_jobs=32,
    random_seed=None,
):
    """Reconstruct time-domain waveforms from posterior samples."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    idxs = np.random.randint(0, samples.shape[0], num_samples)
    
    reconstructions = Parallel(n_jobs=n_jobs)(
        delayed(_perform_injection_td)(template_obj, samples[idx], detector_idx)
        for idx in idxs
    )
    
    return np.array(reconstructions)


# ============================================================
# Credible intervals
# ============================================================

def compute_credible_region(
    waveforms,
    credibility=0.9,
    method="std",
):
    """Compute median and credible region from ensemble of waveforms."""
    delta = (1 + credibility) / 2
    upper_percentile = delta * 100
    lower_percentile = (1 - delta) * 100
    
    median = np.percentile(waveforms, 50, axis=0)
    upper = np.percentile(waveforms, upper_percentile, axis=0)
    lower = np.percentile(waveforms, lower_percentile, axis=0)
    
    if method == "std":
        std_dev = np.std(waveforms, axis=0, ddof=1)
        k = t.ppf((1 + credibility) / 2, len(waveforms) - 1)
        upper_std = median + k * std_dev
        lower_std = median - k * std_dev
    else:
        upper_std = upper
        lower_std = lower
    
    return {
        "median": median,
        "upper": upper,
        "lower": lower,
        "upper_std": upper_std,
        "lower_std": lower_std,
        "std": np.std(waveforms, axis=0, ddof=1),
    }


# ============================================================
# Plot: single reconstruction
# ============================================================

def plot_td_reconstruction(
    times,
    reconstruction_dict,
    signal_td=None,
    data_td=None,
    case="hyperbolic",
    label=None,
    title="",
    xlabel=r"$t-t_{ref}$ [s]",
    ylabel="Strain",
    xlim=None,
    outpath=None,
    show=True,
    black_background=False,
    legend_loc="best",
    legend_fontsize=18,
    panel_scale=0.88,  # IMPORTANT for LaTeX grids
    preset="prd",
):
    """Plot time-domain reconstruction with confidence bands."""
    
    # --- Apply corner style first ---
    if black_background:
        matplotlib.rcParams.update(rcparams1)
        data_color = "lightgray"
        signal_color = SIGNAL_COLOR
        reconstruction_color = default_colors.get(case)
    else:
        matplotlib.rcParams.update(rcparams2)
        data_color = DATA_COLOR
        signal_color = SIGNAL_COLOR
        reconstruction_color = default_colors.get(case)

    # --- Apply HyperWave style (overrides fonts properly) ---
    apply_style(
        preset=preset,
        black_background=black_background,
        transparent=black_background,
        panel_scale=panel_scale,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Plot raw data ---
    if data_td is not None:
        ax.plot(
            times,
            data_td,
            color=data_color,
            lw=0.5,
            alpha=0.3,
            label="Data",
        )

    # --- Plot injected signal ---
    if signal_td is not None:
        ax.plot(
            times,
            signal_td,
            color=signal_color,
            lw=0.7,
            label="Injected Signal",
        )

    # --- Plot median reconstruction ---
    ax.plot(
        times,
        reconstruction_dict["median"],
        color=reconstruction_color,
        lw=0.9,
        label=label if label is not None else f"Reconstruction ({case.title()})",
    )

    # --- Confidence region ---
    ax.fill_between(
        times,
        reconstruction_dict["lower_std"],
        reconstruction_dict["upper_std"],
        alpha=0.2,
        color=reconstruction_color,
    )

    # --- Axes formatting (HyperWave override) ---
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.legend(
        loc=legend_loc,
        frameon=True if not black_background else False,
        fontsize=legend_fontsize,
    )

    ax.grid(False)

    # --- Save ---
    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=black_background)
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", transparent=black_background)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ============================================================
# Plot: multiple reconstructions (comparison)
# ============================================================

def plot_multi_case_reconstruction(
    times,
    reconstruction_dicts,
    case_names=None,
    case_colors=None,
    signal_td=None,
    data_td=None,
    title="",
    xlabel=r"$t-t_{ref}$ [s]",
    ylabel="Strain",
    xlim=None,
    outpath=None,
    show=False,
    black_background=False,
    legend_fontsize=18,
    panel_scale=0.88,
    preset="nature",
):
    """Plot multiple reconstruction cases on same axes for comparison."""
    
    if case_names is None:
        case_names = [f"Case {i}" for i in range(len(reconstruction_dicts))]
    
    if case_colors is None:
        case_colors = [
            default_colors.get("hyperbolic"),
            default_colors.get("whittle"),
        ]

    # --- Apply corner style ---
    if black_background:
        matplotlib.rcParams.update(rcparams1)
        data_color = "lightgray"
        signal_color = SIGNAL_COLOR
    else:
        matplotlib.rcParams.update(rcparams2)
        data_color = DATA_COLOR
        signal_color = SIGNAL_COLOR

    # --- Apply HyperWave style ---
    apply_style(
        preset=preset,
        black_background=black_background,
        transparent=black_background,
        panel_scale=panel_scale,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plot raw data ---
    if data_td is not None:
        ax.plot(times, data_td, color=data_color, lw=0.5, alpha=0.3)

    # --- Plot injected signal ---
    if signal_td is not None:
        ax.plot(
            times,
            signal_td,
            color=signal_color,
            lw=0.7,
            label="Injected Signal",
        )

    # --- Plot each reconstruction ---
    for recon_dict, case_name, color in zip(reconstruction_dicts, case_names, case_colors):
        ax.plot(
            times,
            recon_dict["median"],
            color=color,
            lw=0.9,
            label=f"Reconstruction ({case_name})",
        )
        ax.fill_between(
            times,
            recon_dict["lower_std"],
            recon_dict["upper_std"],
            alpha=0.15,
            color=color,
        )

    # --- Axes formatting ---
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.legend(
        loc="best",
        frameon=True if not black_background else False,
        fontsize=legend_fontsize,
    )

    ax.grid(False)

    # --- Save ---
    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=black_background)
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", transparent=black_background)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
