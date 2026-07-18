"""
Frequency-Domain (FD) Reconstruction plotting utilities for HyperWave.

- Correct Nature-style plotting
- Proper legend with patch + line (median + CI)
- Multi-case comparison
- Grid on/off option
- HyperWave style integration
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# HyperWave styles
from .corners import default_colors, rcparams1, rcparams2
from .style import SIGNAL_COLOR, apply_style, style_axes


class HandlerPatchLine(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        
        patch, line = orig_handle

        # --- SHADE RECTANGLE ---
        p = Rectangle(
            (xdescent, ydescent + height*0.2),
            width,
            height*0.6,
            facecolor=patch.get_facecolor(),
            edgecolor="none",
            alpha=patch.get_alpha(),
            transform=trans,
        )

        # --- MEDIAN LINE ---
        l = Line2D(
            [xdescent, xdescent + width],
            [ydescent + height/2, ydescent + height/2],
            color=line.get_color(),
            lw=line.get_linewidth(),
            transform=trans,
        )

        return [p, l]



# ============================================================
# Helper: waveform generation (frequency domain)
# ============================================================

def _perform_injection_fd(template_obj, params, detector_idx=0):
    """Generate frequency-domain signal for a given detector and parameter set."""
    ifo_name = template_obj.ifos[detector_idx].name
    return np.abs(template_obj.make_injections_to_ifo(params)[ifo_name])


# ============================================================
# Reconstruction from posterior samples
# ============================================================

def reconstruct_fd_waveforms(
    template_obj,
    samples,
    detector_idx=0,
    num_samples=20000,
    n_jobs=32,
    random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    idxs = np.random.randint(0, samples.shape[0], num_samples)

    reconstructions = Parallel(n_jobs=n_jobs)(
        delayed(_perform_injection_fd)(template_obj, samples[idx], detector_idx)
        for idx in idxs
    )

    return np.array(reconstructions)


# ============================================================
# Credible intervals
# ============================================================

def compute_credible_region_fd(waveforms, credibility=0.9):
    delta = (1 + credibility) / 2
    upper_percentile = delta * 100
    lower_percentile = (1 - delta) * 100

    median = np.percentile(waveforms, 50, axis=0)
    upper = np.percentile(waveforms, upper_percentile, axis=0)
    lower = np.percentile(waveforms, lower_percentile, axis=0)

    return {
        "median": median,
        "upper": upper,
        "lower": lower,
        "std": np.std(waveforms, axis=0, ddof=1),
    }


# ============================================================
# Plot: SINGLE FD reconstruction
# ============================================================

def plot_fd_reconstruction(
    frequencies,
    reconstruction_dict,
    signal_fd=None,
    case="hyperbolic",
    label=None,
    title="",
    xlabel="Frequency [Hz]",
    ylabel=r'$|\tilde{h}(f)|~\sqrt{2\Delta f}$ & $\sqrt{S(f)}$ $\quad$ $\left[\sqrt{\rm{Hz^{-1}}}\right]$',
    xlim=(20, 400),
    ylim=(1e-25, 2e-23),
    outpath=None,
    show=True,
    black_background=False,
    legend_loc="best",
    legend_fontsize=18,
    panel_scale=0.9,
    preset="prd",
    grid=True,
):
    # --- Colors ---
    reconstruction_color = default_colors.get(case, "#dc267f")
    signal_color = SIGNAL_COLOR

    # --- Base style ---
    if black_background:
        matplotlib.rcParams.update(rcparams1)
    else:
        matplotlib.rcParams.update(rcparams2)

    apply_style(
        preset=preset,
        black_background=black_background,
        transparent=black_background,
        panel_scale=panel_scale,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plot reconstruction ---
    ax.loglog(frequencies, reconstruction_dict["median"],
              color=reconstruction_color, lw=1.1, label="_nolegend_")

    ax.fill_between(
        frequencies,
        reconstruction_dict["lower"],
        reconstruction_dict["upper"],
        color=reconstruction_color,
        alpha=0.22,
        label="_nolegend_",
    )

    # --- Plot injected signal ---
    if signal_fd is not None:
        ax.loglog(frequencies, signal_fd,
                  color=signal_color, lw=1.0, label="_nolegend_")

    # --- Axes formatting ---
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # --- LEGEND (patch + line, correct way) ---
    patch = Patch(facecolor=reconstruction_color, alpha=0.22, edgecolor=reconstruction_color)
    line = Line2D([0], [0], color=reconstruction_color, lw=1.1)

    legend_handles = [(patch, line)]
    legend_labels = [label if label is not None else f"{case.title()}: median & 90% CI"]

    if signal_fd is not None:
        inj_line = Line2D([0], [0], color=signal_color, lw=1.0)
        legend_handles.append(inj_line)
        legend_labels.append("Injected Signal")
    
        
    legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerPatchLine()},
        handlelength=2.2,
        handletextpad=0.35,
        fontsize=legend_fontsize,
    )

    # --- Grid ---
    ax.grid(grid, which="both", alpha=0.25 if grid else 0.0)

    plt.tight_layout()

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
# Plot: MULTI-CASE FD reconstruction
# ============================================================

def plot_multi_case_fd_reconstruction(
    frequencies,
    reconstruction_dicts,
    case_names,
    signal_fd=None,
    title="",
    xlabel="Frequency [Hz]",
    ylabel=r'$|\tilde{h}(f)|~\sqrt{2\Delta f}$ & $\sqrt{S(f)}$ $\quad$ $\left[\sqrt{\rm{Hz^{-1}}}\right]$',
    xlim=(20, 400),
    ylim=(1e-25, 2e-23),
    outpath=None,
    show=True,
    black_background=False,
    legend_fontsize=18,
    panel_scale=0.9,
    preset="prd",
    grid=True,
):
    case_colors = [default_colors[c.lower()] for c in case_names]

    signal_color = SIGNAL_COLOR

    if black_background:
        matplotlib.rcParams.update(rcparams1)
    else:
        matplotlib.rcParams.update(rcparams2)

    apply_style(
        preset=preset,
        black_background=black_background,
        transparent=black_background,
        panel_scale=panel_scale,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plot each case ---
    for recon_dict, case_name, color in zip(reconstruction_dicts, case_names, case_colors):
        ax.loglog(frequencies, recon_dict["median"], color=color, lw=1.1, label="_nolegend_")
        ax.fill_between(frequencies, recon_dict["lower"], recon_dict["upper"],
                        color=color, alpha=0.22, label="_nolegend_")

    # --- Injected signal ---
    if signal_fd is not None:
        ax.loglog(frequencies, signal_fd, color=signal_color, lw=1.0, label="_nolegend_")

    # --- Axes ---
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # --- custom legend handles ---
    legend_handles = []
    legend_labels = []

    for case_name, color in zip(case_names, case_colors):
        patch = Patch(facecolor=color, alpha=0.22, edgecolor="none", linewidth=0.5)
        line = Line2D([0], [0], color=color, lw=1.3)

        legend_handles.append((patch, line))
        legend_labels.append(f"{case_name}: median & 90% CI")

    if signal_fd is not None:
        line_signal = Line2D([0], [0], color=signal_color, lw=1.2)
        legend_handles.append(line_signal)
        legend_labels.append("Injected Signal")
        
    legend = ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerPatchLine()},
        handlelength=2.2,
        handletextpad=0.35,
        fontsize=legend_fontsize,
    )

    # --- Grid ---
    ax.grid(grid, which="both", alpha=0.25 if grid else 0.0)

    plt.tight_layout()

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
