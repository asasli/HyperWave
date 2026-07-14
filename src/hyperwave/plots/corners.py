from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [4.5,3.5]

rcparams1 = {
    'axes.labelsize': 14,
    'font.size': 16,
    'text.latex.preamble': (r'\usepackage{revtex4-2}'),
    'legend.fontsize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'figure.figsize': fig_size,
    'figure.dpi': 150,
    'figure.autolayout': True,
    'axes.linewidth': 0.5,
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'axes.facecolor': 'none',  # Make the area inside the plot transparent
    'figure.facecolor': 'none',  # Keep the figure background transparent
}

rcparams2 = {
    'axes.labelsize': 14,
    'font.size': 16,
    'text.latex.preamble': (r'\usepackage{revtex4-2}'),
    'legend.fontsize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.5,
    'xtick.color': 'k',
    'xtick.labelsize': 10,
    'ytick.color': 'k',
    'ytick.labelsize': 10,
    'text.usetex': False,
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'text.color': 'k',
    'figure.figsize': fig_size,     
    'figure.dpi': 150,
    'figure.autolayout': True,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
}
default_colors = {
        "hyperbolic": "#dc267f", #"#FF8FFF",  # light orchid
        "gaussian": "#009e73",    # forest green
        "whittle": "#6495ED",     # cornflower blue
        "wavelets": "#6D75CC",    # indigo (Morlet-Gabor wavelet reconstruction)
    }
# --- Corner/ChainConsumer Plotting Utility ---
def plot_posterior(samples, param_names, case="hyperbolic", name=None, color=None, package="corner", show=True, save_dir=None, TAG=None, truths=None, black_background=False):
    """
    Plot a posterior using either corner or chainconsumer.

    Args:
        samples (np.ndarray): The samples to plot (shape: [n_samples, n_params]).
        param_names (list): List of parameter names for the axes.
        case (str): One of 'hyperbolic', 'gaussian', 'whittle'. Determines default color.
        name (str): Optional name for the plot.
        color (str): Optional color for the plot. If not provided, uses default for case.
        package (str): 'corner' or 'chainconsumer'.
        show (bool): Whether to show the plot.
        save_path (str): If provided, saves the plot to this path.
        truths (list): Optional list of truth values for each parameter, used for plotting true values in corner.
    """
    if color is None:
        color = default_colors.get(case, default_colors[str(case)])
    if black_background:
        matplotlib.rcParams.update(rcparams1)
    else:
        matplotlib.rcParams.update(rcparams2)

    if package == "corner":
        import corner
        fig = corner.corner(
            samples,
            labels=param_names,
            color=color,
            truths=truths,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            smooth=True,
            smooth1d=True,
            bins=50,
            plot_datapoints=False,
            label_kwargs={"fontsize": 14},
            spacing=0.02,
        )
        if name:
            fig.suptitle(name, fontsize=16)
        if save_dir:
            plt.savefig(save_dir + f"/{TAG}_posteriors.pdf" if TAG else save_dir)
        if show:
            plt.show()
        else:
            plt.close(fig)
    elif package == "chainconsumer":
        from chainconsumer import ChainConsumer
        c = ChainConsumer()
        c.add_chain(samples, parameters=param_names, name=name)
        c.configure(
            colors=color,
            shade=True,
            shade_alpha=0.25,
            bar_shade=True,
            linewidths=1.4,
            max_ticks=3,
            diagonal_tick_labels=True,
            tick_font_size=10,
            label_font_size=16,
            kde=True,
            summary=False,
        )
        ndim = len(param_names)
        base_size = 1.  # reduce whitespace while keeping content readable
        if truths is not None:
            fig = c.plotter.plot(figsize=(base_size * ndim, base_size * ndim)
, truth=truths, legend=True,
            )
        else:
            fig = c.plotter.plot(figsize=(base_size * ndim, base_size * ndim)
, legend=True,)
        if save_dir:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        raise ValueError(f"Unknown package: {package}")

# --- Noise-only helper ---
def plot_noise_only(samples, param_names, case="hyperbolic", package="corner", show=True, save_dir=None, TAG=None, black_background=False):
    """ Plot a noise-only posterior using either corner or chainconsumer.
    Args:
        samples (np.ndarray): The samples to plot (shape: [n_samples, n_params]).
        param_names (list): List of parameter names for the axes.
        case (str): One of 'hyperbolic', 'gaussian', 'whittle'. Determines default color.
        package (str): 'corner' or 'chainconsumer'.
        save_dir (str): If provided, saves the plot to this directory.
        TAG (str): Optional tag to append to the save path.
    """
    # Use the default color for the case if not specified by the user
    color = default_colors.get(str(case), list(default_colors.values())[0])
    plot_posterior(samples, param_names, case=case, color=color, package=package, show=show, save_dir=save_dir, TAG = TAG+f"_{case}_noise_only.pdf" if save_dir else None, black_background=black_background)


def plot_multi_posteriors(samples_list, param_names, labels, cases=None, colors=None, package="corner", show=True, save_dir=None, truths=None, black_background=False):
    """
    Plot multiple posteriors on the same plot using either corner or chainconsumer.

    Args:
        samples_list (list of np.ndarray): List of samples arrays to plot.
        param_names (list): List of parameter names for the axes.
        labels (list): List of labels for each posterior/case.
        cases (list): List of case types ("hyperbolic", "gaussian", "whittle") for each posterior. If None, defaults to "hyperbolic" for all.
        colors (list): List of colors for each posterior. If None, uses default for each case.
        package (str): 'corner' or 'chainconsumer'.
        show (bool): Whether to show the plot.
        save_path (str): If provided, saves the plot to this path.
        truths (list): Optional list of truth values for each parameter, used for plotting true values in corner
    Usage:
        # samples_list: list of np.ndarray, one for each posterior
        # param_names: list of parameter names
        # labels: list of strings, one for each posterior
        # cases: list of case types (optional, e.g. ["hyperbolic", "whittle"])
        # colors: list of colors (optional)

        plot_multi_posteriors(
            samples_list=[samples1, samples2],
            param_names=param_names,
            labels=["Hyperbolic", "Whittle"],
            cases=["hyperbolic", "whittle"],
            package="corner"
        )
    """
    n_cases = len(samples_list)
    if black_background:
        matplotlib.rcParams.update(rcparams1)
    else:
        matplotlib.rcParams.update(rcparams2)
    if cases is None:
        cases = ["hyperbolic"] * n_cases
    # Use default color for each case if colors not provided
    if colors is None:
        colors = [default_colors.get(str(c), list(default_colors.values())[0]) for c in cases]
    if len(labels) != n_cases:
        raise ValueError("labels must have the same length as samples_list")
    if len(colors) != n_cases:
        raise ValueError("colors must have the same length as samples_list")
    if len(cases) != n_cases:
        raise ValueError("cases must have the same length as samples_list")

    if package == "corner":
        import corner
        fig = None
        for i, (samples, label, color) in enumerate(zip(samples_list, labels, colors)):
            fig = corner.corner(
                samples,
                labels=param_names if i == 0 else None,
                color=color,
                truths=truths,
                show_titles=True if i == 0 else False,
                title_kwargs={"fontsize": 18},
                smooth=True,
                smooth1d=True,
                bins=50,
                plot_datapoints=False,
                label_kwargs={"fontsize": 18},
                fig=fig,
                spacing=0.02,
                hist_kwargs={"label": label, "color": color, "lw": 2, "alpha": 0.7},
                contour_kwargs={"colors": [color]},
                fill_contours=False,
            )
        if labels:
            handles = [plt.Line2D([0], [0], color=col, lw=2, label=lab) for col, lab in zip(colors, labels)]
            fig.legend(handles=handles, loc="upper right", fontsize=18)
        if save_dir:
            plt.savefig(save_dir)
        if show:
            plt.show()
        else:
            plt.close(fig)
        if show:
            plt.show()
            
    elif package == "chainconsumer":
        import logging

        from chainconsumer import ChainConsumer
        logging.getLogger("chainconsumer").setLevel(logging.ERROR)

        c = ChainConsumer()
        for samples, label in zip(samples_list, labels):
            c.add_chain(samples, parameters=param_names, name=label)
        c.configure(
            colors=colors,
            shade=True,
            shade_alpha=0.25,
            bar_shade=True,
            linewidths=1.4,
            max_ticks=3,
            diagonal_tick_labels=True,
            tick_font_size=10,
            label_font_size=16,
            kde=True,
            summary=False,
        )
        ndim = len(param_names)
        base_size = 1.  # reduce whitespace while keeping content readable
        if truths is not None:
            fig = c.plotter.plot(figsize=(base_size * ndim, base_size * ndim)
, truth=truths, legend=True,
            )
        else:
            fig = c.plotter.plot(figsize=(base_size * ndim, base_size * ndim)
, legend=True,)
        if save_dir:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

def half_violin(
    data_left,
    data_right,
    pos=0,
    width=0.8,
    color_left=None,
    color_right=None,
    alpha=0.6,
    ax=None,
):
    """Draw a split violin at a single position."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 4))

    c_left = color_left or default_colors.get("hyperbolic")
    c_right = color_right or default_colors.get("whittle")

    parts_left = ax.violinplot(
        data_left,
        positions=[pos],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    parts_right = ax.violinplot(
        data_right,
        positions=[pos],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for pc in parts_left["bodies"]:
        pc.set_facecolor(c_left)
        pc.set_alpha(alpha)
        pc.set_edgecolor("none")
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(
            pc.get_paths()[0].vertices[:, 0], -np.inf, m
        )

    for pc in parts_right["bodies"]:
        pc.set_facecolor(c_right)
        pc.set_alpha(alpha)
        pc.set_edgecolor("none")
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(
            pc.get_paths()[0].vertices[:, 0], m, np.inf
        )

    return ax


def plot_half_violin_parameter(
    samples_L,
    samples_R,
    param_idx,
    label,
    true_value,
    colors=None,
    outpath=None,
    fig_size=(3.5, 4),
    alpha=0.65,
    show=False,
    background="white",
    legend_labels=("Hyperbolic", "Whittle"),
):
    """Plot and save a half-violin for a single parameter with medians and truth."""
    if outpath is None:
        raise ValueError("outpath is required for saving the violin plot")

    if background == "black":
        matplotlib.rcParams.update(rcparams1)
    else:
        matplotlib.rcParams.update(rcparams2)

    resolved_colors = colors or {
        "left": default_colors.get("hyperbolic"),
        "right": default_colors.get("whittle"),
    }

    x1 = samples_L[:, param_idx]
    x2 = samples_R[:, param_idx]

    fig, ax = plt.subplots(figsize=fig_size)

    half_violin(
        x1,
        x2,
        pos=0,
        color_left=resolved_colors.get("left"),
        color_right=resolved_colors.get("right"),
        alpha=alpha,
        ax=ax,
    )

    ax.grid(False) 
    ax.xaxis.set_visible(False)
    ax.set_ylabel(label)

    median_color = "white" if background == "black" else "black"
    ax.hlines(np.median(x1), -0.15, 0, color=median_color, alpha=0.45, lw=1)
    ax.hlines(np.median(x2), 0.00, 0.15, color=median_color, alpha=0.45, lw=1)

    ax.axhline(
        true_value,
        color="black" if background == "white" else "white",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        zorder=10,
    )
    
    if legend_labels:
        legend_elements = [
            Patch(facecolor=resolved_colors.get("left"), alpha=alpha, label=legend_labels[0]),
            Patch(facecolor=resolved_colors.get("right"), alpha=alpha, label=legend_labels[1]),
        ]

        ax.legend(handles=legend_elements, frameon=False, loc="upper left")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
