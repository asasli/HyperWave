import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [4.5,3.5]

rcparams1 = {
    'axes.labelsize': 10,
    'font.size': 10,
    'text.latex.preamble': (r'\usepackage{revtex4-2}'),
    'legend.fontsize': 10,
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
    'axes.linewidth': 0.5,
}

rcparams2 = {
    'axes.labelsize': 10,
    'font.size': 10,
    'text.latex.preamble': (r'\usepackage{revtex4-2}'),
    'legend.fontsize': 10,
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
    'figure.autolayout': True
}

class Shape:
    def __init__(self, samples, black_background=False, hyperwave='classic', ddims=True, save_dir=None, TAG=None):
        if black_background:
            self.rcparams = rcparams1
            self.triangle_color = 'w'
        else:
            self.rcparams = rcparams2
            self.triangle_color = 'k'
        if save_dir is not None:
            self.save_name = save_dir + f'{TAG}' if TAG is not None else save_dir
        else:
            self.save_name = None
        matplotlib.rcParams.update(self.rcparams)
        self.hyperwave = hyperwave
        self.samples = samples
        if ddims:
            self.ndims = int(self.samples.shape[1]/2)
            self.alpha_dim = int(self.samples.shape[1]/2)
        else:
            self.ndims = int(self.samples.shape[1])
            self.alpha_dim = 1
        cmap = plt.get_cmap('nipy_spectral')  # Using a neon-like colormap
        self.clr = [cmap(i / self.ndims) for i in range(self.ndims)]
        if self.hyperwave == 'classic':
            self.alpha, self.delta = 10**np.median(samples[:, 0:self.alpha_dim]), 10**np.median(samples[:, self.alpha_dim:], axis=0)
            self.ratio = samples[:, self.alpha_dim:]/samples[:, 0:self.alpha_dim]  # delta / alpha
            self.median_sigma = np.median(self.ratio, axis=0)
            self.upper_sigma = np.percentile(self.ratio, 90, axis=0)
            self.lower_sigma = np.percentile(self.ratio, 10, axis=0)
            _, self.ksi, _ = self.xi_ksi(beta=0, alpha=self.alpha, delta=self.delta)
        else:
            self.alpha, self.delta = self.convert_a_bar_to_alpha_delta(samples=self.samples, alpha_dim=self.alpha_dim)
            self.ratio = 10**(samples[:, self.alpha_dim:])  # delta / alpha
            self.median_sigma = np.median(self.ratio, axis=0)
            self.upper_sigma = np.percentile(self.ratio, 90, axis=0)
            self.lower_sigma = np.percentile(self.ratio, 10, axis=0)
            _, self.ksi, _ = self.xi_ksi(beta=0, alpha=self.alpha, delta=self.delta)
        self.shape_triangle(beta=0, alpha=self.alpha, delta=self.delta, clr=self.clr)
        pass

    @staticmethod
    def convert_a_bar_to_alpha_delta(samples, alpha_dim):
        """ Convert a and b to alpha and delta. """
        alpha = 10**np.median(samples[:, 0:alpha_dim], axis=0)  # alpha
        B = 10**np.median(samples[:, alpha_dim:], axis=0)  # delta / alpha
        delta = alpha * B
        return alpha, delta

    @staticmethod
    def xi_ksi(beta, alpha, delta):
        """ Calculate xi and ksi for triangle plot. """
        ksi = 1 / np.sqrt(1 + delta * np.sqrt(alpha**2 - beta**2))
        xi = ksi * beta / alpha
        zeta = delta * np.sqrt(alpha**2 - beta**2)
        rho = beta / alpha
        return xi, ksi, rho

    def shape_triangle(self, beta, alpha, delta, clr):
        """ Plot the shape of the triangle. """
        xi, self.ksi, _ = self.xi_ksi(beta, alpha, delta)
        fig, ax = plt.subplots(1, figsize=[6, 5])
        trianglex = [-1, 1, 0, -1]  # repeated last coordinates to form the outline of the triangle
        triangley = [1, 1, 0, 1]

        ax.plot(trianglex, triangley, '-', color=self.triangle_color)
        plt.ylim(-0.15, 1.15)
        plt.xlim(-1.2, 1.2)
        plt.text(0.0, 1.03, 'skew−Laplace distribution', family='serif', fontsize=10, style='italic', ha='center', wrap=False)
        plt.text(0, -0.06, 'Normal distribution', family='serif', fontsize=10, style='italic', ha='center', wrap=False)
        plt.text(0.55, 0.15, 'left-skewed \n support bounded on left', rotation=56, family='serif', fontsize=10, style='italic', ha='center', wrap=False)
        plt.text(-.55, 0.15, 'left-skewed \n support bounded on right', rotation=-54, family='serif', fontsize=10, style='italic', ha='center', wrap=False)
        plt.xlabel(r'$\chi$')
        plt.ylabel(r'$\xi$')

        for i in range(len(self.ksi)):
            ax.plot(xi[i], self.ksi[i], '*', markersize='10', color=clr[i], alpha=0.6)

        plt.tight_layout()
        if self.save_name is not None:
            plt.savefig(self.save_name + 'shape_triangle.pdf', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        return
    
    def PSD_correction(self, f, Sn, segi):
        """ Plot the PSD correction. """
        matplotlib.rcParams.update(self.rcparams)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        Sn_predicted = [Sn[segi[i]]*(self.median_sigma[i]) for i in range(len(segi))]
        self.Sn_predicted = np.concatenate(Sn_predicted).flatten()
        Sn_predicted_upper = [Sn[segi[i]]*(self.upper_sigma[i]) for i in range(len(segi))]
        self.Sn_predicted_upper = np.concatenate(Sn_predicted_upper).flatten()
        Sn_predicted_lower = [Sn[segi[i]]*(self.lower_sigma[i]) for i in range(len(segi))]
        self.Sn_predicted_lower = np.concatenate(Sn_predicted_lower).flatten()
        
        plt.loglog(f, Sn, color="#80BCD8", lw=0.5, label='initial')
        plt.loglog(f, self.Sn_predicted, color="#f9246f", lw=0.5, label='predicted')
        plt.fill_between(f, self.Sn_predicted_lower, self.Sn_predicted_upper, color="#ca0147", alpha=0.2)
        plt.legend(loc='upper right', fontsize=10)
        plt.xlim(f[0], f[-1])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        if self.save_name is not None:
            plt.savefig(self.save_name + 'PSD_correction.pdf', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

    def data_gaussianity(self, f, data, segi):
        matplotlib.rcParams.update(self.rcparams)
        if self.ksi is None:
            _, self.ksi, _ = self.xi_ksi(self.beta, self.alpha, self.delta)
        ksi = [np.full(len(segi[i]), self.ksi[i]) for i in range(len(segi))]
        self.ksi = np.concatenate(ksi).flatten()
        sigma = [np.full(len(segi[i]), self.median_sigma[i]) for i in range(len(segi))]
        self.sigma = np.concatenate(sigma).flatten()
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.loglog(f, data, color="#80BCD8", alpha=0.5)
        ax1.loglog(f, self.sigma, '--', color="#f9246f", alpha=0.5)
        ax1.set_xlim(f[0], f[-1])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Whitened data', color="#80BCD8")

        # Create a twin y-axis for ksi
        ax2 = ax1.twinx()
        ax2.plot(f, self.ksi, color='orchid')
        ax2.set_ylabel(r'$\xi$', color='orchid')
        ax2.tick_params(axis='y', colors='orchid')
        ax2.set_ylim(0, 1)
        if self.save_name is not None:
            plt.savefig(self.save_name + 'data_gaussianity.pdf', dpi=300, bbox_inches='tight', transparent=True)

        plt.show()

