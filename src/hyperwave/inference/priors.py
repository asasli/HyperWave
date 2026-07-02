import numpy as np
from scipy.stats import rv_continuous


def per_detector_noise_priors(
    ifo_names,
    nsegs,
    alpha_range=(0.0, 30.0),
    delta_range=(0.0, 30.0),
    classic=True,
):
    """Priors for ``GWLikelihoods(..., shape_per_detector=True)``.

    Builds an ordered dict of uniform ``α`` and ``δ`` (or ``ratio``) priors in
    **segment-major** order — exactly the layout the per-detector
    ``_alpha_columns`` / ``_tail_columns`` reshape expects::

        alpha_<ifo0>_seg0, alpha_<ifo1>_seg0, ..., alpha_<ifoK>_seg0,
        alpha_<ifo0>_seg1, ..., alpha_<ifoK>_seg{nsegs-1},
        delta_<ifo0>_seg0, ..., delta_<ifoK>_seg{nsegs-1}      # if classic=True
        # (or ratio_<...> if classic=False)

    Pass the result as ``noise_priors=`` to :class:`LVKinference`. The matching
    likelihood is :meth:`GWLikelihoods.hyperbolic_classic` for ``classic=True``,
    or :meth:`GWLikelihoods.hyperbolic` for ``classic=False``.

    Parameters
    ----------
    ifo_names : sequence of str
        Detector names, in the same order as ``ifos_list`` passed to
        :class:`GWLikelihoods`.
    nsegs : int
        Number of frequency segments (must match ``GWLikelihoods.nsegs``).
    alpha_range, delta_range : (float, float)
        Uniform-prior bounds. Can be a single ``(lo, hi)`` shared across all
        (ifo, seg), or a dict keyed by detector name for per-detector bounds.
    classic : bool, default True
        ``True`` builds priors for :meth:`hyperbolic_classic` (α, δ);
        ``False`` for :meth:`hyperbolic` (α, ratio = δ/α).
    """
    import bilby

    def _bounds(value, name):
        return value[name] if isinstance(value, dict) else value

    priors = {}
    tail_name = "delta" if classic else "ratio"
    # α block — segment-major: outer loop over segments, inner loop over ifos.
    for i in range(nsegs):
        for ifo in ifo_names:
            lo, hi = _bounds(alpha_range, ifo)
            key = f"alpha_{ifo}_{i}"
            priors[key] = bilby.core.prior.Uniform(lo, hi, name=key)
    # δ / ratio block — same ordering.
    for i in range(nsegs):
        for ifo in ifo_names:
            lo, hi = _bounds(delta_range, ifo)
            key = f"{tail_name}_{ifo}_{i}"
            priors[key] = bilby.core.prior.Uniform(lo, hi, name=key)
    return priors


def calibration_node_priors(ifo_names, n_nodes, amplitude_sigma=0.05, phase_sigma=0.05):
    """Gaussian priors for sampled spline calibration nodes (Method A).

    Returns an ordered dict of bilby ``Gaussian`` priors,
    ``recalib_{ifo}_amplitude_{i}`` then ``recalib_{ifo}_phase_{i}`` per
    detector, matching the calibration-node layout expected by
    :class:`~hyperwave.likelihoods.GWLikelihoods` ``*_calsample`` methods
    (``cal_n_nodes=n_nodes``). Append these **last** to the prior dict handed to
    :class:`LVKinference` so the calibration columns land at the end of
    ``theta``. ``amplitude_sigma`` / ``phase_sigma`` may be a float (all
    detectors) or a per-detector dict.
    """
    import bilby

    def _per(value, name):
        return value[name] if isinstance(value, dict) else value

    priors = {}
    for name in ifo_names:
        a = _per(amplitude_sigma, name)
        for i in range(n_nodes):
            key = f"recalib_{name}_amplitude_{i}"
            priors[key] = bilby.core.prior.Gaussian(mu=0.0, sigma=a, name=key)
        p = _per(phase_sigma, name)
        for i in range(n_nodes):
            key = f"recalib_{name}_phase_{i}"
            priors[key] = bilby.core.prior.Gaussian(mu=0.0, sigma=p, name=key)
    return priors


def wrapper_for_eryn(bilby_prior, name="bilby_eryn_wrapper"):
        class Prior(rv_continuous):
            def __init__(self):
                super().__init__(name=name)
                self.minimum = bilby_prior.minimum
                self.maximum = bilby_prior.maximum
                self._width = self.maximum - self.minimum

            def _pdf(self, x):
                # Eryn expects this to be normalized PDF
                ln_prob = np.array([bilby_prior.ln_prob(xi) for xi in np.atleast_1d(x)])
                prob = np.exp(ln_prob)
                return prob

            def _cdf(self, x):
                # Approximate numerically if no cdf in Bilby
                # You can skip this if not needed in Eryn
                cdf_values = np.array([bilby_prior.cdf(xi) for xi in np.atleast_1d(x)])
                return cdf_values
            
            # Define a scipy.stats-like rvs function
            # def rvs(self, size=1, random_state=None):
            #     if random_state is not None:
            #         np.random.seed(random_state)
            #     samples = np.array(bilby_prior.sample(size))
            #     return samples.flatten() if size > 1 else samples[0]
            def rvs(self, size=1, random_state=None):
                if random_state is not None:
                    np.random.seed(random_state)
                samples = bilby_prior.sample(size)
                return np.array(samples)

            def _ppf(self, q):
                # Use inverse transform sampling (optional)
                raise NotImplementedError("PPF not available for this prior.")

            def prob(self, x):
                """Binary mask: 1 if in support, 0 if out"""
                x = np.atleast_1d(x)
                return (self.minimum <= x) & (x <= self.maximum)

            def logpdf(self, x, *args, **kwargs):
                """Log-prior that returns -inf outside the support.

                Some bilby priors (e.g. ``UniformInComponentsMassRatio``) return
                NaN for out-of-bounds inputs, which crashes Eryn when a stretch
                move proposes outside the prior box. Clip to the support first and
                guard against any residual non-finite values.
                """
                x = np.atleast_1d(np.asarray(x, dtype=float))
                out = np.full(x.shape, -np.inf, dtype=float)
                inb = np.isfinite(x) & (x >= self.minimum) & (x <= self.maximum)
                if np.any(inb):
                    lp = np.array([bilby_prior.ln_prob(float(xi)) for xi in x[inb]])
                    out[inb] = np.where(np.isfinite(lp), lp, -np.inf)
                return out

        return Prior()

def wrapper_for_pocomc(bilby_prior, name="bilby_eryn_wrapper"):
    """Convert Bilby priors into POCOMC-compatible format."""
    # Define a scipy.stats-like logpdf function
    def logpdf(x):
        return bilby_prior.ln_prob(x)
        
    # Define a scipy.stats-like rvs function
    def rvs(size=1, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        samples = np.array(bilby_prior.sample(size)).flatten()
        return samples
        
    def support():
        return (float(bilby_prior.minimum), float(bilby_prior.maximum))


    # Create a placeholder distribution object
    prior_dist = type(f'BilbyPriorWrapper_{name}', (), {})()
    prior_dist.logpdf = logpdf
    prior_dist.rvs = rvs
    prior_dist.support = support
    
    return prior_dist