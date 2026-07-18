import numpy as np
from scipy.stats import rv_continuous


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
