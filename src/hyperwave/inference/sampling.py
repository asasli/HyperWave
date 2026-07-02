import multiprocessing as mp
import os
import pickle
import warnings
from typing import Callable, Dict, Optional

import bilby
import numpy as np
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer
from eryn.state import State

# imports for Eryn
from .priors import wrapper_for_eryn, wrapper_for_pocomc

try:
    import pocomc as pc
except ImportError:
    warnings.warn("POCOMC is not installed. Please install it to use the POCOMC sampler.", ImportWarning)
    pc = None

class LVKinference:
    """
    A unified inference runner supporting both Eryn and POCOMC samplers.
    
    Attributes
    ----------
    loglikelihood : Callable
        Log-likelihood function.
    sampler_name : str
        Name of the sampler to use ("eryn" or "pocomc").
    priors : dict
        Dictionary of user-supplied priors.
    noise_priors : dict
        Dictionary of priors for noise-related parameters.
    common_params : dict
        Dictionary of parameters common to both samplers (e.g., cmin, cmax, TAG, etc.).
    sampler_kwargs : dict
        Additional keyword arguments specific to the sampler.
    """

    def __init__(
        self,
        loglikelihood: Callable,
        sampler_name: str,
        priors: Dict[str, bilby.prior.Prior],
        noise_priors: Dict[str, bilby.prior.Prior],
        common_params: Dict,
        sampler_kwargs: Optional[Dict] = None,
        periodic: Optional[list] = None
    ):
        self.loglikelihood = loglikelihood
        self.sampler_name = sampler_name.lower()
        self.user_priors = priors
        self.noise_priors = noise_priors
        self.common = common_params
        self.kwargs = sampler_kwargs or {}
        self.periodic = periodic
        self._prepare_priors()

    def _prepare_priors(self):
        """Construct full Bilby prior dictionary with user and noise parameters."""
        self.priors = bilby.core.prior.PriorDict(self.user_priors)
        if self.common["like"] not in ['gauss', 'gaussian']:
            self.priors.update(self.noise_priors)
        self.ndims = len(self.priors)
        # Normalize `self.periodic` to be a list of integer indices.
        # Allow users to pass parameter NAMES (strings) or indices (ints).
        if self.periodic is not None and len(self.periodic) > 0:
            # If names were provided, convert them to indices according to the priors ordering
            if all(isinstance(p, str) for p in self.periodic):
                name_list = list(self.priors.keys())
                try:
                    self.periodic = [name_list.index(p) for p in self.periodic]
                except ValueError as exc:
                    raise ValueError(f"One or more periodic parameter names not found in priors: {self.periodic}") from exc
            # Otherwise ensure they are ints
            else:
                self.periodic = [int(p) for p in self.periodic]

    def _prepare_eryn_priors(self):
        priors_in = {i: wrapper_for_eryn(self.priors[key], str(key)) for i, key in enumerate(self.priors.keys())}
        priors = ProbDistContainer(priors_in)
        return priors
    
    def _prepare_pocomc_priors(self):
        return pc.Prior([wrapper_for_pocomc(self.priors[key], key) for key in self.priors])        

    def run(self):
        """Dispatch the sampler execution based on the chosen method."""
        if self.sampler_name == "eryn":
            self.eryn_priors = self._prepare_eryn_priors()
            self._run_eryn()
        elif self.sampler_name == "pocomc":
            self._run_pocomc()
        else:
            raise ValueError("Unsupported sampler. Choose 'eryn' or 'pocomc'.")

    def _run_eryn(self):
        """Run inference using the Eryn sampler."""
        # eryn modules for sampling
        # Build Eryn-style periodic mapping: map parameter index -> upper bound
        periodic = {"model_0": {}}
        name_list = list(self.priors.keys())
        for idx in self.periodic or []:
            prior = self.priors[name_list[idx]]
            if not hasattr(prior, "maximum"):
                raise ValueError(f"Periodic prior {prior.name} has no maximum")
            periodic["model_0"][idx] = prior.maximum
        
        nwalkers = self.kwargs.get("nwalkers", 2* self.ndims)  # Default to 2 * ndims if not specified
        ntemps = self.kwargs.get("ntemps", 5)
        burn = self.kwargs.get("burn", 100)
        nsteps = self.kwargs.get("nsteps", 200)
        thin = self.kwargs.get("thin", 1)
        # Optional custom Eryn machinery (e.g. flow proposals): a `moves` list of
        # (move, weight) replaces the default stretch move, and update_fn /
        # update_iterations enable periodic callbacks (e.g. flow retraining).
        extra = {}
        if self.kwargs.get("moves") is not None:
            extra["moves"] = self.kwargs["moves"]
        if self.kwargs.get("update_fn") is not None:
            extra["update_fn"] = self.kwargs["update_fn"]
            extra["update_iterations"] = int(self.kwargs.get("update_iterations", 100))
        # Setup initial positions
        tmp = self.eryn_priors.rvs(size=nwalkers * ntemps)
        coords = tmp.reshape((ntemps, nwalkers, self.ndims))

        logl = self.loglikelihood(coords.reshape(ntemps * nwalkers, self.ndims)).reshape(ntemps, nwalkers)
        state = State(coords[:, :, np.newaxis, :], log_like=logl)

        # Ensure the chains directory exists
        chains_dir = os.path.join(self.common["save_dir"], "chains")
        os.makedirs(chains_dir, exist_ok=True)
        backend_file = self.common["save_dir"] + f"/chains/{self.common['TAG']}_eryn.h5"
        # Runs always start fresh from prior draws (no resume support), so a
        # pre-existing backend file is stale — typically left by a crashed run —
        # and makes Eryn fail comparing key_order against it. Remove it.
        if os.path.exists(backend_file):
            os.remove(backend_file)
        backend = HDFBackend(backend_file)
        sampler = EnsembleSampler(
            nwalkers,
            self.ndims,
            self.loglikelihood,
            self.eryn_priors,
            tempering_kwargs=dict(betas=np.linspace(1.0, 0.0, ntemps), stop_adaptation=burn),
            vectorize=True,
            periodic=periodic,
            backend=backend,
            **extra,
        )

        print("> Running Eryn MCMC...")
        sampler.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin)
        print(f"> Eryn sampling finished. Output at: {backend.filename}")
    
    def get_samples(self, thin=2):
        """Load the samples from the Eryn backend."""
        if self.sampler_name == "eryn":
            print(f"Loading samples from {self.common['save_dir']}/chains/{self.common['TAG']}_eryn.h5")
            # Ensure the chains directory exists
            if not os.path.exists(self.common["save_dir"] + f"/chains/{self.common['TAG']}_eryn.h5"):
                raise FileNotFoundError(f"File {self.common['save_dir']}/chains/{self.common['TAG']}_eryn.h5 does not exist.")
            backend = HDFBackend(self.common["save_dir"] + f"/chains/{self.common['TAG']}_eryn.h5")
            samples = self.get_clean_chain(backend.get_chain(thin=thin)['model_0'], self.ndims)
            return samples
        elif self.sampler_name == "pocomc":
            print(f"Loading samples from {self.common['save_dir']}/chains/POCO_{self.common['TAG']}_pocomc.pkl")
            out_file = f"{self.common['save_dir']}/chains/POCO_{self.common['TAG']}_pocomc.pkl"
            if not os.path.exists(out_file):
                raise FileNotFoundError(f"File {out_file} does not exist.")
            with open(out_file, "rb") as f:
                data = pickle.load(f)
            samples = data['samples']
            return samples
        else:
            raise ValueError("Unsupported sampler. Choose 'eryn' or 'pocomc'.")
    
    def get_clean_chain(self, coords, ndim, temp=0):
        """ A silly function to get the clean parameter chains out from Eryn. 
            In the future it will not be needed.
        """
        naninds    = np.logical_not(np.isnan(coords[:, temp, :, :, 0].flatten()))
        print(np.sum(naninds))
        samples_in = np.zeros((coords[:, temp, :, :, 0].flatten()[naninds].shape[0], ndim))  # init the chains to plot
        # get the samples to plot
        for d in range(ndim):
            givenparam = coords[:, temp, :, :, d].flatten()
            samples_in[:, d] = givenparam[
                np.logical_not(np.isnan(givenparam))
            ]  # Discard the NaNs, each time they change the shape of the samples_in
        return samples_in

    def get_result(self, injection=None, parameter_names=None, thin=2):
        """Return the posterior as a :class:`hyperwave.Result`.

        ``parameter_names`` defaults to the prior order (science parameters
        followed by the noise-shape nuisance parameters). ``injection`` is an
        optional ``{name: truth}`` mapping used by the calibration/PP-tests.
        """
        from ..result import Result

        samples = self.get_samples(thin=thin)
        if parameter_names is None:
            parameter_names = [str(k) for k in self.priors.keys()]
        names = list(parameter_names)[: samples.shape[1]]
        names += [f"param_{i}" for i in range(len(names), samples.shape[1])]
        bounds = {}
        for key in self.priors.keys():
            pr = self.priors[key]
            if hasattr(pr, "minimum") and hasattr(pr, "maximum"):
                bounds[str(key)] = (float(pr.minimum), float(pr.maximum))
        return Result(
            samples, names, injection=injection, sampler=self.sampler_name,
            priors=bounds or None,
            metadata={"tag": self.common.get("TAG"), "ndim": int(self.ndims)},
        )

    def _run_pocomc(self):
        """Run inference using the POCOMC sampler."""
        # POCOMC import
        prior = self._prepare_pocomc_priors()
        n_total = self.kwargs.get("n_total", 50000)
        n_effective = self.kwargs.get("n_effective", 12000) # 2000
        n_active = self.kwargs.get("n_active", 400) #1000
        n_steps = self.kwargs.get("n_steps", max(10, int(0.7 * self.ndims)))

        sampler = pc.Sampler(
            likelihood=self.loglikelihood,
            prior=prior,
            n_effective=n_effective,
            n_active=n_active,
            vectorize=True,
            periodic=self.periodic,
            n_steps=n_steps,
        )

        print("> Running POCOMC sampling...")
        sampler.run(n_total=n_total)
        samples, logl, logp = sampler.posterior(resample=True)

        out_file = f"{self.common['save_dir']}/chains/POCO_{self.common['TAG']}_pocomc.pkl"
        with open(out_file, "wb") as f:
            pickle.dump({'samples': samples, 'logl': logl, 'logp': logp}, f)

        print(f"> POCOMC sampling finished. Output at: {out_file}")

class DataInference:
    """
    A simplified inference runner for data analysis using only noise priors
    supporting both Eryn and POCOMC samplers.
    
    Attributes
    ----------
    loglikelihood : Callable
        Log-likelihood function.
    noise_priors : dict
        Dictionary of priors for noise-related parameters.
    common_params : dict
        Dictionary of parameters common to the sampler (e.g., cmin, cmax, TAG, etc.).
    sampler_kwargs : dict
        Additional keyword arguments specific to the sampler.
    """
    def __init__(
        self,
        loglikelihood: Callable,
        noise_priors: Dict[str, bilby.prior.Prior],
        common_params: Dict,
        sampler_kwargs: Optional[Dict] = None,
        sampler_name: str = "eryn",
        periodic: Optional[list] = None,
        workers: int = 32,
    ):

        self.loglikelihood = loglikelihood
        self.noise_priors = noise_priors
        self.common = common_params
        self.kwargs = sampler_kwargs or {}
        # Default POCOMC kwargs if not provided
        if sampler_name.lower() == "pocomc":
            self.kwargs.setdefault("n_total", 20000)
            self.kwargs.setdefault("n_effective", 1024)
            self.kwargs.setdefault("n_active", 512)
        self.sampler_name = sampler_name.lower()
        self.periodic = periodic
        self.workers = int(workers) if workers is not None else 32
        self.ndims = len(self.noise_priors)
        self.eryn_priors = self._prepare_eryn_priors()
        
        # Prepare POCOMC priors on demand
        if self.sampler_name == "pocomc":
            if pc is None:
                raise ImportError("POCOMC is not installed. Please install it to use the POCOMC sampler.")
            self.poco_priors = self._prepare_pocomc_priors()

    def _prepare_eryn_priors(self):
        priors_in = {i: wrapper_for_eryn(self.noise_priors[key], str(key)) for i, key in enumerate(self.noise_priors.keys())}
        priors = ProbDistContainer(priors_in)
        return priors
    
    def _prepare_pocomc_priors(self):
        return pc.Prior([wrapper_for_pocomc(self.noise_priors[key], key) for key in self.noise_priors])        

    def run(self):
        """Dispatch the sampler execution based on the chosen method."""
        if self.sampler_name == "eryn":
            self._run_eryn()
        elif self.sampler_name == "pocomc":
            self._run_pocomc()
        else:
            raise ValueError("Unsupported sampler. Choose 'eryn' or 'pocomc'.")

    def _run_eryn(self):
        """Run inference using the Eryn sampler."""
        nwalkers = self.kwargs.get("nwalkers", 2* self.ndims)  # Default to 2 * ndims if not specified
        ntemps = self.kwargs.get("ntemps", 5)
        burn = self.kwargs.get("burn", 1000)
        nsteps = self.kwargs.get("nsteps", 2000)
        thin = self.kwargs.get("thin", 1)
        print(f"nwalkers: {nwalkers}")
        print(f"ntemps: {ntemps}")
        print(f"burn: {burn}")
        print(f"nsteps: {nsteps}")

        # Setup initial positions
        tmp = self.eryn_priors.rvs(size=nwalkers * ntemps)
        coords = tmp.reshape((ntemps, nwalkers, self.ndims))

        logl = self.loglikelihood(coords.reshape(ntemps * nwalkers, self.ndims)).reshape(ntemps, nwalkers)
        state = State(coords[:, :, np.newaxis, :], log_like=logl)
        # Ensure the chains directory exists
        chains_dir = os.path.join(self.common["save_dir"], "chains")
        os.makedirs(chains_dir, exist_ok=True)
        backend_file = self.common["save_dir"] + f"/chains/{self.common['TAG']}_eryn.h5"
        # Runs always start fresh from prior draws (no resume support), so a
        # pre-existing backend file is stale — typically left by a crashed run —
        # and makes Eryn fail comparing key_order against it. Remove it.
        if os.path.exists(backend_file):
            os.remove(backend_file)
        backend = HDFBackend(backend_file)
        sampler = EnsembleSampler(
            nwalkers,
            self.ndims,
            self.loglikelihood,
            self.eryn_priors,
            tempering_kwargs=dict(betas=np.linspace(1.0, 0.0, ntemps), stop_adaptation=burn),
            vectorize=True,
            backend=backend
        )

        print("> Running Eryn MCMC...")
        sampler.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin)
        print(f"> Eryn sampling finished. Output at: {backend.filename}")
    
    def get_samples(self, thin=2):
        """Load the samples from the sampler backend."""
        if self.sampler_name == "eryn":
            print(f"Loading samples from {self.common['save_dir']}/chains/{self.common['TAG']}_eryn.h5")
            # Ensure the chains directory exists
            if not os.path.exists(self.common["save_dir"] + f"/chains/{self.common['TAG']}_eryn.h5"):
                raise FileNotFoundError(f"File {self.common['save_dir']}/chains/{self.common['TAG']}_eryn.h5 does not exist.")
            backend = HDFBackend(self.common["save_dir"] + f"/chains/{self.common['TAG']}_eryn.h5")
            samples = self.get_clean_chain(backend.get_chain(thin=thin)['model_0'], self.ndims)
            return samples
        elif self.sampler_name == "pocomc":
            print(f"Loading samples from {self.common['save_dir']}/chains/POCO_{self.common['TAG']}_pocomc.pkl")
            out_file = f"{self.common['save_dir']}/chains/POCO_{self.common['TAG']}_pocomc.pkl"
            if not os.path.exists(out_file):
                raise FileNotFoundError(f"File {out_file} does not exist.")
            with open(out_file, "rb") as f:
                data = pickle.load(f)
            samples = data['samples']
            return samples
        else:
            raise ValueError("Unsupported sampler. Choose 'eryn' or 'pocomc'.")
    
    def get_clean_chain(self, coords, ndim, temp=0):
        """ A silly function to get the clean parameter chains out from Eryn. 
            In the future it will not be needed.
        """
        naninds    = np.logical_not(np.isnan(coords[:, temp, :, :, 0].flatten()))
        print(np.sum(naninds))
        samples_in = np.zeros((coords[:, temp, :, :, 0].flatten()[naninds].shape[0], ndim))  # init the chains to plot
        # get the samples to plot
        for d in range(ndim):
            givenparam = coords[:, temp, :, :, d].flatten()
            samples_in[:, d] = givenparam[
                np.logical_not(np.isnan(givenparam))
            ]  # Discard the NaNs, each time they change the shape of the samples_in
        return samples_in

    def _run_pocomc(self):
        """Run inference using the POCOMC sampler."""
        if pc is None:
            raise ImportError("POCOMC is not installed. Please install it to use the POCOMC sampler.")

        prior = getattr(self, 'poco_priors', None)
        if prior is None:
            prior = self._prepare_pocomc_priors()

        n_total = self.kwargs.get("n_total", 20000)
        n_effective = self.kwargs.get("n_effective", 1024)
        n_active = self.kwargs.get("n_active", 512)

        created_pool = None
        sampler = None
        # Try providing a multiprocessing pool for parallelization
        try:
            pool = self.kwargs.get("pool", None)
            if pool is None and (self.workers or 0) > 1:
                created_pool = mp.Pool(processes=self.workers)
                pool = created_pool

            sampler = pc.Sampler(
                likelihood=self.loglikelihood,
                prior=prior,
                n_effective=n_effective,
                n_active=n_active,
                vectorize=True,
                periodic=self.periodic,
                n_steps=max(10, int(0.7 * self.ndims)),
                pool=pool,
            )
        except TypeError:
            # Fallback: try workers argument
            try:
                sampler = pc.Sampler(
                    likelihood=self.loglikelihood,
                    prior=prior,
                    n_effective=n_effective,
                    n_active=n_active,
                    vectorize=True,
                    periodic=self.periodic,
                    n_steps=max(10, int(0.7 * self.ndims)),
                    workers=self.workers,
                )
            except TypeError:
                # Final fallback: run without pool/workers, optionally set num_jobs on the likelihood object
                if hasattr(self.loglikelihood, "__self__") and hasattr(self.loglikelihood.__self__, "num_jobs"):
                    try:
                        self.loglikelihood.__self__.num_jobs = self.workers
                    except Exception:
                        pass
                sampler = pc.Sampler(
                    likelihood=self.loglikelihood,
                    prior=prior,
                    n_effective=n_effective,
                    n_active=n_active,
                    vectorize=True,
                    periodic=self.periodic,
                    n_steps=max(10, int(0.7 * self.ndims)),
                )

        print("> Running POCOMC sampling...")
        try:
            sampler.run(n_total=n_total)
            samples, logl, logp = sampler.posterior(resample=True)
        finally:
            # Ensure we clean up any pool we created
            if created_pool is not None:
                try:
                    created_pool.close()
                    created_pool.join()
                except Exception:
                    pass

        chains_dir = os.path.join(self.common["save_dir"], "chains")
        os.makedirs(chains_dir, exist_ok=True)
        out_file = f"{self.common['save_dir']}/chains/POCO_{self.common['TAG']}_pocomc.pkl"
        with open(out_file, "wb") as f:
            pickle.dump({'samples': samples, 'logl': logl, 'logp': logp}, f)

        print(f"> POCOMC sampling finished. Output at: {out_file}")

# TODO: ADD evidence calculation methods for both samplers
