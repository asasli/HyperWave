"""Galactic-binary (UCB) waveform template for LISA, GBGPU-backed.

The LISA counterpart of :class:`hyperwave.detectors.lvk.waveform.GW`: it wraps
``gbgpu`` and exposes the standard HyperWave template interface
(``.parameters`` + ``make_injections_to_ifo_batch``), so it drops straight into
:class:`hyperwave.likelihoods.GWLikelihoods` and the ``LVKinference`` pipeline —
no bespoke likelihood needed.

The whole walker block is generated in a single ``run_wave`` call (natively
batched on CPU or GPU); the per-walker Python loop the paper code originally
used is what made the CPU run take ~19 h instead of minutes.

Sampling space (paper ``hyperbolic_likelihood_inv_ucbs.py``)::

    [log10A, f0_mHz, log10fdot, phi0, cosi, psi, lambda, sinb]

is mapped to the physical parameters GBGPU expects (linear amplitude/frequency,
iota = arccos(cosi), beta = arcsin(sinb), fddot = 0).
"""

from __future__ import annotations

import numpy as np

UCB_PARAMETER_NAMES = ["log10A", "f0_mHz", "log10fdot",
                       "phi0", "cosi", "psi", "lambda", "sinb"]
# periodic sampling params (index -> period): phi0, psi (period pi), lambda
UCB_PERIODIC = {"phi0": 2 * np.pi, "psi": np.pi, "lambda": 2 * np.pi}

SECS_PER_YEAR = 365.25 * 86400.0


def _to_numpy(arr):
    """CuPy or NumPy array -> NumPy (no-op on host arrays)."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


class UCBTemplate:
    """GBGPU galactic-binary template in the paper's sampling space.

    Parameters
    ----------
    noise : LISANoise
        Carries the analysis grid (``frequency_array``), ``tobs_years`` and ``dt``.
    parameters : sequence of str, optional
        Sampled parameter names (default :data:`UCB_PARAMETER_NAMES`).
    n_window : int, optional
        Number of GBGPU output bins per source; defaults to the noise grid size.
    use_tdi2, use_gpu : bool
        GBGPU options.
    """

    def __init__(self, noise, parameters=None, n_window=None,
                 use_tdi2=False, use_gpu=False):
        try:
            from gbgpu.gbgpu import GBGPU
        except ImportError as exc:  # pragma: no cover - LISA env only
            raise ImportError(
                "UCBTemplate needs gbgpu (use the hyperwave-dev env on a "
                "v100/Skylake node)."
            ) from exc
        self._noise = noise
        self.parameters = list(parameters) if parameters is not None \
            else list(UCB_PARAMETER_NAMES)
        self.channels = tuple(getattr(noise, "channels", ("A", "E")))
        self._gb = GBGPU(use_gpu=use_gpu)
        n = int(n_window) if n_window is not None else int(len(noise.frequency_array))
        self._kwargs = dict(N=n, dt=float(noise.dt),
                            T=float(noise.tobs_years) * SECS_PER_YEAR, tdi2=use_tdi2)

    # -- sampling-space -> physical (paper statutils.ucb.trans) ----------------
    @staticmethod
    def _sampling_to_physical(with_fddot):
        # with_fddot: (N, 9) = [log10A, f0_mHz, log10fdot, fddot=0,
        #                       phi0, cosi, psi, lambda, sinb]
        phys = with_fddot.copy()
        phys[:, 0] = 10 ** with_fddot[:, 0]                            # A
        phys[:, 1] = with_fddot[:, 1] * 1e-3                           # f0 [Hz]
        phys[:, 2] = 10 ** with_fddot[:, 2]                           # fdot
        # phys[:, 3] = 0.0  (fddot, already filled)  / [:,4]=phi0 unchanged
        phys[:, 5] = np.arccos(np.clip(with_fddot[:, 5], -1.0, 1.0))   # iota
        # phys[:, 6]=psi, [:, 7]=lambda  unchanged
        phys[:, 8] = np.arcsin(np.clip(with_fddot[:, 8], -1.0, 1.0))   # beta
        return phys

    def make_injections_to_ifo_batch(self, thetas):
        """Sampling-space ``(N, 8)`` -> ``(N, nchannels, nfreq)`` A/E signal.

        One ``run_wave`` generates all N sources at once.
        """
        params = np.atleast_2d(np.asarray(thetas, dtype=float))
        with_fddot = np.insert(params, 3, 0.0, axis=1)        # fddot = 0
        phys = self._sampling_to_physical(with_fddot).T       # (9, N)
        self._gb.run_wave(*phys, **self._kwargs)
        a = np.atleast_2d(_to_numpy(self._gb.A))
        e = np.atleast_2d(_to_numpy(self._gb.E))
        return np.stack([a, e], axis=1)                       # (N, 2, nfreq)

    def make_injections_to_ifo(self, theta):
        """Single-vector convenience: returns ``{channel: (nfreq,) array}``."""
        batch = self.make_injections_to_ifo_batch(np.atleast_2d(theta))
        return {c: batch[0, j, :] for j, c in enumerate(self.channels)}
