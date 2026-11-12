"""A standardized inference result container.

:class:`Result` wraps the raw posterior samples produced by the samplers
(Eryn / pocoMC) together with the parameter names, the injected truth (if any),
the priors, the evidence, and free-form metadata, and provides a uniform API for
downstream analysis and I/O -- the same role bilby's ``Result`` plays.

The container depends only on numpy. Optional features degrade gracefully:

* :meth:`Result.to_pandas` needs ``pandas``;
* :meth:`Result.corner` needs ``corner``;
* HDF5 I/O needs ``h5py`` (otherwise ``.npz`` is used).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = ["Result"]


@dataclass
class Result:
    """Posterior samples plus the metadata needed to interpret and persist them.

    Parameters
    ----------
    samples:
        Posterior samples, shape ``(n_samples, n_dim)``.
    parameter_names:
        Length ``n_dim`` list of column names, aligned with ``samples``.
    injection:
        Optional mapping ``{name: true_value}`` of injected parameters (for
        validation / PP-tests). May cover a subset of ``parameter_names``.
    log_evidence, log_evidence_err:
        Optional evidence estimate and its uncertainty (pocoMC provides these).
    sampler:
        Name of the sampler that produced the samples (``"eryn"``/``"pocomc"``).
    priors:
        Optional mapping ``{name: (minimum, maximum)}`` describing the prior
        support (used for plotting ranges and provenance).
    metadata:
        Free-form provenance (seed, wall-clock, package version, command, ...).
    """

    samples: np.ndarray
    parameter_names: list[str]
    injection: Optional[dict[str, float]] = None
    log_evidence: Optional[float] = None
    log_evidence_err: Optional[float] = None
    sampler: Optional[str] = None
    priors: Optional[dict[str, tuple[float, float]]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = np.atleast_2d(np.asarray(self.samples, dtype=float))
        self.parameter_names = list(self.parameter_names)
        if self.samples.shape[1] != len(self.parameter_names):
            raise ValueError(
                f"samples has {self.samples.shape[1]} columns but "
                f"{len(self.parameter_names)} parameter names were given"
            )

    # -- access ----------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def n_dim(self) -> int:
        return self.samples.shape[1]

    def _index(self, name: str) -> int:
        try:
            return self.parameter_names.index(name)
        except ValueError:
            raise KeyError(f"unknown parameter {name!r}; have {self.parameter_names}")

    def __getitem__(self, name: str) -> np.ndarray:
        """1-D marginal samples for a parameter by name."""
        return self.samples[:, self._index(name)]

    def median(self) -> dict[str, float]:
        return {n: float(np.median(self.samples[:, i])) for i, n in enumerate(self.parameter_names)}

    def percentile(self, q: Sequence[float] | float) -> dict[str, np.ndarray]:
        """Per-parameter percentile(s) ``q`` in [0, 100]."""
        return {n: np.percentile(self.samples[:, i], q) for i, n in enumerate(self.parameter_names)}

    def credible_interval(self, level: float = 0.9) -> dict[str, tuple[float, float]]:
        """Symmetric ``level`` credible interval per parameter."""
        lo, hi = 50 * (1 - level), 50 * (1 + level)
        return {n: (float(a), float(b)) for n, (a, b) in self.percentile([lo, hi]).items()}

    # -- validation ------------------------------------------------------------
    def credible_level(self, truth: Optional[Mapping[str, float]] = None) -> dict[str, float]:
        """Quantile of the injected value in each 1-D marginal posterior.

        For parameter ``θ`` this is ``mean(samples_θ < θ_true)`` -- the credible
        level at which the truth sits. For a correctly calibrated posterior these
        values are Uniform(0, 1) across many injections, which is what the
        PP-test checks (:mod:`hyperwave.validation`).
        """
        truth = truth if truth is not None else (self.injection or {})
        if not truth:
            raise ValueError("no injection/truth available to compute credible levels")
        out = {}
        for i, name in enumerate(self.parameter_names):
            if name in truth and truth[name] is not None:
                out[name] = float(np.mean(self.samples[:, i] < truth[name]))
        return out

    # -- interop ---------------------------------------------------------------
    def to_pandas(self):
        """Return the posterior as a ``pandas.DataFrame`` (requires pandas)."""
        import pandas as pd  # optional

        return pd.DataFrame(self.samples, columns=self.parameter_names)

    def corner(self, truths: bool | Mapping[str, float] = True, **kwargs):
        """Corner plot of the posterior (requires ``corner``).

        ``truths=True`` overlays the stored injection; pass a mapping to override,
        or ``False`` for none.
        """
        import corner  # optional

        t = None
        if truths is True and self.injection is not None:
            inj = self.injection
            if isinstance(inj, Mapping):
                t = [inj.get(n) for n in self.parameter_names]
            else:  # array-like in parameter order
                vals = list(np.atleast_1d(inj))
                t = vals[: len(self.parameter_names)] if len(vals) else None
        elif isinstance(truths, Mapping):
            t = [truths.get(n) for n in self.parameter_names]
        kwargs.setdefault("labels", self.parameter_names)
        # samples may carry extra (noise-shape) columns beyond the named science
        # parameters; plot only the named ones so labels/truths line up.
        k = len(self.parameter_names)
        return corner.corner(self.samples[:, :k], truths=t, **kwargs)

    def skymap(self, lon: str = "lam", sinlat: Optional[str] = "sin_beta",
               lat: Optional[str] = None, **kwargs):
        """Sky posterior (:class:`hyperwave.plots.SkyPosterior`) from two columns.

        Defaults to the LISA sampling names ``lam``/``sin_beta``; pass ``lat=``
        instead of ``sinlat=`` for a latitude column (e.g. ``dec``).  Extra
        keyword arguments go to :class:`SkyPosterior` (``nbins``, ``smooth``).
        """
        from .plots.skymap import SkyPosterior

        return SkyPosterior.from_result(self, lon=lon, sinlat=sinlat, lat=lat, **kwargs)

    # -- I/O -------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Persist to ``path``. ``.h5``/``.hdf5`` use h5py; otherwise ``.npz``."""
        meta = json.dumps(self._meta_dict())
        if path.endswith((".h5", ".hdf5")):
            import h5py  # optional

            with h5py.File(path, "w") as f:
                f.create_dataset("samples", data=self.samples)
                f.attrs["parameter_names"] = json.dumps(self.parameter_names)
                f.attrs["meta"] = meta
        else:
            np.savez_compressed(
                path, samples=self.samples,
                parameter_names=json.dumps(self.parameter_names), meta=meta,
            )
        return path

    @classmethod
    def load(cls, path: str) -> "Result":
        if path.endswith((".h5", ".hdf5")):
            import h5py

            with h5py.File(path, "r") as f:
                samples = np.asarray(f["samples"])
                names = json.loads(f.attrs["parameter_names"])
                meta = json.loads(f.attrs["meta"])
        else:
            with np.load(path, allow_pickle=False) as d:
                samples = d["samples"]
                names = json.loads(str(d["parameter_names"]))
                meta = json.loads(str(d["meta"]))
        return cls(samples=samples, parameter_names=names, **cls._meta_from_dict(meta))

    def _meta_dict(self) -> dict:
        return {
            "injection": self.injection,
            "log_evidence": self.log_evidence,
            "log_evidence_err": self.log_evidence_err,
            "sampler": self.sampler,
            "priors": self.priors,
            "metadata": self.metadata,
        }

    @staticmethod
    def _meta_from_dict(meta: dict) -> dict:
        # tuples are stored as lists in JSON; restore prior bounds to tuples
        priors = meta.get("priors")
        if priors:
            priors = {k: tuple(v) for k, v in priors.items()}
        return {
            "injection": meta.get("injection"),
            "log_evidence": meta.get("log_evidence"),
            "log_evidence_err": meta.get("log_evidence_err"),
            "sampler": meta.get("sampler"),
            "priors": priors,
            "metadata": meta.get("metadata", {}),
        }

    def __repr__(self) -> str:
        ev = "" if self.log_evidence is None else f", logZ={self.log_evidence:.2f}"
        return (f"Result(n_samples={self.n_samples}, n_dim={self.n_dim}, "
                f"sampler={self.sampler!r}{ev})")
