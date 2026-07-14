"""Adaptive normalizing-flow proposals for Eryn moves.

This module keeps Eryn in charge of the actual RJMCMC machinery and plugs a
learned proposal distribution into Eryn's official distribution-based moves.
The flow backend is provided by ``pocomc.Flow``, which itself wraps ``zuko``
normalizing flows.
"""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.special import expit, logsumexp


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def flow_backend_available() -> bool:
    """Return ``True`` when ``pocomc`` and ``torch`` are importable."""
    return _module_available("pocomc") and _module_available("torch")


def require_flow_backend():
    """Import the flow backend used by the adaptive proposal layer."""
    try:
        import pocomc as pc
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "Adaptive flow proposals require both 'pocomc' and 'torch'. "
            "Install HyperWave with `pip install .[sampling]` and a compatible "
            "PyTorch build."
        ) from exc

    return torch, pc


class _JointProposalAdapter:
    """Expose an ``AdaptiveFlowProposal`` with the API Eryn expects."""

    def __init__(self, proposal: "AdaptiveFlowProposal"):
        self.proposal = proposal
        self.use_cupy = False
        self.return_gpu = False

    def rvs(self, size=1, random_state: Optional[int] = None):
        if isinstance(size, tuple):
            if len(size) != 1:
                raise ValueError("Joint flow proposals only support one leading sample dimension.")
            size = size[0]
        return self.proposal.rvs(size=size, random_state=random_state)

    def logpdf(self, x):
        return self.proposal.logpdf(x)


def _support_bounds(distribution) -> tuple[float, float]:
    if hasattr(distribution, "minimum") and hasattr(distribution, "maximum"):
        return float(distribution.minimum), float(distribution.maximum)
    if hasattr(distribution, "support"):
        lower, upper = distribution.support()
        return float(lower), float(upper)
    raise ValueError(
        "AdaptiveFlowProposal requires bounded one-dimensional distributions "
        "with either (minimum, maximum) attributes or a support() method."
    )


def _random_seed_from_state(random) -> Optional[int]:
    if random is None:
        return None
    for name in ("integers", "randint"):
        method = getattr(random, name, None)
        if method is None:
            continue
        return int(method(0, 2**31 - 1))
    return None


def _build_context(proposal, active_points: np.ndarray):
    if hasattr(proposal, "build_context"):
        return proposal.build_context(active_points)
    return active_points


def _proposal_rvs(proposal, *, size: int, context=None, random=None) -> np.ndarray:
    random_state = _random_seed_from_state(random)
    if hasattr(proposal, "rvs_with_context"):
        return np.asarray(
            proposal.rvs_with_context(size=size, context=context, random_state=random_state),
            dtype=float,
        )
    try:
        return np.asarray(proposal.rvs(size=size, context=context, random_state=random_state), dtype=float)
    except TypeError:
        return np.asarray(proposal.rvs(size=size, random_state=random_state), dtype=float)


def _proposal_logpdf(proposal, x, *, context=None) -> np.ndarray | float:
    if hasattr(proposal, "logpdf_with_context"):
        return proposal.logpdf_with_context(x, context=context)
    try:
        return proposal.logpdf(x, context=context)
    except TypeError:
        return proposal.logpdf(x)


def _as_2d_array(x: np.ndarray | Sequence[float], ndim: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    squeezed = arr.ndim == 1
    if squeezed:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != ndim:
        raise ValueError(f"Expected an array of shape (n, {ndim}); got {arr.shape}.")
    return arr, squeezed


@dataclass
class FlowFitReport:
    """Small summary of the latest adaptive proposal refit."""

    num_samples: int
    history: Optional[dict]


class AdaptiveFlowProposal:
    """Bounded proposal distribution that adapts using ``pocomc.Flow``.

    The wrapped proposal remains an exact MH/RJ proposal because it exposes both
    ``rvs`` and ``logpdf`` in the original parameterization. Before enough
    training data are available, or when the optional flow backend is missing,
    the proposal falls back to the independent prior.
    """

    def __init__(
        self,
        parameter_distributions: Mapping[Any, Any],
        *,
        flow: str = "nsf6",
        train_config: Optional[Mapping[str, Any]] = None,
        min_training_samples: int = 256,
        eps: float = 1e-6,
        device: Optional[str] = None,
        periodic_parameters: Optional[Mapping[Any, float]] = None,
        mixture_prior_weight: float = 0.15,
        precondition: str = "identity",
        precondition_regularization: float = 1e-4,
    ):
        ordered_items = list(parameter_distributions.items())
        if not ordered_items:
            raise ValueError("AdaptiveFlowProposal needs at least one parameter distribution.")

        self.parameter_keys = [key for key, _ in ordered_items]
        self.parameter_distributions = [dist for _, dist in ordered_items]
        self.ndim = len(self.parameter_distributions)
        self.lower = np.array([_support_bounds(dist)[0] for dist in self.parameter_distributions], dtype=float)
        self.upper = np.array([_support_bounds(dist)[1] for dist in self.parameter_distributions], dtype=float)
        self.width = self.upper - self.lower
        if np.any(~np.isfinite(self.lower)) or np.any(~np.isfinite(self.upper)) or np.any(self.width <= 0.0):
            raise ValueError("All proposal bounds must be finite and strictly ordered.")

        self.flow_spec = flow
        self.train_config = dict(
            validation_split=0.25,
            epochs=500,
            batch_size=256,
            patience=max(self.ndim, 20),
            learning_rate=1e-3,
            annealing=True,
            clip_grad_norm=1.0,
            verbose=0,
        )
        if train_config is not None:
            self.train_config.update(train_config)

        self.min_training_samples = int(min_training_samples)
        self.eps = float(eps)
        self.device = device
        self.mixture_prior_weight = float(mixture_prior_weight)
        if not (0.0 <= self.mixture_prior_weight < 1.0):
            raise ValueError("mixture_prior_weight must satisfy 0 <= w < 1.")
        self.precondition = str(precondition)
        if self.precondition not in {"identity", "whiten"}:
            raise ValueError("precondition must be 'identity' or 'whiten'.")
        self.precondition_regularization = float(precondition_regularization)
        if self.precondition_regularization < 0.0:
            raise ValueError("precondition_regularization must be non-negative.")

        self.periodic_periods = np.zeros(self.ndim, dtype=float)
        self.periodic_centers = 0.5 * (self.lower + self.upper)
        if periodic_parameters is not None:
            index_lookup = {key: idx for idx, key in enumerate(self.parameter_keys)}
            for key, period in periodic_parameters.items():
                idx = index_lookup.get(key, key if isinstance(key, int) else None)
                if idx is None or not (0 <= int(idx) < self.ndim):
                    raise KeyError(f"Unknown periodic parameter key/index: {key!r}")
                period = float(period)
                if not np.isfinite(period) or period <= 0.0:
                    raise ValueError(f"Invalid periodic range for {key!r}: {period}")
                self.periodic_periods[int(idx)] = period

        self.fit_lower = self.lower.copy()
        self.fit_upper = self.upper.copy()
        self.fit_width = self.width.copy()
        self._flow_shift = np.zeros(self.ndim, dtype=float)
        self._flow_whitener = np.eye(self.ndim, dtype=float)
        self._flow_unwhitener = np.eye(self.ndim, dtype=float)
        self._flow_logabsdet = 0.0

        self._flow = None
        self._torch = None
        self.is_trained = False
        self.fit_calls = 0
        self.last_fit_report: Optional[FlowFitReport] = None

        if not flow_backend_available():
            warnings.warn(
                "Adaptive flow proposals will fall back to the prior because "
                "'pocomc' and/or 'torch' are not installed.",
                ImportWarning,
                stacklevel=2,
            )

    def _ensure_flow(self):
        if self._flow is not None:
            return self._flow

        torch, pc = require_flow_backend()
        self._torch = torch
        self._flow = pc.Flow(self.ndim, flow=self.flow_spec)
        if self.device is not None and hasattr(self._flow, "to"):
            self._flow = self._flow.to(self.device)
        return self._flow

    def _sample_prior(self, size: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        columns = []
        for dist in self.parameter_distributions:
            seed = None if random_state is None else int(rng.integers(0, 2**31 - 1))
            draw = np.asarray(dist.rvs(size=size, random_state=seed), dtype=float).reshape(size)
            columns.append(draw)
        return np.column_stack(columns)

    def _prior_logpdf(self, x: np.ndarray) -> np.ndarray:
        values = np.zeros(x.shape[0], dtype=float)
        for idx, dist in enumerate(self.parameter_distributions):
            contrib = np.asarray(dist.logpdf(x[:, idx]), dtype=float).reshape(-1)
            values += contrib
        return values

    def _update_fit_support(self, samples: np.ndarray) -> None:
        self.fit_lower = self.lower.copy()
        self.fit_upper = self.upper.copy()
        self.fit_width = self.width.copy()

        for idx, period in enumerate(self.periodic_periods):
            if period <= 0.0:
                continue

            base = self.lower[idx]
            phase = (samples[:, idx] - base) * (2.0 * np.pi / period)
            resultant = np.exp(1j * phase).mean()
            if not np.isfinite(resultant.real) or not np.isfinite(resultant.imag) or np.abs(resultant) < 1e-12:
                center = base + 0.5 * period
            else:
                center = base + (np.angle(resultant) % (2.0 * np.pi)) * period / (2.0 * np.pi)

            self.periodic_centers[idx] = center
            self.fit_lower[idx] = center - 0.5 * period
            self.fit_upper[idx] = center + 0.5 * period
            self.fit_width[idx] = period

    def _wrap_periodic_to_fit_interval(self, x: np.ndarray) -> np.ndarray:
        values = np.array(x, dtype=float, copy=True)
        for idx, period in enumerate(self.periodic_periods):
            if period <= 0.0:
                continue
            center = self.periodic_centers[idx]
            values[:, idx] = center + np.mod(values[:, idx] - center + 0.5 * period, period) - 0.5 * period
        return values

    def _wrap_periodic_to_support(self, x: np.ndarray) -> np.ndarray:
        values = np.array(x, dtype=float, copy=True)
        for idx, period in enumerate(self.periodic_periods):
            if period <= 0.0:
                continue
            values[:, idx] = self.lower[idx] + np.mod(values[:, idx] - self.lower[idx], period)
        return values

    def _update_preconditioner(self, transformed: np.ndarray) -> None:
        self._flow_shift = np.zeros(self.ndim, dtype=float)
        self._flow_whitener = np.eye(self.ndim, dtype=float)
        self._flow_unwhitener = np.eye(self.ndim, dtype=float)
        self._flow_logabsdet = 0.0

        if self.precondition != "whiten" or transformed.shape[0] < 2:
            return

        centered = np.asarray(transformed, dtype=float)
        shift = centered.mean(axis=0)
        centered = centered - shift
        covariance = np.cov(centered, rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance = covariance + self.precondition_regularization * np.eye(self.ndim)

        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return

        whitener = np.linalg.solve(chol, np.eye(self.ndim))
        self._flow_shift = shift
        self._flow_whitener = whitener
        self._flow_unwhitener = chol
        self._flow_logabsdet = float(np.linalg.slogdet(whitener)[1])

    def _to_preconditioned_space(self, transformed: np.ndarray) -> np.ndarray:
        return (np.asarray(transformed, dtype=float) - self._flow_shift) @ self._flow_whitener.T

    def _from_preconditioned_space(self, whitened: np.ndarray) -> np.ndarray:
        return self._flow_shift + np.asarray(whitened, dtype=float) @ self._flow_unwhitener.T

    def _to_flow_space(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._wrap_periodic_to_fit_interval(x)
        unit = (values - self.fit_lower) / self.fit_width
        unit = np.clip(unit, self.eps, 1.0 - self.eps)
        z = np.log(unit) - np.log1p(-unit)
        log_jac = (-np.log(self.fit_width) - np.log(unit) - np.log1p(-unit)).sum(axis=1)
        return z, log_jac

    def _from_flow_space(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unit = expit(z)
        x = self.fit_lower + self.fit_width * unit
        log_jac = (-np.log(self.fit_width) - np.log(unit) - np.log1p(-unit)).sum(axis=1)
        return self._wrap_periodic_to_support(x), log_jac

    def fit(self, samples: np.ndarray, weights: Optional[np.ndarray] = None) -> Optional[FlowFitReport]:
        """Train or update the internal normalizing flow."""
        samples, _ = _as_2d_array(samples, self.ndim)
        if samples.shape[0] < self.min_training_samples:
            return None
        if not flow_backend_available():
            return None

        flow = self._ensure_flow()
        torch = self._torch
        self._update_fit_support(samples)
        transformed, _ = self._to_flow_space(samples)
        self._update_preconditioner(transformed)
        transformed = self._to_preconditioned_space(transformed)

        x_tensor = torch.as_tensor(transformed, dtype=torch.float32, device=self.device)
        w_tensor = None
        if weights is not None:
            w_tensor = torch.as_tensor(
                np.asarray(weights, dtype=float),
                dtype=torch.float32,
                device=self.device,
            )

        history = flow.fit(x_tensor, weights=w_tensor, **self.train_config)
        self.is_trained = True
        self.fit_calls += 1
        self.last_fit_report = FlowFitReport(num_samples=samples.shape[0], history=history)
        return self.last_fit_report

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Draw proposals in the original bounded parameterization."""
        size = int(size)
        if size < 0:
            raise ValueError("size must be >= 0")
        if size == 0:
            return np.empty((0, self.ndim), dtype=float)

        if not self.is_trained or self._flow is None or self.mixture_prior_weight >= 1.0:
            return self._sample_prior(size=size, random_state=random_state)

        rng = np.random.default_rng(random_state)
        use_prior = rng.random(size) < self.mixture_prior_weight
        draws = np.empty((size, self.ndim), dtype=float)

        n_prior = int(use_prior.sum())
        if n_prior:
            draws[use_prior] = self._sample_prior(
                size=n_prior,
                random_state=None if random_state is None else int(rng.integers(0, 2**31 - 1)),
            )

        n_flow = size - n_prior
        if n_flow:
            if random_state is not None:
                self._torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
            samples_z, _ = self._flow.sample(n_flow)
            samples = np.asarray(samples_z.detach().cpu().numpy(), dtype=float)
            samples = self._from_preconditioned_space(samples)
            original, _ = self._from_flow_space(samples)
            draws[~use_prior] = original

        return draws

    def _flow_logpdf(self, x: np.ndarray) -> np.ndarray:
        inside = np.all((x >= self.lower) & (x <= self.upper), axis=1)
        logq = np.full(x.shape[0], -np.inf, dtype=float)
        if np.any(inside):
            transformed, log_jac = self._to_flow_space(x[inside])
            transformed = self._to_preconditioned_space(transformed)
            z_tensor = self._torch.as_tensor(
                transformed,
                dtype=self._torch.float32,
                device=self.device,
            )
            log_prob_z = np.asarray(self._flow.log_prob(z_tensor).detach().cpu().numpy(), dtype=float)
            logq[inside] = log_prob_z + self._flow_logabsdet + log_jac
        return logq

    def logpdf(self, x: np.ndarray | Sequence[float]) -> np.ndarray | float:
        """Evaluate the proposal density in the original parameterization."""
        values, squeezed = _as_2d_array(x, self.ndim)

        if not self.is_trained or self._flow is None:
            logq = self._prior_logpdf(values)
            return float(logq[0]) if squeezed else logq

        logq_flow = self._flow_logpdf(values)
        if self.mixture_prior_weight > 0.0:
            logq_prior = self._prior_logpdf(values)
            logq = logsumexp(
                np.column_stack(
                    [
                        np.log(self.mixture_prior_weight) + logq_prior,
                        np.log1p(-self.mixture_prior_weight) + logq_flow,
                    ]
                ),
                axis=1,
            )
        else:
            logq = logq_flow

        return float(logq[0]) if squeezed else logq


class FlowTrainingCallback:
    """Periodic Eryn ``update_fn`` that retrains branch proposal flows."""

    def __init__(
        self,
        branch_proposals: Mapping[str, AdaptiveFlowProposal],
        *,
        every: int = 25,
        stop_after: Optional[int] = None,
        temperature_indices: Optional[Sequence[int]] = (0,),
        max_samples: Optional[int] = 4096,
        buffer_size: Optional[int] = 16384,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.branch_proposals = dict(branch_proposals)
        self.every = int(every)
        self.stop_after = None if stop_after is None else int(stop_after)
        self.temperature_indices = None if temperature_indices is None else tuple(int(i) for i in temperature_indices)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.buffer_size = None if buffer_size is None else int(buffer_size)
        self.rng = np.random.default_rng(random_state)
        self.verbose = bool(verbose)
        self._frozen = False
        self._buffers = {branch_name: None for branch_name in self.branch_proposals}

    def _select_temperatures(self, ntemps: int) -> Sequence[int]:
        if self.temperature_indices is None:
            return range(ntemps)
        return [idx for idx in self.temperature_indices if 0 <= idx < ntemps]

    def _collect_branch_samples(self, last_sample, branch_name: str) -> Optional[np.ndarray]:
        coords = np.asarray(last_sample.branches_coords[branch_name], dtype=float)
        inds = np.asarray(last_sample.branches_inds[branch_name], dtype=bool)

        collected = []
        for temp_index in self._select_temperatures(coords.shape[0]):
            active = coords[temp_index][inds[temp_index]]
            if active.size:
                collected.append(active.reshape(-1, coords.shape[-1]))

        if not collected:
            return None

        return np.concatenate(collected, axis=0)

    def _merge_buffer(self, branch_name: str, samples: np.ndarray) -> np.ndarray:
        existing = self._buffers.get(branch_name)
        if existing is None:
            merged = np.asarray(samples, dtype=float)
        else:
            merged = np.concatenate([existing, np.asarray(samples, dtype=float)], axis=0)

        if self.buffer_size is not None and merged.shape[0] > self.buffer_size:
            keep = self.rng.choice(merged.shape[0], size=self.buffer_size, replace=False)
            merged = merged[keep]

        self._buffers[branch_name] = merged
        return merged

    def _subsample_for_fit(self, samples: np.ndarray) -> np.ndarray:
        if self.max_samples is None or samples.shape[0] <= self.max_samples:
            return samples
        keep = self.rng.choice(samples.shape[0], size=self.max_samples, replace=False)
        return samples[keep]

    def __call__(self, iteration: int, last_sample, sampler, *args, **kwargs) -> None:
        if self.every <= 0 or iteration <= 0 or iteration % self.every != 0:
            return

        if self.stop_after is not None and iteration > self.stop_after:
            self._frozen = True
            return
        if self._frozen:
            return

        for branch_name, proposal in self.branch_proposals.items():
            samples = self._collect_branch_samples(last_sample, branch_name)
            if samples is None:
                continue
            buffered = self._merge_buffer(branch_name, samples)
            fit_samples = self._subsample_for_fit(buffered)
            report = proposal.fit(fit_samples)
            if self.verbose and report is not None:
                print(
                    f"[flow-update] iteration={iteration} branch={branch_name} "
                    f"buffer={buffered.shape[0]} samples={report.num_samples} "
                    f"fit_calls={proposal.fit_calls}"
                )


class ContextAwareBirthRJMove:
    """RJ move that uses a dedicated, state-aware birth proposal per branch.

    Each proposal entry must provide ``rvs`` and ``logpdf`` methods. Proposals
    may optionally implement ``build_context(active_points)``, where
    ``active_points`` is the array of currently active leaves for one walker.
    The returned context is then passed back to ``rvs``/``logpdf`` when those
    methods accept a ``context=...`` keyword.
    """

    def __new__(cls, birth_generate_dist, *args, **kwargs):
        try:
            from eryn.moves.rj import ReversibleJumpMove
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError("Eryn is required to build context-aware RJ moves.") from exc

        class _ContextAwareBirthRJMove(ReversibleJumpMove):
            def __init__(self, birth_generate_dist, *inner_args, **inner_kwargs):
                self.birth_generate_dist = dict(birth_generate_dist)
                super().__init__(*inner_args, **inner_kwargs)

            def get_model_change_proposal(self, inds, random, nleaves_min, nleaves_max):
                ntemps, nwalkers, _ = inds.shape
                nleaves = inds.sum(axis=-1)

                if self.fix_change is None:
                    change = random.choice([-1, +1], size=nleaves.shape)
                else:
                    change = np.full(nleaves.shape, self.fix_change)

                change = (
                    change * ((nleaves != nleaves_min) & (nleaves != nleaves_max))
                    + (+1) * (nleaves == nleaves_min)
                    + (-1) * (nleaves == nleaves_max)
                )

                inds_for_change = {
                    "+1": np.zeros((np.sum(change == +1), 3), dtype=int),
                    "-1": np.zeros((np.sum(change == -1), 3), dtype=int),
                }

                increase_i = 0
                decrease_i = 0
                for t in range(ntemps):
                    for w in range(nwalkers):
                        change_tw = change[t][w]
                        inds_tw = inds[t][w]
                        if change_tw == +1:
                            inds_false = np.where(inds_tw == False)[0]
                            ind_change = random.choice(inds_false)
                            inds_for_change["+1"][increase_i] = np.array([t, w, ind_change], dtype=int)
                            increase_i += 1
                        elif change_tw == -1:
                            inds_true = np.where(inds_tw == True)[0]
                            ind_change = random.choice(inds_true)
                            if inds_for_change["-1"].shape[0] > 0:
                                inds_for_change["-1"][decrease_i] = np.array([t, w, ind_change], dtype=int)
                                decrease_i += 1
                return inds_for_change

            def get_proposal(
                self, all_coords, all_inds, nleaves_min_all, nleaves_max_all, random, **kwargs
            ):
                q = {}
                new_inds = {}
                all_inds_for_change = {}

                assert len(nleaves_min_all)
                assert len(all_coords.keys()) == len(nleaves_max_all.keys())

                for name, inds in all_inds.items():
                    nleaves_max = nleaves_max_all[name]
                    nleaves_min = nleaves_min_all[name]
                    if nleaves_min == nleaves_max:
                        continue
                    if nleaves_min > nleaves_max:
                        raise ValueError("nleaves_min is greater than nleaves_max. Not allowed.")
                    all_inds_for_change[name] = self.get_model_change_proposal(
                        inds, random, nleaves_min, nleaves_max
                    )

                for i, (name, coords, inds) in enumerate(
                    zip(all_coords.keys(), all_coords.values(), all_inds.values())
                ):
                    ntemps, nwalkers, _, ndim = coords.shape
                    new_inds[name] = inds.copy()
                    q[name] = coords.copy()

                    if i == 0:
                        factors = np.zeros((ntemps, nwalkers))

                    if name not in all_inds_for_change:
                        continue

                    proposal = self.birth_generate_dist[name]
                    inds_for_change = all_inds_for_change[name]

                    for t, w, leaf in inds_for_change["-1"]:
                        removed = np.asarray(q[name][t, w, leaf], dtype=float)
                        new_inds[name][t, w, leaf] = False
                        active_after_death = np.asarray(
                            q[name][t, w][new_inds[name][t, w]], dtype=float
                        ).reshape(-1, ndim)
                        context = _build_context(proposal, active_after_death)
                        factors[t, w] += float(_proposal_logpdf(proposal, removed, context=context))

                    for t, w, leaf in inds_for_change["+1"]:
                        active_before_birth = np.asarray(
                            q[name][t, w][new_inds[name][t, w]], dtype=float
                        ).reshape(-1, ndim)
                        context = _build_context(proposal, active_before_birth)
                        proposal_draw = _proposal_rvs(proposal, size=1, context=context, random=random).reshape(
                            1, ndim
                        )[0]
                        q[name][t, w, leaf] = proposal_draw
                        new_inds[name][t, w, leaf] = True
                        factors[t, w] += -float(
                            _proposal_logpdf(proposal, proposal_draw, context=context)
                        )

                return q, new_inds, factors

        return _ContextAwareBirthRJMove(birth_generate_dist, *args, **kwargs)


def _as_prob_dist_container(proposal):
    try:
        from eryn.prior import ProbDistContainer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("Eryn is required to build flow proposal moves.") from exc

    if isinstance(proposal, ProbDistContainer):
        return proposal

    if isinstance(proposal, AdaptiveFlowProposal):
        return ProbDistContainer({tuple(range(proposal.ndim)): _JointProposalAdapter(proposal)})

    raise TypeError(
        "Flow proposal entries must be AdaptiveFlowProposal or eryn.prior.ProbDistContainer instances."
    )


class _BilbyDistAdapter:
    """Wrap a bilby prior as a bounded dist for :class:`AdaptiveFlowProposal`."""

    def __init__(self, prior):
        self._p = prior
        self.minimum = float(prior.minimum)
        self.maximum = float(prior.maximum)

    def rvs(self, size=1, random_state=None):
        return np.asarray(self._p.sample(size), dtype=float).reshape(size)

    def logpdf(self, x):
        return np.asarray(self._p.ln_prob(np.asarray(x, dtype=float)), dtype=float)


def build_pe_flow_proposal(priors_ordered, periodic_indices=None, device=None,
                           min_training_samples=512, **flow_kwargs):
    """Adaptive-flow proposal over a standard PE parameter vector.

    ``priors_ordered`` is the list of bilby priors in sampling order (science +
    noise-shape parameters, i.e. exactly the vector the likelihood receives).
    The returned :class:`AdaptiveFlowProposal` is an exact independence
    proposal: it exposes ``rvs``/``logpdf`` and falls back to the prior until
    the flow has trained on enough chain history (via
    :class:`FlowTrainingCallback`).

    Wire into Eryn through ``hyperwave.inference.make_flow_distribution_move``
    mixed with the default stretch move, e.g. 30% flow / 70% stretch.
    """
    dists = {i: _BilbyDistAdapter(p) for i, p in enumerate(priors_ordered)}
    periodic = None
    if periodic_indices:
        periodic = {int(i): float(priors_ordered[i].maximum - priors_ordered[i].minimum)
                    for i in periodic_indices}
    return AdaptiveFlowProposal(dists, periodic_parameters=periodic, device=device,
                                min_training_samples=min_training_samples, **flow_kwargs)


def make_flow_distribution_move(branch_proposals: Mapping[str, AdaptiveFlowProposal], **kwargs):
    """Build Eryn's in-model distribution proposal move from adaptive flows."""
    try:
        from eryn.moves import DistributionGenerate
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("Eryn is required to build flow proposal moves.") from exc
    return DistributionGenerate(
        generate_dist={name: _as_prob_dist_container(proposal) for name, proposal in branch_proposals.items()},
        **kwargs,
    )


def make_flow_rj_move(branch_proposals: Mapping[str, AdaptiveFlowProposal], **kwargs):
    """Build Eryn's RJ distribution proposal move from adaptive flows."""
    try:
        from eryn.moves import DistributionGenerateRJ
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("Eryn is required to build flow RJ moves.") from exc
    return DistributionGenerateRJ(
        generate_dist={name: _as_prob_dist_container(proposal) for name, proposal in branch_proposals.items()},
        **kwargs,
    )


__all__ = [
    "AdaptiveFlowProposal",
    "build_pe_flow_proposal",
    "ContextAwareBirthRJMove",
    "FlowFitReport",
    "FlowTrainingCallback",
    "flow_backend_available",
    "make_flow_distribution_move",
    "make_flow_rj_move",
    "require_flow_backend",
]
