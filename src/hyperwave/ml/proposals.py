"""Eryn-compatible birth proposal driven by a trained wavelet flow.

Wraps :class:`hyperwave.ml.flow_birth.FlowBirthModel` so it can be passed to
:class:`eryn.moves.DistributionGenerateRJ` as the per-leaf birth distribution.
The Hastings ratio in DistributionGenerateRJ relies on both ``rvs`` (to sample
a candidate) and ``logpdf`` (to score it) — both are honest evaluations of the
flow's conditional density, so the resulting RJMCMC remains exact-MH.

Usage sketch (in the example script):

    from hyperwave.ml.proposals import WaveletFlowDistribution, load_flow_model
    model = load_flow_model("results/wavelet_flow/.../flow_birth.pt")
    birth = WaveletFlowDistribution(model)
    rj_move = DistributionGenerateRJ({"model_0": birth}, ...)
    # before each call to eryn's step:
    birth.set_residual(current_residual_fd)   # (n_ifo, n_freq) complex
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .flow_birth import (
    _WAVELET_DIM,
    FlowBirthConfig,
    FlowBirthModel,
    _tf_image,
    denormalize_wavelet,
    normalize_wavelet,
)

try:
    import torch
    from eryn.prior import ProbDistContainer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "hyperwave.ml.proposals needs torch + eryn. "
        "Install with `pip install hyperwave[flows]`."
    ) from exc


def load_flow_model(path: str, device: str = "cpu") -> FlowBirthModel:
    """Restore a trained FlowBirthModel from a `train_wavelet_flow.py` checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = FlowBirthConfig(**ckpt["cfg"])
    model = FlowBirthModel(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


class WaveletFlowDistribution(ProbDistContainer):
    """ProbDistContainer-compatible wrapper around FlowBirthModel.

    Eryn calls ``rvs(size=N)`` when birthing N candidate wavelets and
    ``logpdf(x)`` when scoring them in the Hastings ratio. We translate both
    onto the flow's normalised parameter space, conditioning every call on the
    *current residual* (passed in via :meth:`set_residual`).

    Falling back to the flow's prior — a uniform draw in normalised space when
    no residual has been set — keeps the proposal usable during the warm-up
    phase when there is no residual yet (the Hastings ratio still goes through
    because logpdf is consistent with rvs).
    """

    def __init__(self, model: FlowBirthModel, *, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self._cached_ctx: Optional[torch.Tensor] = None
        self.ndim = _WAVELET_DIM
        # Mimic ProbDistContainer attributes for the moves that inspect them.
        self.priors_in = {tuple(range(_WAVELET_DIM)): self}
        self.priors = [[np.arange(_WAVELET_DIM), self]]
        self.has_ints = True
        self.has_strings = False
        self.use_cupy = False
        self.return_gpu = False

    # -- context management ---------------------------------------------------
    @torch.no_grad()
    def set_residual(self, residual_fd: np.ndarray):
        """Cache the encoded TF context for the *current* residual.

        ``residual_fd`` is the same ``(n_ifo, n_freq)`` complex whitened FD data
        the flow was trained on (data minus the current model wavelets).
        """
        img = _tf_image(residual_fd, n_t=self.model.cfg.tf_t, n_f=self.model.cfg.tf_f)
        x = _to_tensor(img[None].astype(np.float32), self.device)
        self._cached_ctx = self.model.context(x)  # (1, ctx_dim)

    def clear_residual(self):
        self._cached_ctx = None

    # -- sampling -------------------------------------------------------------
    @torch.no_grad()
    def rvs(self, size=1, keys=None):
        if isinstance(size, (tuple, list)):
            n = int(np.prod(size))
            shape = tuple(size) + (_WAVELET_DIM,)
        else:
            n = int(size)
            shape = (n, _WAVELET_DIM)

        if self._cached_ctx is None:
            # Uniform fallback in normalised space.
            z = np.random.uniform(-1.0, 1.0, size=(n, _WAVELET_DIM)).astype(np.float32)
        else:
            ctx = self._cached_ctx.expand(n, -1)
            z = self.model.flow(ctx).sample().cpu().numpy().astype(np.float32)

        # Denormalize: [-1, 1]^5 -> physical (t0, f0, Q, snr, phi0).
        cfg = self.model.cfg
        phys = denormalize_wavelet(z, duration=cfg.duration, f_min=cfg.f_min,
                                   f_max=cfg.f_max, q_bounds=cfg.q_bounds,
                                   snr_bounds=cfg.snr_bounds)
        return phys.reshape(shape)

    @torch.no_grad()
    def logpdf(self, x, keys=None):
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        cfg = self.model.cfg
        z_np = normalize_wavelet(x, duration=cfg.duration, f_min=cfg.f_min,
                                 f_max=cfg.f_max, q_bounds=cfg.q_bounds,
                                 snr_bounds=cfg.snr_bounds)
        # Jacobian of normalize_wavelet: physical -> normalised.
        # d z0 / d t0 = 2 / T
        # d z1 / d f0 = 2 / (f0 * log(fmax/fmin))
        # d z2 / d Q  = 2 / (Q_hi - Q_lo)
        # d z3 / d snr = 2 / ((1 + snr) * log1p(snr_hi))
        # d z4 / d phi0 = -sin(phi0)
        T = cfg.duration
        log_fmax_fmin = np.log(cfg.f_max) - np.log(cfg.f_min)
        dQ = cfg.q_bounds[1] - cfg.q_bounds[0]
        log1p_snrhi = np.log1p(cfg.snr_bounds[1])
        log_jac = (
            np.log(2.0 / T)
            + np.log(2.0 / (np.clip(x[:, 1], cfg.f_min, cfg.f_max) * log_fmax_fmin))
            + np.log(2.0 / dQ)
            + np.log(2.0 / ((1.0 + np.clip(x[:, 3], *cfg.snr_bounds)) * log1p_snrhi))
            + np.log(np.abs(np.sin(x[:, 4])) + 1e-12)
        )
        if self._cached_ctx is None:
            # Uniform fallback: -log(2)^5 minus Jacobian.
            base = -_WAVELET_DIM * np.log(2.0)
            return base - log_jac
        ctx = self._cached_ctx.expand(z_np.shape[0], -1)
        z = _to_tensor(z_np, self.device)
        logp_z = self.model.flow(ctx).log_prob(z).cpu().numpy().astype(np.float64)
        return logp_z - log_jac

    # -- ProbDistContainer compatibility shims --------------------------------
    def __len__(self):
        return _WAVELET_DIM


__all__ = ["WaveletFlowDistribution", "load_flow_model"]
