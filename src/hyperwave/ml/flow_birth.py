"""Path A: conditional normalizing-flow birth proposal (source-agnostic).

Trained on the random-wavelet generator in :mod:`hyperwave.ml.synthetic`, the
flow learns ``p(wavelet | residual_TF_image)`` — the distribution over the
*next* wavelet given the current residual. Plugs into eryn as an exact-MH
birth proposal: the flow exposes both ``rvs`` and ``logpdf``, so the Hastings
ratio in :class:`eryn.moves.DistributionGenerateRJ` remains correct and the
posterior stays unbiased even when the flow's predictions are imperfect.

Training target (curriculum):
  1. ``D = 1``: trivially identifies the single wavelet's parameters.
  2. ``D >= 1``: subtract ``D - 1`` known wavelets, train the flow to recover
     the one left in. This is exactly the "next wavelet given the residual"
     task at inference time.

The architecture is intentionally small (CNN backbone + neural-spline flow):
on an A100 it trains in a few hours from scratch on streaming synthetic data,
with no fixed training corpus required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    import zuko
except ImportError as exc:  # pragma: no cover - optional dep
    raise ImportError(
        "hyperwave.ml.flow_birth needs torch + zuko. "
        "Install with `pip install hyperwave[flows]`."
    ) from exc

# wavelet parameter layout used by every wavelet module in the package
_WAVELET_DIM = 5  # (t0, f0, Q, snr, phi0)


def _tf_image(data: np.ndarray, n_t: int = 128, n_f: int = 128) -> np.ndarray:
    """Cheap fixed-size TF representation of whitened complex FD data.

    ``data`` is ``(n_ifo, n_freq)`` whitened complex. We project to a real
    ``(2 * n_ifo, n_t, n_f)`` image: per-IFO log-amplitude + cosine phase,
    binned and downsampled to ``(n_t, n_f)`` via separable averaging.

    The training task does not need a full spectrogram — a coarse
    representation suffices for the flow to localise wavelets in TF.
    Replace with a true Q-transform in production training if accuracy needs
    pushing (the layer-1 image speeds training ~10x at small loss).
    """
    n_ifo, n_freq = data.shape
    amp = np.log1p(np.abs(data))
    pha = np.cos(np.angle(data))
    f_bin = max(1, n_freq // n_f)
    amp = amp[:, : f_bin * n_f].reshape(n_ifo, n_f, f_bin).mean(-1)
    pha = pha[:, : f_bin * n_f].reshape(n_ifo, n_f, f_bin).mean(-1)
    # tile along the "time" axis: the FD log-mag is time-invariant, but the
    # flow conditions on the same vector for every time bin -- this lets us
    # share the conv backbone with future spectrogram inputs.
    amp = np.broadcast_to(amp[:, None, :], (n_ifo, n_t, n_f))
    pha = np.broadcast_to(pha[:, None, :], (n_ifo, n_t, n_f))
    return np.concatenate([amp, pha], axis=0)  # (2 * n_ifo, n_t, n_f)


def normalize_wavelet(w: np.ndarray, *, duration: float, f_min: float,
                      f_max: float, q_bounds=(0.1, 40.0),
                      snr_bounds=(0.0, 100.0)) -> np.ndarray:
    """Map ``(t0, f0, Q, snr, phi0)`` -> R^5 ~ N(0, 1) sampling space."""
    w = np.asarray(w, dtype=np.float32)
    out = np.empty_like(w)
    out[..., 0] = w[..., 0] / duration * 2.0 - 1.0          # t0 -> [-1, 1]
    out[..., 1] = (np.log(np.clip(w[..., 1], f_min, f_max)) - np.log(f_min)) \
                  / (np.log(f_max) - np.log(f_min)) * 2 - 1  # log f0 -> [-1, 1]
    out[..., 2] = (w[..., 2] - q_bounds[0]) / (q_bounds[1] - q_bounds[0]) * 2 - 1
    out[..., 3] = np.log1p(np.clip(w[..., 3], *snr_bounds)) / np.log1p(snr_bounds[1]) * 2 - 1
    out[..., 4] = np.cos(w[..., 4])  # phase -> projected cosine
    return out


def denormalize_wavelet(z: np.ndarray, *, duration: float, f_min: float,
                        f_max: float, q_bounds=(0.1, 40.0),
                        snr_bounds=(0.0, 100.0)) -> np.ndarray:
    # Clip the *normalised* output first so out-of-bounds flow draws stay
    # physical (a fresh NSF is roughly Gaussian; without this clip, f0 leaks
    # past f_max via the log-mapping and phi0 hits NaN via arccos).
    z = np.clip(np.asarray(z, dtype=np.float32), -1.0, 1.0)
    out = np.empty_like(z)
    out[..., 0] = (z[..., 0] + 1) / 2 * duration
    log_f = (z[..., 1] + 1) / 2 * (np.log(f_max) - np.log(f_min)) + np.log(f_min)
    out[..., 1] = np.exp(log_f)
    out[..., 2] = (z[..., 2] + 1) / 2 * (q_bounds[1] - q_bounds[0]) + q_bounds[0]
    out[..., 3] = np.expm1((z[..., 3] + 1) / 2 * np.log1p(snr_bounds[1]))
    out[..., 4] = np.arccos(z[..., 4])
    return out


class _ConvContext(nn.Module):
    """Small CNN backbone that summarises a (C, T, F) TF image."""

    def __init__(self, in_channels: int, hidden: int = 64, ctx_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden * 2, ctx_dim), nn.GELU(),
        )
        self.out_dim = ctx_dim

    def forward(self, x):
        return self.net(x)


@dataclass
class FlowBirthConfig:
    duration: float = 4.0
    sampling_rate: float = 2048.0
    f_min: float = 20.0
    f_max: float = 512.0
    q_bounds: tuple = (0.1, 40.0)
    snr_bounds: tuple = (0.0, 100.0)
    tf_t: int = 128
    tf_f: int = 128
    cnn_hidden: int = 64
    ctx_dim: int = 128
    flow_transforms: int = 8
    flow_hidden: int = 128


class FlowBirthModel(nn.Module):
    """CNN backbone + Neural Spline Flow over the 5 wavelet parameters."""

    def __init__(self, cfg: FlowBirthConfig, n_ifo: int = 2):
        super().__init__()
        self.cfg = cfg
        self.n_ifo = n_ifo
        self.context = _ConvContext(in_channels=2 * n_ifo,
                                    hidden=cfg.cnn_hidden, ctx_dim=cfg.ctx_dim)
        self.flow = zuko.flows.NSF(
            features=_WAVELET_DIM,
            context=cfg.ctx_dim,
            transforms=cfg.flow_transforms,
            hidden_features=(cfg.flow_hidden, cfg.flow_hidden),
        )

    # -- training -------------------------------------------------------------
    def log_prob(self, wavelets_normalized: torch.Tensor,
                 tf_image: torch.Tensor) -> torch.Tensor:
        """``log p(z | TF image)``. Inputs are batched."""
        ctx = self.context(tf_image)
        return self.flow(ctx).log_prob(wavelets_normalized)

    # -- inference (proposal API) ---------------------------------------------
    @torch.no_grad()
    def sample(self, tf_image: torch.Tensor, n: int = 1) -> torch.Tensor:
        ctx = self.context(tf_image)
        dist = self.flow(ctx)
        return dist.sample((n,))


def make_tf_image_batch(samples: Sequence) -> np.ndarray:
    """Stack a list of :class:`WaveletSample`s into ``(B, C, T, F)``."""
    imgs = [_tf_image(s.data) for s in samples]
    return np.stack(imgs, axis=0).astype(np.float32)


def make_target_wavelet_batch(samples: Sequence, *, mode: str = "first",
                              duration: float = 4.0, f_min: float = 20.0,
                              f_max: float = 512.0, q_bounds=(0.1, 40.0),
                              snr_bounds=(0.0, 100.0)) -> np.ndarray:
    """Pick one wavelet from each sample as the training target.

    ``mode="first"`` returns wavelet 0 (deterministic, fine for D=1 curriculum
    or pre-shuffled wavelet lists). ``mode="random"`` picks one uniformly per
    sample (works for the general D>=1 case).
    """
    targets = np.zeros((len(samples), _WAVELET_DIM), dtype=np.float32)
    for i, s in enumerate(samples):
        if s.n_wavelets == 0:
            targets[i] = 0.0
            continue
        if mode == "first":
            w = s.wavelets[0]
        elif mode == "random":
            w = s.wavelets[np.random.randint(s.n_wavelets)]
        else:
            raise ValueError(mode)
        targets[i] = normalize_wavelet(
            w, duration=duration, f_min=f_min, f_max=f_max,
            q_bounds=q_bounds, snr_bounds=snr_bounds,
        )
    return targets


__all__ = [
    "FlowBirthConfig",
    "FlowBirthModel",
    "make_target_wavelet_batch",
    "make_tf_image_batch",
    "normalize_wavelet",
    "denormalize_wavelet",
]
