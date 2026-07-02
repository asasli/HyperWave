"""LISA detector helpers."""

from .aet import (
    AET_CHANNELS,
    LISAAETTemplate,
    build_lisa_aet_likelihood,
    lisatools_available,
    prepare_lisa_aet_inputs,
)
from .noise import LISANoise, lisa_aet_psd

# UCBTemplate needs gbgpu (LISA env only); keep the import optional so the
# package still imports on machines without the LISA stack.
try:
    from .ucb import UCBTemplate, UCB_PARAMETER_NAMES, UCB_PERIODIC
except ImportError:  # pragma: no cover - gbgpu not installed
    UCBTemplate = None
    UCB_PARAMETER_NAMES = UCB_PERIODIC = None

__all__ = [
    "AET_CHANNELS",
    "LISAAETTemplate",
    "build_lisa_aet_likelihood",
    "lisatools_available",
    "prepare_lisa_aet_inputs",
    "LISANoise",
    "lisa_aet_psd",
    "UCBTemplate",
    "UCB_PARAMETER_NAMES",
    "UCB_PERIODIC",
]
