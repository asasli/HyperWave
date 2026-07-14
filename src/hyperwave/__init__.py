"""HyperWave: hyperbolic likelihood tools for gravitational-wave data analysis."""

from importlib import metadata

__version__ = "2.0.0"

try:  # Prefer distribution version if installed
    __version__ = metadata.version("hyperwave")
except metadata.PackageNotFoundError:
    pass

from . import detectors, plots
from .detectors import (
    Detector,
    Interferometer,
    InterferometerList,
    PowerSpectralDensity,
    StrainData,
    Template,
    WaveletTemplate,
)
from .detectors.lvk import GW, DetectorNoise
from .inference import (
    AdaptiveFlowProposal,
    ContextAwareBirthRJMove,
    DataInference,
    FlowTrainingCallback,
    InferenceRunner,
    LVKinference,
    SNRPrior,
    build_wavelet_priors,
    flow_backend_available,
    make_flow_distribution_move,
    make_flow_rj_move,
)
from .likelihoods import (
    GWLikelihoods,
    HeterodyneLikelihood,
    LogLike,
    WaveletLikelihood,
    gpu_backend_available,
    loglike,
)
from .ml4gw import ml4gw_available, torch_cuda_available
from .result import Result
from .utils import load_object, save_object
from . import validation

__all__ = [
    "__version__",
    # io / results
    "load_object",
    "save_object",
    "Result",
    # inference (bilby priors retained here)
    "LVKinference",
    "InferenceRunner",
    "DataInference",
    "AdaptiveFlowProposal",
    "ContextAwareBirthRJMove",
    "FlowTrainingCallback",
    "flow_backend_available",
    "make_flow_distribution_move",
    "make_flow_rj_move",
    # likelihoods
    "GWLikelihoods",
    "HeterodyneLikelihood",
    "WaveletLikelihood",
    "LogLike",
    "loglike",
    "gpu_backend_available",
    # detectors / waveforms (bilby-free)
    "Detector",
    "PowerSpectralDensity",
    "StrainData",
    "Interferometer",
    "InterferometerList",
    "Template",
    "WaveletTemplate",
    "DetectorNoise",
    "GW",
    # wavelet reconstruction
    "build_wavelet_priors",
    "SNRPrior",
    # acceleration probes
    "ml4gw_available",
    "torch_cuda_available",
    # subpackages
    "detectors",
    "plots",
    "validation",
]
