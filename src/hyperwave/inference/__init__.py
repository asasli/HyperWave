"""Inference helpers for HyperWave."""

from .convergence import WaveletConvergenceStopping
from .priors import calibration_node_priors, per_detector_noise_priors
from .sampling import DataInference, LVKinference

# Optional submodules — re-exported only if their source files are present.
# (wavelet_proposals, fastjumps, flow_proposals, wavelet_priors are pending upload.)
try:
    from .wavelet_proposals import (
        DataInformedMarginal,
        MatchedFilterBirth,
        WaveletFisherMove,
        WaveletHalfCycleMove,
        WaveletSkyRingMove,
        build_flow_proposal,
        build_guided_birth,
        build_mf_birth,
        guided_initial_wavelets,
    )
except ImportError:
    DataInformedMarginal = MatchedFilterBirth = None
    WaveletFisherMove = WaveletHalfCycleMove = WaveletSkyRingMove = None
    build_flow_proposal = build_guided_birth = build_mf_birth = None
    guided_initial_wavelets = None

try:
    from .fastjumps import FastJumpInference, FastJumpModel, FastJumpResult, PocoFastJumps
except ImportError:
    FastJumpInference = FastJumpModel = FastJumpResult = PocoFastJumps = None

try:
    from .flow_proposals import (
        AdaptiveFlowProposal,
        ContextAwareBirthRJMove,
        FlowFitReport,
        FlowTrainingCallback,
        flow_backend_available,
        make_flow_distribution_move,
        make_flow_rj_move,
    )
except ImportError:
    AdaptiveFlowProposal = ContextAwareBirthRJMove = None
    FlowFitReport = FlowTrainingCallback = None
    make_flow_distribution_move = make_flow_rj_move = None

    def flow_backend_available():
        return False

try:
    from .wavelet_priors import CosinePrior, SNRPrior, build_wavelet_priors
except ImportError:
    CosinePrior = SNRPrior = build_wavelet_priors = None

InferenceRunner = LVKinference

__all__ = [
    "LVKinference",
    "InferenceRunner",
    "DataInference",
    "calibration_node_priors",
    "per_detector_noise_priors",
    "FastJumpInference",
    "FastJumpModel",
    "FastJumpResult",
    "PocoFastJumps",
    "AdaptiveFlowProposal",
    "ContextAwareBirthRJMove",
    "FlowFitReport",
    "FlowTrainingCallback",
    "flow_backend_available",
    "make_flow_distribution_move",
    "make_flow_rj_move",
    # wavelet reconstruction
    "build_wavelet_priors",
    "SNRPrior",
    "CosinePrior",
    "WaveletConvergenceStopping",
    "build_guided_birth",
    "build_mf_birth",
    "WaveletFisherMove",
    "WaveletHalfCycleMove",
    "WaveletSkyRingMove",
    "build_flow_proposal",
    "guided_initial_wavelets",
    "DataInformedMarginal",
    "MatchedFilterBirth",
]
