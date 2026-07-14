"""Machine-learning wavelet reconstruction (source-agnostic).

Two paths sharing one synthetic data generator:

* **Path A** (``flow_birth``): a conditional normalizing flow trained on random
  Morlet-Gabor superpositions, used as an exact-MH birth proposal inside the
  existing eryn RJMCMC. Faster mixing; posteriors stay exact.
* **Path B** (``amortized``, in progress): a DETR-style set transformer that
  predicts the full posterior over ``(D, {wavelets})`` in one forward pass.
  Milliseconds per inference; validated against Path A's exact posterior.

Both are source-agnostic by construction: the training corpus is random wavelet
superpositions with random sky, SNR and noise realisations, so the network
learns the inverse of the wavelet generative model rather than any specific
astrophysical source. Any signal that admits a Morlet-Gabor decomposition can
be reconstructed at inference time.
"""

from .synthetic import RandomWaveletDataset, WaveletSample, generate_synthetic_signal

__all__ = ["RandomWaveletDataset", "WaveletSample", "generate_synthetic_signal"]
