# Detector-Dependent Noise Parameters

## Goal

Allow each detector to use its own hyperbolic noise-distribution parameters.

## Assumptions

- Work happens on `codex/detector_dependent_noise`, rebased on
  `codex/calibration_uncertainties` at `b8d917f` (`add response curve marg`).
- The `codex/calibration_uncertainties` worktree is not modified.
- HyperWave keeps positional, batched likelihood parameters rather than bilby
  mutable likelihood-parameter dictionaries.
- Heterodyned hyperbolic is a separate approximation path and is not extended
  in this change.

## Decision

Use an opt-in `detector_dependent_noise` flag. When enabled, alpha/delta noise
parameters are detector-major in the same order as `ifos_list`, and the
likelihood sums factorized per-detector hyperbolic log densities.

## Approaches Checked

- Bilby-style named parameters such as `alpha_H1_0` and `delta_L1_1`.
  Rejected because HyperWave likelihood calls use positional batched arrays.
- A separate likelihood object per detector, then sum outside the class.
  Rejected because it pushes detector bookkeeping into callers and duplicates
  waveform generation.
- Detector-major positional blocks inside the existing likelihood classes.
  Chosen because it keeps the HyperWave API shape and makes the likelihood
  equal to a sum of single-detector hyperbolic likelihoods.

## Tasks

- [x] Add detector-dependent hyperbolic noise parameters to `GWLikelihoods`.
- [x] Add the same opt-in path to data-only `LogLike`.
- [x] Keep existing shared-network hyperbolic behaviour as the default.
- [x] Document the positional parameter layout for `ddims=False` and
  `ddims=True`.
- [x] Add focused tests against the sum of single-detector likelihoods.
- [x] Check that shared and detector-dependent hyperbolic paths still use the
  batched template call.
- [x] Check that detector-dependent hyperbolic works with calibration
  marginalization.
- [x] Check the GPU-requested path. If CuPy/CUDA is absent, the test verifies
  the existing NumPy fallback and the same batched call path.
- [x] Add a mixed-detector model mask so selected detectors can stay Gaussian
  while others use detector-dependent hyperbolic noise.
- [x] Add a fast example with Gaussian H1 noise and heavy-tailed L1 noise,
  recovered with H1 Gaussian and L1 hyperbolic.

## Verification

- `python -m compileall -q src/hyperwave/likelihoods/distributions_fd.py src/hyperwave/likelihoods/gwparallel.py tests/test_detector_noise_parameters.py`
- `PYTHONPATH=src pytest tests/test_detector_noise_parameters.py -q`: 12 passed.
- `PYTHONPATH=src pytest --ignore=tests/test_pp.py`: 43 passed, 1 skipped.
- `PYTHONPATH=src mkdocs build --strict`: passed. Material for MkDocs printed
  its upstream MkDocs 2.0 warning, but the strict build completed.
- `PYTHONPATH=src python examples/noise/mixed_detector_noise_recovery.py --no-plot`:
  true amplitude 1.0000, all-Gaussian recovery 1.0614, mixed recovery 1.0185.
- `PYTHONPATH=src pytest tests/test_pp.py -q` still fails on
  `ModuleNotFoundError: No module named 'hyperwave.validation'` on the untouched
  calibration checkout, so this is a pre-existing calibration-branch collection
  issue. The `repo_scan` bug-scan commit addresses this separately and is not
  included in this branch.

## Left Out

- Heterodyned hyperbolic likelihoods are not extended here.
- There is no named-parameter adapter. Callers must order noise parameters
  according to the documented detector-major layout.
