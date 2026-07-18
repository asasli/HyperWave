# GPU acceleration

HyperWave's GPU story has two independent layers; each falls back to CPU
automatically when no device is present.

## 1. Likelihood algebra (CuPy)

Pass `gpu=True` to any likelihood and the residual/likelihood arithmetic runs
on CuPy:

```python
like = GWLikelihoods(..., gpu=True)     # falls back to NumPy if CuPy/GPU absent
```

Install with `pip install "hyperwave[gpu]"` (CuPy for CUDA 12; use
`cupy-cuda11x` on CUDA 11 systems).

## 2. Waveform generation

### LVK: ml4gw (Torch)

The `ml4gw` backend generates whole batches of IMRPhenomD / IMRPhenomPv2 /
TaylorF2 polarisations in one Torch call:

```python
template = GW(noise, approximant="IMRPhenomPv2",
              waveform_backend="ml4gw", gpu=True, torch_device="cuda")
```

!!! note "Convention status"
    HyperWave corrects ml4gw's coalescence-time and phase conventions to match
    the LAL backend exactly (overlap 1.0) for **zero and aligned-positive
    spins**. Two known residual issues are tracked in `TODO.md`: a constant
    phase offset for anti-aligned/precessing systems (couples to `phi_jl`) and
    a high-frequency amplitude rolloff above ~512 Hz. PE in the
    zero/aligned-spin configuration is verified unbiased.

### LISA: bbhx and GBGPU

Both engines are natively batched and integrate through the same
`make_injections_to_ifo_batch` fast path — see [LISA](lisa.md).
**bbhx runs on A100 GPUs** (cuda12x source build): SMBHB PE end-to-end in 57.9 s (vs 111 s CPU), batch-128 waveform generation 6.3× faster. GBGPU's GPU path awaits its new-API migration (tracked in `TODO.md`).

## Vectorization (why batching matters more than the device)

Both samplers evaluate the likelihood with `vectorize=True`: the entire walker
population arrives as one `(N, ndim)` batch, the template generates all `N`
waveforms in one backend call, and the likelihood reduces them in one
vectorized pass. Converting the LISA bridge to this path alone took the UCB
example from 116 s to **33.9 s** with no hardware change.

## Cluster node-architecture pitfalls (MSI)

Hard-won facts, recorded so nobody rediscovers them:

| partition | CPU | GPU | gotcha |
|---|---|---|---|
| `a100-4/8` | AMD Milan (Zen3) | A100 sm_80 | **no AVX-512** — PyPI wheels of bbhx/gbgpu/lisatools SIGILL at import |
| `v100` | Intel Skylake | V100 sm_70 | AVX-512 OK — prebuilt LISA wheels run here |
| `msigpu` h100/l40s | AMD Genoa (Zen4) | sm_90 | needs `cupy-cuda12x` |

The fix for the A100 nodes is building the LISA stack from source with
`-march=haswell` (AVX2 baseline) — the verified recipe lives in
`ENVIRONMENT.md`.