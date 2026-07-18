# Wavelet reconstruction

Model-agnostic signal reconstruction with Morlet–Gabor wavelets and
reversible-jump MCMC (Eryn), in the spirit of BayesWave: the number of wavelets is itself sampled, an SNR prior supplies the Occam penalty, and an optional extrinsic branch samples the sky position (`ra`, `dec`, `psi`, ellipticity).

```bash
python examples/bbh_wavelet_reconstruction.py --proposal mffisher --sample-sky \
    --nwalkers 50 --ntemps 10 --nsteps 30000 --burn 10000 --device gpu
```

## Proposal cascade

Reconstruction quality is set almost entirely by the proposals. HyperWave ships, in increasing order of sophistication (`--proposal`):

| mode | births | in-model |
|---|---|---|
| `standard` | prior draws | stretch |
| `guided` | data-informed \((t_0, f_0)\) placement | stretch |
| `fisher` | data-informed placement | **Fisher** + half-cycle + sky-ring |
| `mffisher` | **matched-filter** (data-fitted SNR + phase) | Fisher + half-cycle + sky-ring |
| `flow` / `flowfisher` | learned normalizing flow | stretch / Fisher |

### Matched-filter births (`mffisher`)

Placement alone is not enough: a wavelet born at the right \((t_0, f_0)\) with prior-random amplitude and phase still mismatches the data and dies. The matched-filter birth fits the *linear* parameters from the data at the proposed location,

$$ z_k = 4\,\Delta f \sum_f \frac{d_k(f)\, \bar w_0(f)}{S_k(f)}, $$

and proposes \(\mathrm{SNR} \sim \mathcal{N}_{\rm trunc}(\hat\rho, 1.5)\),
\(\phi_0 \sim \mathrm{VonMises}(\arg z, 8)\), each mixed 20% with the prior so death moves keep finite reverse densities. `rvs` and `logpdf` evaluate the same deterministic fit, so the reversible-jump Hastings ratio remains **exact**.

### In-model moves

- **Fisher** — local jumps scaled by the per-wavelet Fisher matrix (BayesWave's
  workhorse).
- **Half-cycle** — \(t_0 \to t_0 \pm 1/(2 f_0)\) with a phase flip (the wavelet
  degeneracy).
- **Sky-ring** — rotates the sky position about the two-detector baseline,
  updating arrival-time-consistent parameters.

## Reading the output

The example writes an `.npz` with the reconstruction posterior:
per-draw network overlap with the injection, the recovered SNR, the
\(p(n_\text{wavelets})\) posterior, and acceptance rates (`im_acc`, `rj_acc`).

Current benchmark (BBH injection, network SNR ~46, sampled sky): overlap
0.94 with the `fisher` cascade; BayesWave reaches 0.994 on the same data — the matched-filter births are the active work to close that gap (RJ acceptance 0.07 → 0.11 already in smoke tests).
