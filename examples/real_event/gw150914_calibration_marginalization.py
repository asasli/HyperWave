"""GW150914 calibration marginalization from public GWTC-1 envelopes.

This example downloads the event-specific GW150914 calibration uncertainty
envelopes from the LIGO DCC GWTC-1 release, samples response curves from those
envelopes, and passes the curves to HyperWave's calibration-marginalized
likelihood.

It is intentionally a setup/evaluation example, not a full sampler run:

    python examples/real_event/gw150914_calibration_marginalization.py
    python examples/real_event/gw150914_calibration_marginalization.py --n-curves 128

Public calibration source:
https://dcc.ligo.org/public/0158/P1900040/001/GWTC1_GW150914_CalEnv.tar.gz
"""

from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

os.environ.setdefault("GWPY_RCPARAMS", "0")

import bilby
import numpy as np

from hyperwave.detectors.calibration import CubicSpline
from hyperwave.detectors.lvk import GW, DetectorNoise
from hyperwave.likelihoods import GWLikelihoods

GW150914_TIME = 1126259462.4
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data"
CALIBRATION_URL = (
    "https://dcc.ligo.org/public/0158/P1900040/001/GWTC1_GW150914_CalEnv.tar.gz"
)
CALIBRATION_MEMBERS = {
    "H1": "./GWTC1_GW150914_CalEnv/GWTC1_GW150914_H_CalEnv.txt",
    "L1": "./GWTC1_GW150914_CalEnv/GWTC1_GW150914_L_CalEnv.txt",
}
PARAMETER_NAMES = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
    "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
    "cos_tilt_2", "phi_12", "phi_jl", "geocent_time",
]


def download_gw150914_calibration_envelopes(cache_dir):
    """Download and cache the public GWTC-1 GW150914 H1/L1 envelope files."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "GWTC1_GW150914_CalEnv.tar.gz"
    if not archive.exists():
        print(f"> downloading calibration envelopes: {CALIBRATION_URL}")
        urllib.request.urlretrieve(CALIBRATION_URL, archive)

    envelope_files = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers()}
        for ifo, member_name in CALIBRATION_MEMBERS.items():
            out = cache_dir / Path(member_name).name
            if not out.exists():
                extracted = tar.extractfile(members[member_name])
                if extracted is None:
                    raise FileNotFoundError(member_name)
                out.write_bytes(extracted.read())
            envelope_files[ifo] = out
    return envelope_files


def calibration_draws_from_envelopes(envelope_files, f, n_curves, n_nodes, seed):
    """Sample template-side response curves from public calibration envelopes."""
    bilby.core.utils.random.seed(seed)
    f = np.asarray(f, dtype=float)
    draws = {}
    for ifo, envelope_file in envelope_files.items():
        priors = bilby.gw.prior.CalibrationPriorDict.from_envelope_file(
            envelope_file=str(envelope_file),
            minimum_frequency=float(f[0]),
            maximum_frequency=float(f[-1]),
            n_nodes=int(n_nodes),
            label=ifo,
            correction_type="data",
        )
        samples = priors.sample(int(n_curves))
        spline = CubicSpline(
            prefix=f"recalib_{ifo}_",
            minimum_frequency=float(f[0]),
            maximum_frequency=float(f[-1]),
            n_points=int(n_nodes),
        )
        curves = np.asarray(spline.get_calibration_factor(f, **samples), dtype=complex)
        draws[ifo] = np.atleast_2d(curves)
    return draws


def build_problem(args):
    detectors = ["H1", "L1"]
    noise = DetectorNoise(
        args.duration,
        args.sampling_frequency,
        GW150914_TIME,
        detectors,
        minimum_frequency=args.fmin,
        maximum_frequency=args.fmax,
    )
    print("> downloading GW150914 open strain data (GWOSC)")
    noise.generate_noise(real_noise=True)

    template = GW(
        noise,
        approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        parameters=PARAMETER_NAMES,
        static_parameters={},
    )
    f, asd0 = template.detector_asd_masked(0)
    psd = np.array([asd0**2, template.detector_asd_masked(1)[1]**2])
    data = np.array([template.detector_data_fd(0), template.detector_data_fd(1)])
    return template, f, psd, data


def representative_gw150914_points():
    """Two plausible points near GW150914, only for a smoke likelihood eval."""
    q = 29.0 / 36.0
    eta = 36.0 * 29.0 / (36.0 + 29.0) ** 2
    chirp = (36.0 + 29.0) * eta**0.6
    return np.array([
        [chirp, q, 450.0, 0.8, 1.3, 1.95, -1.2,
         0.0, 0.0, np.cos(2.7), 1.0, 1.0, 0.0, 0.0, GW150914_TIME],
        [chirp * 1.01, 0.78, 520.0, 1.1, 1.0, 2.1, -1.0,
         0.0, 0.0, np.cos(2.4), 1.0, 1.0, 0.0, 0.0, GW150914_TIME + 0.01],
    ])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--n-curves", type=int, default=32)
    parser.add_argument("--n-nodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=150914)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--sampling-frequency", type=float, default=2048.0)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=512.0)
    parser.add_argument("--calibration-chunk-size", type=int, default=16)
    args = parser.parse_args()

    envelope_files = download_gw150914_calibration_envelopes(args.cache_dir)
    template, f, psd, data = build_problem(args)
    calibration_draws = calibration_draws_from_envelopes(
        envelope_files=envelope_files,
        f=f,
        n_curves=args.n_curves,
        n_nodes=args.n_nodes,
        seed=args.seed,
    )

    nominal = GWLikelihoods(
        data=data, f=f, ifos_list=["H1", "L1"], noise=psd, template=template,
        ddims=False, nsegs=4, gpu=False,
    )
    marginalized = GWLikelihoods(
        data=data, f=f, ifos_list=["H1", "L1"], noise=psd, template=template,
        ddims=False, nsegs=4, gpu=False,
        calibration_marginalization=True,
        calibration_draws=calibration_draws,
        calibration_chunk_size=args.calibration_chunk_size,
    )

    theta = representative_gw150914_points()
    print(f"> H1 calibration draws: {calibration_draws['H1'].shape}")
    print(f"> L1 calibration draws: {calibration_draws['L1'].shape}")
    print("> nominal Gaussian logL:")
    print(nominal.gaussian(theta))
    print("> calibration-marginalized Gaussian logL:")
    print(marginalized.gaussian(theta))


if __name__ == "__main__":
    main()
