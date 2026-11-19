"""Calibration factors match bilby and are applied in projection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from conftest import requires_bilby
from hyperwave.detectors.calibration import CubicSpline, _batch_calibration_factor
from hyperwave.detectors.waveforms.template import Template
from hyperwave.likelihoods import GWLikelihoods


def _params(prefix, n_points, amplitudes, phases):
    out = {}
    for ii in range(n_points):
        out[f"{prefix}amplitude_{ii}"] = amplitudes[ii]
        out[f"{prefix}phase_{ii}"] = phases[ii]
    return out


@requires_bilby
def test_cubic_spline_matches_bilby_scalar():
    import bilby

    prefix = "recalib_H1_"
    n_points = 5
    f = np.geomspace(20.0, 800.0, 31)
    amplitudes = np.linspace(-0.08, 0.05, n_points)
    phases = np.linspace(0.03, -0.04, n_points)
    params = _params(prefix, n_points, amplitudes, phases)

    ours = CubicSpline(prefix, 20.0, 800.0, n_points).get_calibration_factor(f, **params)
    ref = bilby.gw.detector.calibration.CubicSpline(
        prefix, 20.0, 800.0, n_points
    ).get_calibration_factor(f, **params)

    np.testing.assert_allclose(ours, ref, rtol=1e-14, atol=1e-14)


@requires_bilby
def test_bilby_cubic_spline_can_be_batched():
    import bilby

    prefix = "recalib_H1_"
    n_points = 4
    f = np.geomspace(20.0, 800.0, 23)
    amplitudes = np.array([
        [0.01, -0.02, 0.03, -0.04],
        [0.05, 0.01, -0.03, 0.02],
    ])
    phases = np.array([
        [0.03, -0.01, 0.02, -0.04],
        [-0.02, 0.04, 0.01, -0.03],
    ])
    params = _params(prefix, n_points, amplitudes.T, phases.T)

    model = bilby.gw.detector.calibration.CubicSpline(prefix, 20.0, 800.0, n_points)
    batched = _batch_calibration_factor(model, f, params, n=2)
    expected = []
    for jj in range(2):
        scalar_params = {key: np.asarray(value)[jj] for key, value in params.items()}
        expected.append(model.get_calibration_factor(f, **scalar_params))

    np.testing.assert_allclose(batched, np.asarray(expected), rtol=1e-14, atol=1e-14)


def test_projection_applies_calibration_factor():
    class FakeDetector:
        def antenna_response(self, ra, dec, psi, gps):
            return np.ones_like(ra), np.zeros_like(ra)

        def time_delay_from_geocenter(self, ra, dec, gps):
            return np.zeros_like(ra)

    prefix = "recalib_H1_"
    n_points = 4
    f = np.array([20.0, 40.0, 80.0, 160.0])
    amps = np.array([0.10, -0.05])
    phases = np.array([0.20, -0.10])

    template = Template.__new__(Template)
    template.mask = np.ones(f.size, dtype=bool)
    template._f_masked = f
    template.frequency_array = f
    template.start_time = 0.0
    template.trigger_time = 0.0
    template.detectors = [FakeDetector()]
    template.detector_names = ["H1"]
    template.calibration_models = [CubicSpline(prefix, f[0], f[-1], n_points)]

    hp = np.ones((2, f.size), dtype=complex)
    hc = np.zeros_like(hp)
    named = {
        "ra": np.zeros(2),
        "dec": np.zeros(2),
        "psi": np.zeros(2),
        "geocent_time": np.zeros(2),
    }
    for ii in range(n_points):
        named[f"{prefix}amplitude_{ii}"] = amps
        named[f"{prefix}phase_{ii}"] = phases

    projected = template._project(hp, hc, named, masked=True)
    expected = (1 + amps) * (2 + 1j * phases) / (2 - 1j * phases)

    np.testing.assert_allclose(projected[:, 0, :], np.repeat(expected[:, None], f.size, axis=1))


def test_template_appends_calibration_parameters_before_noise_terms():
    prefix = "recalib_H1_"
    model = CubicSpline(prefix, 20.0, 160.0, 4)
    template = Template.__new__(Template)
    template.static_parameters = {}
    template.calibration_models = [model]

    ordered = template._append_calibration_parameters(["chirp_mass", "mass_ratio"])

    assert ordered[:2] == ["chirp_mass", "mass_ratio"]
    assert ordered[2:] == model.parameter_names


def test_gaussian_calibration_marginalization_matches_explicit_average():
    class FakeTemplate:
        parameters = ["amplitude"]

        def make_injections_to_ifo_batch(self, theta):
            theta = np.atleast_2d(theta)
            base = np.array([1.0 + 0.5j, -0.2 + 0.3j])
            return theta[:, 0, None, None] * base[None, None, :]

    f = np.array([20.0, 21.0])
    data = np.array([[1.2 + 0.1j, -0.1 + 0.4j]])
    psd = np.ones_like(data.real)
    draws = {"H1": np.array([[1.0 + 0.0j, 1.0 + 0.0j],
                             [1.1 + 0.1j, 0.8 - 0.2j]])}
    theta = np.array([[0.9], [1.2]])

    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=["H1"],
        noise=psd,
        template=FakeTemplate(),
        nsegs=1,
        calibration_marginalization=True,
        calibration_draws=draws,
    )

    out = likelihood.gaussian(theta)
    expected = []
    signal = FakeTemplate().make_injections_to_ifo_batch(theta)
    for ii in range(theta.shape[0]):
        per_draw = []
        for curve in draws["H1"]:
            residual = data[0] - curve * signal[ii, 0]
            per_draw.append(-0.5 * np.sum(4.0 * (f[1] - f[0]) * np.abs(residual) ** 2))
        expected.append(logsumexp(per_draw) - np.log(len(per_draw)))

    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=1e-14)


def test_gaussian_calibration_marginalization_uses_joint_curve_index():
    class FakeTemplate:
        parameters = ["amplitude"]

        def make_injections_to_ifo_batch(self, theta):
            theta = np.atleast_2d(theta)
            base = np.array([
                [1.0 + 0.5j, -0.2 + 0.3j],
                [0.4 - 0.1j, 0.2 + 0.6j],
            ])
            return theta[:, 0, None, None] * base[None, :, :]

    f = np.array([20.0, 21.0])
    data = np.array([
        [1.2 + 0.1j, -0.1 + 0.4j],
        [0.3 - 0.2j, 0.5 + 0.1j],
    ])
    psd = np.array([[1.0, 2.0], [1.5, 0.8]])
    draws = {
        "H1": np.array([
            [1.0 + 0.0j, 1.0 + 0.0j],
            [1.1 + 0.1j, 0.8 - 0.2j],
            [0.9 - 0.1j, 1.2 + 0.1j],
        ]),
        "L1": np.array([
            [0.9 + 0.0j, 1.1 + 0.0j],
            [1.0 - 0.2j, 0.7 + 0.1j],
            [1.2 + 0.1j, 0.8 - 0.1j],
        ]),
    }
    theta = np.array([[0.9], [1.2]])

    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=["H1", "L1"],
        noise=psd,
        template=FakeTemplate(),
        nsegs=1,
        calibration_marginalization=True,
        calibration_draws=draws,
        calibration_chunk_size=1,
    )

    out = likelihood.gaussian(theta)
    signal = FakeTemplate().make_injections_to_ifo_batch(theta)
    expected = []
    for ii in range(theta.shape[0]):
        per_draw = []
        for kk in range(draws["H1"].shape[0]):
            yy = 0.0
            for ifo, name in enumerate(["H1", "L1"]):
                residual = data[ifo] - draws[name][kk] * signal[ii, ifo]
                yy += np.sum(np.abs(residual) ** 2 / psd[ifo])
            per_draw.append(-0.5 * 4.0 * (f[1] - f[0]) * yy)
        expected.append(logsumexp(per_draw) - np.log(draws["H1"].shape[0]))

    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=1e-14)


def test_hyperbolic_calibration_marginalization_matches_explicit_average():
    class FakeTemplate:
        parameters = ["amplitude"]

        def __init__(self, calibration=None):
            self.calibration = calibration

        def make_injections_to_ifo_batch(self, theta):
            theta = np.atleast_2d(theta)
            base = np.array([0.5 + 0.2j, -0.3 + 0.1j, 0.1 - 0.2j])
            signal = theta[:, 0, None, None] * base[None, None, :]
            if self.calibration is not None:
                signal = signal * self.calibration[None, None, :]
            return signal

    f = np.array([20.0, 21.0, 22.0])
    data = np.array([[0.7 + 0.1j, -0.2 + 0.3j, 0.05 - 0.1j]])
    psd = np.ones_like(data.real)
    curves = np.array([[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
                       [0.9 + 0.1j, 1.1 - 0.1j, 1.0 + 0.2j]])
    theta = np.array([[0.8, 2.0, 0.7], [1.1, 1.5, 0.9]])

    marginalized = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=["H1"],
        noise=psd,
        template=FakeTemplate(),
        ddims=False,
        nsegs=1,
        calibration_marginalization=True,
        calibration_draws={"H1": curves},
    ).hyperbolic_classic(theta)

    per_draw = []
    for curve in curves:
        likelihood = GWLikelihoods(
            data=data,
            f=f,
            ifos_list=["H1"],
            noise=psd,
            template=FakeTemplate(calibration=curve),
            ddims=False,
            nsegs=1,
        )
        per_draw.append(np.atleast_1d(likelihood.hyperbolic_classic(theta)))
    expected = logsumexp(np.stack(per_draw, axis=1), axis=1) - np.log(curves.shape[0])

    np.testing.assert_allclose(marginalized, expected, rtol=1e-14, atol=1e-14)


def test_calibration_marginalization_rejects_sampled_calibration_parameters():
    class FakeTemplate:
        parameters = ["amplitude", "recalib_H1_amplitude_0"]

        def make_injections_to_ifo_batch(self, theta):
            return np.ones((np.atleast_2d(theta).shape[0], 1, 2), dtype=complex)

    with pytest.raises(ValueError, match="response-curve draws"):
        GWLikelihoods(
            data=np.ones((1, 2), dtype=complex),
            f=np.array([20.0, 21.0]),
            ifos_list=["H1"],
            noise=np.ones((1, 2)),
            template=FakeTemplate(),
            nsegs=1,
            calibration_marginalization=True,
            calibration_draws={"H1": np.ones((2, 2), dtype=complex)},
        )


def test_calibration_marginalization_rejects_template_calibration_state():
    class StaticTemplate:
        parameters = ["amplitude"]
        static_parameters = {"recalib_H1_amplitude_0": 0.0}

        def make_injections_to_ifo_batch(self, theta):
            return np.ones((np.atleast_2d(theta).shape[0], 1, 2), dtype=complex)

    with pytest.raises(ValueError, match="static_parameters"):
        GWLikelihoods(
            data=np.ones((1, 2), dtype=complex),
            f=np.array([20.0, 21.0]),
            ifos_list=["H1"],
            noise=np.ones((1, 2)),
            template=StaticTemplate(),
            nsegs=1,
            calibration_marginalization=True,
            calibration_draws={"H1": np.ones((2, 2), dtype=complex)},
        )

    class CalibratedTemplate(StaticTemplate):
        static_parameters = {}
        calibration_models = [CubicSpline("recalib_H1_", 20.0, 21.0, 4)]

    with pytest.raises(ValueError, match="double application"):
        GWLikelihoods(
            data=np.ones((1, 2), dtype=complex),
            f=np.array([20.0, 21.0]),
            ifos_list=["H1"],
            noise=np.ones((1, 2)),
            template=CalibratedTemplate(),
            nsegs=1,
            calibration_marginalization=True,
            calibration_draws={"H1": np.ones((2, 2), dtype=complex)},
        )
