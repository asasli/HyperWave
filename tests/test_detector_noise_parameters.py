from __future__ import annotations

import numpy as np
import pytest

from hyperwave.likelihoods import GWLikelihoods, LogLike


class ToyTemplate:
    parameters = ["amplitude"]
    detector_scale = {"H1": 1.0, "L1": 1.2}

    def __init__(self, ifos, f, calibration=None):
        self.ifos = list(ifos)
        self.f = np.asarray(f)
        self.calibration = calibration or {}
        self.profile = (self.f / self.f[0]) * np.exp(0.03j * self.f)
        self.single_calls = 0
        self.batch_calls = 0

    def make_injections_to_ifo(self, theta):
        self.single_calls += 1
        theta = np.asarray(theta)
        signals = {}
        for ifo in self.ifos:
            signal = self.detector_scale[ifo] * theta[0] * self.profile
            if ifo in self.calibration:
                signal = signal * self.calibration[ifo]
            signals[ifo] = signal
        return signals

    def make_injections_to_ifo_batch(self, theta):
        self.batch_calls += 1
        theta = np.atleast_2d(np.asarray(theta))
        signal = np.zeros((theta.shape[0], len(self.ifos), self.f.size), dtype=complex)
        for i, ifo in enumerate(self.ifos):
            signal[:, i, :] = self.detector_scale[ifo] * theta[:, :1] * self.profile[None, :]
            if ifo in self.calibration:
                signal[:, i, :] *= self.calibration[ifo][None, :]
        return signal


def _detector_parameters(alpha, delta, ddims):
    if ddims:
        return np.concatenate([alpha.ravel(), delta.ravel()])
    return np.concatenate([alpha[:, 0], delta.ravel()])


def _repeat_noise(theta0, noise):
    return np.column_stack([theta0, np.repeat(noise[None, :], theta0.size, axis=0)])


def _logmeanexp(values):
    vmax = np.max(values, axis=1, keepdims=True)
    return vmax[:, 0] + np.log(np.mean(np.exp(values - vmax), axis=1))


@pytest.fixture
def detector_problem():
    f = np.linspace(20.0, 40.0, 6)
    data = np.array([
        0.8 + 0.3j * (f / f[0]),
        1.2 - 0.2j * (f / f[0]),
    ])
    psd = np.array([
        np.linspace(1.0, 1.4, f.size),
        np.linspace(0.7, 1.1, f.size),
    ])
    return f, data, psd


@pytest.mark.parametrize("ddims", [False, True])
def test_gw_hyperbolic_detector_dependent_noise_is_sum_of_single_detectors(
    detector_problem, ddims
):
    f, data, psd = detector_problem
    ifos = ["H1", "L1"]
    alpha = np.array([[4.0, 6.0], [8.0, 5.0]])
    delta = np.array([[0.7, 1.3], [2.0, 0.9]])
    amplitude = 0.4

    theta = np.concatenate([[amplitude], _detector_parameters(alpha, delta, ddims)])
    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=ToyTemplate(ifos, f),
        ddims=ddims,
        nsegs=2,
        cpu_cores=1,
        detector_dependent_noise=True,
    )

    expected = 0.0
    for i, ifo in enumerate(ifos):
        single_theta = np.concatenate([
            [amplitude],
            _detector_parameters(alpha[i:i + 1], delta[i:i + 1], ddims),
        ])
        single = GWLikelihoods(
            data=data[i:i + 1],
            f=f,
            ifos_list=[ifo],
            noise=psd[i:i + 1],
            template=ToyTemplate([ifo], f),
            ddims=ddims,
            nsegs=2,
            cpu_cores=1,
        )
        expected += float(single.hyperbolic_classic(single_theta))

    assert np.isclose(likelihood.hyperbolic_classic(theta), expected)


def test_shared_hyperbolic_keeps_batched_template_path(detector_problem):
    f, data, psd = detector_problem
    ifos = ["H1", "L1"]
    template = ToyTemplate(ifos, f)
    theta = np.array([
        [0.4, 3.0, 0.7, 1.1],
        [0.5, 3.0, 0.7, 1.1],
    ])
    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=template,
        ddims=False,
        nsegs=2,
        cpu_cores=1,
    )

    out = likelihood.hyperbolic_classic(theta)

    assert np.shape(out) == (2,)
    assert template.batch_calls == 1
    assert template.single_calls == 0


def test_detector_dependent_noise_keeps_batched_template_path(detector_problem):
    f, data, psd = detector_problem
    ifos = ["H1", "L1"]
    alpha = np.array([[4.0, 6.0], [8.0, 5.0]])
    delta = np.array([[0.7, 1.3], [2.0, 0.9]])
    theta = _repeat_noise(
        np.array([0.4, 0.5]),
        _detector_parameters(alpha, delta, ddims=False),
    )
    template = ToyTemplate(ifos, f)
    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=template,
        ddims=False,
        nsegs=2,
        cpu_cores=1,
        detector_dependent_noise=True,
    )

    batched = likelihood.hyperbolic_classic(theta)
    likelihood._batched_template = False
    per_walker = likelihood.hyperbolic_classic(theta)

    np.testing.assert_allclose(batched, per_walker, rtol=1e-14, atol=1e-14)
    assert template.batch_calls == 1
    assert template.single_calls == theta.shape[0]


@pytest.mark.parametrize("ddims", [False, True])
def test_mixed_detector_noise_is_gaussian_plus_hyperbolic(detector_problem, ddims):
    f, data, psd = detector_problem
    ifos = ["H1", "L1"]
    alpha = np.array([[8.0, 5.0]])
    delta = np.array([[2.0, 0.9]])
    amplitude = 0.4

    theta = np.concatenate([[amplitude], _detector_parameters(alpha, delta, ddims)])
    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=ToyTemplate(ifos, f),
        ddims=ddims,
        nsegs=2,
        cpu_cores=1,
        detector_dependent_noise=True,
        detector_noise_models=["gaussian", "hyperbolic"],
    )

    gaussian = GWLikelihoods(
        data=data[:1],
        f=f,
        ifos_list=["H1"],
        noise=psd[:1],
        template=ToyTemplate(["H1"], f),
        ddims=ddims,
        nsegs=2,
        cpu_cores=1,
    )
    hyperbolic = GWLikelihoods(
        data=data[1:],
        f=f,
        ifos_list=["L1"],
        noise=psd[1:],
        template=ToyTemplate(["L1"], f),
        ddims=ddims,
        nsegs=2,
        cpu_cores=1,
    )
    expected = (
        float(gaussian.gaussian([amplitude]))
        + float(hyperbolic.hyperbolic_classic(theta))
    )

    assert np.isclose(likelihood.hyperbolic_classic(theta), expected)


def test_mixed_detector_noise_keeps_batched_template_path(detector_problem):
    f, data, psd = detector_problem
    ifos = ["H1", "L1"]
    alpha = np.array([[8.0, 5.0]])
    delta = np.array([[2.0, 0.9]])
    theta = _repeat_noise(
        np.array([0.4, 0.5]),
        _detector_parameters(alpha, delta, ddims=False),
    )
    template = ToyTemplate(ifos, f)
    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=template,
        ddims=False,
        nsegs=2,
        cpu_cores=1,
        detector_dependent_noise=True,
        detector_noise_models=["gaussian", "hyperbolic"],
    )

    batched = likelihood.hyperbolic_classic(theta)
    likelihood._batched_template = False
    per_walker = likelihood.hyperbolic_classic(theta)

    np.testing.assert_allclose(batched, per_walker, rtol=1e-14, atol=1e-14)
    assert template.batch_calls == 1
    assert template.single_calls == theta.shape[0]


def test_detector_noise_models_require_detector_dependent_noise(detector_problem):
    f, data, psd = detector_problem

    with pytest.raises(ValueError, match="detector_dependent_noise=True"):
        GWLikelihoods(
            data=data,
            f=f,
            ifos_list=["H1", "L1"],
            noise=psd,
            template=ToyTemplate(["H1", "L1"], f),
            ddims=False,
            nsegs=2,
            detector_noise_models=["gaussian", "hyperbolic"],
        )


def test_detector_dependent_noise_gpu_request_matches_cpu(detector_problem):
    from hyperwave.likelihoods import gpu_backend_available

    f, data, psd = detector_problem
    ifos = ["H1", "L1"]
    alpha = np.array([[4.0, 6.0], [8.0, 5.0]])
    delta = np.array([[0.7, 1.3], [2.0, 0.9]])
    theta = _repeat_noise(
        np.array([0.4, 0.5]),
        _detector_parameters(alpha, delta, ddims=False),
    )
    cpu = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=ToyTemplate(ifos, f),
        ddims=False,
        nsegs=2,
        cpu_cores=1,
        detector_dependent_noise=True,
    ).hyperbolic_classic(theta)

    template = ToyTemplate(ifos, f)
    kwargs = dict(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=template,
        ddims=False,
        nsegs=2,
        gpu=True,
        detector_dependent_noise=True,
    )
    if gpu_backend_available():
        likelihood = GWLikelihoods(**kwargs)
        assert likelihood._use_gpu is True
    else:
        with pytest.warns(RuntimeWarning, match="GPU backend requested"):
            likelihood = GWLikelihoods(**kwargs)
        assert likelihood._use_gpu is False

    out = likelihood.hyperbolic_classic(theta)

    np.testing.assert_allclose(out, cpu, rtol=1e-14, atol=1e-14)
    assert template.batch_calls == 1
    assert template.single_calls == 0


def test_detector_dependent_calibration_marginalization_matches_explicit_average(
    detector_problem,
):
    f, data, psd = detector_problem
    ifos = ["H1", "L1"]
    alpha = np.array([[4.0, 6.0], [8.0, 5.0]])
    delta = np.array([[0.7, 1.3], [2.0, 0.9]])
    theta = _repeat_noise(
        np.array([0.4, 0.5]),
        _detector_parameters(alpha, delta, ddims=False),
    )
    draws = {
        "H1": np.array([
            np.ones(f.size, dtype=complex),
            0.9 + 0.1j * (f / f[-1]),
        ]),
        "L1": np.array([
            1.1 - 0.05j * (f / f[-1]),
            0.95 + 0.02j * (f / f[-1]),
        ]),
    }
    template = ToyTemplate(ifos, f)
    likelihood = GWLikelihoods(
        data=data,
        f=f,
        ifos_list=ifos,
        noise=psd,
        template=template,
        ddims=False,
        nsegs=2,
        calibration_marginalization=True,
        calibration_draws=draws,
        detector_dependent_noise=True,
    )

    marginalized = likelihood.hyperbolic_classic(theta)
    per_draw = []
    for draw in range(draws["H1"].shape[0]):
        draw_logl = np.zeros(theta.shape[0])
        for i, ifo in enumerate(ifos):
            single_noise = _detector_parameters(alpha[i:i + 1], delta[i:i + 1], ddims=False)
            single_theta = _repeat_noise(theta[:, 0], single_noise)
            single = GWLikelihoods(
                data=data[i:i + 1],
                f=f,
                ifos_list=[ifo],
                noise=psd[i:i + 1],
                template=ToyTemplate([ifo], f, calibration={ifo: draws[ifo][draw]}),
                ddims=False,
                nsegs=2,
                cpu_cores=1,
            )
            draw_logl += np.atleast_1d(single.hyperbolic_classic(single_theta))
        per_draw.append(draw_logl)
    expected = _logmeanexp(np.stack(per_draw, axis=1))

    np.testing.assert_allclose(marginalized, expected, rtol=1e-14, atol=1e-14)
    assert template.batch_calls == 1
    assert template.single_calls == 0


@pytest.mark.parametrize("ddims", [False, True])
def test_data_hyperbolic_detector_dependent_noise_is_sum_of_single_detectors(
    detector_problem, ddims
):
    f, data, _psd = detector_problem
    ifos = ["H1", "L1"]
    alpha = np.array([[4.0, 6.0], [8.0, 5.0]])
    delta = np.array([[0.7, 1.3], [2.0, 0.9]])
    theta = _detector_parameters(alpha, delta, ddims)

    likelihood = LogLike(
        data=data,
        f=f,
        ifos_list=ifos,
        ddims=ddims,
        nsegs=2,
        detector_dependent_noise=True,
    )

    expected = 0.0
    for i, ifo in enumerate(ifos):
        single_theta = _detector_parameters(alpha[i:i + 1], delta[i:i + 1], ddims)
        single = LogLike(data=data[i], f=f, ifos_list=[ifo], ddims=ddims, nsegs=2)
        if ddims:
            expected += float(single.hyperbolic_classic2D(single_theta))
        else:
            expected += float(single.hyperbolic_classic1D(single_theta))

    if ddims:
        actual = likelihood.hyperbolic_classic2D(theta)
    else:
        actual = likelihood.hyperbolic_classic1D(theta)

    assert np.isclose(actual, expected)
