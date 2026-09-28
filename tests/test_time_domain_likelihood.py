from __future__ import annotations

import numpy as np

from hyperwave.likelihoods.time_domain import TimeDomainGWLikelihoods


class ToyFullFDTemplate:
    parameters = ["amplitude"]

    def __init__(self, ifos, sampling_rate, n_time, scales=None):
        self.ifos = list(ifos)
        self.sampling_rate = float(sampling_rate)
        self.n_time = int(n_time)
        self.minimum_frequency = 2.0
        self.maximum_frequency = sampling_rate / 2.0 - 1.0
        t = np.arange(n_time) / sampling_rate
        profile = np.sin(2.0 * np.pi * 3.0 * t) + 0.4 * np.cos(2.0 * np.pi * 5.0 * t)
        self.profile_fd = np.fft.rfft(profile) / sampling_rate
        self.scales = (
            np.linspace(1.0, 1.3, len(ifos))
            if scales is None
            else np.asarray(scales, dtype=float)
        )
        self.calls = 0

    def make_injections_to_ifo_batch(self, theta, masked=False):
        assert masked is False
        self.calls += 1
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        out = np.zeros(
            (theta.shape[0], len(self.ifos), self.n_time // 2 + 1),
            dtype=complex,
        )
        for i, scale in enumerate(self.scales):
            out[:, i, :] = theta[:, :1] * scale * self.profile_fd[None, :]
        return out


def _problem(ifos=("H1", "L1"), n_time=32, sampling_rate=32.0):
    rng = np.random.default_rng(10)
    ifos = list(ifos)
    template = ToyFullFDTemplate(ifos, sampling_rate, n_time)
    data = rng.normal(0.0, 0.2, size=(len(ifos), n_time))
    acf = np.vstack([
        (1.4 + 0.2 * i) * 0.35 ** np.arange(n_time)
        for i in range(len(ifos))
    ])
    return template, data, acf, sampling_rate


def test_td_covariance_methods_agree_for_gaussian():
    template, data, acf, sampling_rate = _problem()
    theta = np.array([[0.2], [0.35]])
    values = []
    for method in [
        "direct-inversion",
        "cholesky-solve-triangular",
        "toeplitz-inversion",
        "gohberg-semencul",
    ]:
        likelihood = TimeDomainGWLikelihoods(
            data=data,
            sampling_rate=sampling_rate,
            ifos_list=["H1", "L1"],
            acf=acf,
            template=template,
            likelihood_method=method,
        )
        calls_before = template.calls
        values.append(likelihood.gaussian(theta))
        assert template.calls == calls_before + 1

    for value in values[1:]:
        np.testing.assert_allclose(value, values[0], rtol=1e-10, atol=1e-10)

    split = TimeDomainGWLikelihoods(
        data=data,
        sampling_rate=sampling_rate,
        ifos_list=["H1", "L1"],
        acf=acf,
        template=template,
        likelihood_method="toeplitz",
        split_inner_products=True,
    )
    calls_before = template.calls
    split_value = split.gaussian(theta)
    assert template.calls == calls_before + 1
    np.testing.assert_allclose(split_value, values[0], rtol=1e-10, atol=1e-10)


def test_seconds_time_bands_match_uniform_student_t_bands():
    template, data, acf, sampling_rate = _problem(n_time=16, sampling_rate=16.0)
    theta = np.array([[0.2, 6.0, 9.0]])
    uniform = TimeDomainGWLikelihoods(
        data=data,
        sampling_rate=sampling_rate,
        ifos_list=["H1", "L1"],
        acf=acf,
        template=template,
        time_bands=2,
    )
    seconds = TimeDomainGWLikelihoods(
        data=data,
        sampling_rate=sampling_rate,
        ifos_list=["H1", "L1"],
        acf=acf,
        template=template,
        time_band_boundaries=[0.5],
    )

    assert seconds.time_bands == [0.5]
    np.testing.assert_allclose(seconds.student_t(theta), uniform.student_t(theta))


def test_detector_dependent_hyperbolic_is_sum_of_single_detectors():
    template, data, acf, sampling_rate = _problem()
    alpha = np.array([[4.0, 6.0], [8.0, 5.0]])
    delta = np.array([[0.7, 1.3], [2.0, 0.9]])
    theta = np.concatenate([[0.2], alpha.ravel(), delta.ravel()])
    likelihood = TimeDomainGWLikelihoods(
        data=data,
        sampling_rate=sampling_rate,
        ifos_list=["H1", "L1"],
        acf=acf,
        template=template,
        time_bands=2,
        detector_dependent_noise=True,
        detector_likelihoods="hyperbolic",
    )

    expected = 0.0
    for i, ifo in enumerate(["H1", "L1"]):
        one_template = ToyFullFDTemplate(
            [ifo], sampling_rate, data.shape[1], scales=[template.scales[i]]
        )
        single = TimeDomainGWLikelihoods(
            data=data[i:i + 1],
            sampling_rate=sampling_rate,
            ifos_list=[ifo],
            acf=acf[i:i + 1],
            template=one_template,
            time_bands=2,
            detector_dependent_noise=True,
            detector_likelihoods="hyperbolic",
        )
        single_theta = np.concatenate([[0.2], alpha[i], delta[i]])
        expected += float(single.hyperbolic_classic(single_theta))

    np.testing.assert_allclose(likelihood.hyperbolic_classic(theta), expected)


def test_mixed_detector_likelihood_matches_manual_sum():
    template, data, acf, sampling_rate = _problem()
    theta = np.array([0.2, 6.0, 1.4])
    mixed = TimeDomainGWLikelihoods(
        data=data,
        sampling_rate=sampling_rate,
        ifos_list=["H1", "L1"],
        acf=acf,
        template=template,
        detector_likelihoods={"H1": "gaussian", "L1": "hyperbolic"},
        detector_dependent_noise=True,
    )
    h1 = TimeDomainGWLikelihoods(
        data=data[:1],
        sampling_rate=sampling_rate,
        ifos_list=["H1"],
        acf=acf[:1],
        template=ToyFullFDTemplate(
            ["H1"], sampling_rate, data.shape[1], scales=[template.scales[0]]
        ),
    )
    l1 = TimeDomainGWLikelihoods(
        data=data[1:],
        sampling_rate=sampling_rate,
        ifos_list=["L1"],
        acf=acf[1:],
        template=ToyFullFDTemplate(
            ["L1"], sampling_rate, data.shape[1], scales=[template.scales[1]]
        ),
        detector_dependent_noise=True,
        detector_likelihoods="hyperbolic",
    )

    expected = float(h1.gaussian([theta[0]])) + float(l1.hyperbolic_classic(theta))
    np.testing.assert_allclose(mixed.mixed(theta), expected)
    assert mixed.mixed_ndims == 3


def test_psd_patching_matches_bilby_td_rule():
    sampling_rate = 16.0
    n_time = 16
    f = np.fft.rfftfreq(n_time, d=1.0 / sampling_rate)
    psd = np.linspace(1.0, 5.0, len(f))
    psd[1] = np.inf
    psd[2] = 0.0
    template = ToyFullFDTemplate(["H1"], sampling_rate, n_time)
    likelihood = TimeDomainGWLikelihoods(
        data=np.zeros((1, n_time)),
        sampling_rate=sampling_rate,
        ifos_list=["H1"],
        f=f,
        noise=psd[None, :],
        template=template,
        minimum_frequency=3.0,
        maximum_frequency=6.0,
    )

    expected = psd.copy()
    finite = np.isfinite(expected) & (expected > 0.0)
    expected[~finite] = float(np.max(expected[finite]))
    active = (f >= 3.0) & (f <= 6.0)
    expected[f < f[active][0]] = 10.0 * float(np.max(expected[active]))
    expected[f > f[active][-1]] = 10.0 * float(np.max(expected[f >= f[active][-1]]))

    np.testing.assert_allclose(likelihood.psd[0], expected, rtol=0.0, atol=0.0)
