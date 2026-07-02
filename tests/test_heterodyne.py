"""Heterodyne (relative-binning) likelihood: accuracy vs the exact Gaussian logL.

Uses an analytic PN-like chirp (closed form at any frequency, so the exact
likelihood and the edge evaluations share one waveform definition) on a dense
grid. The heterodyned logL must reproduce exact logL *differences* (the dropped
<d|d>/2 constant cancels) across parameter offsets around the reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from hyperwave.likelihoods import HeterodyneLikelihood, heterodyne_bin_edges

F = np.linspace(20.0, 512.0, 60_000)
DF = F[1] - F[0]
IFOS = ["H1", "L1"]
PSD = np.vstack([np.full(F.size, 1e-46), np.full(F.size, 1.2e-46)])
THETA_REF = np.array([30.0, 1.0e-21, 0.0])  # (mc-like, amplitude, phase offset)


def waveform(theta, f):
    """Analytic chirp: amplitude * f^{-7/6} * exp(i Psi(f; mc, phi))."""
    mc, amp, phi = theta
    psi = 3.0 / 128.0 * (np.pi * mc * 4.93e-6 * f) ** (-5.0 / 3.0) + phi
    h = amp * (f / 100.0) ** (-7.0 / 6.0) * np.exp(1j * psi)
    return np.vstack([h, 0.9 * h])  # (n_ifo, n_freq)


def waveform_edges_factory(f_edges):
    def waveform_edges(thetas):
        thetas = np.atleast_2d(thetas)
        return np.stack([waveform(t, f_edges) for t in thetas], axis=0)
    return waveform_edges


def exact_logl(theta, data):
    """<d|h> - <h|h>/2 with the same inner product convention."""
    h = waveform(theta, F)
    dh = float(np.sum((4.0 * DF * data * np.conj(h) / PSD).real))
    hh = float(np.sum((4.0 * DF * np.abs(h) ** 2 / PSD).real))
    return dh - 0.5 * hh


@pytest.fixture(scope="module")
def problem():
    data = waveform(THETA_REF, F)  # zero-noise injection at the reference point
    h0 = waveform(THETA_REF, F)
    like = HeterodyneLikelihood(
        data=data, f=F, psd=PSD, ifos_list=IFOS, h0=h0,
        waveform_edges=None, eps=0.1,
    )
    like._waveform_edges = waveform_edges_factory(like.f_edges)
    return like, data


def test_bin_edges_cover_band_and_are_sparse():
    edges = heterodyne_bin_edges(F, eps=0.1)
    assert edges[0] == 0 and edges[-1] == F.size - 1
    assert np.all(np.diff(edges) > 0)
    assert edges.size < F.size / 50  # the whole point: orders of magnitude fewer

def test_matches_exact_loglikelihood_differences(problem):
    like, data = problem
    rng = np.random.default_rng(7)
    # parameter offsets spanning the posterior bulk around the reference
    trials = THETA_REF[None, :] + np.column_stack([
        rng.uniform(-2e-4, 2e-4, 12),     # chirp-mass-like shifts
        rng.uniform(-2e-23, 2e-23, 12),   # amplitude shifts
        rng.uniform(-0.3, 0.3, 12),       # phase shifts
    ])
    het = np.atleast_1d(like.logl(trials))
    het_ref = float(like.logl(THETA_REF))
    exact = np.array([exact_logl(t, data) for t in trials])
    exact_ref = exact_logl(THETA_REF, data)
    err = (het - het_ref) - (exact - exact_ref)
    scale = np.maximum(np.abs(exact - exact_ref), 1.0)
    assert np.max(np.abs(err) / scale) < 5e-3, (err, exact - exact_ref)

def test_reference_point_is_maximum_for_zero_noise(problem):
    like, _ = problem
    l_ref = float(like.logl(THETA_REF))
    worse = float(like.logl(THETA_REF + np.array([5e-4, 0.0, 0.0])))
    assert l_ref > worse

def test_batch_and_single_agree(problem):
    like, _ = problem
    t = THETA_REF + np.array([1e-4, 1e-23, 0.05])
    single = float(like.logl(t))
    batch = np.atleast_1d(like.logl(np.vstack([t, THETA_REF])))
    assert np.isclose(single, batch[0], rtol=0, atol=1e-6)


def test_real_lvk_waveform_agreement():
    """End-to-end with real IMRPhenomPv2 via the lal sequence backend.

    Heterodyned logL differences must track the exact GWLikelihoods.gaussian
    differences (the <d|d>/2 constants cancel) for perturbations around the
    injection.
    """
    pytest.importorskip("lalsimulation")
    pytest.importorskip("bilby")
    from hyperwave.detectors.lvk import GW, DetectorNoise
    from hyperwave.likelihoods import GWLikelihoods

    names = ["chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
             "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
             "cos_tilt_2", "phi_12", "phi_jl"]
    m1, m2 = 36.0, 29.0
    mc = (m1 + m2) * (m1 * m2 / (m1 + m2) ** 2) ** 0.6
    theta_ref = np.array([mc, m2 / m1, 600.0, 1.1, 0.9, 1.375, -0.2108,
                          0.0, 0.0, np.cos(0.4), 1.0, 1.0, 0.0, 0.0])
    trigger = 1268189526.951953

    noise = DetectorNoise(4.0, 2048.0, trigger, ["H1", "L1"],
                          minimum_frequency=20.0, maximum_frequency=512.0)
    noise.generate_noise(real_noise=False, seed=3)
    template = GW(noise, approximant="IMRPhenomPv2", reference_frequency=50.0,
                  parameters=names, static_parameters={"geocent_time": trigger})
    template.make_injections_to_ifo(theta_ref)

    f, asd0 = template.detector_asd_masked(0)
    psd = np.array([asd0 ** 2, template.detector_asd_masked(1)[1] ** 2])
    data = np.array([template.detector_data_fd(0), template.detector_data_fd(1)])

    exact = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"], noise=psd,
                          template=template, ddims=False, nsegs=1, gpu=False)
    het = HeterodyneLikelihood.from_lvk_template(
        template, data=data, f=f, psd=psd, ifos_list=["H1", "L1"],
        theta_ref=theta_ref, eps=0.1,
    )
    assert het.f_edges.size < f.size / 4  # sparse edge grid

    rng = np.random.default_rng(11)
    trials = np.tile(theta_ref, (8, 1))
    trials[:, 0] += rng.uniform(-0.02, 0.02, 8)     # chirp mass [Msun]
    trials[:, 2] *= 1.0 + rng.uniform(-0.05, 0.05, 8)  # distance
    trials[:, 4] = (trials[:, 4] + rng.uniform(-0.2, 0.2, 8)) % (2 * np.pi)  # phase

    g = np.atleast_1d(exact.gaussian(np.column_stack([trials, np.full(8, 1.0)])[:, :14]))
    g_ref = float(exact.gaussian(theta_ref))
    h = np.atleast_1d(het.logl(trials))
    h_ref = float(het.logl(theta_ref))
    err = (h - h_ref) - (g - g_ref)
    scale = np.maximum(np.abs(g - g_ref), 1.0)
    assert np.max(np.abs(err) / scale) < 2e-2, (err, g - g_ref)


def test_interpolated_waveform_template_hyperbolic():
    """'Heterodyne the waveform, not the likelihood': the edge-evaluated,
    ratio-interpolated template must reproduce exact hyperbolic logL
    differences (the sqrt prevents true binning; this path keeps the
    likelihood exact and bins only the waveform)."""
    pytest.importorskip("lalsimulation")
    pytest.importorskip("bilby")
    from hyperwave.detectors.lvk import GW, DetectorNoise
    from hyperwave.likelihoods import GWLikelihoods
    from hyperwave.likelihoods.heterodyne import InterpolatedWaveformTemplate

    names = ["chirp_mass", "mass_ratio", "luminosity_distance", "phase"]
    trigger = 1268189526.951953
    static = dict(psi=1.1, ra=1.375, dec=-0.2108, chi_1=0.0, chi_2=0.0,
                  cos_theta_jn=np.cos(0.4), cos_tilt_1=1.0, cos_tilt_2=1.0,
                  phi_12=0.0, phi_jl=0.0, geocent_time=trigger)
    m1, m2 = 36.0, 29.0
    theta_ref = np.array([(m1 + m2) * (m1 * m2 / (m1 + m2) ** 2) ** 0.6,
                          m2 / m1, 600.0, 0.9])
    noise = DetectorNoise(4.0, 2048.0, trigger, ["H1", "L1"],
                          minimum_frequency=20.0, maximum_frequency=512.0)
    noise.generate_noise(real_noise=False, seed=3)
    tmpl = GW(noise, approximant="IMRPhenomPv2", reference_frequency=50.0,
              parameters=names, static_parameters=static)
    tmpl.make_injections_to_ifo(theta_ref)
    f, a0 = tmpl.detector_asd_masked(0)
    psd = np.array([a0 ** 2, tmpl.detector_asd_masked(1)[1] ** 2])
    data = np.array([tmpl.detector_data_fd(0), tmpl.detector_data_fd(1)])

    exact = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"], noise=psd,
                          template=tmpl, ddims=False, nsegs=2)
    itmpl = InterpolatedWaveformTemplate(tmpl, f, theta_ref, eps=0.1)
    fast = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"], noise=psd,
                         template=itmpl, ddims=False, nsegs=2)

    rng = np.random.default_rng(11)
    trials = np.tile(theta_ref, (8, 1))
    trials[:, 0] += rng.uniform(-0.02, 0.02, 8)
    trials[:, 2] *= 1 + rng.uniform(-0.05, 0.05, 8)
    full = np.column_stack([trials, np.full(8, 5.0), np.full(8, 1.0), np.full(8, 1.0)])
    ref = np.concatenate([theta_ref, [5.0, 1.0, 1.0]])[None, :]
    ge = np.atleast_1d(exact.hyperbolic_classic(full)) - float(exact.hyperbolic_classic(ref))
    gf = np.atleast_1d(fast.hyperbolic_classic(full)) - float(fast.hyperbolic_classic(ref))
    assert np.max(np.abs(gf - ge) / np.maximum(np.abs(ge), 1.0)) < 5e-3


def test_heterodyned_hyperbolic_likelihood():
    """Route 2: true heterodyned hyperbolic (new likelihood; exact one remains).

    First-order expansion of sqrt(delta^2+yy) around the reference residual,
    bin summaries with reference-fixed weights, sampled delta via grid+spline
    interpolation. Tolerances reflect the validated trust region: ~0.2 logL in
    the posterior bulk (shared LAL sequence-vs-grid systematic included)."""
    pytest.importorskip("lalsimulation")
    pytest.importorskip("bilby")
    from hyperwave.detectors.lvk import GW, DetectorNoise
    from hyperwave.likelihoods import GWLikelihoods
    from hyperwave.likelihoods.heterodyne import HeterodynedHyperbolicLikelihood

    names = ["chirp_mass", "mass_ratio", "luminosity_distance", "phase"]
    trigger = 1268189526.951953
    static = dict(psi=1.1, ra=1.375, dec=-0.2108, chi_1=0.0, chi_2=0.0,
                  cos_theta_jn=np.cos(0.4), cos_tilt_1=1.0, cos_tilt_2=1.0,
                  phi_12=0.0, phi_jl=0.0, geocent_time=trigger)
    m1, m2 = 36.0, 29.0
    theta_ref = np.array([(m1 + m2) * (m1 * m2 / (m1 + m2) ** 2) ** 0.6,
                          m2 / m1, 600.0, 0.9])
    noise = DetectorNoise(4.0, 2048.0, trigger, ["H1", "L1"],
                          minimum_frequency=20.0, maximum_frequency=512.0)
    noise.generate_noise(real_noise=False, seed=3)
    tmpl = GW(noise, approximant="IMRPhenomPv2", reference_frequency=50.0,
              parameters=names, static_parameters=static)
    tmpl.make_injections_to_ifo(theta_ref)
    f, a0 = tmpl.detector_asd_masked(0)
    psd = np.array([a0 ** 2, tmpl.detector_asd_masked(1)[1] ** 2])
    data = np.array([tmpl.detector_data_fd(0), tmpl.detector_data_fd(1)])

    exact = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"], noise=psd,
                          template=tmpl, ddims=False, nsegs=2)
    het = HeterodynedHyperbolicLikelihood.from_lvk_template(
        tmpl, data=data, f=f, psd=psd, ifos_list=["H1", "L1"],
        theta_ref=theta_ref, nsegs=2, eps=0.1)

    rng = np.random.default_rng(11)
    # at the reference waveform, across the (alpha, delta) plane
    th0 = np.tile(np.concatenate([theta_ref, [5.0, 1.0, 1.0]]), (16, 1))
    th0[:, -3] = rng.uniform(0.5, 20.0, 16)
    th0[:, -2] = rng.uniform(0.05, 25.0, 16)
    th0[:, -1] = rng.uniform(0.05, 25.0, 16)
    e0 = np.atleast_1d(exact.hyperbolic_classic(th0))
    h0v = np.atleast_1d(het.logl(th0))
    assert np.all(np.isfinite(h0v))
    assert np.max(np.abs(h0v - e0)) < 1.0

    # posterior-bulk waveform perturbations
    tr = np.tile(theta_ref, (16, 1))
    tr[:, 0] += rng.uniform(-0.005, 0.005, 16)
    tr[:, 2] *= 1 + rng.uniform(-0.015, 0.015, 16)
    full = np.column_stack([tr, rng.uniform(1.0, 15.0, 16),
                            rng.uniform(0.3, 15.0, 16), rng.uniform(0.3, 15.0, 16)])
    refr = full.copy()
    refr[:, :4] = theta_ref
    ge = np.atleast_1d(exact.hyperbolic_classic(full)) - np.atleast_1d(exact.hyperbolic_classic(refr))
    gh = np.atleast_1d(het.logl(full)) - np.atleast_1d(het.logl(refr))
    assert np.max(np.abs(gh - ge)) < 0.3
