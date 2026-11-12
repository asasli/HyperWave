"""Tests for hyperwave.plots.skymap."""

import importlib.util

import numpy as np
import pytest

from hyperwave import Result
from hyperwave.plots.skymap import DEG2, SkyPosterior, compare_skymaps, plot_skymap

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _bimodal(n=200_000, f_dom=0.8, sigma_deg=3.0, seed=0):
    """Two Gaussian blobs: a dominant mode and its antipode."""
    rng = np.random.default_rng(seed)
    lon0, lat0 = np.radians(232.0), np.radians(-40.0)
    s = np.radians(sigma_deg)
    n1 = int(f_dom * n)
    lon = np.concatenate([lon0 + rng.normal(0, s / np.cos(lat0), n1),
                          lon0 + np.pi + rng.normal(0, s / np.cos(lat0), n - n1)])
    lat = np.concatenate([lat0 + rng.normal(0, s, n1), -lat0 + rng.normal(0, s, n - n1)])
    return lon % (2 * np.pi), lat, (lon0, lat0)


def test_credible_area_gaussian():
    rng = np.random.default_rng(1)
    s = np.radians(2.0)
    lon = np.radians(90.0) + rng.normal(0, s, 400_000)
    lat = rng.normal(0, s, 400_000)
    sky = SkyPosterior.from_samples(lon, lat=lat, smooth=0.0)
    expected = 2 * np.pi * s**2 * np.log(10) * DEG2      # 90% of an isotropic 2-D Gaussian
    assert abs(sky.credible_area(0.9) / expected - 1) < 0.08
    assert sky.credible_area(0.5) < sky.credible_area(0.9)


def test_matches_canonical_estimator():
    smbhb = pytest.importorskip("hyperwave.detectors.lisa.smbhb")
    lon, lat, _ = _bimodal(n=50_000)
    sky = SkyPosterior.from_samples(lon, lat=lat, periodic=False)
    ref = smbhb.sky_credible_area(lon, np.sin(lat), 0.9)
    assert abs(sky.credible_area(0.9) / ref - 1) < 0.02


def test_modes_and_antipode():
    lon, lat, truth = _bimodal(f_dom=0.8)
    sky = SkyPosterior.from_samples(lon, lat=lat)
    m = sky.modes(0.9)
    assert len(m) == 2
    assert abs(m[0]["mass_total"] - 0.8) < 0.02
    assert abs(sum(d["mass_total"] for d in m) - 1.0) < 1e-6
    assert abs(sum(d["mass"] for d in m) - 0.9) < 0.02
    assert m[1]["antipode_of"] == 0
    assert abs(np.degrees(m[0]["lat"]) + 40) < 1.0
    assert abs(sky.dominant_mode_fraction(0.9) - 0.8) < 0.02
    assert 0.0 <= sky.truth_credible_level(*truth) <= 0.3     # truth sits at the dominant peak
    assert sky.searched_area(*truth) < sky.credible_area(0.5)


def test_seam_mode_is_not_split():
    rng = np.random.default_rng(3)
    lon = rng.normal(0, np.radians(3), 100_000) % (2 * np.pi)      # straddles lon = 0
    lat = rng.normal(0, np.radians(3), 100_000)
    sky = SkyPosterior.from_samples(lon, lat=lat)
    assert len(sky.modes(0.9)) == 1


def test_weights_and_result_api():
    lon, lat, _ = _bimodal(n=20_000)
    w = np.ones_like(lon)
    sky_w = SkyPosterior.from_samples(lon, lat=lat, weights=w)
    sky = SkyPosterior.from_samples(lon, lat=lat)
    assert np.isclose(sky_w.credible_area(0.9), sky.credible_area(0.9))
    r = Result(np.column_stack([lon, np.sin(lat)]), ["lam", "sin_beta"])
    assert np.isclose(r.skymap().credible_area(0.9), sky.credible_area(0.9))


@pytest.mark.skipif(importlib.util.find_spec("healpy") is None, reason="healpy not installed")
def test_healpix_export(tmp_path):
    import healpy as hp

    lon, lat, _ = _bimodal(n=20_000)
    sky = SkyPosterior.from_samples(lon, lat=lat)
    m = sky.to_healpix(nside=32)
    assert m.size == hp.nside2npix(32) and abs(m.sum() - 1) < 1e-9
    path = sky.write_fits(str(tmp_path / "sky.fits"), nside=32)
    back = hp.read_map(path)
    assert abs(back.sum() - 1) < 1e-6


def test_fragmented_island_is_one_mode():
    """Two blobs 14 degrees apart plus their antipodal images: two modes, not four."""
    rng = np.random.default_rng(5)
    s = np.radians(1.5)
    lon0, lat0 = np.radians(290.0), np.radians(-17.0)
    parts = [(lon0, lat0, 50_000), (lon0 + np.radians(11.0), lat0 - np.radians(10.0), 30_000),
             (lon0 + np.pi, -lat0, 12_000), (lon0 + np.pi + np.radians(11.0), -lat0 + np.radians(10.0), 8_000)]
    lon = np.concatenate([lc + rng.normal(0, s / np.cos(bc), n) for lc, bc, n in parts]) % (2 * np.pi)
    lat = np.concatenate([bc + rng.normal(0, s, n) for lc, bc, n in parts])
    sky = SkyPosterior.from_samples(lon, lat=lat)
    m = sky.modes(0.9)
    assert len(m) == 2
    assert abs(m[0]["mass_total"] - 0.8) < 0.02
    assert m[1]["antipode_of"] == 0
    assert abs(sky.dominant_mode_fraction(0.9) - 0.8) < 0.02
    assert len(sky.modes(0.9, merge_deg=0.0)) == 4
    part = sky.mode_partition(0.9)
    assert part.shape == sky.prob.shape and set(np.unique(part)) == {0, 1}


def test_display_smoothing_leaves_areas_alone():
    lon, lat, _ = _bimodal(n=30_000)
    sky = SkyPosterior.from_samples(lon, lat=lat)
    a90 = sky.credible_area(0.9)
    d = sky.density(3.0)
    assert d.shape == sky.prob.shape and abs(d.sum() - 1) < 1e-9
    assert sky.density() is sky.prob
    assert sky.credible_area(0.9) == a90


def test_plots_run(tmp_path):
    import matplotlib.pyplot as plt

    from hyperwave.plots.skymap import sky_style

    rc_before = dict(matplotlib.rcParams)
    lon, lat, truth = _bimodal(n=30_000)
    sky = SkyPosterior.from_samples(lon, lat=lat)
    fig, axes = plot_skymap(sky, truth=truth, fov={"Rubin": 9.6}, save=str(tmp_path / "a.png"))
    assert len(axes) == 2
    assert axes[1].get_legend() is not None
    labels = [t.get_text() for t in axes[0].texts]
    assert "dominant" in labels and "antipodal" in labels
    fig, axes = plot_skymap(sky, dark=True, zoom=False)
    assert len(axes) == 1
    lon2, lat2, _ = _bimodal(n=30_000, sigma_deg=5.0, seed=7)
    sky2 = SkyPosterior.from_samples(lon2, lat=lat2)
    fig, axes = compare_skymaps([sky, sky2], ["a", "b"], truth=truth)
    assert len(axes) == 2
    lon3, lat3, _ = _bimodal(n=30_000, sigma_deg=12.0, seed=9)
    fig, axes = compare_skymaps([sky, sky2], ["a", "b"], truth=truth, allsky=False,
                                reference=SkyPosterior.from_samples(lon3, lat=lat3),
                                window=(200.0, 260.0, -60.0, -20.0), save=str(tmp_path / "b.pdf"))
    assert len(axes) == 1
    assert axes[0].get_xlim() == (200.0, 260.0) and axes[0].get_ylim() == (-60.0, -20.0)
    legend = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert len(legend) == 4 and legend[2].startswith("unseeded") and legend[3] == "injection"
    fig, axes = compare_skymaps([sky, sky2], ["a", "b"], level=0.9, zoom=False)
    assert len(axes) == 1
    sky.plot(truth=truth, astro=True, lon_labels=True)
    sky.plot_zoom(mode_index=1, label="antipode")
    with sky_style():
        assert matplotlib.rcParams["xtick.direction"] == "in"
    assert dict(matplotlib.rcParams) == rc_before          # the global style is left alone
    plt.close("all")
