"""Sky-map demo: synthetic bimodal LISA posterior -> Mollweide + zoom, light and dark.

Usage: python examples/lisa/skymap_demo.py <output_dir>
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
from hyperwave.plots.skymap import SkyPosterior, plot_skymap, compare_skymaps
out = sys.argv[1]
lon0, lat0 = 4.052962883, -1.081082148          # Case-1 injection (ecliptic, rad)

def posterior(n, f_dom, sig_deg, elong=2.5, seed=0):
    r = np.random.default_rng(seed)
    s = np.radians(sig_deg); n1 = int(f_dom * n)
    def blob(lc, bc, m):
        u = r.normal(0, s * elong, m); v = r.normal(0, s, m); th = np.radians(35.0)
        dx = u * np.cos(th) - v * np.sin(th); dy = u * np.sin(th) + v * np.cos(th)
        return lc + dx / np.cos(bc), bc + dy
    l1, b1 = blob(lon0, lat0, n1); l2, b2 = blob(lon0 + np.pi, -lat0, n - n1)
    return np.concatenate([l1, l2]) % (2*np.pi), np.clip(np.concatenate([b1, b2]), -np.pi/2, np.pi/2)

sky_sub = SkyPosterior.from_samples(*posterior(60000, 0.81, 2.2, seed=1)[:1], lat=posterior(60000, 0.81, 2.2, seed=1)[1])
sky_fg  = SkyPosterior.from_samples(*posterior(60000, 0.53, 2.2, seed=2)[:1], lat=posterior(60000, 0.53, 2.2, seed=2)[1])
sky_cl  = SkyPosterior.from_samples(*posterior(60000, 0.72, 2.6, seed=3)[:1], lat=posterior(60000, 0.72, 2.6, seed=3)[1])
s = sky_sub.summary(truth=(lon0, lat0))
print("A90=%.0f A50=%.0f f_dom=%.2f n_modes=%d truthCL=%.2f searched=%.1f" % (s["area_90"], s["area_50"], s["f_dominant"], s["n_modes"], s["truth_credible_level"], s["searched_area"]))
for m in s["modes"]:
    print("MODE p=%.2f hpd=%.2f area=%.0f at (%.1f,%.1f) antipode_of=%s" % (m["mass_total"], m["mass"], m["area"], m["lon_deg"], m["lat_deg"], m["antipode_of"]))
fov = {"Rubin": 9.6, "Athena": 0.4}
for dark in (False, True):
    tag = "dark" if dark else "light"
    print("plot_skymap", tag); plot_skymap(sky_sub, truth=(lon0, lat0), fov=fov, dark=dark, save=f"{out}/skymap_{tag}.png")
    print("compare", tag); compare_skymaps([sky_sub, sky_fg, sky_cl], ["HERON-subtracted", "foreground", "clean"], truth=(lon0, lat0), dark=dark, fov=fov, save=f"{out}/compare_{tag}.png")
# zoom panel alone, with a broad reference posterior as a dotted outline
sky_ref = SkyPosterior.from_samples(*posterior(60000, 0.5, 9.0, seed=4)[:1], lat=posterior(60000, 0.5, 9.0, seed=4)[1])
compare_skymaps([sky_sub, sky_fg, sky_cl], ["HERON-subtracted", "foreground", "clean"], truth=(lon0, lat0),
                reference=sky_ref, allsky=False, save=f"{out}/compare_zoom.pdf")
try:
    sky_sub.write_fits(f"{out}/alert.fits", nside=64); print("fits ok")
except ImportError:
    print("healpy not installed: skipping the HEALPix FITS export")
