"""Sky-localization maps from posterior samples.

Turns ``(longitude, sin latitude)`` posterior samples -- e.g. the LISA
``(lam, sin_beta)`` columns of an SMBHB run -- into a :class:`SkyPosterior`
that knows how to

* measure highest-posterior-density (HPD) credible areas with the *canonical*
  HyperWave estimator (equal-area ``(lon, sin lat)`` grid at 0.5 degree,
  one-cell Gaussian smoothing, greedy HPD), matching
  :func:`hyperwave.detectors.lisa.smbhb.sky_credible_area`;
* find the distinct sky modes inside a credible region, their probability
  mass and whether they are antipodal images of each other (the classic LISA
  degeneracy);
* report the credible level and searched area at which a known position is
  recovered;
* export a HEALPix probability map (FITS) for telescope teams (optional,
  needs ``healpy``);
* draw publication-quality figures in the style of the HERON pointing figure:
  an all-sky Mollweide map (density on a dark sky, labelled sky modes), a
  zoom on the dominant mode with filled 50/90% regions and instrument fields
  of view, and multi-run contour overlays.  Every plot has a ``dark`` switch
  (dark page) for talks.

Areas are always measured with the canonical estimator; only the *drawn*
density and contours are smoothed further (``DISPLAY_SMOOTH`` cells).

Only numpy, scipy and matplotlib are required.

Example
-------
>>> sky = SkyPosterior.from_samples(lam, sinlat=sin_beta)
>>> sky.credible_area(0.9)                    # deg^2
>>> sky.modes(0.9)                            # [{'mass': 0.81, ...}, {'mass': 0.19, ...}]
>>> fig, axes = plot_skymap(sky, truth=(lam0, beta0), fov={"Rubin": 9.6, "Athena": 0.4})
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "SkyPosterior",
    "plot_skymap",
    "compare_skymaps",
    "sky_style",
    "DEG2",
]

DEG2 = (180.0 / np.pi) ** 2
TWO_PI = 2.0 * np.pi

#: page colours (the sky itself is always drawn with ``SKY_CMAP``)
_PALETTE = {
    "light": dict(fg="#222222", muted="#737373", face="white"),
    "dark": dict(fg="#eeeeee", muted="#a0a0a0", face="#121212"),
}
#: density colour map of the all-sky panel (dark sky, as in the HERON pointing figure)
SKY_CMAP = "magma"
#: colour of a single posterior in the zoom panel (coral)
HERO_COLOR = "#E63946"
#: colours / line styles for multi-run overlays (coral, violet, cornflower, amber, green)
OVERLAY_COLORS = ["#E63946", "#9467BD", "#6495ED", "#f2ab15", "#009e73"]
OVERLAY_STYLES = ["solid", "dashed", "solid", "dashdot", "solid"]
#: smoothing (cells) of the *drawn* density and contours; areas always use ``SkyPosterior.smooth``
DISPLAY_SMOOTH = 3.0


def _wrap_pi(lon):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(lon, dtype=float) + np.pi) % TWO_PI - np.pi


def _circ_mean(lon, w):
    return np.arctan2(np.sum(w * np.sin(lon)), np.sum(w * np.cos(lon))) % TWO_PI


def _unit(lon, lat):
    """Unit vectors, shape ``(..., 3)``."""
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def _chord(deg):
    """Chord length on the unit sphere subtending ``deg`` degrees."""
    return 2.0 * np.sin(0.5 * np.radians(deg))


def _hpd_levels(P, ps):
    """Greedy-HPD thresholds of the array ``P`` for the credible masses ``ps``."""
    flat = np.sort(np.asarray(P).ravel())[::-1]
    c = np.cumsum(flat) / flat.sum()
    return [float(flat[min(int(np.searchsorted(c, q)), flat.size - 1)]) for q in ps]


def _fmt_area(a):
    """Area in deg^2 with a sensible number of digits."""
    return f"{a:.0f}" if a >= 10 else (f"{a:.1f}" if a >= 1 else f"{a:.2g}")


def _pct(p):
    import matplotlib as mpl

    return f"{int(round(100 * p))}" + (r"\%" if mpl.rcParams["text.usetex"] else "%")


@contextmanager
def sky_style(usetex: bool = False):
    """Matplotlib style of the sky-map figures (serif, inward ticks on all sides).

    The plotting functions of this module apply it themselves (on top of
    matplotlib's defaults), so the figures do not depend on the global
    ``rcParams`` and leave them unchanged.  Use it for your own additions to
    a figure, so that they share its fonts and ticks::

        fig, axes = plot_skymap(sky)
        with sky_style():
            axes[1].set_title("alert at merger minus 30 days")
            fig.savefig("sky.pdf", bbox_inches="tight")

    ``usetex=True`` renders the text with LaTeX (needs a LaTeX installation).
    """
    import matplotlib as mpl
    from matplotlib import font_manager

    # a concrete font name (not the generic "serif") stays with the figure after the context exits
    installed = {f.name for f in font_manager.fontManager.ttflist}
    serif = next((n for n in ("Times New Roman", "Times", "Nimbus Roman", "STIXGeneral")
                  if n in installed), "DejaVu Serif")
    rc = {
        "text.usetex": bool(usetex),
        "font.family": [serif],
        "mathtext.fontset": "stix",
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "axes.linewidth": 0.9, "axes.grid": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "savefig.dpi": 300,
    }
    if usetex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    with mpl.rc_context():
        mpl.rcdefaults()          # start from matplotlib's defaults, whatever was set globally
        mpl.rcParams.update(rc)
        yield


@dataclass
class SkyPosterior:
    """Gridded sky posterior on an equal-area ``(lon, sin lat)`` grid.

    Build it with :meth:`from_samples` or :meth:`from_result`.  ``prob`` holds
    the probability per cell (sums to one), shape ``(n_lon, n_sinlat)``;
    longitude runs ``[0, 2 pi)`` along axis 0 and ``sin(lat)`` runs ``[-1, 1]``
    along axis 1.

    Parameters
    ----------
    lon, sinlat:
        Posterior samples (radians / dimensionless).
    weights:
        Optional importance weights.
    nbins:
        ``(n_lon, n_sinlat)``; the default is the canonical 0.5 degree grid.
    smooth:
        Gaussian smoothing in cells (the canonical value is 1.0).
    periodic:
        Smooth across the longitude seam (recommended; the canonical scalar
        estimator does not, which differs only when mass sits at lon = 0).
    """

    lon: np.ndarray
    sinlat: np.ndarray
    weights: Optional[np.ndarray] = None
    nbins: tuple[int, int] = (720, 360)
    smooth: float = 1.0
    periodic: bool = True
    # derived
    prob: np.ndarray = field(init=False, repr=False)
    lon_edges: np.ndarray = field(init=False, repr=False)
    sinlat_edges: np.ndarray = field(init=False, repr=False)

    # -- construction ----------------------------------------------------------
    def __post_init__(self) -> None:
        from scipy.ndimage import gaussian_filter

        self.lon = np.asarray(self.lon, dtype=float).ravel() % TWO_PI
        self.sinlat = np.clip(np.asarray(self.sinlat, dtype=float).ravel(), -1.0, 1.0)
        if self.lon.shape != self.sinlat.shape:
            raise ValueError("lon and sinlat must have the same length")
        w = None if self.weights is None else np.asarray(self.weights, dtype=float).ravel()
        H, xe, ye = np.histogram2d(self.lon, self.sinlat, bins=list(self.nbins),
                                   range=[[0.0, TWO_PI], [-1.0, 1.0]], weights=w)
        tot = H.sum()
        if tot <= 0:
            raise ValueError("no probability mass on the sky grid")
        self._hist = H / tot
        self._cache = {}
        if self.smooth:
            mode = ("wrap", "nearest") if self.periodic else "reflect"
            H = gaussian_filter(H, self.smooth, mode=mode)
        self.prob = H / H.sum()
        self.lon_edges, self.sinlat_edges = xe, ye

    def density(self, smooth: Optional[float] = None) -> np.ndarray:
        """Probability per cell with ``smooth`` cells of Gaussian smoothing.

        ``None`` returns the canonical map ``prob``.  A larger value gives the
        smoother map used for drawing (it is never used for areas).
        """
        if smooth is None or smooth == self.smooth:
            return self.prob
        key = ("density", float(smooth))
        if key not in self._cache:
            from scipy.ndimage import gaussian_filter

            H = gaussian_filter(self._hist, smooth, mode=("wrap", "nearest")) if smooth else self._hist
            self._cache[key] = H / H.sum()
        return self._cache[key]

    @classmethod
    def from_samples(cls, lon, lat=None, *, sinlat=None, weights=None, degrees=False,
                     **kwargs) -> "SkyPosterior":
        """From longitude and either ``lat`` or ``sinlat`` samples.

        ``degrees=True`` converts ``lon``/``lat`` from degrees.
        """
        lon = np.asarray(lon, dtype=float)
        if degrees:
            lon = np.radians(lon)
        if sinlat is None:
            if lat is None:
                raise ValueError("give either lat or sinlat")
            lat = np.asarray(lat, dtype=float)
            sinlat = np.sin(np.radians(lat) if degrees else lat)
        return cls(lon, np.asarray(sinlat, dtype=float), weights=weights, **kwargs)

    @classmethod
    def from_result(cls, result, lon: str = "lam", sinlat: Optional[str] = "sin_beta",
                    lat: Optional[str] = None, weights=None, **kwargs) -> "SkyPosterior":
        """From a :class:`hyperwave.Result` (default LISA names ``lam``/``sin_beta``)."""
        lon_s = result[lon]
        if lat is not None:
            return cls.from_samples(lon_s, lat=result[lat], weights=weights, **kwargs)
        return cls.from_samples(lon_s, sinlat=result[sinlat], weights=weights, **kwargs)

    # -- geometry helpers --------------------------------------------------------
    @property
    def cell_area_deg2(self) -> float:
        """Area of one grid cell in deg^2 (all cells are equal-area)."""
        return float((self.lon_edges[1] - self.lon_edges[0])
                     * (self.sinlat_edges[1] - self.sinlat_edges[0]) * DEG2)

    @property
    def lon_centers(self) -> np.ndarray:
        return 0.5 * (self.lon_edges[1:] + self.lon_edges[:-1])

    @property
    def lat_centers(self) -> np.ndarray:
        """Latitude (radians) of the cell centres along axis 1."""
        return np.arcsin(0.5 * (self.sinlat_edges[1:] + self.sinlat_edges[:-1]))

    def _cell(self, lon, lat):
        i = np.searchsorted(self.lon_edges, float(lon) % TWO_PI, side="right") - 1
        j = np.searchsorted(self.sinlat_edges, np.sin(float(lat)), side="right") - 1
        return int(np.clip(i, 0, self.nbins[0] - 1)), int(np.clip(j, 0, self.nbins[1] - 1))

    # -- credible regions --------------------------------------------------------
    def level(self, p: float) -> float:
        """Probability-per-cell threshold enclosing credible mass ``p`` (greedy HPD)."""
        flat = np.sort(self.prob.ravel())[::-1]
        c = np.cumsum(flat)
        n = int(np.searchsorted(c, p) + 1)
        return float(flat[min(n, flat.size) - 1])

    def credible_mask(self, p: float) -> np.ndarray:
        return self.prob >= self.level(p)

    def credible_area(self, p: float = 0.9) -> float:
        """HPD credible area [deg^2] at level ``p`` (canonical estimator)."""
        return float(self.credible_mask(p).sum() * self.cell_area_deg2)

    def credible_areas(self, levels: Sequence[float] = (0.5, 0.9)) -> dict[float, float]:
        return {float(p): self.credible_area(p) for p in levels}

    def truth_credible_level(self, lon, lat) -> float:
        """Smallest HPD credible level that contains the point ``(lon, lat)`` [rad].

        This is the searched probability; for a calibrated posterior it is
        uniform over repeated injections.
        """
        i, j = self._cell(lon, lat)
        p0 = self.prob[i, j]
        return float(self.prob[self.prob >= p0].sum())

    def searched_area(self, lon, lat) -> float:
        """Area [deg^2] a greedy search covers before reaching ``(lon, lat)``."""
        i, j = self._cell(lon, lat)
        return float((self.prob >= self.prob[i, j]).sum() * self.cell_area_deg2)

    # -- modes -------------------------------------------------------------------
    def _modes_full(self, p: float, min_mass: float, merge_deg: float):
        """``(modes, region, partition)``; the maps hold mode indices (-1 = none)."""
        key = ("modes", float(p), float(min_mass), float(merge_deg))
        if key in self._cache:
            return self._cache[key]
        from scipy.ndimage import label
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        mask = self.credible_mask(p)
        lab, n = label(mask)
        empty = ([], np.full(mask.shape, -1), np.full(mask.shape, -1))
        if n == 0:
            self._cache[key] = empty
            return empty
        ii, jj = np.nonzero(mask)
        comp = lab[ii, jj]
        # coarse points: mean direction of each island inside ~5 degree super-cells
        cf = max(1, int(round(self.nbins[0] * 5.0 / 360.0)))
        ckey = (comp.astype(np.int64) * 8192 + ii // cf) * 8192 + jj // cf
        _, inv = np.unique(ckey, return_inverse=True)
        inv = np.asarray(inv).ravel()
        npts = int(inv.max()) + 1
        pts = np.zeros((npts, 3))
        np.add.at(pts, inv, _unit(self.lon_centers[ii], self.lat_centers[jj]))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pcomp = np.zeros(npts, dtype=int)
        pcomp[inv] = comp
        # islands are one mode when they touch across the seam or lie within merge_deg
        rows, cols = [], []
        if self.periodic:
            both = (lab[0] > 0) & (lab[-1] > 0)
            rows += list(lab[0][both] - 1)
            cols += list(lab[-1][both] - 1)
        if merge_deg > 0 and npts > 1:
            pairs = cKDTree(pts).query_pairs(_chord(merge_deg), output_type="ndarray")
            rows += list(pcomp[pairs[:, 0]] - 1)
            cols += list(pcomp[pairs[:, 1]] - 1)
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, root = connected_components(graph, directed=False)
        lut = np.concatenate([[0], root + 1])
        lab = lut[lab]
        plabel = lut[pcomp]

        LON, LAT = np.meshgrid(self.lon_centers, self.lat_centers, indexing="ij")
        out = []
        for k in np.unique(lab[lab > 0]):
            sel = lab == k
            w = self.prob[sel]
            mass = float(w.sum())
            if mass < min_mass:
                continue
            lon_c = float(_circ_mean(LON[sel], w))
            lat_c = float(np.sum(w * LAT[sel]) / mass)
            ipk = int(np.argmax(w))
            out.append(dict(mass=mass, area=float(sel.sum() * self.cell_area_deg2),
                            lon=lon_c, lat=lat_c, lon_deg=np.degrees(lon_c),
                            lat_deg=np.degrees(lat_c), lon_peak=float(LON[sel][ipk]),
                            lat_peak=float(LAT[sel][ipk]), _label=int(k)))
        if not out:
            self._cache[key] = empty
            return empty
        out.sort(key=lambda d: -d["mass"])
        index_of = np.full(int(lab.max()) + 1, -1)
        for i, d in enumerate(out):
            index_of[d["_label"]] = i
        region = index_of[lab]
        region[lab == 0] = -1
        # every direction on the sky belongs to the nearest kept mode, so that
        # ``mass_total`` partitions the whole posterior (sums to one)
        keep = index_of[plabel] >= 0
        tree = cKDTree(pts[keep])
        owner = index_of[plabel[keep]]
        partition = owner[tree.query(_unit(LON, LAT).reshape(-1, 3))[1]].reshape(lab.shape)
        tol = _chord(max(merge_deg, 15.0))
        for i, d in enumerate(out):
            d.pop("_label")
            d["mass_total"] = float(self.prob[partition == i].sum())
            d["antipode_of"] = None
        for a_idx, a in enumerate(out):
            dist, q = tree.query(_unit(a["lon"] + np.pi, -a["lat"]))
            b_idx = int(owner[q])
            if dist < tol and b_idx > a_idx and out[b_idx]["antipode_of"] is None:
                out[b_idx]["antipode_of"] = a_idx
        self._cache[key] = (out, region, partition)
        return self._cache[key]

    def modes(self, p: float = 0.9, min_mass: float = 0.005, merge_deg: float = 45.0) -> list[dict]:
        """Sky modes inside the ``p`` credible region, by decreasing mass.

        Islands of the credible region that lie within ``merge_deg`` degrees of
        each other (or touch across the longitude seam) form one mode, so a
        fragmented island is not counted several times; ``merge_deg=0`` keeps
        every connected island separate.

        Each entry has ``mass`` (posterior probability inside the mode's HPD
        cells, summing over modes to about ``p``), ``mass_total`` (probability
        of the whole sky attributed to this mode by nearest-mode partition,
        summing to one -- the quantity to quote as a mode probability),
        ``area`` [deg^2], ``lon``/``lat`` (probability-weighted centre,
        radians), ``lon_deg``/``lat_deg``, ``lon_peak``/``lat_peak`` (densest
        cell, radians) and ``antipode_of`` -- the index of the mode whose
        antipodal point ``(lon + pi, -lat)`` falls on this one, or ``None``.
        """
        return [dict(d) for d in self._modes_full(p, min_mass, merge_deg)[0]]

    def mode_partition(self, p: float = 0.9, min_mass: float = 0.005,
                       merge_deg: float = 45.0) -> np.ndarray:
        """Index of the nearest mode of :meth:`modes` for every grid cell."""
        return self._modes_full(p, min_mass, merge_deg)[2]

    def dominant_mode_fraction(self, p: float = 0.9) -> float:
        """Posterior probability attributed to the strongest mode (``mass_total``)."""
        m = self.modes(p)
        return float(m[0]["mass_total"]) if m else float("nan")

    def summary(self, levels: Sequence[float] = (0.5, 0.9), truth=None) -> dict:
        """Dictionary of the numbers a paper or alert would quote."""
        s = {f"area_{int(round(100 * p))}": self.credible_area(p) for p in levels}
        m = self.modes(max(levels))
        s["n_modes"] = len(m)
        s["f_dominant"] = self.dominant_mode_fraction(max(levels))
        s["modes"] = m
        if truth is not None:
            s["truth_credible_level"] = self.truth_credible_level(*truth)
            s["searched_area"] = self.searched_area(*truth)
        return s

    # -- HEALPix export ----------------------------------------------------------
    def to_healpix(self, nside: int = 64, nest: bool = False) -> np.ndarray:
        """Probability per HEALPix pixel (sums to one). Requires ``healpy``.

        The samples are binned directly with ``ang2pix`` (no smoothing); use a
        coarser ``nside`` for small sample sets.  Coordinates are the input
        frame (ecliptic for LISA); longitude is ``phi`` and colatitude
        ``pi/2 - lat``.
        """
        import healpy as hp  # optional

        theta = np.pi / 2 - np.arcsin(self.sinlat)
        pix = hp.ang2pix(nside, theta, self.lon, nest=nest)
        m = np.bincount(pix, weights=self.weights, minlength=hp.nside2npix(nside)).astype(float)
        return m / m.sum()

    def write_fits(self, path: str, nside: int = 64, nest: bool = False,
                   coord: str = "E", extra_header: Optional[Mapping[str, object]] = None) -> str:
        """Write a HEALPix ``PROB`` map in FITS (LIGO/Virgo-style). Requires ``healpy``."""
        import healpy as hp  # optional

        m = self.to_healpix(nside=nside, nest=nest)
        hdr = [("PIXTYPE", "HEALPIX"), ("ORDERING", "NESTED" if nest else "RING"),
               ("COORDSYS", coord), ("NSIDE", nside), ("CREATOR", "HyperWave"),
               ("AREA90", self.credible_area(0.9)), ("AREA50", self.credible_area(0.5))]
        if extra_header:
            hdr += [(str(k)[:8].upper(), v) for k, v in extra_header.items()]
        hp.write_map(path, m, nest=nest, coord=coord, column_names=["PROB"],
                     extra_header=hdr, overwrite=True, dtype=np.float64)
        return path

    # -- plotting ----------------------------------------------------------------
    def _rolled(self, center_lon: float, P: Optional[np.ndarray] = None):
        """Grid re-centred on ``center_lon`` so that x = wrap(lon - center) is monotonic."""
        nlon = self.nbins[0]
        k = int(np.round(center_lon / TWO_PI * nlon)) % nlon
        shift = nlon // 2 - k
        P = np.roll(self.prob if P is None else P, shift, axis=0)
        x_edges = self.lon_edges - np.pi            # [-pi, pi]
        return x_edges, P

    def plot(self, ax=None, *, projection: str = "mollweide", levels: Sequence[float] = (0.5, 0.9),
             truth=None, dark: bool = False, center_lon: float = 0.0, astro: bool = False,
             cmap=None, contours: bool = True, label_modes: bool = True,
             modes_p: float = 0.9, title: Optional[str] = None, grid: bool = True,
             smooth: float = DISPLAY_SMOOTH, lon_labels: bool = False,
             vmax_quantile: float = 1.0, usetex: bool = False):
        """All-sky map: posterior density on a dark sky, HPD contours, labelled modes.

        ``projection`` is any matplotlib geographic projection (``'mollweide'``,
        ``'hammer'``, ``'aitoff'``).  Longitude increases to the right and the
        map is centred on ``center_lon``; ``astro=True`` puts east on the left.
        ``truth=(lon, lat)`` in radians marks the injected position.  The
        density and the thin ``levels`` contours are drawn with ``smooth``
        cells of smoothing.  The strongest mode is labelled "dominant", its
        antipodal image "antipodal".  ``dark`` switches to a dark page.
        Returns the axes.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import patheffects

        pal = _PALETTE["dark" if dark else "light"]
        with sky_style(usetex):
            if ax is None:
                fig = plt.figure(figsize=(6.4, 3.6))
                fig.patch.set_facecolor(pal["face"])
                ax = fig.add_subplot(111, projection=projection)
            cm = matplotlib.colormaps[cmap or SKY_CMAP] if isinstance(cmap or SKY_CMAP, str) else cmap
            sky_face = cm(0.0)
            ax.set_facecolor(sky_face)
            D = self.density(smooth)
            x_edges, P = self._rolled(center_lon, D)
            sgn = -1.0 if astro else 1.0
            xc = sgn * 0.5 * (x_edges[1:] + x_edges[:-1])
            yc = self.lat_centers
            if astro:
                xc, P = xc[::-1], P[::-1]
            vmax = np.quantile(P[P > 0], vmax_quantile) if vmax_quantile < 1.0 else P.max()
            ax.pcolormesh(xc, yc, P.T, cmap=cm, vmin=0, vmax=vmax, shading="gouraud", rasterized=True)
            if contours and levels:
                lv = sorted(set(_hpd_levels(D, levels)))
                lws = [0.4, 0.8] if len(lv) == 2 else 0.6
                ax.contour(xc, yc, P.T, levels=lv, colors="white", linewidths=lws, alpha=0.85)
            if truth is not None:
                ax.plot(sgn * _wrap_pi(truth[0] - center_lon), truth[1], marker="*", ms=15, mfc="white",
                        mec="k", mew=1.0, ls="none", zorder=6)
            if label_modes:
                stroke = [patheffects.withStroke(linewidth=1.8, foreground=sky_face)]
                for i, m in enumerate(self.modes(modes_p)[:6]):
                    if i == 0:
                        name = "dominant"
                    elif m["antipode_of"] == 0:
                        name = "antipodal"
                    else:
                        name = f"mode {i + 1}"
                    mx, my = sgn * _wrap_pi(m["lon_peak"] - center_lon), m["lat_peak"]
                    # leader line away from the equator, towards it for high-latitude modes
                    up = (my >= 0) == (abs(my) < 0.45)
                    ax.annotate(name, xy=(mx, my), xytext=(mx, my + (0.6 if up else -0.6)),
                                fontsize=8, color="white", ha="center", va="bottom" if up else "top",
                                arrowprops=dict(arrowstyle="-", color="0.8", lw=0.5,
                                                shrinkA=1, shrinkB=3),
                                path_effects=stroke, zorder=7)
            ax.grid(grid, lw=0.3, color="white", alpha=0.25)
            ax.tick_params(axis="both", which="both", labelsize=7, colors=pal["muted"], length=0)
            if lon_labels:
                ticks = np.radians(np.arange(-150, 180, 30))
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{int(round(np.degrees(sgn * t + center_lon))) % 360}°" for t in ticks],
                                   fontsize=7, color="0.75")
            else:
                ax.set_xticklabels([])
            for sp in ax.spines.values():
                sp.set_edgecolor(pal["muted"])
                sp.set_linewidth(0.6)
            if title:
                ax.set_title(title, color=pal["fg"], pad=8)
        return ax

    def _zoom_window(self, mode, cells_dlon_deg, cells_lat_deg, pad):
        """Axis limits (unwrapped longitude degrees, latitude degrees) around mode cells."""
        x0, x1 = float(np.min(cells_dlon_deg)), float(np.max(cells_dlon_deg))
        y0, y1 = float(np.min(cells_lat_deg)), float(np.max(cells_lat_deg))
        hx = max(0.5 * (x1 - x0), 1.0) * pad
        hy = max(0.5 * (y1 - y0), 1.0) * pad
        cx, cy = mode["lon_deg"] + 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        return cx - hx, cx + hx, max(cy - hy, -90.0), min(cy + hy, 90.0)

    def _mode_cells(self, region_mask, mode):
        """Longitude offset from the mode centre and latitude (degrees) of the masked cells."""
        LON, LAT = np.meshgrid(self.lon_centers, self.lat_centers, indexing="ij")
        return np.degrees(_wrap_pi(LON[region_mask] - mode["lon"])), np.degrees(LAT[region_mask])

    def _draw_zoom(self, ax, mode, levels, smooth, color, linestyle="solid", filled=True,
                   linewidths=(1.0, 1.6), alpha=1.0):
        """HPD contours (optionally filled) in unwrapped degrees around ``mode``; returns vertices."""
        D = self.density(smooth)
        x_edges, P = self._rolled(mode["lon"], D)
        xc = np.degrees(0.5 * (x_edges[1:] + x_edges[:-1])) + mode["lon_deg"]
        yc = np.degrees(self.lat_centers)
        lv = sorted(set(_hpd_levels(D, levels)))
        top = P.max() * 1.0001
        if filled:
            ax.contourf(xc, yc, P.T, levels=[lv[0], top], colors=[color], alpha=0.22)
            if len(lv) > 1:
                ax.contourf(xc, yc, P.T, levels=[lv[-1], top], colors=[color], alpha=0.5)
        lws = list(linewidths)[: len(lv)] if len(lv) <= 2 else 1.2
        if len(lv) == 1:
            lws = [linewidths[-1]]
        cs = ax.contour(xc, yc, P.T, levels=lv, colors=[color], linewidths=lws,
                        linestyles=linestyle, alpha=alpha)
        paths = cs.get_paths() if hasattr(cs, "get_paths") else [q for c in cs.collections
                                                                 for q in c.get_paths()]
        verts = [q.vertices for q in paths if len(q.vertices)]
        return np.concatenate(verts) if verts else np.empty((0, 2))

    def plot_zoom(self, ax=None, *, levels: Sequence[float] = (0.5, 0.9), truth=None, dark: bool = False,
                  pad: float = 1.5, mode_index: int = 0, fov: Optional[Mapping[str, float]] = None,
                  filled: bool = True, color: Optional[str] = None, linestyle: str = "solid",
                  label: Optional[str] = None, title: Optional[str] = None,
                  window: Optional[Sequence[float]] = None, smooth: float = DISPLAY_SMOOTH,
                  legend: bool = True, legend_title: Optional[str] = None, usetex: bool = False):
        """Zoom (degrees) on one mode: filled HPD regions, injection, fields of view.

        The ``levels`` regions are drawn with ``smooth`` cells of smoothing
        (areas are not affected).  ``fov`` maps instrument name to
        field-of-view area in deg^2 (e.g. ``{"Rubin": 9.6, "Athena": 0.4}``);
        each is drawn as a circle of equal area centred on the mode.  ``pad``
        scales the window around the mode's extent; ``window=(lon_min,
        lon_max, lat_min, lat_max)`` in degrees sets it explicitly.  Returns
        the axes.
        """
        import matplotlib.pyplot as plt

        pal = _PALETTE["dark" if dark else "light"]
        modes, region, _ = self._modes_full(max(levels), 0.005, 45.0)
        if not modes:
            raise ValueError("no mode found at the requested level")
        mi = min(mode_index, len(modes) - 1)
        m = modes[mi]
        color = color or HERO_COLOR
        with sky_style(usetex):
            if ax is None:
                fig, ax = plt.subplots(figsize=(6.0, 4.6))
                fig.patch.set_facecolor(pal["face"])
            verts = self._draw_zoom(ax, m, levels, smooth, color, linestyle, filled)
            grow = window is None
            if window is None:
                window = self._zoom_window(m, *self._mode_cells(region == mi, m), pad)
            handles = _zoom_decorate(ax, m, pal, window, truth=truth, fov=fov)
            if label:
                from matplotlib.lines import Line2D

                handles.insert(0, Line2D([0], [0], color=color, lw=2.4, ls=linestyle, label=label))
            if title:
                ax.set_title(title, color=pal["fg"])
            if legend and handles:
                _zoom_legend(ax, handles, pal, verts, title=legend_title, grow=grow)
        return ax


def _zoom_decorate(ax, mode, pal, window, truth=None, fov=None):
    """Limits, labels, ticks, injection star and fields of view; returns legend handles."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import FuncFormatter

    handles = []
    cosb = max(np.cos(mode["lat"]), 0.05)
    ax.set_facecolor(pal["face"])
    if fov:
        ls_cycle = ["--", ":", "-."]
        for k, (name, area) in enumerate(fov.items()):
            r = np.sqrt(float(area) / np.pi)                            # equal-area radius [deg]
            ax.add_patch(Ellipse((np.degrees(mode["lon_peak"] - mode["lon"]) + mode["lon_deg"],
                                  np.degrees(mode["lat_peak"])), 2 * r / cosb, 2 * r, fill=False,
                                 lw=1.0, ls=ls_cycle[k % 3], ec=pal["fg"], zorder=5))
            handles.append(Line2D([0], [0], color=pal["fg"], lw=1.0, ls=ls_cycle[k % 3],
                                  label=f"{name} FoV ({area:g}$\\,$deg$^2$)"))
    if truth is not None:
        tx = mode["lon_deg"] + np.degrees(_wrap_pi(truth[0] - mode["lon"]))
        ax.plot(tx, np.degrees(truth[1]), marker="*", ms=18, mfc="white", mec="k", mew=1.3,
                ls="none", zorder=6)
        handles.append(Line2D([0], [0], marker="*", ms=13, mfc="white", mec="k", mew=1.1,
                              ls="none", label="injection"))
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect(1.0 / cosb)      # conformal at the mode: 1 deg of longitude spans cos(beta) deg on the sky
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v % 360:g}"))
    ax.set_xlabel(r"ecliptic longitude $\lambda$ [deg]", color=pal["fg"])
    ax.set_ylabel(r"ecliptic latitude $\beta$ [deg]", color=pal["fg"])
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, colors=pal["fg"])
    for sp in ax.spines.values():
        sp.set_edgecolor(pal["fg"])
    return handles


def _zoom_legend(ax, handles, pal, verts, title=None, grow=True):
    """Legend in a corner free of contours.

    If every corner is busy and ``grow`` is set, the window is extended
    downwards (or upwards) to make room; failing that, or when the legend is
    wider than the axes, it goes below the axes.
    """
    from matplotlib.axes import Axes

    def make(loc, **kw):
        # the base-class legend: Axes subclasses registered by other packages
        # (e.g. gwpy) restyle the legend handles
        leg = Axes.legend(ax, handles=handles, loc=loc, frameon=False, title=title, **kw)
        leg._legend_box.align = "left"
        for t in leg.get_texts() + [leg.get_title()]:
            t.set_color(pal["fg"])
        return leg

    leg = make("lower left")
    if not len(verts):
        return leg
    try:                                    # legend size in axes fractions
        fig = ax.figure
        renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else fig._get_renderer()
        ax.apply_aspect()
        box = leg.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        w, h = box.width + 0.03, box.height + 0.03
    except Exception:
        w, h = 0.6, min(0.07 * (len(handles) + (1 if title else 0)) + 0.03, 0.6)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    def busy(loc, ylo, yhi):
        fx = (verts[:, 0] - x0) / (x1 - x0)
        fy = (verts[:, 1] - ylo) / (yhi - ylo)
        inx = fx < w if "left" in loc else fx > 1 - w
        iny = fy < h if "lower" in loc else fy > 1 - h
        return int(np.sum(inx & iny & (fx > 0) & (fx < 1) & (fy > 0) & (fy < 1)))

    loc = None
    if w <= 1.0:
        locs = ["lower left", "lower right", "upper left", "upper right"]
        counts = [busy(c, y0, y1) for c in locs]
        if min(counts) == 0 or not grow:
            loc = locs[int(np.argmin(counts))]
        else:                               # extend the window by up to 60% of its height
            step = 0.05 * (y1 - y0)
            for k in range(1, 13):
                if y0 - k * step >= -90.0 and busy("lower left", y0 - k * step, y1) == 0:
                    loc = "lower left"
                    ax.set_ylim(y0 - k * step, y1)
                    break
                if y1 + k * step <= 90.0 and busy("upper left", y0, y1 + k * step) == 0:
                    loc = "upper left"
                    ax.set_ylim(y0, y1 + k * step)
                    break
    if loc != "lower left":
        leg.remove()
        # no free corner: below the axes, clear of the tick and axis labels
        leg = make(loc) if loc else make("upper left", bbox_to_anchor=(0.0, 0.0), borderaxespad=3.8)
    return leg


def _as_sky(obj, **kwargs) -> SkyPosterior:
    if isinstance(obj, SkyPosterior):
        return obj
    if hasattr(obj, "parameter_names"):           # hyperwave.Result
        return SkyPosterior.from_result(obj, **kwargs)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return SkyPosterior.from_samples(arr[:, 0], sinlat=arr[:, 1], **kwargs)
    raise TypeError("pass a SkyPosterior, a hyperwave.Result or an (N, 2) array of (lon, sinlat)")


def _area_label(levels):
    """``A_90 / A_50`` for the given credible levels (largest first)."""
    return r"$\,/\,$".join(f"$A_{{{int(round(100 * p))}}}$" for p in sorted(levels, reverse=True))


def _areas_text(sky, levels):
    return r"$\,/\,$".join(_fmt_area(sky.credible_area(p)) for p in sorted(levels, reverse=True)) \
        + r"$\,$deg$^2$"


def _n_modes_text(n):
    return {1: "one sky mode", 2: "both sky modes"}.get(n, f"all {n} sky modes")


def _figure(pal, allsky, zoom):
    import matplotlib.pyplot as plt

    if allsky and zoom:
        fig = plt.figure(figsize=(12.5, 4.3))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.16)
        axes = (fig.add_subplot(gs[0], projection="mollweide"), fig.add_subplot(gs[1]))
    elif allsky:
        fig = plt.figure(figsize=(6.4, 3.6))
        axes = (fig.add_subplot(111, projection="mollweide"),)
    else:
        fig = plt.figure(figsize=(7.6, 5.4))
        axes = (fig.add_subplot(111),)
    fig.patch.set_facecolor(pal["face"])
    return fig, axes


def plot_skymap(obj, *, truth=None, levels: Sequence[float] = (0.5, 0.9), dark: bool = False,
                fov: Optional[Mapping[str, float]] = None, zoom: bool = True, title: Optional[str] = None,
                save: Optional[str] = None, usetex: bool = False, **kwargs):
    """One-call figure: Mollweide all-sky map plus a zoom on the dominant mode.

    ``obj`` is a :class:`SkyPosterior`, a :class:`hyperwave.Result` (uses the
    ``lam``/``sin_beta`` columns) or an ``(N, 2)`` array of ``(lon, sinlat)``.
    The quoted areas are the canonical ones; the drawn regions are smoothed
    for display.  Returns ``(fig, axes)``.
    """
    sky = _as_sky(obj, **kwargs)
    pal = _PALETTE["dark" if dark else "light"]
    top = max(levels)
    with sky_style(usetex):
        fig, axes = _figure(pal, True, zoom)
        s = sky.summary(levels, truth=truth)
        if title is None:
            title = (f"sky posterior ($A_{{{int(round(100 * top))}}}={_fmt_area(sky.credible_area(top))}\\,$"
                     f"deg$^2$, {_n_modes_text(s['n_modes'])})")
        sky.plot(axes[0], levels=levels, truth=truth, dark=dark, title=title, usetex=usetex)
        if zoom:
            label = _areas_text(sky, levels) + f", $f_{{\\rm dom}}={s['f_dominant']:.2f}$"
            sky.plot_zoom(axes[1], levels=levels, truth=truth, dark=dark, fov=fov, label=label,
                          legend_title=f"{_area_label(levels)}, {_n_modes_text(s['n_modes'])}",
                          usetex=usetex)
        if save:
            fig.savefig(save, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    return fig, axes


def compare_skymaps(objs: Sequence, labels: Sequence[str], *, colors: Optional[Sequence[str]] = None,
                    linestyles: Optional[Sequence[str]] = None, levels: Sequence[float] = (0.5, 0.9),
                    level: Optional[float] = None, truth=None, dark: bool = False, allsky: bool = True,
                    zoom: bool = True, fov: Optional[Mapping[str, float]] = None, reference=None,
                    reference_label: str = "unseeded", window: Optional[Sequence[float]] = None,
                    pad: float = 1.3, legend_title: Optional[str] = None,
                    smooth: float = DISPLAY_SMOOTH, save: Optional[str] = None,
                    usetex: bool = False, **kwargs):
    """Overlay the HPD regions of several runs (e.g. subtracted / foreground / clean).

    The first run is the reference posterior: it is drawn filled, the zoom is
    centred on its dominant mode and the all-sky panel shows its density.  The
    other runs are drawn as contour lines.  The legend quotes the canonical
    areas of every run and ``f_dom``, the probability each run assigns to the
    part of the sky that belongs to that dominant mode.  ``reference`` is an
    optional extra posterior (e.g. an unseeded run) drawn as a dotted outer
    contour.  ``allsky=False`` gives the zoom panel alone; ``level=0.9`` is a
    shortcut for ``levels=(0.9,)``.  Returns ``(fig, axes)``.
    """
    from matplotlib.lines import Line2D

    if level is not None:
        levels = (level,)
    if not (allsky or zoom):
        raise ValueError("at least one of allsky and zoom must be True")
    skies = [_as_sky(o, **kwargs) for o in objs]
    ref = None if reference is None else _as_sky(reference, **kwargs)
    colors = list(colors or OVERLAY_COLORS)
    styles = list(linestyles or OVERLAY_STYLES)
    pal = _PALETTE["dark" if dark else "light"]
    top = max(levels)
    hero = skies[0]
    modes, region, partition = hero._modes_full(top, 0.005, 45.0)
    if not modes:
        raise ValueError("no mode found at the requested level")
    m = modes[0]
    island = partition == 0

    def f_dom(sky):
        return float(sky.prob[island].sum()) if sky.prob.shape == island.shape else float("nan")

    with sky_style(usetex):
        fig, axes = _figure(pal, allsky, zoom)
        if allsky:
            ax0 = axes[0]
            hero.plot(ax0, levels=levels, truth=truth, dark=dark, contours=False, smooth=smooth,
                      usetex=usetex)
            for sky, col, ls in zip(skies, colors, styles):
                D = sky.density(smooth)
                x_edges, P = sky._rolled(0.0, D)
                xc = 0.5 * (x_edges[1:] + x_edges[:-1])
                ax0.contour(xc, sky.lat_centers, P.T, levels=_hpd_levels(D, [top]), colors=[col],
                            linewidths=1.0, linestyles=ls)
            ax0.set_title(f"{_pct(top)} regions on the sky", color=pal["fg"], pad=8)
        if zoom:
            ax1 = axes[-1]
            verts, cells = [], [hero._mode_cells(region == 0, m)]
            for i, (sky, col, ls) in enumerate(zip(skies, colors, styles)):
                verts.append(sky._draw_zoom(ax1, m, levels, smooth, col, ls, filled=(i == 0),
                                            linewidths=(1.0, 1.6) if i == 0 else (0.9, 1.4),
                                            alpha=1.0 if i == 0 else 0.9))
                if i and sky.prob.shape == island.shape:
                    cells.append(sky._mode_cells(sky.credible_mask(top) & island, m))
            if ref is not None:
                ref._draw_zoom(ax1, m, [top], smooth, "0.6", "dotted", filled=False, linewidths=(1.0,))
            grow = window is None
            if window is None:
                window = hero._zoom_window(m, np.concatenate([c[0] for c in cells]),
                                           np.concatenate([c[1] for c in cells]), pad)
            extra = _zoom_decorate(ax1, m, pal, window, truth=truth, fov=fov)
            handles = []
            for sky, lab, col, ls in zip(skies, labels, colors, styles):
                handles.append(Line2D([0], [0], color=col, lw=2.4, ls=ls,
                                      label=f"{lab}:  {_areas_text(sky, levels)},"
                                            f"  $f_{{\\rm dom}}={f_dom(sky):.2f}$"))
            if ref is not None:
                handles.append(Line2D([0], [0], color="0.6", lw=1.0, ls="dotted",
                                      label=f"{reference_label}:  {_fmt_area(ref.credible_area(top))}"
                                            f"$\\,$deg$^2$ ({_pct(top)})"))
            if legend_title is None:
                legend_title = f"{_area_label(levels)}, all sky modes"
            _zoom_legend(ax1, handles + extra, pal, np.concatenate(verts), title=legend_title, grow=grow)
        if save:
            fig.savefig(save, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    return fig, axes
