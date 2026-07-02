"""Detector calibration response helpers.

The bilby-compatible classes apply calibration uncertainty as a complex,
template-side factor after antenna projection.  The native
:class:`SplineCalibration` path draws response-curve banks for marginalized
likelihoods without requiring bilby at runtime.
"""

from __future__ import annotations

import numpy as np

from ..backends import get_array_backend


def read_calibration_file(
    filename, frequency_array, number_of_response_curves, starting_index=0, correction_type=None
):
    """Read bilby/LVK-style HDF5 calibration response curves via bilby.

    The file must contain ``deltaR/draws_amp_rel``, ``deltaR/draws_phase`` and
    ``deltaR/freq``.  Curves are interpolated onto ``frequency_array`` and are
    returned in bilby's internal template-side convention.  LVK calibration
    products are usually data-side corrections, so ``correction_type='data'``
    inverts them.
    """
    from bilby.gw.detector.calibration import read_calibration_file as bilby_read

    return bilby_read(
        filename=filename,
        frequency_array=frequency_array,
        number_of_response_curves=number_of_response_curves,
        starting_index=starting_index,
        correction_type=correction_type,
    )


def _batch_calibration_factor(model, frequency_array, params, n):
    """Return a calibration factor with shape ``(n, n_freq)``.

    Bilby-like scalar models return ``(n_freq,)``.  Bilby cubic splines given
    batched parameter arrays return ``(n_freq, n)`` because their implementation
    indexes the frequency axis first.  HyperWave's template batch uses
    ``(n, n_freq)``, so both cases are normalized here.
    """
    frequency_array = np.asarray(frequency_array, dtype=float)
    try:
        factor = model.get_calibration_factor(frequency_array, **params)
    except (TypeError, ValueError):
        factors = []
        for jj in range(n):
            single = {}
            for key, value in params.items():
                array = np.asarray(value)
                if array.ndim > 0 and array.shape[0] == n:
                    single[key] = array[jj]
                else:
                    single[key] = value
            factors.append(model.get_calibration_factor(frequency_array, **single))
        return np.asarray(factors, dtype=complex)

    factor = np.asarray(factor, dtype=complex)
    nfreq = frequency_array.size

    if factor.shape == ():
        return np.full((n, nfreq), complex(factor))
    if factor.shape == (nfreq,):
        return factor[None, :]
    if factor.shape == (n, nfreq):
        return factor
    if factor.shape == (nfreq, n):
        return factor.T
    raise ValueError(
        f"Calibration factor has shape {factor.shape}; expected ({nfreq},), "
        f"({n}, {nfreq}), or ({nfreq}, {n})."
    )


def calibration_parameter_names(model):
    """Return sampled calibration parameter names for a bilby-like model."""
    names = getattr(model, "parameter_names", None)
    if names is not None:
        return list(names)

    prefix = getattr(model, "prefix", None)
    n_points = getattr(model, "n_points", None)
    if prefix is not None and n_points is not None:
        return (
            [f"{prefix}amplitude_{ii}" for ii in range(int(n_points))]
            + [f"{prefix}phase_{ii}" for ii in range(int(n_points))]
        )
    if prefix is not None and getattr(model, "name", None) == "precomputed":
        return [prefix]
    return []


class Recalibrate:
    """Identity calibration model."""

    name = "none"

    def __init__(self, prefix="recalib_"):
        self.params = {}
        self.prefix = prefix

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(prefix={self.prefix!r})"

    def get_calibration_factor(self, frequency_array, **params):
        return np.ones_like(frequency_array, dtype=complex)

    @property
    def parameter_names(self):
        return []

    def set_calibration_parameters(self, **params):
        self.params.update({
            key[len(self.prefix):]: params[key]
            for key in params
            if isinstance(key, str) and key.startswith(self.prefix)
        })

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class CubicSpline(Recalibrate):
    """Bilby-compatible cubic-spline calibration model."""

    name = "cubic_spline"

    def __init__(self, prefix, minimum_frequency, maximum_frequency, n_points):
        super().__init__(prefix=prefix)
        if n_points < 4:
            raise ValueError("Cubic spline calibration requires at least 4 spline nodes.")
        self.n_points = int(n_points)
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = float(maximum_frequency)
        self._log_spline_points = np.linspace(
            np.log10(self.minimum_frequency), np.log10(self.maximum_frequency), self.n_points
        )

    @property
    def parameter_names(self):
        return (
            [f"{self.prefix}amplitude_{ii}" for ii in range(self.n_points)]
            + [f"{self.prefix}phase_{ii}" for ii in range(self.n_points)]
        )

    @property
    def log_spline_points(self):
        return self._log_spline_points

    @property
    def delta_log_spline_points(self):
        if not hasattr(self, "_delta_log_spline_points"):
            self._delta_log_spline_points = self._log_spline_points[1] - self._log_spline_points[0]
        return self._delta_log_spline_points

    @property
    def nodes_to_spline_coefficients(self):
        if not hasattr(self, "_nodes_to_spline_coefficients"):
            self._setup_spline_coefficients()
        return self._nodes_to_spline_coefficients

    def _setup_spline_coefficients(self):
        tmp1 = np.zeros((self.n_points, self.n_points))
        tmp1[0, 0] = -1
        tmp1[0, 1] = 2
        tmp1[0, 2] = -1
        tmp1[-1, -3] = -1
        tmp1[-1, -2] = 2
        tmp1[-1, -1] = -1
        for ii in range(1, self.n_points - 1):
            tmp1[ii, ii - 1] = 1 / 6
            tmp1[ii, ii] = 2 / 3
            tmp1[ii, ii + 1] = 1 / 6

        tmp2 = np.zeros((self.n_points, self.n_points))
        for ii in range(1, self.n_points - 1):
            tmp2[ii, ii - 1] = 1
            tmp2[ii, ii] = -2
            tmp2[ii, ii + 1] = 1
        self._nodes_to_spline_coefficients = np.linalg.solve(tmp1, tmp2)

    def _node_value(self, kind, ii, params):
        full = f"{self.prefix}{kind}_{ii}"
        short = f"{kind}_{ii}"
        if full in params:
            return params[full]
        if short in params:
            return params[short]
        if short in self.params:
            return self.params[short]
        raise KeyError(f"Missing calibration parameter {full!r}.")

    def _node_array(self, kind, params):
        nodes = [np.asarray(self._node_value(kind, ii, params), dtype=float)
                 for ii in range(self.n_points)]
        sizes = [value.size for value in nodes if value.ndim > 0]
        if not sizes:
            return np.array([float(value) for value in nodes])
        n = sizes[0]
        if any(size != n for size in sizes):
            raise ValueError("Batched calibration node arrays must have the same length.")
        out = np.zeros((n, self.n_points))
        for ii, value in enumerate(nodes):
            out[:, ii] = value if value.ndim else float(value)
        return out

    def _evaluate_spline(self, kind, a, b, c, d, previous_nodes, params):
        parameters = self._node_array(kind, params)
        next_nodes = previous_nodes + 1
        if parameters.ndim == 1:
            coefficients = self.nodes_to_spline_coefficients.dot(parameters)
            return (
                a * parameters[previous_nodes]
                + b * parameters[next_nodes]
                + c * coefficients[previous_nodes]
                + d * coefficients[next_nodes]
            )

        coefficients = parameters.dot(self.nodes_to_spline_coefficients.T)
        return (
            a[None, :] * parameters[:, previous_nodes]
            + b[None, :] * parameters[:, next_nodes]
            + c[None, :] * coefficients[:, previous_nodes]
            + d[None, :] * coefficients[:, next_nodes]
        )

    def get_calibration_factor(self, frequency_array, **params):
        self.set_calibration_parameters(**params)
        frequency_array = np.asarray(frequency_array, dtype=float)
        log10f_per_deltalog10f = (
            np.log10(frequency_array) - self.log_spline_points[0]
        ) / self.delta_log_spline_points
        previous_nodes = np.clip(
            np.floor(log10f_per_deltalog10f).astype(int),
            a_min=0,
            a_max=self.n_points - 2,
        )
        b = log10f_per_deltalog10f - previous_nodes
        a = 1 - b
        c = (a**3 - a) / 6
        d = (b**3 - b) / 6

        delta_amplitude = self._evaluate_spline("amplitude", a, b, c, d, previous_nodes, params)
        delta_phase = self._evaluate_spline("phase", a, b, c, d, previous_nodes, params)
        return (1 + delta_amplitude) * (2 + 1j * delta_phase) / (2 - 1j * delta_phase)


class Precomputed(Recalibrate):
    """Select from precomputed template-side calibration response curves."""

    name = "precomputed"

    def __init__(self, label, curves, frequency_array, parameters=None):
        self.label = str(label)
        self.curves = np.asarray(curves, dtype=complex)
        self.frequency_array = np.asarray(frequency_array, dtype=float)
        self.parameters = parameters
        super().__init__(prefix=f"recalib_index_{self.label}")

    @property
    def parameter_names(self):
        return [self.prefix]

    def get_calibration_factor(self, frequency_array, **params):
        if self.prefix not in params:
            raise KeyError(f"Calibration index for {self.label} not found.")
        frequency_array = np.asarray(frequency_array, dtype=float)
        if not np.array_equal(frequency_array, self.frequency_array):
            raise ValueError("Frequency grid passed to calibrator does not match.")
        idx = np.asarray(params[self.prefix], dtype=int)
        return self.curves[int(idx)] if idx.ndim == 0 else self.curves[idx]

    @classmethod
    def from_calibration_file(
        cls, label, filename, frequency_array, n_curves, starting_index=0, correction_type=None
    ):
        curves, parameters = read_calibration_file(
            filename=filename,
            frequency_array=frequency_array,
            number_of_response_curves=n_curves,
            starting_index=starting_index,
            correction_type=correction_type,
        )
        return cls(
            label=label, curves=curves, frequency_array=frequency_array, parameters=parameters
        )


def _nodes_to_spline_coefficients(n_points):
    """Matrix mapping node values to natural-cubic-spline second derivatives.

    Port of bilby's ``CubicSpline._setup_spline_coefficients`` (Eq. 9 of
    LIGO-T2300140). Small (``n_points x n_points``), built on the host.
    """
    tmp1 = np.zeros((n_points, n_points))
    tmp1[0, 0] = -1
    tmp1[0, 1] = 2
    tmp1[0, 2] = -1
    tmp1[-1, -3] = -1
    tmp1[-1, -2] = 2
    tmp1[-1, -1] = -1
    for i in range(1, n_points - 1):
        tmp1[i, i - 1] = 1 / 6
        tmp1[i, i] = 2 / 3
        tmp1[i, i + 1] = 1 / 6
    tmp2 = np.zeros((n_points, n_points))
    for i in range(1, n_points - 1):
        tmp2[i, i - 1] = 1
        tmp2[i, i] = -2
        tmp2[i, i + 1] = 1
    return np.linalg.solve(tmp1, tmp2)


def _spline_basis_matrix(log_frequencies, log_nodes):
    """Linear map ``B`` such that ``delta(f) = B @ node_values``.

    Because the cubic spline is linear in its node values (for fixed nodes and
    evaluation grid), the whole evaluation collapses to one matrix ``B`` of
    shape ``(n_freq, n_nodes)``. This is what keeps the calibration factor a
    cheap, fully-batched matmul on the GPU. Reproduces bilby's
    ``CubicSpline.get_calibration_factor`` evaluation.
    """
    n = log_nodes.size
    delta_log = log_nodes[1] - log_nodes[0]
    x = (log_frequencies - log_nodes[0]) / delta_log
    prev = np.clip(np.floor(x).astype(int), 0, n - 2)
    nxt = prev + 1
    b = x - prev
    a = 1 - b
    c = (a ** 3 - a) / 6
    d = (b ** 3 - b) / 6

    M = _nodes_to_spline_coefficients(n)
    B = np.zeros((log_frequencies.size, n))
    rows = np.arange(log_frequencies.size)
    B[rows, prev] += a
    B[rows, nxt] += b
    B += c[:, None] * M[prev, :] + d[:, None] * M[nxt, :]
    return B


class SplineCalibration:
    """Native (NumPy/CuPy) cubic-spline detector calibration model.

    Parameters
    ----------
    frequency_array : array-like
        The (masked) analysis frequency grid, i.e. the same ``f`` the likelihood
        uses. Values must lie within ``[minimum_frequency, maximum_frequency]``.
    n_nodes : int
        Number of spline nodes (>= 4), log-spaced over the band.
    minimum_frequency, maximum_frequency : float, optional
        Node band edges; default to ``frequency_array[0]`` / ``[-1]``.
    gpu : bool
        Build the basis matrix on CuPy when available.
    """

    def __init__(self, frequency_array, n_nodes=10, minimum_frequency=None,
                 maximum_frequency=None, gpu=False):
        if n_nodes < 4:
            raise ValueError("Cubic spline calibration requires at least 4 nodes.")
        self._backend = get_array_backend(gpu=gpu)
        self.xp = self._backend.xp

        f = np.asarray(frequency_array, dtype=float)
        fmin = float(f[0]) if minimum_frequency is None else float(minimum_frequency)
        fmax = float(f[-1]) if maximum_frequency is None else float(maximum_frequency)
        self.n_nodes = int(n_nodes)
        self.minimum_frequency = fmin
        self.maximum_frequency = fmax
        self.log_nodes = np.linspace(np.log10(fmin), np.log10(fmax), n_nodes)

        B = _spline_basis_matrix(np.log10(f), self.log_nodes)  # (n_freq, n_nodes)
        self.basis = self._backend.asarray(B)
        self._n_freq = f.size

    def factor(self, amplitude_nodes, phase_nodes):
        """Calibration factor ``C(f)`` from node values.

        ``amplitude_nodes`` / ``phase_nodes`` have shape ``(..., n_nodes)``;
        returns ``(..., n_freq)`` complex. ``dA`` and ``dphi`` are each linear in
        the nodes (``@ basis.T``); the factor combines them as
        ``(1 + dA)(2 + i dphi)/(2 - i dphi)``.
        """
        a = self._backend.asarray(amplitude_nodes)
        p = self._backend.asarray(phase_nodes)
        delta_amplitude = a @ self.basis.T
        delta_phase = p @ self.basis.T
        return (1 + delta_amplitude) * (2 + 1j * delta_phase) / (2 - 1j * delta_phase)

    def draw_bank(self, n_curves, amplitude_sigma, phase_sigma, seed=None):
        """Draw ``n_curves`` response curves from constant Gaussian node priors.

        Returns ``(n_curves, n_freq)`` complex on the active backend.
        """
        rng = np.random.default_rng(seed)
        amp = rng.normal(0.0, amplitude_sigma, size=(n_curves, self.n_nodes))
        phase = rng.normal(0.0, phase_sigma, size=(n_curves, self.n_nodes))
        return self.factor(amp, phase)

    def draw_bank_from_envelope(self, envelope_file, n_curves, seed=None):
        """Draw curves from an LVK calibration envelope file (no bilby).

        Envelope columns: ``freq median_amp median_phase -1sigma_amp
        -1sigma_phase +1sigma_amp +1sigma_phase``. Per-node Gaussian
        ``(mu, sigma)`` are interpolated (in log-frequency) from the envelope,
        matching bilby's :meth:`CalibrationPriorDict.from_envelope_file`.
        """
        from scipy.interpolate import InterpolatedUnivariateSpline

        data = np.loadtxt(envelope_file).T
        log_f = np.log10(data[0])
        amp_median = data[1] - 1.0
        phase_median = data[2]
        amp_sigma = (data[5] - data[3]) / 2.0
        phase_sigma = (data[6] - data[4]) / 2.0

        def at_nodes(values):
            return InterpolatedUnivariateSpline(log_f, values)(self.log_nodes)

        amp_mu, amp_sd = at_nodes(amp_median), at_nodes(amp_sigma)
        phase_mu, phase_sd = at_nodes(phase_median), at_nodes(phase_sigma)

        rng = np.random.default_rng(seed)
        amp = rng.normal(amp_mu, amp_sd, size=(n_curves, self.n_nodes))
        phase = rng.normal(phase_mu, phase_sd, size=(n_curves, self.n_nodes))
        return self.factor(amp, phase)


def make_calibration_bank(
    ifo_names,
    frequency_array,
    n_curves=1000,
    n_nodes=10,
    amplitude_sigma=0.05,
    phase_sigma=0.05,
    envelope_files=None,
    seed=None,
    gpu=False,
):
    """Draw a bank of calibration response curves, one stack per detector.

    Native NumPy/CuPy implementation (no bilby at runtime).

    Parameters
    ----------
    ifo_names : sequence of str
        Detector labels, e.g. ``["H1", "L1"]``. The order fixes axis 0 of the
        returned array and must match the ``ifos_list`` / ``data`` ordering of
        the likelihood the bank is handed to.
    frequency_array : array-like
        The (masked) analysis frequency grid the likelihood uses.
    n_curves : int
        Number of response curves per detector (the marginalization sum runs
        over these). 1000 is the common LVK choice.
    n_nodes : int
        Number of cubic-spline nodes (>= 4), log-spaced over the band.
    amplitude_sigma, phase_sigma : float or dict
        Gaussian 1-sigma uncertainty in fractional amplitude and in phase
        (radians). Float = same for all detectors; dict = per detector. Ignored
        for a detector that has an entry in ``envelope_files``.
    envelope_files : dict, optional
        ``{detector_name: path}`` to LVK calibration envelope files; when given
        for a detector, its prior is built from the envelope instead of the
        constant ``*_sigma`` values.
    seed : int, optional
        Base seed; detector ``j`` uses ``seed + j`` so detectors are decorrelated
        yet reproducible.
    gpu : bool
        Return the bank on CuPy when available.

    Returns
    -------
    array-like
        Complex array of shape ``(n_ifo, n_curves, n_freq)`` on the active
        backend. Curve index ``k`` is a *joint* calibration sample across
        detectors (same index in every detector), matching the LVK convention.
    """
    spline = SplineCalibration(frequency_array, n_nodes=n_nodes, gpu=gpu)
    envelope_files = envelope_files or {}

    def _per_ifo(value, name):
        return value[name] if isinstance(value, dict) else value

    curves = []
    for j, name in enumerate(ifo_names):
        sub_seed = None if seed is None else int(seed) + j
        if name in envelope_files:
            curves.append(spline.draw_bank_from_envelope(
                envelope_files[name], n_curves, seed=sub_seed))
        else:
            curves.append(spline.draw_bank(
                n_curves,
                _per_ifo(amplitude_sigma, name),
                _per_ifo(phase_sigma, name),
                seed=sub_seed,
            ))
    return spline.xp.stack(curves, axis=0)


__all__ = [
    "CubicSpline",
    "Precomputed",
    "Recalibrate",
    "SplineCalibration",
    "_batch_calibration_factor",
    "calibration_parameter_names",
    "make_calibration_bank",
    "read_calibration_file",
]
