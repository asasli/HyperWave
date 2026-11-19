"""Bilby-style calibration factors for detector-projected templates.

Bilby applies calibration uncertainty as a complex, template-side factor

    d(f) = alpha(f) h(f)

after antenna projection and the geocentric time shift.  The classes here mirror
that behaviour without requiring bilby objects in the waveform hot path, while
still accepting bilby-like objects with ``get_calibration_factor``.
"""

from __future__ import annotations

import numpy as np


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


__all__ = [
    "CubicSpline",
    "Precomputed",
    "Recalibrate",
    "_batch_calibration_factor",
    "calibration_parameter_names",
    "read_calibration_file",
]
