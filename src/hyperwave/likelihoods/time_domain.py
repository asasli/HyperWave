"""Covariance-based time-domain likelihoods.

The covariance is Toeplitz, with first row built from the one-sided PSD using
the bilby_greg/pyRing convention

    acf = 0.5 * irfft(psd * df, n=n_time) * n_time.

Waveforms are generated on the full detector frequency grid and transformed
back to the time domain. For HyperWave LVK templates this reuses the same
frequency-domain phase-delay convention as the existing FD likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
import scipy.fft as sf
from scipy.linalg import inv, solve_toeplitz, solve_triangular, toeplitz
from scipy.special import gammaln, kve

_LOG_2PI = np.log(2.0 * np.pi)


def _ensure_2d(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None, :]
    return theta


def _as_1d_float_array(name, value, check_finite=False):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty")
    if check_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _log_scaled_bessel_second_kind_asymptotic(order, argument):
    scaled_argument = argument / order
    eta_sqrt = np.sqrt(1.0 + scaled_argument * scaled_argument)
    eta = eta_sqrt + np.log(scaled_argument / (1.0 + eta_sqrt))
    return (
        argument
        - order * eta
        + 0.5 * np.log(np.pi / (2.0 * order))
        - 0.25 * np.log1p(scaled_argument * scaled_argument)
    )


def _log_scaled_bessel_second_kind(order, argument):
    argument = np.asarray(argument, dtype=float)
    out = np.empty_like(argument, dtype=float)
    flat = argument.ravel()
    out_flat = out.ravel()
    for i, value in enumerate(flat):
        if order <= 0.0 or not np.isfinite(order) or value <= 0.0 or not np.isfinite(value):
            out_flat[i] = np.nan
            continue
        if value < 0.2 * order:
            x2_over4 = 0.25 * value * value
            term = 1.0
            series = 1.0
            for index in range(1, 10000):
                denominator = index * (index - order)
                if denominator == 0.0:
                    break
                term *= x2_over4 / denominator
                series += term
                if (not np.isfinite(term)) or (not np.isfinite(series)):
                    break
                if abs(term) <= 1e-15 * abs(series):
                    break
            if series > 0.0 and np.isfinite(series):
                out_flat[i] = (
                    value
                    + np.log(0.5)
                    + lgamma(order)
                    + order * np.log(2.0 / value)
                    + np.log(series)
                )
                continue
        if order >= 100.0:
            out_flat[i] = _log_scaled_bessel_second_kind_asymptotic(order, value)
            continue
        scaled_bessel = kve(order, value)
        if scaled_bessel > 0.0 and np.isfinite(scaled_bessel):
            out_flat[i] = np.log(scaled_bessel)
        else:
            out_flat[i] = _log_scaled_bessel_second_kind_asymptotic(order, value)
    return out


def _gaussian_log_likelihood_from_inner_product(q, log_normalisation):
    return -0.5 * q + log_normalisation


def _student_t_log_likelihood_from_inner_product(q, logdet, dimension, nu):
    q = np.asarray(q, dtype=float)
    nu = np.asarray(nu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logl = (
            gammaln(0.5 * (nu + dimension))
            - gammaln(0.5 * nu)
            - 0.5 * (dimension * np.log(nu * np.pi) + logdet)
            - 0.5 * (nu + dimension) * np.log1p(q / nu)
        )
    return np.where((nu > 0.0) & np.isfinite(nu) & (q >= 0.0), logl, np.nan)


def _hyperbolic_log_likelihood_from_inner_product(q, logdet, dimension, alpha, delta):
    q = np.asarray(q, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    delta = np.asarray(delta, dtype=float)
    order = 0.5 * (dimension + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        argument = alpha * delta
        log_scaled_bessel = _log_scaled_bessel_second_kind(order, argument)
        radial_shift = q / (np.sqrt(delta * delta + q) + delta)
        logl = (
            order * np.log(alpha / delta)
            + 0.5 * (1.0 - dimension) * _LOG_2PI
            - np.log(2.0 * alpha)
            - log_scaled_bessel
            - alpha * radial_shift
            - 0.5 * logdet
        )
    good = (
        (alpha > 0.0)
        & (delta > 0.0)
        & np.isfinite(alpha)
        & np.isfinite(delta)
        & (q >= 0.0)
        & np.isfinite(q)
    )
    return np.where(good, logl, np.nan)


def _toeplitz_kernel_rfft(column, row, fft_length):
    column = np.asarray(column, dtype=float)
    row = np.asarray(row, dtype=float)
    kernel = np.zeros(fft_length, dtype=float)
    kernel[: len(column)] = column
    if len(column) > 1:
        kernel[fft_length - len(column) + 1 :] = row[:0:-1]
    return sf.rfft(kernel)


def _multiply_fft_kernel(kernel_fft, vector_fft):
    if vector_fft.ndim == 1:
        return kernel_fft * vector_fft
    return kernel_fft[:, None] * vector_fft


def _irfft_head(product_fft, fft_length, n):
    return sf.irfft(product_fft, fft_length, axis=0)[:n]


@dataclass(frozen=True)
class _GohbergSemenculToeplitzInverse:
    acf: np.ndarray
    x: np.ndarray
    x0: float
    fft_length: int
    lower_x_fft: np.ndarray
    upper_x_fft: np.ndarray
    lower_tail_reverse_x_fft: np.ndarray
    upper_tail_reverse_x_fft: np.ndarray

    @classmethod
    def from_acf(cls, acf, check_finite=False):
        acf = _as_1d_float_array("acf", acf, check_finite=check_finite)
        basis_vector = np.zeros_like(acf)
        basis_vector[0] = 1.0
        x = solve_toeplitz(acf, basis_vector, check_finite=check_finite)
        return cls.from_acf_and_generator(acf, x, check_finite=check_finite)

    @classmethod
    def from_acf_and_generator(cls, acf, x, check_finite=False):
        acf = _as_1d_float_array("acf", acf, check_finite=check_finite)
        x = _as_1d_float_array("x", x, check_finite=check_finite)
        if len(acf) != len(x):
            raise ValueError("ACF and Gohberg-Semencul generator lengths do not agree")
        if abs(x[0]) <= np.finfo(float).eps:
            raise ValueError("The first Gohberg-Semencul generator entry is zero")

        zeros = np.zeros_like(x)
        zeros_with_x0 = np.zeros_like(x)
        zeros_with_x0[0] = x[0]
        tail_reverse_x = np.zeros_like(x)
        tail_reverse_x[1:] = x[:0:-1]
        fft_length = sf.next_fast_len(2 * len(x) - 1, real=True)

        return cls(
            acf=acf,
            x=x,
            x0=float(x[0]),
            fft_length=fft_length,
            lower_x_fft=_toeplitz_kernel_rfft(x, zeros, fft_length),
            upper_x_fft=_toeplitz_kernel_rfft(zeros_with_x0, x, fft_length),
            lower_tail_reverse_x_fft=_toeplitz_kernel_rfft(
                tail_reverse_x, zeros, fft_length
            ),
            upper_tail_reverse_x_fft=_toeplitz_kernel_rfft(
                zeros, tail_reverse_x, fft_length
            ),
        )

    def matvec(self, vector, check_finite=False):
        vector = np.asarray(vector, dtype=float)
        if vector.ndim not in (1, 2):
            raise ValueError("vector must be one- or two-dimensional")
        if vector.shape[0] != len(self.x):
            raise ValueError("Gohberg-Semencul factor and target vector lengths differ")
        if check_finite and not np.all(np.isfinite(vector)):
            raise ValueError("vector must contain only finite values")
        if len(self.x) == 1:
            return vector * self.x0

        vector_fft = sf.rfft(vector, self.fft_length, axis=0)
        upper_x_vector = _irfft_head(
            _multiply_fft_kernel(self.upper_x_fft, vector_fft),
            self.fft_length,
            len(self.x),
        )
        upper_tail_vector = _irfft_head(
            _multiply_fft_kernel(self.upper_tail_reverse_x_fft, vector_fft),
            self.fft_length,
            len(self.x),
        )
        upper_x_fft = sf.rfft(upper_x_vector, self.fft_length, axis=0)
        upper_tail_fft = sf.rfft(upper_tail_vector, self.fft_length, axis=0)
        return _irfft_head(
            _multiply_fft_kernel(self.lower_x_fft, upper_x_fft)
            - _multiply_fft_kernel(self.lower_tail_reverse_x_fft, upper_tail_fft),
            self.fft_length,
            len(self.x),
        ) / self.x0


def _toeplitz_slogdet(acf):
    acf = np.asarray(acf, dtype=float)
    dimension = len(acf)
    r0 = acf[0]
    normalized = np.concatenate((acf, np.array([r0], dtype=float))) / r0
    logdet = dimension * np.log(abs(r0))
    sign = np.sign(r0) ** dimension
    if dimension == 1:
        return sign, logdet

    y = np.zeros(dimension, dtype=float)
    x = np.zeros(dimension, dtype=float)
    b = -normalized[1 : dimension + 1]
    r = normalized[:dimension]
    y[0] = -r[1]
    x[0] = b[0]
    beta = 1.0
    alpha = -r[1]
    determinant_update = 1.0 + (-b[0]) * x[0]
    sign *= np.sign(determinant_update)
    logdet += np.log(abs(determinant_update))

    for index in range(0, dimension - 2):
        beta = (1.0 - alpha * alpha) * beta
        mu = (b[index + 1] - np.dot(r[1 : index + 2], x[index::-1])) / beta
        x[0 : index + 1] = x[0 : index + 1] + mu * y[index::-1]
        x[index + 1] = mu
        determinant_update = 1.0 + np.dot(-b[0 : index + 2], x[0 : index + 2])
        sign *= np.sign(determinant_update)
        logdet += np.log(abs(determinant_update))
        if index < dimension - 2:
            alpha = -(r[index + 2] + np.dot(r[1 : index + 2], y[index::-1])) / beta
            y[0 : index + 1] = y[0 : index + 1] + alpha * y[index::-1]
            y[index + 1] = alpha
    return sign, logdet


def _is_time_band_cut_list(time_bands):
    return isinstance(time_bands, (list, tuple, np.ndarray))


def _coerce_time_band_boundaries(time_band_boundaries):
    if not _is_time_band_cut_list(time_band_boundaries):
        raise ValueError("time_band_boundaries must be a 1D list, tuple, or array")
    boundaries = np.asarray(time_band_boundaries, dtype=float)
    if boundaries.ndim != 1:
        raise ValueError("time_band_boundaries must be one-dimensional")
    return boundaries.tolist()


def _resolve_time_bands(time_bands, time_band_boundaries=None):
    if time_band_boundaries is None:
        if _is_time_band_cut_list(time_bands):
            return _coerce_time_band_boundaries(time_bands)
        return int(time_bands)

    boundaries = _coerce_time_band_boundaries(time_band_boundaries)
    if _is_time_band_cut_list(time_bands):
        if _coerce_time_band_boundaries(time_bands) != boundaries:
            raise ValueError("time_bands and time_band_boundaries must match")
        return boundaries

    number_of_time_bands = int(time_bands)
    if number_of_time_bands not in (1, len(boundaries) + 1):
        raise ValueError("time_bands and time_band_boundaries are inconsistent")
    return boundaries


def _time_band_count(time_bands):
    if _is_time_band_cut_list(time_bands):
        return len(time_bands) + 1
    return int(time_bands)


def _time_band_sample_slices(dimension, time_bands, sampling_rate=None):
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("The time-band dimension must be positive")

    if _is_time_band_cut_list(time_bands):
        if sampling_rate is None:
            raise ValueError("A sampling rate is required for second-based time bands")
        sampling_rate = float(sampling_rate)
        cuts = np.asarray(time_bands, dtype=float)
        if cuts.size == 0:
            raise ValueError("Time-band cut-time lists cannot be empty")
        if (not np.all(np.isfinite(cuts))) or np.any(cuts <= 0.0):
            raise ValueError("Time-band cut times must be positive and finite")
        if np.any(np.diff(cuts) <= 0.0):
            raise ValueError("Time-band cut times must be strictly increasing")
        sample_times = np.arange(dimension, dtype=float) / sampling_rate
        if cuts[-1] > sample_times[-1]:
            raise ValueError("The last time-band cut exceeds the segment")
        edges = np.concatenate(
            ([0], np.searchsorted(sample_times, cuts, side="left"), [dimension])
        ).astype(int)
    else:
        number_of_bands = int(time_bands)
        if number_of_bands < 1:
            raise ValueError("The number of time bands must be positive")
        if number_of_bands > dimension:
            raise ValueError("The number of time bands cannot exceed samples")
        edges = np.linspace(0, dimension, number_of_bands + 1, dtype=int)

    if np.any(np.diff(edges) <= 0):
        raise ValueError("Time bands must contain at least one sample each")
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


@dataclass
class _LikelihoodCache:
    start: int
    end: int
    acf: np.ndarray
    logdet: float
    log_normalisation: float
    inverse_covariance: np.ndarray | None = None
    cholesky: np.ndarray | None = None
    gohberg_semencul_inverse: _GohbergSemenculToeplitzInverse | None = None


def _resolve_likelihood_method(likelihood_method):
    aliases = {
        "direct-inversion": "direct-inversion",
        "direct": "direct-inversion",
        "cholesky-solve-triangular": "cholesky-solve-triangular",
        "cholesky": "cholesky-solve-triangular",
        "toeplitz-inversion": "toeplitz-inversion",
        "toeplitz": "toeplitz-inversion",
        "gohberg-semencul": "gohberg-semencul",
        "gohberg_semencul": "gohberg-semencul",
        "gohberg": "gohberg-semencul",
        "gs": "gohberg-semencul",
    }
    key = str(likelihood_method).lower()
    if key not in aliases:
        raise ValueError("Unknown time-domain likelihood method")
    return aliases[key]


def _make_likelihood_cache(acf, likelihood_method, no_lognorm=False):
    acf = np.asarray(acf, dtype=float)
    sign, logdet = _toeplitz_slogdet(acf)
    if sign <= 0:
        raise ValueError("The Toeplitz covariance determinant must be positive")
    cache = _LikelihoodCache(
        start=0,
        end=len(acf),
        acf=acf,
        logdet=float(logdet),
        log_normalisation=0.0 if no_lognorm else -0.5 * logdet - 0.5 * len(acf) * _LOG_2PI,
    )
    if likelihood_method == "direct-inversion":
        cache.inverse_covariance = inv(toeplitz(acf))
    elif likelihood_method == "cholesky-solve-triangular":
        cache.cholesky = np.linalg.cholesky(toeplitz(acf))
    elif likelihood_method == "gohberg-semencul":
        cache.gohberg_semencul_inverse = _GohbergSemenculToeplitzInverse.from_acf(
            acf, check_finite=False
        )
    return cache


def _make_time_band_likelihood_cache(
    acf, likelihood_method, time_bands, no_lognorm=False, sampling_rate=None
):
    band_cache = []
    for band_start, band_end in _time_band_sample_slices(
        len(acf), time_bands, sampling_rate
    ):
        band_acf = np.asarray(acf[: band_end - band_start], dtype=float)
        cache = _make_likelihood_cache(
            acf=band_acf,
            likelihood_method=likelihood_method,
            no_lognorm=no_lognorm,
        )
        cache.start = band_start
        cache.end = band_end
        band_cache.append(cache)
    return band_cache


def _quadratic_forms(residuals, cache, likelihood_method):
    residuals = np.atleast_2d(np.asarray(residuals, dtype=float))
    if likelihood_method == "direct-inversion":
        return np.einsum("ni,ij,nj->n", residuals, cache.inverse_covariance, residuals)
    if likelihood_method == "cholesky-solve-triangular":
        whitened = solve_triangular(
            cache.cholesky, residuals.T, lower=True, check_finite=False
        )
        return np.sum(whitened * whitened, axis=0)
    if likelihood_method == "toeplitz-inversion":
        solved = solve_toeplitz(cache.acf, residuals.T, check_finite=False)
        return np.sum(residuals.T * solved, axis=0)
    if likelihood_method == "gohberg-semencul":
        solved = cache.gohberg_semencul_inverse.matvec(residuals.T, check_finite=False)
        return np.sum(residuals.T * solved, axis=0)
    raise ValueError("Unknown likelihood method requested")


def _patch_psd_outside_active_band(psd, frequencies, active_frequencies):
    psd = np.asarray(psd, dtype=float).copy()
    frequencies = np.asarray(frequencies, dtype=float)
    active_frequencies = np.asarray(active_frequencies, dtype=float)
    if len(active_frequencies) == 0:
        raise ValueError("Cannot patch a PSD without active frequencies")
    low_frequency = float(active_frequencies[0])
    high_frequency = float(active_frequencies[-1])
    active_band_mask = (frequencies >= low_frequency) & (frequencies <= high_frequency)
    low_patch_value = 10.0 * float(np.max(psd[active_band_mask]))
    high_patch_value = 10.0 * float(np.max(psd[frequencies >= high_frequency]))
    psd[frequencies < low_frequency] = low_patch_value
    psd[frequencies > high_frequency] = high_patch_value
    return psd


def _patch_psd_pyring(
    psd,
    frequencies,
    f_min_bp,
    f_max_bp,
    f_min_patch=None,
    f_max_patch=None,
    f_turn_over_patch=None,
    patch_value=None,
):
    psd = np.asarray(psd, dtype=float).copy()
    frequencies = np.asarray(frequencies, dtype=float)
    if f_min_patch is None:
        f_min_patch = float(f_min_bp) * 1.05
    if f_max_patch is None:
        f_max_patch = float(f_max_bp) * 0.995
    if f_turn_over_patch is None:
        f_turn_over_patch = min(2000.0, float(frequencies[-1]))
    if patch_value is None:
        patch_value = (
            10.0 * float(np.max(psd[(frequencies >= f_min_patch) & (frequencies <= f_max_patch)])),
            10.0 * float(np.max(psd[frequencies >= f_turn_over_patch])),
        )
    elif isinstance(patch_value, (int, float)):
        patch_value = (float(patch_value), float(patch_value))
    elif not (isinstance(patch_value, tuple) and len(patch_value) == 2):
        raise ValueError("patch_value must be a scalar or a pair")
    psd[frequencies <= f_min_patch] = patch_value[0]
    psd[frequencies >= f_turn_over_patch] = patch_value[1]
    return psd


def _normalise_likelihood_type(likelihood_type):
    aliases = {
        "gaussian": "gaussian",
        "normal": "gaussian",
        "student-t": "student-t",
        "student_t": "student-t",
        "studentt": "student-t",
        "hyperbolic": "hyperbolic",
        "hyperbolic_classic": "hyperbolic",
    }
    key = str(likelihood_type).lower()
    if key not in aliases:
        raise ValueError("Detector likelihoods must be gaussian, student-t, or hyperbolic")
    return aliases[key]


class TimeDomainGWLikelihoods:
    """Batched time-domain GW likelihoods.

    Parameters are positional, matching the rest of HyperWave:

    - ``gaussian(theta)`` uses only waveform columns.
    - ``student_t(theta)`` appends one ``nu`` per time band, or per detector and
      time band when ``detector_dependent_noise=True``.
    - ``hyperbolic_classic(theta)`` appends alpha block then delta block with
      the same layout. ``hyperbolic(theta)`` uses alpha then ratio, with
      ``delta = alpha * ratio``.
    - ``mixed(theta)`` appends Student-t columns for Student-t detectors, then
      hyperbolic alpha and delta columns for hyperbolic detectors.
    """

    def __init__(
        self,
        data,
        sampling_rate,
        ifos_list,
        noise=None,
        template=None,
        f=None,
        psd=None,
        acf=None,
        time_bands=1,
        time_band_boundaries=None,
        likelihood_method="cholesky-solve-triangular",
        minimum_frequency=None,
        maximum_frequency=None,
        psd_patch=True,
        f_min_patch=None,
        f_max_patch=None,
        f_turn_over_patch=None,
        patch_value=None,
        detector_dependent_noise=False,
        detector_likelihoods=None,
        likelihood_types=None,
        split_inner_products=False,
        no_lognorm=False,
        infs=-1e300,
    ):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data[None, :]
        if self.data.ndim != 2:
            raise ValueError("data must have shape (n_ifo, n_time)")

        self.sampling_rate = float(sampling_rate)
        self.ifos = list(ifos_list)
        self._nchannels = len(self.ifos)
        if self.data.shape[0] != self._nchannels:
            raise ValueError("data and ifos_list disagree on detector count")

        self._template = template
        if self._template is None:
            raise ValueError("A template with make_injections_to_ifo_batch is required")
        if getattr(self._template, "parameters", None) is None:
            raise TypeError("Template must define its parameter list")
        self._wfdims = len(self._template.parameters)

        self._ntime = self.data.shape[1]
        self.duration = self._ntime / self.sampling_rate
        self._nfreq = self._ntime // 2 + 1
        self.f = (
            np.fft.rfftfreq(self._ntime, d=1.0 / self.sampling_rate)
            if f is None
            else np.asarray(f, dtype=float)
        )
        if self.f.shape != (self._nfreq,):
            raise ValueError(f"f must have length {self._nfreq}")

        self.minimum_frequency = (
            float(getattr(self._template, "minimum_frequency", self.f[0]))
            if minimum_frequency is None
            else float(minimum_frequency)
        )
        self.maximum_frequency = (
            float(getattr(self._template, "maximum_frequency", self.f[-1]))
            if maximum_frequency is None
            else float(maximum_frequency)
        )

        self.likelihood_method = _resolve_likelihood_method(likelihood_method)
        self.time_bands = _resolve_time_bands(time_bands, time_band_boundaries)
        self.time_band_boundaries = (
            self.time_bands if _is_time_band_cut_list(self.time_bands) else None
        )
        self._number_of_time_bands = _time_band_count(self.time_bands)
        self.detector_dependent_noise = bool(detector_dependent_noise)
        self.split_inner_products = bool(split_inner_products)
        self._inf = float(infs)

        detector_likelihoods = (
            likelihood_types if likelihood_types is not None else detector_likelihoods
        )
        self.detector_likelihoods = self._resolve_detector_likelihoods(detector_likelihoods)

        if acf is None:
            psd = noise if psd is None else psd
            if psd is None:
                psd = self._psd_from_template()
            self.psd = self._build_psd_array(
                psd,
                psd_patch=psd_patch,
                f_min_patch=f_min_patch,
                f_max_patch=f_max_patch,
                f_turn_over_patch=f_turn_over_patch,
                patch_value=patch_value,
            )
            self.acf = self._acf_from_psd(self.psd)
        else:
            self.psd = None
            self.acf = self._coerce_detector_array(acf, self._ntime, "acf")

        self._detector_likelihood_caches = self._build_detector_likelihood_caches(
            no_lognorm=no_lognorm
        )

        self.student_t_ndims = self._wfdims + self._noise_width(
            "student-t", range(self._nchannels)
        )
        hyper_width = self._noise_width("hyperbolic", range(self._nchannels))
        self.hyperbolic_ndims = self._wfdims + 2 * hyper_width
        self.mixed_ndims = self._wfdims + self._mixed_noise_width()

    def _resolve_detector_likelihoods(self, detector_likelihoods):
        if detector_likelihoods is None:
            return tuple("gaussian" for _ in self.ifos)
        if isinstance(detector_likelihoods, str):
            like = _normalise_likelihood_type(detector_likelihoods)
            return tuple(like for _ in self.ifos)
        if isinstance(detector_likelihoods, dict):
            missing = [ifo for ifo in self.ifos if ifo not in detector_likelihoods]
            if missing:
                raise ValueError("detector_likelihoods is missing detectors: " + ", ".join(missing))
            return tuple(_normalise_likelihood_type(detector_likelihoods[ifo]) for ifo in self.ifos)
        values = tuple(_normalise_likelihood_type(value) for value in detector_likelihoods)
        if len(values) != self._nchannels:
            raise ValueError("detector_likelihoods must match ifos_list")
        return values

    def _coerce_detector_array(self, values, expected_length, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = np.repeat(values[None, :], self._nchannels, axis=0)
        if values.shape != (self._nchannels, expected_length):
            raise ValueError(
                f"{name} must have shape ({self._nchannels}, {expected_length})"
            )
        return values

    def _psd_from_template(self):
        if hasattr(self._template, "ifo_object"):
            return np.asarray(
                [
                    ifo.power_spectral_density.power_spectral_density_interpolated(self.f)
                    for ifo in self._template.ifo_object()
                ],
                dtype=float,
            )
        if hasattr(self._template, "ifos"):
            try:
                return np.asarray(
                    [
                        ifo.power_spectral_density.power_spectral_density_interpolated(self.f)
                        for ifo in self._template.ifos
                    ],
                    dtype=float,
                )
            except Exception:
                pass
        raise ValueError("noise/psd is required when it cannot be read from the template")

    def _finite_positive_psd(self, psd):
        psd = np.asarray(psd, dtype=float).copy()
        finite = np.isfinite(psd) & (psd > 0.0)
        if not np.any(finite):
            raise ValueError("PSD must contain at least one positive finite value")
        psd[~finite] = float(np.max(psd[finite]))
        return psd

    def _build_psd_array(
        self,
        psd,
        psd_patch=True,
        f_min_patch=None,
        f_max_patch=None,
        f_turn_over_patch=None,
        patch_value=None,
    ):
        psd = self._coerce_detector_array(psd, self._nfreq, "psd")
        active = self.f[(self.f >= self.minimum_frequency) & (self.f <= self.maximum_frequency)]
        out = np.zeros_like(psd)
        for ifo in range(self._nchannels):
            finite_psd = self._finite_positive_psd(psd[ifo])
            if psd_patch == "pyring":
                finite_psd = _patch_psd_pyring(
                    finite_psd,
                    self.f,
                    f_min_bp=self.minimum_frequency,
                    f_max_bp=self.maximum_frequency,
                    f_min_patch=f_min_patch,
                    f_max_patch=f_max_patch,
                    f_turn_over_patch=f_turn_over_patch,
                    patch_value=patch_value,
                )
            elif psd_patch:
                finite_psd = _patch_psd_outside_active_band(finite_psd, self.f, active)
            out[ifo] = finite_psd
        return out

    def _acf_from_psd(self, psd):
        df = 1.0 / self.duration
        return 0.5 * np.real(np.fft.irfft(psd * df, n=self._ntime, axis=-1)) * self._ntime

    def _build_detector_likelihood_caches(self, no_lognorm=False):
        caches = {}
        for ifo, name in enumerate(self.ifos):
            cache = _make_likelihood_cache(
                acf=self.acf[ifo],
                likelihood_method=self.likelihood_method,
                no_lognorm=no_lognorm,
            )
            time_band_cache = None
            if self._number_of_time_bands > 1:
                time_band_cache = _make_time_band_likelihood_cache(
                    acf=self.acf[ifo],
                    likelihood_method=self.likelihood_method,
                    time_bands=self.time_bands,
                    sampling_rate=self.sampling_rate,
                    no_lognorm=no_lognorm,
                )
            caches[name] = {"full": cache, "time_bands": time_band_cache}
        return caches

    def _full_frequency_signal_batch(self, theta):
        theta = _ensure_2d(theta)
        physics = theta[:, : self._wfdims]
        targets = [getattr(self._template, "template", None), self._template]
        for target in targets:
            if target is None:
                continue
            func = getattr(target, "make_injections_to_ifo_batch", None)
            if not callable(func):
                continue
            try:
                signal = func(physics, masked=False)
            except TypeError:
                signal = func(physics)
            signal = np.asarray(signal, dtype=complex)
            if signal.shape == (physics.shape[0], self._nchannels, self._nfreq):
                return signal
            mask = getattr(target, "mask", getattr(self._template, "mask", None))
            if mask is not None and signal.shape[:2] == (physics.shape[0], self._nchannels):
                mask = np.asarray(mask, dtype=bool)
                if mask.shape == (self._nfreq,) and signal.shape[-1] == int(np.sum(mask)):
                    full = np.zeros(
                        (physics.shape[0], self._nchannels, self._nfreq),
                        dtype=complex,
                    )
                    full[:, :, mask] = signal
                    return full
        raise ValueError("Template must provide full-grid batched detector waveforms")

    def _signal_time_domain_batch(self, theta):
        signal_fd = self._full_frequency_signal_batch(theta)
        return np.fft.irfft(signal_fd, n=self._ntime, axis=-1).real * self.sampling_rate

    def _signal_and_residual_batch(self, theta):
        signal = self._signal_time_domain_batch(theta)
        return signal, self.data[None, :, :] - signal

    def _finish(self, logl):
        return np.nan_to_num(
            np.asarray(logl, dtype=float),
            copy=True,
            nan=self._inf,
            posinf=self._inf,
            neginf=self._inf,
        ).squeeze()

    def _caches_for_detector(self, detector_index, use_time_bands):
        detector_cache = self._detector_likelihood_caches[self.ifos[detector_index]]
        if use_time_bands and detector_cache["time_bands"] is not None:
            return detector_cache["time_bands"]
        return [detector_cache["full"]]

    def _quadratic_forms_for_cache(self, detector_index, cache, residuals, signal=None):
        if not self.split_inner_products:
            return _quadratic_forms(residuals, cache, self.likelihood_method)
        if cache.inverse_covariance is None:
            cache.inverse_covariance = inv(toeplitz(cache.acf))
        data = self.data[detector_index, cache.start : cache.end]
        signal = np.asarray(signal, dtype=float)
        dd = float(np.dot(data, np.dot(cache.inverse_covariance, data)))
        dh = np.einsum("i,ij,nj->n", data, cache.inverse_covariance, signal)
        hh = np.einsum("ni,ij,nj->n", signal, cache.inverse_covariance, signal)
        return dd - 2.0 * dh + hh

    def _noise_width(self, family, detector_indices):
        del family
        if self.detector_dependent_noise:
            return len(tuple(detector_indices)) * self._number_of_time_bands
        return self._number_of_time_bands

    def _mixed_noise_width(self):
        width = 0
        student_indices = self._detector_indices("student-t")
        hyperbolic_indices = self._detector_indices("hyperbolic")
        if student_indices:
            width += self._noise_width("student-t", student_indices)
        if hyperbolic_indices:
            width += 2 * self._noise_width("hyperbolic", hyperbolic_indices)
        return width

    def _detector_indices(self, family):
        return [i for i, value in enumerate(self.detector_likelihoods) if value == family]

    def _take_noise_block(self, theta, start, detector_indices):
        width = self._noise_width("noise", detector_indices)
        stop = start + width
        if theta.shape[1] < stop:
            raise ValueError(f"theta has {theta.shape[1]} columns but at least {stop} are needed")
        block = theta[:, start:stop]
        if self.detector_dependent_noise:
            block = block.reshape(theta.shape[0], len(tuple(detector_indices)), self._number_of_time_bands)
        return block, stop

    def _band_parameter(self, values, local_detector_index, band_index):
        if self.detector_dependent_noise:
            return values[:, local_detector_index, band_index]
        return values[:, band_index]

    def _gaussian_logl_from_residuals(self, residuals, detector_indices, signal=None):
        logl = np.zeros(residuals.shape[0], dtype=float)
        for detector_index in detector_indices:
            cache = self._detector_likelihood_caches[self.ifos[detector_index]]["full"]
            signal_band = None if signal is None else signal[:, detector_index, :]
            q = self._quadratic_forms_for_cache(
                detector_index, cache, residuals[:, detector_index, :], signal_band
            )
            logl += _gaussian_log_likelihood_from_inner_product(q, cache.log_normalisation)
        return logl

    def _student_t_logl_from_residuals(
        self, theta, residuals, detector_indices, start, signal=None
    ):
        nu, stop = self._take_noise_block(theta, start, detector_indices)
        logl = np.zeros(theta.shape[0], dtype=float)
        for local_index, detector_index in enumerate(detector_indices):
            for band_index, cache in enumerate(
                self._caches_for_detector(detector_index, use_time_bands=True)
            ):
                band_residuals = residuals[:, detector_index, cache.start : cache.end]
                signal_band = (
                    None
                    if signal is None
                    else signal[:, detector_index, cache.start : cache.end]
                )
                q = self._quadratic_forms_for_cache(
                    detector_index, cache, band_residuals, signal_band
                )
                logl += _student_t_log_likelihood_from_inner_product(
                    q,
                    logdet=cache.logdet,
                    dimension=cache.end - cache.start,
                    nu=self._band_parameter(nu, local_index, band_index),
                )
        return logl, stop

    def _hyperbolic_logl_from_residuals(
        self, theta, residuals, detector_indices, start, classic, signal=None
    ):
        alpha, stop = self._take_noise_block(theta, start, detector_indices)
        tail, stop = self._take_noise_block(theta, stop, detector_indices)
        delta = tail if classic else alpha * tail
        logl = np.zeros(theta.shape[0], dtype=float)
        for local_index, detector_index in enumerate(detector_indices):
            for band_index, cache in enumerate(
                self._caches_for_detector(detector_index, use_time_bands=True)
            ):
                band_residuals = residuals[:, detector_index, cache.start : cache.end]
                signal_band = (
                    None
                    if signal is None
                    else signal[:, detector_index, cache.start : cache.end]
                )
                q = self._quadratic_forms_for_cache(
                    detector_index, cache, band_residuals, signal_band
                )
                logl += _hyperbolic_log_likelihood_from_inner_product(
                    q,
                    logdet=cache.logdet,
                    dimension=cache.end - cache.start,
                    alpha=self._band_parameter(alpha, local_index, band_index),
                    delta=self._band_parameter(delta, local_index, band_index),
                )
        return logl, stop

    def gaussian(self, theta):
        theta = _ensure_2d(theta)
        signal, residuals = self._signal_and_residual_batch(theta)
        return self._finish(
            self._gaussian_logl_from_residuals(
                residuals, range(self._nchannels), signal=signal
            )
        )

    def student_t(self, theta):
        theta = _ensure_2d(theta)
        signal, residuals = self._signal_and_residual_batch(theta)
        logl, _ = self._student_t_logl_from_residuals(
            theta, residuals, list(range(self._nchannels)), self._wfdims, signal=signal
        )
        return self._finish(logl)

    def hyperbolic_classic(self, theta):
        theta = _ensure_2d(theta)
        signal, residuals = self._signal_and_residual_batch(theta)
        logl, _ = self._hyperbolic_logl_from_residuals(
            theta,
            residuals,
            list(range(self._nchannels)),
            self._wfdims,
            classic=True,
            signal=signal,
        )
        return self._finish(logl)

    def hyperbolic(self, theta):
        theta = _ensure_2d(theta)
        signal, residuals = self._signal_and_residual_batch(theta)
        logl, _ = self._hyperbolic_logl_from_residuals(
            theta,
            residuals,
            list(range(self._nchannels)),
            self._wfdims,
            classic=False,
            signal=signal,
        )
        return self._finish(logl)

    def mixed(self, theta):
        theta = _ensure_2d(theta)
        signal, residuals = self._signal_and_residual_batch(theta)
        logl = np.zeros(theta.shape[0], dtype=float)
        gaussian_indices = self._detector_indices("gaussian")
        student_indices = self._detector_indices("student-t")
        hyperbolic_indices = self._detector_indices("hyperbolic")

        if gaussian_indices:
            logl += self._gaussian_logl_from_residuals(
                residuals, gaussian_indices, signal=signal
            )
        start = self._wfdims
        if student_indices:
            student_logl, start = self._student_t_logl_from_residuals(
                theta, residuals, student_indices, start, signal=signal
            )
            logl += student_logl
        if hyperbolic_indices:
            hyper_logl, start = self._hyperbolic_logl_from_residuals(
                theta, residuals, hyperbolic_indices, start, classic=True, signal=signal
            )
            logl += hyper_logl
        return self._finish(logl)

    def log_likelihood(self, theta):
        families = set(self.detector_likelihoods)
        if families == {"gaussian"}:
            return self.gaussian(theta)
        if families == {"student-t"}:
            return self.student_t(theta)
        if families == {"hyperbolic"}:
            return self.hyperbolic_classic(theta)
        return self.mixed(theta)


__all__ = [
    "TimeDomainGWLikelihoods",
    "_GohbergSemenculToeplitzInverse",
    "_gaussian_log_likelihood_from_inner_product",
    "_student_t_log_likelihood_from_inner_product",
    "_hyperbolic_log_likelihood_from_inner_product",
    "_patch_psd_outside_active_band",
    "_patch_psd_pyring",
]
