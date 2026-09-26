"""Baseline-only AR, DFA and wavelet–Kuramoto graph hypotheses."""

import numpy as np

from .config import GraphConfig
from .models import GraphResult


def _validate(baseline: np.ndarray) -> np.ndarray:
    data = np.asarray(baseline, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError("baseline must be a finite [time, channel] array")
    return data


def _adjacency(dependence: np.ndarray, threshold: float) -> np.ndarray:
    adjacency = dependence >= threshold
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, True)
    return adjacency


def _ar_residuals(data: np.ndarray, order: int) -> np.ndarray:
    residuals = []
    for channel in data.T:
        target = channel[order:]
        design = np.column_stack(
            [channel[order - lag - 1 : -lag - 1] for lag in range(order)]
        )
        design = np.column_stack([np.ones(len(design)), design])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals.append(target - design @ coefficients)
    return np.column_stack(residuals)


def innovation_graph(baseline: np.ndarray, config: GraphConfig) -> GraphResult:
    """Create a graph from dependence between separate-channel AR innovations."""
    data = _validate(baseline)
    if len(data) <= config.ar_order + 2:
        raise ValueError("baseline is too short for requested AR order")
    residuals = _ar_residuals(data, config.ar_order)
    dependence = np.atleast_2d(np.nan_to_num(np.abs(np.corrcoef(residuals, rowvar=False))))
    return GraphResult(_adjacency(dependence, config.edge_threshold), dependence,
                       "ar_innovation_correlation", {"ar_order": config.ar_order})


def _detrended_segments(profile: np.ndarray, scale: int) -> np.ndarray:
    """Return locally linear-detrended, non-overlapping profile segments."""
    count = len(profile) // scale
    if count < 2:
        raise ValueError("baseline is too short for a requested DFA scale")
    segments = profile[: count * scale].reshape(count, scale)
    x = np.arange(scale, dtype=float)
    design = np.column_stack([x, np.ones(scale)])
    coefficients = np.linalg.lstsq(design, segments.T, rcond=None)[0]
    trends = (design @ coefficients).T
    return segments - trends


def dfa_graph(baseline: np.ndarray, config: GraphConfig) -> GraphResult:
    """Build a cross-DFA graph from multiscale detrended covariance.

    Similarity combines the mean absolute detrended cross-correlation (DCCA-like
    coefficient) with closeness of the per-channel DFA exponents.
    """
    data = _validate(baseline)
    scales = np.asarray(sorted(set(config.dfa_scales)), dtype=int)
    if scales[-1] * 2 > len(data):
        raise ValueError("each DFA scale requires at least two baseline windows")
    profiles = np.cumsum(data - data.mean(axis=0), axis=0)
    channels = data.shape[1]
    fluctuation = np.empty((len(scales), channels))
    cross = np.zeros((len(scales), channels, channels))
    for scale_index, scale in enumerate(scales):
        residuals = [_detrended_segments(profiles[:, node], int(scale))
                     for node in range(channels)]
        flat = [residual.reshape(-1) for residual in residuals]
        fluctuation[scale_index] = [np.sqrt(np.mean(values**2)) for values in flat]
        cross[scale_index] = np.nan_to_num(np.abs(np.corrcoef(flat)))
    exponents = np.array([
        np.polyfit(np.log(scales), np.log(np.maximum(fluctuation[:, node], 1e-12)), 1)[0]
        for node in range(channels)
    ])
    exponent_similarity = np.exp(-np.abs(exponents[:, None] - exponents[None, :]))
    dependence = 0.5 * cross.mean(axis=0) + 0.5 * exponent_similarity
    return GraphResult(
        _adjacency(dependence, config.edge_threshold), dependence, "cross_dfa",
        {"scales": scales, "alpha": exponents, "fluctuation": fluctuation},
    )


def _morlet_phases(data: np.ndarray, periods: tuple[float, ...]) -> np.ndarray:
    phases = np.empty((len(periods), len(data), data.shape[1]))
    for band, period in enumerate(periods):
        radius = min(int(np.ceil(4 * period)), (len(data) - 1) // 2)
        time = np.arange(-radius, radius + 1, dtype=float)
        sigma = period
        kernel = np.exp(2j * np.pi * time / period) * np.exp(-(time**2) / (2 * sigma**2))
        kernel /= np.sqrt(np.sum(np.abs(kernel) ** 2))
        for node in range(data.shape[1]):
            phases[band, :, node] = np.angle(np.convolve(data[:, node], kernel, mode="same"))
    return phases


def _phase_locking(phases: np.ndarray) -> np.ndarray:
    differences = phases[:, :, :, None] - phases[:, :, None, :]
    return np.abs(np.mean(np.exp(1j * differences), axis=(0, 1)))


def kuramoto_wavelet_graph(baseline: np.ndarray, config: GraphConfig) -> GraphResult:
    """Build a phase-locking graph with Morlet phases and circular-shift surrogates."""
    data = _validate(baseline)
    phases = _morlet_phases(data, config.wavelet_periods)
    dependence = _phase_locking(phases)
    rng = np.random.default_rng(config.random_state)
    surrogates = np.empty((config.surrogate_count, data.shape[1], data.shape[1]))
    minimum_shift = max(1, len(data) // 10)
    for sample in range(config.surrogate_count):
        shifted = phases.copy()
        for node in range(data.shape[1]):
            shift = int(rng.integers(minimum_shift, max(minimum_shift + 1, len(data) - minimum_shift)))
            shifted[:, :, node] = np.roll(shifted[:, :, node], shift, axis=1)
        surrogates[sample] = _phase_locking(shifted)
    surrogate_threshold = np.quantile(surrogates, config.surrogate_quantile, axis=0)
    adjacency = (dependence >= config.edge_threshold) & (dependence > surrogate_threshold)
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, True)
    order_parameter = np.abs(np.mean(np.exp(1j * phases), axis=2))
    return GraphResult(
        adjacency, dependence, "kuramoto_wavelet_phase_locking",
        {"periods": np.asarray(config.wavelet_periods),
         "surrogate_threshold": surrogate_threshold,
         "order_parameter_mean": order_parameter.mean(axis=1)},
    )


def _approximately_stationary(data: np.ndarray) -> bool:
    """Conservative rolling location/scale heuristic used only by auto mode."""
    chunks = np.array_split(data, 4, axis=0)
    means = np.array([chunk.mean(axis=0) for chunk in chunks])
    variances = np.array([chunk.var(axis=0) for chunk in chunks])
    full_scale = np.maximum(data.std(axis=0), 1e-12)
    location_stable = np.ptp(means, axis=0) < full_scale
    scale_ratio = variances.max(axis=0) / np.maximum(variances.min(axis=0), 1e-12)
    return bool(np.all(location_stable & (scale_ratio < 4)))


def build_graph(baseline: np.ndarray, config: GraphConfig) -> GraphResult:
    """Build the configured graph, or a prespecified intersection ensemble."""
    data = _validate(baseline)
    method = config.method
    if method == "auto":
        method = "ar" if _approximately_stationary(data) else "ensemble"
    if method == "ar":
        return innovation_graph(data, config)
    if method == "dfa":
        return dfa_graph(data, config)
    if method == "kuramoto":
        return kuramoto_wavelet_graph(data, config)
    dfa = dfa_graph(data, config)
    kuramoto = kuramoto_wavelet_graph(data, config)
    adjacency = dfa.adjacency & kuramoto.adjacency
    dependence = np.sqrt(dfa.dependence * kuramoto.dependence)
    return GraphResult(adjacency, dependence, "dfa_kuramoto_intersection",
                       {"dfa": dfa.diagnostics, "kuramoto": kuramoto.diagnostics})
