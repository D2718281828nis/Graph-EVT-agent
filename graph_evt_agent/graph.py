"""Baseline-only graph hypotheses.

AR innovations, cross-DFA and wavelet–Kuramoto (original), plus correlation, partial
correlation, lagged and prewhitened cross-correlation and conditional Granger strength.
"""

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


def _edges(dependence: np.ndarray, config: GraphConfig, mask: np.ndarray | None = None) -> np.ndarray:
    """Apply the configured edge rule; ``threshold`` is exactly the original ``_adjacency``."""
    if config.edge_rule == "topk":
        nodes = len(dependence)
        scores = np.array(dependence, dtype=float)
        np.fill_diagonal(scores, -np.inf)
        adjacency = np.zeros((nodes, nodes), dtype=bool)
        for node in range(nodes):
            partners = np.argsort(-scores[node], kind="stable")[: min(config.top_k_edges, nodes - 1)]
            adjacency[node, partners] = True
        adjacency = np.logical_or(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, True)
    else:
        adjacency = _adjacency(dependence, config.edge_threshold)
    if mask is not None:
        adjacency &= mask
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
    return GraphResult(_edges(dependence, config), dependence,
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
        _edges(dependence, config), dependence, "cross_dfa",
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


def _standardize(data: np.ndarray) -> np.ndarray:
    centred = data - data.mean(axis=0)
    return centred / np.maximum(centred.std(axis=0), 1e-12)


def _lagged_correlation(data: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Best corr(x_i(t), x_j(t + lag)) over |lag| <= max_lag, and the lag where it is reached.

    Each lag uses only the overlapping samples, standardised on that overlap. The matrix is
    symmetric because the lag window covers both directions; ``lag[j, i] == -lag[i, j]``.
    """
    length, nodes = data.shape
    if length <= 2 * max_lag + 20:
        raise ValueError("baseline is too short for the requested max_lag")
    best = np.full((nodes, nodes), -np.inf)
    best_lag = np.zeros((nodes, nodes), dtype=int)
    for lag in range(-max_lag, max_lag + 1):
        first = _standardize(data[max(0, -lag): length - max(0, lag)])
        second = _standardize(data[max(0, lag): length - max(0, -lag)])
        correlation = first.T @ second / len(first)
        better = correlation > best
        best[better] = correlation[better]
        best_lag[better] = lag
    best = np.maximum(best, 0.0)
    np.fill_diagonal(best, 1.0)
    np.fill_diagonal(best_lag, 0)
    return best, best_lag


def _partial_correlation(data: np.ndarray) -> np.ndarray:
    precision = np.linalg.pinv(np.corrcoef(data, rowvar=False))
    scale = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    partial = np.abs(-precision / np.outer(scale, scale))
    np.fill_diagonal(partial, 1.0)
    return np.nan_to_num(partial)


def _granger_strength(data: np.ndarray, max_lag: int, lag_step: int, ridge: float) -> np.ndarray:
    """Directed conditional Granger strength ``G[i, m] = ln(RSS_m without i / RSS_m with all)``.

    Every target ``m`` is regressed on lags ``lag_step, 2*lag_step, ..., max_lag`` of all channels
    (own past included), with a ridge penalty on standardised regressors. Dropping channel ``i``
    measures what ``i`` adds once every other channel's past is known, so a two-step path
    ``i -> j -> m`` is explained by ``j`` and does not create an ``i``–``m`` link.
    """
    length, nodes = data.shape
    lags = np.arange(lag_step, max_lag + 1, lag_step)
    if length - max_lag <= nodes * len(lags) + 10:
        raise ValueError("baseline is too short for the requested Granger lags")
    standard = _standardize(data)
    blocks = [np.column_stack([standard[max_lag - lag: length - lag, channel] for lag in lags])
              for channel in range(nodes)]

    def rss(target: np.ndarray, channels: list[int]) -> float:
        design = np.column_stack([blocks[channel] for channel in channels])
        design = np.column_stack([np.ones(len(design)), design])
        gram = design.T @ design
        gram[1:, 1:] += ridge * np.eye(gram.shape[0] - 1)
        coefficients = np.linalg.solve(gram, design.T @ target)
        residual = target - design @ coefficients
        return float(residual @ residual)

    strength = np.zeros((nodes, nodes))
    for target_node in range(nodes):
        target = standard[max_lag:, target_node]
        full = rss(target, list(range(nodes)))
        for dropped in range(nodes):
            if dropped != target_node:
                reduced = rss(target, [channel for channel in range(nodes) if channel != dropped])
                strength[dropped, target_node] = max(0.0, np.log(reduced / max(full, 1e-300)))
    return strength


def _shift_surrogate_threshold(data: np.ndarray, dependence_of, config: GraphConfig) -> np.ndarray:
    """Quantile of the dependence under independent circular shifts of every channel."""
    rng = np.random.default_rng(config.random_state)
    minimum_shift = max(1, len(data) // 10)
    samples = np.empty((config.surrogate_count, data.shape[1], data.shape[1]))
    for sample in range(config.surrogate_count):
        shifted = np.column_stack([
            np.roll(data[:, node], int(rng.integers(minimum_shift, max(minimum_shift + 1, len(data) - minimum_shift))))
            for node in range(data.shape[1])
        ])
        samples[sample] = dependence_of(shifted)
    return np.quantile(samples, config.surrogate_quantile, axis=0)


def _prune_indirect(adjacency: np.ndarray, dependence: np.ndarray, lag: np.ndarray, tolerance: int) -> list[tuple[int, int, int]]:
    """Remove i–j when a node k gives stronger i–k and k–j links with |lag_ik| + |lag_kj| ≈ |lag_ij|."""
    significant = adjacency.copy()
    pruned = []
    nodes = len(adjacency)
    for first in range(nodes):
        for second in range(first + 1, nodes):
            if not significant[first, second]:
                continue
            for middle in range(nodes):
                if middle in (first, second) or not (significant[first, middle] and significant[middle, second]):
                    continue
                stronger = (dependence[first, middle] > dependence[first, second]
                            and dependence[middle, second] > dependence[first, second])
                path = abs(lag[first, middle]) + abs(lag[middle, second])
                if stronger and abs(path - abs(lag[first, second])) <= tolerance:
                    adjacency[first, second] = adjacency[second, first] = False
                    pruned.append((first, second, middle))
                    break
    return pruned


def correlation_graph(baseline: np.ndarray, config: GraphConfig) -> GraphResult:
    """|corr| of the channels at lag 0."""
    data = _validate(baseline)
    measure = lambda values: np.nan_to_num(np.abs(np.corrcoef(values, rowvar=False)))
    dependence = measure(data)
    threshold = _shift_surrogate_threshold(data, measure, config) if config.surrogate_test else None
    mask = dependence > threshold if threshold is not None else None
    return GraphResult(_edges(dependence, config, mask), dependence, "correlation",
                       {"surrogate_threshold": threshold})


def partial_correlation_graph(baseline: np.ndarray, config: GraphConfig) -> GraphResult:
    """|partial correlation| from the inverse correlation matrix (removes lag-0 paths through others)."""
    data = _validate(baseline)
    dependence = _partial_correlation(data)
    return GraphResult(_edges(dependence, config), dependence, "partial_correlation", {})


def lagged_correlation_graph(baseline: np.ndarray, config: GraphConfig, prewhiten: bool = False) -> GraphResult:
    """Best lagged cross-correlation; with ``prewhiten`` on AR(``ar_order``) innovations (Box–Jenkins)."""
    data = _validate(baseline)
    if prewhiten:
        if len(data) <= config.ar_order + 2:
            raise ValueError("baseline is too short for requested AR order")
        data = _ar_residuals(data, config.ar_order)
    dependence, lag = _lagged_correlation(data, config.max_lag)
    threshold = None
    mask = None
    if config.surrogate_test:
        threshold = _shift_surrogate_threshold(data, lambda values: _lagged_correlation(values, config.max_lag)[0], config)
        mask = dependence > threshold
    adjacency = _edges(dependence, config, mask)
    pruned = _prune_indirect(adjacency, dependence, lag, config.lag_tolerance) if config.prune_indirect else []
    name = "prewhitened_lagged_cross_correlation" if prewhiten else "lagged_cross_correlation"
    return GraphResult(adjacency, dependence, name,
                       {"lag": lag, "max_lag": config.max_lag, "surrogate_threshold": threshold,
                        "pruned": pruned, "ar_order": config.ar_order if prewhiten else None})


def granger_graph(baseline: np.ndarray, config: GraphConfig) -> GraphResult:
    """Undirected graph from conditional Granger strength, ``max(G[i, j], G[j, i])``."""
    data = _validate(baseline)
    directed = _granger_strength(data, config.max_lag, config.lag_step, config.ridge)
    dependence = np.maximum(directed, directed.T)
    return GraphResult(_edges(dependence, config), dependence, "conditional_granger",
                       {"directed_strength": directed, "lags": np.arange(config.lag_step, config.max_lag + 1, config.lag_step),
                        "ridge": config.ridge})


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
    if method == "corr":
        return correlation_graph(data, config)
    if method == "partial":
        return partial_correlation_graph(data, config)
    if method == "xcorr":
        return lagged_correlation_graph(data, config)
    if method == "pwxcorr":
        return lagged_correlation_graph(data, config, prewhiten=True)
    if method == "granger":
        return granger_graph(data, config)
    dfa = dfa_graph(data, config)
    kuramoto = kuramoto_wavelet_graph(data, config)
    adjacency = dfa.adjacency & kuramoto.adjacency
    dependence = np.sqrt(dfa.dependence * kuramoto.dependence)
    return GraphResult(adjacency, dependence, "dfa_kuramoto_intersection",
                       {"dfa": dfa.diagnostics, "kuramoto": kuramoto.diagnostics})
