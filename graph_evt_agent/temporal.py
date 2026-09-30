"""Temporal graphs that make GNN/GAT training useful for 1-D series.

Multichannel graphs use sensors as nodes and learn the source sensor. A
univariate series has no sensor graph, so this module uses overlapping time
windows as nodes and learns the window containing the event onset instead.

Two topologies are available:

* ``edge_method="neighborhood"`` – the original banded graph that links each
  window to its ``neighborhood`` nearest windows with unit (boolean) edges.
* ``edge_method="dfa" | "wavelet" | "ar"`` – a **fully connected** graph whose
  edge weights in ``[0, 1]`` measure how similar two windows are in their
  detrended fluctuation scaling, discrete-wavelet energy spectrum, or
  AR-innovation dynamics. An optional pruning step (``prune=...``) then turns
  the dense graph into a sparse one while keeping the weights of the kept edges.

Weighted graphs are returned as a symmetric float matrix with a unit diagonal;
zero means "no edge". :class:`~graph_evt_agent.learning.GraphProcessModel`
uses the weights for GNN aggregation and as an attention prior in the GAT.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .config import GraphConfig
from .graph import dfa_graph
from .learning import GraphEpisode

EDGE_METHODS = ("neighborhood", "dfa", "wavelet", "ar")
PRUNE_METHODS = ("none", "threshold", "knn", "mst", "disparity")
TEMPORAL_FEATURE_NAMES = ("peak", "energy", "latency", "degree", "neighbor")

# Orthonormal Daubechies low-pass (scaling) filters; high-pass filters follow
# from the quadrature-mirror relation in :func:`_dwt_energies`.
WAVELET_FILTERS = {
    "haar": np.array([1.0, 1.0]) / np.sqrt(2.0),
    "db2": np.array([1 + np.sqrt(3), 3 + np.sqrt(3), 3 - np.sqrt(3), 1 - np.sqrt(3)])
    / (4 * np.sqrt(2.0)),
    "db4": np.array([0.2303778133088964, 0.7148465705529154, 0.6308807679298587,
                     -0.0279837694168599, -0.1870348117190931, 0.0308413818355607,
                     0.0328830116668852, -0.0105974017850690]),
}


@dataclass(frozen=True)
class TemporalGraphConfig:
    """Settings for a 1-D graph whose nodes are time windows.

    ``edge_method`` selects the topology (see the module docstring).
    ``dfa_scales`` defaults to powers of two from 4 to ``window_size // 2``;
    ``wavelet_levels`` defaults to the deepest level the filter supports.
    ``ar_order`` is the order of the AR model fitted on the baseline prefix.

    Pruning of the fully connected graph (``prune``):

    * ``"threshold"`` keeps edges with weight ``>= prune_threshold``;
    * ``"knn"`` keeps each node's ``prune_k`` strongest edges (symmetric union);
    * ``"mst"`` keeps the maximum spanning tree plus the ``prune_k`` strongest
      edges per node (``prune_k=0`` gives the bare tree) – always connected;
    * ``"disparity"`` keeps edges significant under the disparity filter
      (Serrano, Boguñá & Vespignani, 2009) at level ``prune_alpha``.

    ``prune_connected=True`` adds the maximum spanning tree to any pruning
    result so no window becomes isolated.
    """

    window_size: int = 16
    stride: int = 8
    neighborhood: int = 1
    baseline_size: int = 100
    edge_method: Literal["neighborhood", "dfa", "wavelet", "ar"] = "neighborhood"
    dfa_scales: tuple[int, ...] | None = None
    wavelet: Literal["haar", "db2", "db4"] = "db2"
    wavelet_levels: int | None = None
    ar_order: int = 2
    prune: Literal["none", "threshold", "knn", "mst", "disparity"] = "none"
    prune_threshold: float = 0.5
    prune_k: int = 3
    prune_alpha: float = 0.05
    prune_connected: bool = False

    def __post_init__(self) -> None:
        if min(self.window_size, self.stride, self.neighborhood, self.baseline_size) < 1:
            raise ValueError("temporal graph parameters must be positive")
        if self.edge_method not in EDGE_METHODS:
            raise ValueError(f"edge_method must be one of {EDGE_METHODS}")
        if self.prune not in PRUNE_METHODS:
            raise ValueError(f"prune must be one of {PRUNE_METHODS}")
        if self.edge_method == "neighborhood" and (self.prune != "none" or self.prune_connected):
            raise ValueError("pruning applies only to weighted (dfa/wavelet/ar) graphs")
        if self.wavelet not in WAVELET_FILTERS:
            raise ValueError(f"wavelet must be one of {tuple(WAVELET_FILTERS)}")
        if self.wavelet_levels is not None and self.wavelet_levels < 1:
            raise ValueError("wavelet_levels must be positive")
        if self.ar_order < 1:
            raise ValueError("ar_order must be positive")
        if not 0 <= self.prune_threshold <= 1:
            raise ValueError("prune_threshold must be in [0, 1]")
        if self.prune_k < 0 or (self.prune == "knn" and self.prune_k < 1):
            raise ValueError("prune_k must be non-negative (positive for knn)")
        if not 0 < self.prune_alpha < 1:
            raise ValueError("prune_alpha must be in (0, 1)")

    def resolved_dfa_scales(self) -> tuple[int, ...]:
        """DFA scales used for this window size (validated)."""
        scales = self.dfa_scales
        if scales is None:
            scales = tuple(2 ** power for power in range(2, 32)
                           if 2 ** power <= self.window_size // 2)
        scales = tuple(sorted({int(scale) for scale in scales}))
        if len(scales) < 2 or scales[0] < 4 or 2 * scales[-1] > self.window_size:
            raise ValueError("DFA edges need at least two scales in [4, window_size / 2]; "
                             "use window_size >= 16 or pass dfa_scales")
        return scales


@dataclass(frozen=True)
class TemporalGraph:
    """A 1-D temporal graph: window nodes, their features and (weighted) edges.

    ``adjacency`` is boolean for the neighborhood topology and a symmetric
    float weight matrix (unit diagonal, zero = pruned) otherwise.
    ``full_weights`` holds the dense weights before pruning.
    """

    features: np.ndarray
    adjacency: np.ndarray
    window_starts: np.ndarray
    method: str
    full_weights: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    window_size: int | None = None

    @property
    def edge_mask(self) -> np.ndarray:
        """Boolean ``[node, node]`` mask of kept edges, including self-loops."""
        return np.asarray(self.adjacency, dtype=float) > 0

    @property
    def density(self) -> float:
        """Fraction of the ``n (n - 1) / 2`` possible window pairs that are linked."""
        nodes = len(self.window_starts)
        if nodes < 2:
            return 0.0
        return float(np.triu(self.edge_mask, 1).sum() / (nodes * (nodes - 1) / 2))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return TEMPORAL_FEATURE_NAMES


def _node_segments(data: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    """``[window, node]`` samples; tail windows are shifted back to full length."""
    clipped = np.minimum(starts, len(data) - window)
    return np.column_stack([data[start:start + window] for start in clipped])


def _dfa_weights(segments: np.ndarray, config: TemporalGraphConfig):
    """Cross-DFA similarity with windows playing the role of channels."""
    scales = config.resolved_dfa_scales()
    result = dfa_graph(segments, GraphConfig(method="dfa", dfa_scales=scales,
                                             edge_threshold=0.0))
    return result.dependence, {"dfa_scales": np.asarray(scales),
                               "alpha": result.diagnostics["alpha"]}


def _dwt_energies(segments: np.ndarray, wavelet: str, levels: int | None):
    """Energy per detail level (finest first) plus the final approximation.

    Uses a periodized orthonormal DWT, so energies sum to the window energy
    (up to the edge sample repeated when a level has odd length).
    """
    low = WAVELET_FILTERS[wavelet]
    high = low[::-1] * (-1.0) ** np.arange(len(low))
    length = len(segments)
    maximum = max(1, int(np.floor(np.log2(length / len(low)))) + 1)
    levels = maximum if levels is None else levels
    if levels > maximum:
        raise ValueError(f"wavelet_levels={levels} exceeds {maximum} for this window/filter")
    approximation = segments.T
    energies = []
    for _ in range(levels):
        if approximation.shape[1] % 2:
            approximation = np.column_stack([approximation, approximation[:, -1]])
        size = approximation.shape[1]
        index = (2 * np.arange(size // 2)[:, None] + np.arange(len(low))[None, :]) % size
        blocks = approximation[:, index]
        detail = blocks @ high
        approximation = blocks @ low
        energies.append(np.sum(detail ** 2, axis=1))
    energies.append(np.sum(approximation ** 2, axis=1))
    return np.column_stack(energies), levels


def _wavelet_weights(segments: np.ndarray, config: TemporalGraphConfig):
    """Spectral (Jensen–Shannon) × amplitude similarity of DWT energies."""
    energies, levels = _dwt_energies(segments, config.wavelet, config.wavelet_levels)
    energies = np.maximum(energies, 1e-12)
    total = energies.sum(axis=1)
    spectra = energies / total[:, None]
    p, q = spectra[:, None, :], spectra[None, :, :]
    mixture = 0.5 * (p + q)
    divergence = 0.5 * (np.sum(p * np.log2(p / mixture), axis=2)
                        + np.sum(q * np.log2(q / mixture), axis=2))
    spectral = 1 - np.sqrt(np.clip(divergence, 0, 1))
    amplitude = np.sqrt(np.minimum(total[:, None], total[None, :])
                        / np.maximum(total[:, None], total[None, :]))
    return spectral * amplitude, {"wavelet": config.wavelet, "wavelet_levels": levels,
                                  "relative_energy": spectra}


def _ar_innovations(data: np.ndarray, baseline_size: int, order: int):
    """One-step innovations of an AR(order) model fitted on the baseline only."""
    baseline = data[:baseline_size]
    if len(baseline) <= 2 * order + 2:
        raise ValueError("baseline is too short for requested AR order")

    def design(series: np.ndarray) -> np.ndarray:
        lags = [series[order - lag - 1:len(series) - lag - 1] for lag in range(order)]
        return np.column_stack([np.ones(len(series) - order), *lags])

    coefficients, *_ = np.linalg.lstsq(design(baseline), baseline[order:], rcond=None)
    innovations = np.zeros_like(data)
    innovations[order:] = data[order:] - design(data) @ coefficients
    return innovations, coefficients


def _ar_weights(data: np.ndarray, starts: np.ndarray, config: TemporalGraphConfig):
    """Absolute correlation between windows of baseline-AR innovations."""
    innovations, coefficients = _ar_innovations(data, config.baseline_size, config.ar_order)
    segments = _node_segments(innovations, starts, config.window_size)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.nan_to_num(np.abs(np.corrcoef(segments, rowvar=False)))
    return np.atleast_2d(weights), {"ar_order": config.ar_order,
                                    "ar_coefficients": coefficients}


def _maximum_spanning_tree(weights: np.ndarray) -> np.ndarray:
    cost = (weights.max() + 1.0) - weights
    np.fill_diagonal(cost, 0.0)
    tree = minimum_spanning_tree(cost).toarray() != 0
    return tree | tree.T


def _knn_mask(weights: np.ndarray, k: int) -> np.ndarray:
    nodes = len(weights)
    mask = np.zeros((nodes, nodes), dtype=bool)
    k = min(k, nodes - 1)
    if k < 1:
        return mask
    ranked = weights.copy()
    np.fill_diagonal(ranked, -np.inf)
    neighbors = np.argsort(-ranked, axis=1, kind="stable")[:, :k]
    mask[np.repeat(np.arange(nodes), k), neighbors.ravel()] = True
    return mask | mask.T


def _disparity_mask(weights: np.ndarray, alpha: float) -> np.ndarray:
    off = np.clip(weights, 0, None)
    np.fill_diagonal(off, 0.0)
    strength = off.sum(axis=1, keepdims=True)
    degree = (off > 0).sum(axis=1, keepdims=True)
    share = np.divide(off, strength, out=np.zeros_like(off), where=strength > 0)
    p_value = (1 - share) ** np.maximum(degree - 1, 0)
    # A node with a single edge keeps it (Serrano et al. convention).
    significant = (off > 0) & ((p_value < alpha) | (degree <= 1))
    return significant | significant.T


def prune_weighted_graph(weights: np.ndarray, method: str = "knn", *,
                         threshold: float = 0.5, k: int = 3, alpha: float = 0.05,
                         connected: bool = False) -> np.ndarray:
    """Sparsify a dense symmetric similarity matrix; kept edges keep their weight.

    Returns a symmetric float matrix with a unit diagonal and zeros for
    removed edges. See :class:`TemporalGraphConfig` for the methods.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or not np.isfinite(w).all():
        raise ValueError("weights must be a finite square matrix")
    if not np.allclose(w, w.T) or (w < 0).any():
        raise ValueError("weights must be symmetric and non-negative")
    if method not in PRUNE_METHODS:
        raise ValueError(f"method must be one of {PRUNE_METHODS}")
    nodes = len(w)
    if method == "none":
        mask = np.ones((nodes, nodes), dtype=bool)
    elif method == "threshold":
        mask = w >= threshold
    elif method == "knn":
        mask = _knn_mask(w, k)
    elif method == "mst":
        mask = _maximum_spanning_tree(w) | _knn_mask(w, k)
    else:
        mask = _disparity_mask(w, alpha)
    if connected and nodes > 1:
        mask = mask | _maximum_spanning_tree(w)
    # A kept edge must stay an edge even if its similarity is exactly zero.
    pruned = np.where(mask, np.maximum(w, 1e-6), 0.0)
    np.fill_diagonal(pruned, 1.0)
    return pruned


def build_temporal_graph(values: np.ndarray,
                         config: TemporalGraphConfig | None = None) -> TemporalGraph:
    """Build window nodes, features and edges for a 1-D recording.

    Features follow the transparent ranker's five-column schema: robust peak,
    energy, within-window latency, graph degree, and neighboring peak. On a
    weighted graph ``degree`` is the node strength (sum of edge weights) and
    ``neighbor`` the weight-averaged peak over the node and its neighbors.
    """
    config = config or TemporalGraphConfig()
    data = np.asarray(values, dtype=float)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("values must be a non-empty finite 1-D series")
    if config.baseline_size >= len(data):
        raise ValueError("baseline_size must be smaller than the series")

    baseline = data[:config.baseline_size]
    location = float(np.median(baseline))
    scale = max(float(np.median(np.abs(baseline - location)) * 1.4826), 1e-8)
    starts = np.arange(0, len(data), config.stride, dtype=int)

    full_weights, diagnostics = None, {}
    if config.edge_method == "neighborhood":
        adjacency = np.eye(len(starts), dtype=bool)
        for distance in range(1, config.neighborhood + 1):
            indices = np.arange(len(starts) - distance)
            adjacency[indices, indices + distance] = True
            adjacency[indices + distance, indices] = True
        weights = adjacency.astype(float)
    else:
        if config.window_size > len(data) or len(starts) < 2:
            raise ValueError("weighted temporal graphs need at least two full windows")
        standardized = (data - location) / scale
        segments = _node_segments(standardized, starts, config.window_size)
        if config.edge_method == "dfa":
            full_weights, diagnostics = _dfa_weights(segments, config)
        elif config.edge_method == "wavelet":
            full_weights, diagnostics = _wavelet_weights(segments, config)
        else:
            full_weights, diagnostics = _ar_weights(data, starts, config)
        full_weights = np.clip(0.5 * (full_weights + full_weights.T), 0.0, 1.0)
        np.fill_diagonal(full_weights, 1.0)
        adjacency = prune_weighted_graph(
            full_weights, config.prune, threshold=config.prune_threshold,
            k=config.prune_k, alpha=config.prune_alpha, connected=config.prune_connected)
        weights = adjacency

    peak, energy, latency = [], [], []
    for start in starts:
        z = np.abs(data[start:min(start + config.window_size, len(data))] - location) / scale
        peak.append(float(z.max()))
        energy.append(float(np.mean(z ** 2)))
        latency.append(float(np.argmax(z)))
    peak_array = np.asarray(peak)
    degree = weights.sum(axis=1) - 1
    neighbor = weights @ peak_array / weights.sum(axis=1)
    features = np.column_stack([peak_array, energy, latency, degree, neighbor])
    method = config.edge_method if config.edge_method == "neighborhood" else (
        f"{config.edge_method}+{config.prune}")
    return TemporalGraph(features, adjacency, starts, method, full_weights, diagnostics,
                         config.window_size)


def univariate_temporal_graph(values: np.ndarray,
                              config: TemporalGraphConfig | None = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(features, adjacency, window_starts)`` for a 1-D recording.

    ``adjacency`` is boolean for ``edge_method="neighborhood"`` and a weighted
    matrix otherwise; both can be passed to ``GraphProcessModel``. Use
    :func:`build_temporal_graph` for dense weights and diagnostics.
    """
    graph = build_temporal_graph(values, config)
    return graph.features, graph.adjacency, graph.window_starts


def univariate_graph_episode(values: np.ndarray, event_time: int,
                             config: TemporalGraphConfig | None = None) -> GraphEpisode:
    """Convert a labelled 1-D recording into a temporal-node graph episode."""
    config = config or TemporalGraphConfig()
    features, adjacency, starts = univariate_temporal_graph(values, config)
    length = np.asarray(values).shape[0]
    if not 0 <= int(event_time) < length:
        raise ValueError("event_time must identify a sample in the series")
    contains = np.flatnonzero((starts <= event_time)
                              & (event_time < starts + config.window_size))
    candidates = contains if len(contains) else np.arange(len(starts))
    centres = starts[candidates] + config.window_size / 2
    source = int(candidates[np.argmin(np.abs(centres - event_time))])
    return GraphEpisode(features, adjacency, source)
