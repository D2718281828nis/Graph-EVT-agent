"""Temporal graphs that make GNN/GAT training useful for 1-D series.

Multichannel graphs use sensors as nodes and learn the source sensor. A
univariate series has no sensor graph, so this module uses overlapping time
windows as nodes and learns the window containing the event onset instead.
"""

from dataclasses import dataclass

import numpy as np

from .learning import GraphEpisode


@dataclass(frozen=True)
class TemporalGraphConfig:
    """Settings for a 1-D graph whose nodes are time windows."""

    window_size: int = 16
    stride: int = 8
    neighborhood: int = 1
    baseline_size: int = 100

    def __post_init__(self) -> None:
        if min(self.window_size, self.stride, self.neighborhood, self.baseline_size) < 1:
            raise ValueError("temporal graph parameters must be positive")


def univariate_temporal_graph(values: np.ndarray,
                              config: TemporalGraphConfig | None = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(features, adjacency, window_starts)`` for a 1-D recording.

    Features follow the transparent ranker's five-column schema: robust peak,
    energy, within-window latency, graph degree, and neighboring peak.
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
    adjacency = np.eye(len(starts), dtype=bool)
    for distance in range(1, config.neighborhood + 1):
        indices = np.arange(len(starts) - distance)
        adjacency[indices, indices + distance] = True
        adjacency[indices + distance, indices] = True

    peak, energy, latency = [], [], []
    for start in starts:
        z = np.abs(data[start:min(start + config.window_size, len(data))] - location) / scale
        peak.append(float(z.max()))
        energy.append(float(np.mean(z ** 2)))
        latency.append(float(np.argmax(z)))
    peak_array = np.asarray(peak)
    degree = adjacency.sum(axis=1) - 1
    neighbor = adjacency @ peak_array / adjacency.sum(axis=1)
    features = np.column_stack([peak_array, energy, latency, degree, neighbor])
    return features, adjacency, starts


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
