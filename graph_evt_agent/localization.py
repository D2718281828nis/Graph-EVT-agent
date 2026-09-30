"""Transparent source ranking usable before a supervised GAT is available."""

import numpy as np

from .models import SourceRanking


def heuristic_probabilities(features: np.ndarray) -> np.ndarray:
    """Softmax of the fixed score over ``[peak, energy, latency, degree, neighbor]``.

    Extra columns (e.g. temporal onset context features) are ignored.
    """
    peak, energy, latency, _, neighbor = np.asarray(features, dtype=float)[:, :5].T
    raw = peak + np.sqrt(energy) + 0.15 * neighbor - 0.5 * latency
    raw -= raw.max()
    return np.exp(raw) / np.exp(raw).sum()


def rank_sources(values: np.ndarray, event_time: int, adjacency: np.ndarray,
                 location: np.ndarray, scale: np.ndarray, window: int = 12) -> SourceRanking:
    """Rank early, strong responses and reward agreement with graph neighbors."""
    data = np.asarray(values, dtype=float)
    stop = min(len(data), event_time + window)
    z = np.abs(data[event_time:stop] - location) / scale
    peak = z.max(axis=0)
    energy = np.mean(z**2, axis=0)
    cutoff = np.maximum(3.0, 0.5 * peak)
    latency = np.array([np.argmax(z[:, j] >= cutoff[j]) for j in range(z.shape[1])])
    degree = adjacency.sum(axis=1) - 1
    neighbor = adjacency @ peak / np.maximum(adjacency.sum(axis=1), 1)
    features = np.column_stack([peak, energy, latency, degree, neighbor])
    probabilities = heuristic_probabilities(features)
    order = np.argsort(-probabilities)
    return SourceRanking(order, probabilities[order], features)

