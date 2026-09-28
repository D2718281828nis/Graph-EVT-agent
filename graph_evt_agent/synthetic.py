"""Reproducible synthetic episodes with a known onset and source channel.

These generators exist for tutorials, tests and sanity checks of the
labelling/training/evaluation workflow. A pulse starts at the source channel
and reaches every other channel after a delay proportional to its hop
distance on a known topology, with amplitude decaying per hop. Results on
synthetic data demonstrate behavior, not performance on real systems.
"""

from dataclasses import dataclass

import numpy as np

from .input import TimeSeriesInput


@dataclass(frozen=True)
class SyntheticEpisode:
    """A time series with ground truth. ``source`` is ``None`` for 1-D data."""

    values: np.ndarray
    event_times: tuple[int, ...]
    sources: tuple[int | None, ...]
    channel_names: tuple[str, ...]
    topology: np.ndarray

    def as_input(self) -> TimeSeriesInput:
        """Wrap values with channel names for the pipeline, labeller or evaluator."""
        return TimeSeriesInput(self.values, channel_names=self.channel_names)

    @property
    def event_time(self) -> int | None:
        return self.event_times[0] if self.event_times else None

    @property
    def source(self) -> int | None:
        return self.sources[0] if self.sources else None


def _topology(n_channels: int, kind: str) -> np.ndarray:
    adjacency = np.zeros((n_channels, n_channels), dtype=bool)
    for node in range(n_channels - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = True
    if kind == "ring" and n_channels > 2:
        adjacency[0, -1] = adjacency[-1, 0] = True
    elif kind not in {"ring", "chain"}:
        raise ValueError("topology must be 'ring' or 'chain'")
    return adjacency


def _hops(adjacency: np.ndarray, source: int) -> np.ndarray:
    distance = np.full(len(adjacency), len(adjacency), dtype=int)
    distance[source] = 0
    frontier = [source]
    while frontier:
        following = []
        for node in frontier:
            for neighbor in np.flatnonzero(adjacency[node]):
                if distance[neighbor] > distance[node] + 1:
                    distance[neighbor] = distance[node] + 1
                    following.append(int(neighbor))
        frontier = following
    return distance


def _background(rng: np.random.Generator, length: int, adjacency: np.ndarray,
                coupling: float, period: float) -> np.ndarray:
    channels = len(adjacency)
    degree = np.maximum(adjacency.sum(axis=1, keepdims=True), 1)
    mixing = np.eye(channels) + coupling * adjacency / degree
    noise = rng.normal(0, 1, (length, channels)) @ mixing.T
    phases = rng.uniform(0, 0.4, channels)
    time = np.arange(length)[:, None]
    return noise + 0.8 * np.sin(2 * np.pi * time / period + phases)


def _add_pulse(values: np.ndarray, onset: int, source: int, adjacency: np.ndarray,
               amplitude: float, delay_per_hop: int, decay: float, width: int,
               gains: np.ndarray) -> None:
    shape = np.linspace(1.0, 0.4, width)
    for node, hop in enumerate(_hops(adjacency, source)):
        start = onset + delay_per_hop * hop
        stop = min(len(values), start + width)
        if start < len(values):
            values[start:stop, node] += (amplitude * gains[node] * decay**hop
                                         * shape[: stop - start])


def _gains(rng: np.random.Generator, channels: int, jitter: float) -> np.ndarray:
    """Per-channel event responsiveness; jitter > 0 lets a neighbor outshine the source."""
    if jitter < 0:
        raise ValueError("gain_jitter must be non-negative")
    return np.exp(rng.normal(0, jitter, channels)) if jitter else np.ones(channels)


def make_propagation_episode(n_channels: int = 6, source: int = 0, *,
                             baseline_size: int = 400, length: int = 600,
                             event_time: int | None = None, topology: str = "ring",
                             coupling: float = 0.6, amplitude: float = 10.0,
                             delay_per_hop: int = 3, decay: float = 0.65,
                             width: int = 25, period: float = 16.0,
                             gain_jitter: float = 0.0,
                             seed: int | None = None) -> SyntheticEpisode:
    """One multichannel episode whose pulse spreads from ``source``.

    ``gain_jitter`` draws a log-normal event gain per channel, so the loudest
    channel is not always the source – the case where learning can help.
    """
    if not 0 <= source < n_channels or n_channels < 2:
        raise ValueError("need at least two channels and a valid source")
    onset = baseline_size + 20 if event_time is None else event_time
    if not baseline_size <= onset < length:
        raise ValueError("event_time must lie after the baseline")
    rng = np.random.default_rng(seed)
    adjacency = _topology(n_channels, topology)
    values = _background(rng, length, adjacency, coupling, period)
    _add_pulse(values, onset, source, adjacency, amplitude, delay_per_hop, decay, width,
               _gains(rng, n_channels, gain_jitter))
    names = tuple(f"ch{node}" for node in range(n_channels))
    return SyntheticEpisode(values, (onset,), (source,), names, adjacency)


def make_recording(n_events: int = 5, n_channels: int = 6, *, baseline_size: int = 400,
                   gap: int = 250, topology: str = "ring", amplitude: float = 10.0,
                   seed: int | None = None, **kwargs: float) -> SyntheticEpisode:
    """A long recording holding ``n_events`` well-separated propagation events.

    Extra keyword arguments (``coupling``, ``period``, ``delay_per_hop``,
    ``decay``, ``width``, ``gain_jitter``) match :func:`make_propagation_episode`.
    """
    if n_events < 1 or gap < 60:
        raise ValueError("n_events must be positive and gap at least 60 samples")
    rng = np.random.default_rng(seed)
    adjacency = _topology(n_channels, topology)
    length = baseline_size + gap * n_events + gap // 2
    values = _background(rng, length, adjacency, float(kwargs.get("coupling", 0.6)),
                         float(kwargs.get("period", 16.0)))
    onsets = tuple(baseline_size + gap // 2 + gap * index for index in range(n_events))
    sources = tuple(int(node) for node in rng.integers(0, n_channels, n_events))
    for onset, source in zip(onsets, sources):
        _add_pulse(values, onset, source, adjacency, amplitude,
                   int(kwargs.get("delay_per_hop", 3)), float(kwargs.get("decay", 0.65)),
                   int(kwargs.get("width", 25)),
                   _gains(rng, n_channels, float(kwargs.get("gain_jitter", 0.0))))
    names = tuple(f"ch{node}" for node in range(n_channels))
    return SyntheticEpisode(values, onsets, sources, names, adjacency)


def make_univariate_episode(*, baseline_size: int = 400, length: int = 600,
                            event_time: int | None = None, amplitude: float = 8.0,
                            width: int = 25, seed: int | None = None) -> SyntheticEpisode:
    """A single-channel AR(1) series with one extreme pulse, or none if ``amplitude=0``."""
    rng = np.random.default_rng(seed)
    values = np.empty(length)
    values[0] = rng.normal()
    for index in range(1, length):
        values[index] = 0.5 * values[index - 1] + rng.normal()
    onset = baseline_size + 20 if event_time is None else event_time
    if amplitude > 0:
        stop = min(length, onset + width)
        values[onset:stop] += amplitude * np.linspace(1.0, 0.4, width)[: stop - onset]
    events = (onset,) if amplitude > 0 else ()
    return SyntheticEpisode(values[:, None], events, (None,) * len(events), ("value",),
                            np.ones((1, 1), dtype=bool))
