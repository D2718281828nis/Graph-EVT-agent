"""Turn recordings into labelled graph episodes using EVT and the baseline graph.

For each recording the labeller freezes the robust EVT threshold and the
channel graph on the baseline, finds every declustered persistent alarm, and
builds the node features used by :class:`GraphProcessModel`. Every episode
gets a label origin:

* ``"manual"`` – an expert annotation matched the detected onset;
* ``"pseudo"`` – no annotation, and the transparent ranker was confident
  (top weight and top-vs-second margin above thresholds);
* ``"abstained"`` – no annotation and an ambiguous ranking; not used for
  training.

Pseudo-labels copy the transparent ranker. A model trained only on them learns
to imitate that ranker through the graph; it cannot become more accurate than
its teacher by itself. Use them to bootstrap or pre-train, and always measure
quality against manual labels (see :mod:`graph_evt_agent.evaluation`).
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evt import detect, find_events
from .graph import build_graph
from .io import load_timeseries
from .learning import GraphEpisode
from .localization import rank_sources
from .pipeline import GraphEVTPipeline
from .progress import Progress, ProgressCallback

ORIGINS = ("manual", "pseudo", "abstained")


@dataclass(frozen=True)
class LabelingConfig:
    """Acceptance rules for pseudo-labels and matching of annotations."""

    min_confidence: float = 0.5
    min_margin: float = 0.2
    refractory: int = 50
    window: int = 12
    match_tolerance: int = 25
    max_events_per_series: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.min_margin <= 1:
            raise ValueError("min_confidence and min_margin must be in [0, 1]")
        if self.refractory < 0 or self.window < 1 or self.match_tolerance < 0:
            raise ValueError("invalid refractory, window or match_tolerance")


@dataclass(frozen=True)
class LabelledEpisode:
    """One detected event with its graph, node features and (maybe) a source."""

    series_id: str
    event_time: int
    source: int | None
    origin: str
    confidence: float
    margin: float
    features: np.ndarray
    adjacency: np.ndarray
    ranking_probabilities: np.ndarray
    channel_names: tuple[str, ...]
    graph_method: str

    @property
    def labelled(self) -> bool:
        return self.source is not None

    @property
    def source_name(self) -> str | None:
        return None if self.source is None else self.channel_names[self.source]

    def to_graph_episode(self) -> GraphEpisode:
        if self.source is None:
            raise ValueError("an abstained episode has no source label")
        return GraphEpisode(self.features, self.adjacency, self.source)


@dataclass(frozen=True)
class EpisodeLabels:
    """All labelled episodes plus recordings that could not be labelled."""

    episodes: tuple[LabelledEpisode, ...]
    skipped: tuple[tuple[str, str], ...] = ()

    def select(self, origins: Iterable[str] = ("manual", "pseudo")) -> list[LabelledEpisode]:
        wanted = set(origins)
        return [item for item in self.episodes if item.origin in wanted and item.labelled]

    def training_episodes(self, origins: Iterable[str] = ("manual", "pseudo")
                          ) -> list[GraphEpisode]:
        return [item.to_graph_episode() for item in self.select(origins)]

    def groups(self, origins: Iterable[str] = ("manual", "pseudo")) -> list[str]:
        """Series identifiers aligned with :meth:`training_episodes`."""
        return [item.series_id for item in self.select(origins)]

    def summary(self) -> dict[str, int]:
        counts = {origin: 0 for origin in ORIGINS}
        for item in self.episodes:
            counts[item.origin] += 1
        return {"episodes": len(self.episodes), **counts, "skipped_series": len(self.skipped)}

    def to_records(self) -> list[dict[str, Any]]:
        """Flat rows for ``pandas.DataFrame`` or CSV export."""
        return [{
            "series_id": item.series_id, "event_time": item.event_time,
            "source": item.source, "source_name": item.source_name,
            "origin": item.origin, "confidence": item.confidence, "margin": item.margin,
            "graph_method": item.graph_method,
        } for item in self.episodes]


class EpisodeLabeler:
    """Detect all events per recording and label their source channel.

    The pipeline supplies the EVT/graph/input configuration; its optional
    process model is not used, so labelling never depends on a learned model.
    """

    def __init__(self, pipeline: GraphEVTPipeline, config: LabelingConfig | None = None,
                 verbose: bool | int = False,
                 progress_callback: ProgressCallback | None = None):
        self.pipeline = pipeline
        self.config = config or LabelingConfig()
        self.verbose = verbose
        self.progress_callback = progress_callback

    @staticmethod
    def _source_index(source: int | str, names: tuple[str, ...]) -> int:
        if isinstance(source, str):
            if source not in names:
                raise ValueError(f"unknown source channel {source!r}")
            return names.index(source)
        if not 0 <= int(source) < len(names):
            raise ValueError(f"source index {source} is out of range")
        return int(source)

    def label_series(self, values: Any, series_id: str = "series_0",
                     annotations: Sequence[tuple[int, int | str]] = ()) -> EpisodeLabels:
        """Label every event in one recording.

        ``annotations`` are expert ``(event_time, source)`` pairs; a detected
        onset within ``match_tolerance`` samples inherits the manual source.
        """
        if isinstance(values, str) or hasattr(values, "__fspath__"):
            values = load_timeseries(values)
        prepared = self.pipeline.inspector.inspect(values)
        data, names = prepared.values, prepared.profile.channel_names
        if data.shape[1] == 1:
            return EpisodeLabels((), ((series_id, "univariate: source is not identifiable"),))
        evt_config = self.pipeline.evt_config
        detection = detect(data, evt_config)
        onsets = find_events(detection, evt_config, self.config.refractory)
        if self.config.max_events_per_series is not None:
            onsets = onsets[: self.config.max_events_per_series]
        if not onsets:
            return EpisodeLabels((), ((series_id, "no EVT event detected"),))
        graph = build_graph(data[: evt_config.baseline_size], self.pipeline.graph_config)
        manual = [(int(time), self._source_index(source, names)) for time, source in annotations]
        episodes = []
        for onset in onsets:
            ranking = rank_sources(data, onset, graph.adjacency, detection.location,
                                   detection.scale, self.config.window)
            weights = np.empty(len(names))
            weights[ranking.node_ids] = ranking.probabilities
            ordered = np.sort(weights)[::-1]
            confidence = float(ordered[0])
            margin = float(ordered[0] - ordered[1])
            matched = [source for time, source in manual
                       if abs(time - onset) <= self.config.match_tolerance]
            if matched:
                source, origin = matched[0], "manual"
            elif confidence >= self.config.min_confidence and margin >= self.config.min_margin:
                source, origin = ranking.source, "pseudo"
            else:
                source, origin = None, "abstained"
            episodes.append(LabelledEpisode(
                series_id, onset, source, origin, confidence, margin, ranking.features,
                graph.adjacency, weights, names, graph.method,
            ))
        return EpisodeLabels(tuple(episodes))

    def label(self, recordings: Mapping[str, Any] | Sequence[Any],
              annotations: Mapping[str, Sequence[tuple[int, int | str]]] | None = None
              ) -> EpisodeLabels:
        """Label many recordings; a mapping's keys become ``series_id`` values.

        A sequence is treated as a list of recordings (use
        :meth:`label_series` for a single recording).
        """
        items = (list(recordings.items()) if isinstance(recordings, Mapping)
                 else [(f"series_{index}", value) for index, value in enumerate(recordings)])
        annotations = annotations or {}
        episodes: list[LabelledEpisode] = []
        skipped: list[tuple[str, str]] = []
        with Progress(len(items), "Labelling episodes", self.verbose,
                      self.progress_callback) as progress:
            for series_id, values in items:
                result = self.label_series(values, str(series_id),
                                           annotations.get(str(series_id), ()))
                episodes.extend(result.episodes)
                skipped.extend(result.skipped)
                labelled = sum(item.labelled for item in episodes)
                progress.advance(f"{len(episodes)} events, {labelled} labelled")
        return EpisodeLabels(tuple(episodes), tuple(skipped))
