"""Train :class:`GraphProcessModel` from labelled episodes and measure quality.

The trainer holds out whole recordings (groups) for validation, fits the GNN
and GAT, and compares both with the untrained transparent ranker on the same
episodes. The resulting :class:`TrainingReport` feeds the plots in
:mod:`graph_evt_agent.visualization`.
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence
import warnings

import numpy as np

from .labeling import EpisodeLabels
from .learning import GraphEpisode, GraphLearningConfig, GraphProcessModel
from .localization import heuristic_probabilities
from .progress import ProgressCallback

METHODS = ("heuristic", "gnn", "gat")


@dataclass(frozen=True)
class RankingMetrics:
    """Source-localization quality over independent episodes."""

    n: int
    top1: float
    topk: float
    mrr: float
    nll: float
    k: int = 3

    def as_dict(self) -> dict[str, float]:
        return {"n": self.n, "top1": self.top1, f"top{self.k}": self.topk,
                "mrr": self.mrr, "nll": self.nll}


def ranking_metrics(probabilities: Sequence[np.ndarray], sources: Sequence[int],
                    k: int = 3) -> RankingMetrics:
    """Top-1, Top-k, mean reciprocal rank and negative log-likelihood."""
    if not len(sources):
        return RankingMetrics(0, np.nan, np.nan, np.nan, np.nan, k)
    ranks, losses = [], []
    for weights, source in zip(probabilities, sources):
        weights = np.asarray(weights, dtype=float)
        ranks.append(1 + int(np.sum(weights > weights[source])))
        losses.append(-np.log(max(float(weights[source]), 1e-12)))
    ranks_ = np.asarray(ranks)
    return RankingMetrics(len(ranks_), float(np.mean(ranks_ == 1)), float(np.mean(ranks_ <= k)),
                          float(np.mean(1 / ranks_)), float(np.mean(losses)), k)


def split_episodes(episodes: Sequence[GraphEpisode], groups: Sequence[str] | None = None,
                   validation_fraction: float = 0.25, random_state: int = 0):
    """Split by group so no recording contributes to both train and validation.

    With fewer than two groups the split falls back to individual episodes and
    warns, because events of one recording share a baseline graph.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    count = len(episodes)
    labels = list(groups) if groups is not None else [str(i) for i in range(count)]
    if len(labels) != count:
        raise ValueError("groups must align with episodes")
    unique = sorted(set(labels))
    unit = "group"
    if len(unique) < 2 and validation_fraction > 0:
        warnings.warn("only one group available; splitting individual episodes instead",
                      stacklevel=2)
        labels, unique, unit = [str(i) for i in range(count)], [str(i) for i in range(count)], "episode"
    rng = np.random.default_rng(random_state)
    held = int(round(validation_fraction * len(unique)))
    if validation_fraction > 0:
        held = min(max(held, 1), len(unique) - 1)
    validation_groups = set(rng.permutation(unique)[:held].tolist())
    train = [i for i, label in enumerate(labels) if label not in validation_groups]
    validation = [i for i, label in enumerate(labels) if label in validation_groups]
    return train, validation, unit


@dataclass(frozen=True)
class TrainingReport:
    """Fitted model, learning curves and train/validation comparison."""

    model: GraphProcessModel
    history: dict[str, list[float]]
    train_metrics: dict[str, RankingMetrics]
    validation_metrics: dict[str, RankingMetrics]
    validation_predictions: dict[str, list[int]] = field(default_factory=dict)
    validation_sources: list[int] = field(default_factory=list)
    split_unit: str = "group"
    n_train: int = 0
    n_validation: int = 0

    def summary_table(self) -> list[dict[str, float | str]]:
        """Rows ``{split, method, n, top1, top3, mrr, nll}`` for display."""
        rows = []
        for split, metrics in (("train", self.train_metrics),
                               ("validation", self.validation_metrics)):
            for method in METHODS:
                if method in metrics:
                    rows.append({"split": split, "method": method, **metrics[method].as_dict()})
        return rows

    @property
    def best_epoch(self) -> int | None:
        """Epoch with the lowest mean GNN/GAT validation loss (early-stopping hint)."""
        losses = (np.asarray(self.history.get("gnn_val_loss", []), dtype=float)
                  + np.asarray(self.history.get("gat_val_loss", []), dtype=float)) / 2
        if not len(losses) or not np.isfinite(losses).any():
            return None
        return int(self.history["epoch"][int(np.nanargmin(losses))])

    def confusion(self, method: str = "gat") -> np.ndarray | None:
        """Validation ``[true, predicted]`` counts when all graphs share a node count."""
        predicted = self.validation_predictions.get(method)
        if not predicted:
            return None
        nodes = 1 + max(max(predicted), max(self.validation_sources))
        matrix = np.zeros((nodes, nodes), dtype=int)
        for truth, guess in zip(self.validation_sources, predicted):
            matrix[truth, guess] += 1
        return matrix


class ProcessModelTrainer:
    """Split labelled episodes, train GNN/GAT, and score against the heuristic."""

    def __init__(self, config: GraphLearningConfig | None = None,
                 validation_fraction: float = 0.25, k: int = 3, random_state: int = 0,
                 verbose: bool | int = False,
                 progress_callback: ProgressCallback | None = None):
        self.config = config or GraphLearningConfig()
        self.validation_fraction = validation_fraction
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.progress_callback = progress_callback

    def _score(self, model: GraphProcessModel, episodes: list[GraphEpisode]):
        weights: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        for item in episodes:
            prediction = model.predict(item.features, item.adjacency)
            weights["heuristic"].append(heuristic_probabilities(item.features))
            weights["gnn"].append(prediction.gnn_probabilities)
            weights["gat"].append(prediction.gat_probabilities)
        sources = [item.source for item in episodes]
        metrics = {method: ranking_metrics(values, sources, self.k)
                   for method, values in weights.items()}
        predicted = {method: [int(np.argmax(v)) for v in values]
                     for method, values in weights.items()}
        return metrics, predicted, sources

    def train(self, episodes: EpisodeLabels | Sequence[GraphEpisode],
              groups: Sequence[str] | None = None,
              validation: Sequence[GraphEpisode] | None = None,
              origins: Sequence[str] = ("manual", "pseudo")) -> TrainingReport:
        """Train on labelled episodes.

        ``episodes`` may be :class:`EpisodeLabels` (groups = recording ids) or
        plain :class:`GraphEpisode` objects. Pass ``validation`` explicitly,
        e.g. manually labelled recordings, to skip the automatic split.
        """
        if isinstance(episodes, EpisodeLabels):
            groups = episodes.groups(origins)
            episodes = episodes.training_episodes(origins)
        episodes = list(episodes)
        if not episodes:
            raise ValueError("no labelled episodes to train on")
        unit = "explicit"
        if validation is None:
            train_index, validation_index, unit = split_episodes(
                episodes, groups, self.validation_fraction, self.random_state)
            train = [episodes[i] for i in train_index]
            held_out = [episodes[i] for i in validation_index]
        else:
            train, held_out = episodes, list(validation)
        model = GraphProcessModel(self.config).fit(train, held_out, self.verbose,
                                                   self.progress_callback)
        train_metrics, _, _ = self._score(model, train)
        validation_metrics, predicted, sources = self._score(model, held_out)
        return TrainingReport(model, model.history_, train_metrics, validation_metrics,
                              predicted, sources, unit, len(train), len(held_out))

    def train_1d(self, recordings: Sequence[np.ndarray] | Mapping[str, np.ndarray],
                 event_times: Sequence[int] | Mapping[str, int], *, temporal_config=None,
                 validation: Sequence[tuple[np.ndarray, int]] | None = None
                 ) -> TrainingReport:
        """Train GNN/GAT on temporal graphs from labelled 1-D recordings.

        Graph nodes are time windows and the target node contains the event
        onset. Mapping keys become split groups. Explicit validation entries
        are ``(series, event_time)`` pairs.
        """
        from .temporal import TemporalGraphConfig, univariate_graph_episode

        config = temporal_config or TemporalGraphConfig()
        if isinstance(recordings, Mapping):
            keys = list(recordings)
            if not isinstance(event_times, Mapping):
                raise ValueError("event_times must be a mapping when recordings is a mapping")
            missing = [key for key in keys if key not in event_times]
            if missing:
                raise ValueError(f"event_times is missing recording {missing[0]!r}")
            pairs = [(recordings[key], event_times[key]) for key in keys]
            groups = [str(key) for key in keys]
        else:
            if isinstance(event_times, Mapping):
                raise ValueError("event_times must be a sequence when recordings is a sequence")
            values, times = list(recordings), list(event_times)
            if len(values) != len(times):
                raise ValueError("event_times must align with recordings")
            pairs = list(zip(values, times))
            groups = None
        episodes = [univariate_graph_episode(values, time, config) for values, time in pairs]
        held_out = (None if validation is None else
                    [univariate_graph_episode(values, time, config)
                     for values, time in validation])
        return self.train(episodes, groups=groups, validation=held_out)
