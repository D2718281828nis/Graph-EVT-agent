"""Small supervised GNN/GAT source models without a deep-learning dependency."""

from dataclasses import asdict, dataclass
import json
from os import PathLike

import numpy as np

from .models import GraphProcessPrediction
from .progress import Progress, ProgressCallback


@dataclass(frozen=True)
class GraphEpisode:
    """One labelled event graph used to train source-node classifiers."""

    features: np.ndarray
    adjacency: np.ndarray
    source: int


@dataclass(frozen=True)
class GraphLearningConfig:
    hidden_dim: int = 8
    epochs: int = 300
    learning_rate: float = 0.02
    l2: float = 1e-4
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("invalid graph-learning configuration")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")


def _validate(features: np.ndarray, adjacency: np.ndarray, source: int | None = None):
    x = np.asarray(features, dtype=float)
    a = np.asarray(adjacency, dtype=bool)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite [node, feature] array")
    if a.shape != (len(x), len(x)) or not np.array_equal(a, a.T):
        raise ValueError("adjacency must be a symmetric [node, node] array")
    if source is not None and not 0 <= source < len(x):
        raise ValueError("source must identify a node")
    return x, np.logical_or(a, np.eye(len(a), dtype=bool))


def _softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=axis, keepdims=True)


def _normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    graph = np.logical_or(adjacency, np.eye(len(adjacency), dtype=bool)).astype(float)
    return graph / graph.sum(axis=1, keepdims=True)


class GraphProcessModel:
    """Train compact nonlinear GNN and GAT node classifiers.

    The GNN consumes zero-, one-, and two-hop node features. The GAT learns
    episode-specific attention over accepted edges. Both models require source
    labels; inference alone never fabricates training targets.
    """

    def __init__(self, config: GraphLearningConfig | None = None):
        self.config = config or GraphLearningConfig()
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._gnn: list[np.ndarray] | None = None
        self._gat: list[np.ndarray] | None = None
        self.history_: dict[str, list[float]] = {}

    @staticmethod
    def _init(rng: np.random.Generator, input_dim: int, hidden: int):
        return [rng.normal(0, 0.15, (input_dim, hidden)), np.zeros(hidden),
                rng.normal(0, 0.15, hidden), np.zeros(1)]

    @staticmethod
    def _adam(parameters, gradients, moments, variances, step, rate):
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            moments[index] = 0.9 * moments[index] + 0.1 * gradient
            variances[index] = 0.999 * variances[index] + 0.001 * gradient**2
            estimate = moments[index] / (1 - 0.9**step)
            variance = variances[index] / (1 - 0.999**step)
            parameter -= rate * estimate / (np.sqrt(variance) + 1e-8)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return (features - self._mean) / self._scale

    @staticmethod
    def _gnn_input(x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        transition = _normalized_adjacency(adjacency)
        one_hop = transition @ x
        return np.column_stack([x, one_hop, transition @ one_hop])

    @staticmethod
    def _dense_forward(inputs: np.ndarray, parameters: list[np.ndarray]):
        weight, bias, output, output_bias = parameters[:4]
        hidden = np.tanh(inputs @ weight + bias)
        logits = hidden @ output + output_bias[0]
        return hidden, logits

    def _dense_grad(self, inputs, hidden, logits, source, parameters):
        probabilities = _softmax(logits)
        delta = probabilities
        delta[source] -= 1
        weight, _, output, _ = parameters[:4]
        del weight
        hidden_delta = delta[:, None] * output[None, :] * (1 - hidden**2)
        regularization = self.config.l2
        return [inputs.T @ hidden_delta + regularization * parameters[0],
                hidden_delta.sum(axis=0), hidden.T @ delta + regularization * output,
                np.array([delta.sum()])]

    @staticmethod
    def _attention(x: np.ndarray, adjacency: np.ndarray, vectors: np.ndarray):
        source_score = x @ vectors[:, 0]
        target_score = x @ vectors[:, 1]
        raw = source_score[:, None] + target_score[None, :]
        activated = np.where(raw >= 0, raw, 0.2 * raw)
        masked = np.where(adjacency, activated, -1e9)
        return _softmax(masked, axis=1), raw

    def _logits(self, parameters: list[np.ndarray], x: np.ndarray,
                adjacency: np.ndarray, is_gat: bool) -> np.ndarray:
        if not is_gat:
            return self._dense_forward(self._gnn_input(x, adjacency), parameters)[1]
        attention, _ = self._attention(x, adjacency, parameters[4])
        return self._dense_forward(np.column_stack([x, attention @ x]), parameters)[1]

    def _score(self, parameters, data, is_gat: bool) -> tuple[float, float]:
        """Mean cross-entropy and top-1 accuracy over standardized episodes."""
        losses, hits = [], []
        for x, adjacency, source in data:
            probabilities = _softmax(self._logits(parameters, x, adjacency, is_gat))
            losses.append(-np.log(max(probabilities[source], 1e-12)))
            hits.append(float(np.argmax(probabilities) == source))
        return float(np.mean(losses)), float(np.mean(hits))

    def _prepare(self, episodes: list[GraphEpisode], feature_count: int):
        checked = [_validate(item.features, item.adjacency, item.source) for item in episodes]
        if any(x.shape[1] != feature_count for x, _ in checked):
            raise ValueError("all episodes must use the same feature schema")
        return [(self._standardize(x), a, item.source) for (x, a), item in zip(checked, episodes)]

    def fit(self, episodes: list[GraphEpisode],
            validation: list[GraphEpisode] | None = None,
            verbose: bool | int = False,
            progress_callback: ProgressCallback | None = None) -> "GraphProcessModel":
        """Train both models; ``history_`` records loss/accuracy for every epoch.

        ``validation`` episodes are only scored, never used for gradients or
        standardization.
        """
        if not episodes:
            raise ValueError("at least one labelled episode is required")
        checked = [_validate(item.features, item.adjacency, item.source) for item in episodes]
        feature_count = checked[0][0].shape[1]
        stacked = np.vstack([x for x, _ in checked])
        self._mean = stacked.mean(axis=0)
        self._scale = np.maximum(stacked.std(axis=0), 1e-8)
        data = self._prepare(episodes, feature_count)
        held_out = self._prepare(list(validation), feature_count) if validation else []
        rng = np.random.default_rng(self.config.random_state)
        self._gnn = self._init(rng, feature_count * 3, self.config.hidden_dim)
        # GAT dense parameters plus two feature-space attention vectors.
        self._gat = self._init(rng, feature_count * 2, self.config.hidden_dim)
        self._gat.append(rng.normal(0, 0.1, (feature_count, 2)))
        models = [(self._gnn, False, "gnn"), (self._gat, True, "gat")]
        optimizer = {name: ([np.zeros_like(v) for v in parameters],
                            [np.zeros_like(v) for v in parameters])
                     for parameters, _, name in models}
        keys = ["epoch"] + [f"{name}_{split}_{metric}" for name in ("gnn", "gat")
                            for split in ("train", "val") for metric in ("loss", "accuracy")]
        self.history_ = {key: [] for key in keys}
        with Progress(self.config.epochs, "Training GNN/GAT", verbose,
                      progress_callback) as progress:
            for step in range(1, self.config.epochs + 1):
                for parameters, is_gat, name in models:
                    self._epoch(parameters, is_gat, data, feature_count,
                                *optimizer[name], step)
                self.history_["epoch"].append(float(step))
                for parameters, is_gat, name in models:
                    for split, episodes_ in (("train", data), ("val", held_out)):
                        loss, accuracy = (self._score(parameters, episodes_, is_gat)
                                          if episodes_ else (np.nan, np.nan))
                        self.history_[f"{name}_{split}_loss"].append(loss)
                        self.history_[f"{name}_{split}_accuracy"].append(accuracy)
                message = (f"GNN loss {self.history_['gnn_train_loss'][-1]:.3f} "
                           f"GAT loss {self.history_['gat_train_loss'][-1]:.3f}")
                if held_out:
                    message += (f" | val acc {self.history_['gnn_val_accuracy'][-1]:.2f}"
                                f"/{self.history_['gat_val_accuracy'][-1]:.2f}")
                progress.advance(message)
        return self

    def _epoch(self, parameters, is_gat, data, feature_count, moments, variances, step):
        gradients = [np.zeros_like(value) for value in parameters]
        for x, adjacency, source in data:
            if not is_gat:
                inputs = self._gnn_input(x, adjacency)
                hidden, logits = self._dense_forward(inputs, parameters)
                local = self._dense_grad(inputs, hidden, logits, source, parameters)
            else:
                attention, raw = self._attention(x, adjacency, parameters[4])
                neighbor = attention @ x
                inputs = np.column_stack([x, neighbor])
                hidden, logits = self._dense_forward(inputs, parameters)
                local = self._dense_grad(inputs, hidden, logits, source, parameters)
                delta = _softmax(logits)
                delta[source] -= 1
                hidden_delta = delta[:, None] * parameters[2][None, :] * (1 - hidden**2)
                input_delta = hidden_delta @ parameters[0].T
                neighbor_delta = input_delta[:, feature_count:]
                attention_delta = neighbor_delta @ x.T
                raw_delta = attention * (
                    attention_delta - (attention_delta * attention).sum(axis=1, keepdims=True)
                )
                raw_delta *= np.where(raw >= 0, 1.0, 0.2) * adjacency
                local.append(np.column_stack([
                    x.T @ raw_delta.sum(axis=1),
                    x.T @ raw_delta.sum(axis=0),
                ]) + self.config.l2 * parameters[4])
            for total, value in zip(gradients, local):
                total += value / len(data)
        self._adam(parameters, gradients, moments, variances, step, self.config.learning_rate)

    def save(self, path: str | PathLike) -> None:
        """Store weights, standardization and config in one ``.npz`` file."""
        if self._mean is None or self._gnn is None or self._gat is None:
            raise RuntimeError("fit must be called before save")
        arrays = {"mean": self._mean, "scale": self._scale,
                  "config": np.array(json.dumps(asdict(self.config)))}
        arrays.update({f"gnn_{i}": value for i, value in enumerate(self._gnn)})
        arrays.update({f"gat_{i}": value for i, value in enumerate(self._gat)})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | PathLike) -> "GraphProcessModel":
        """Restore a model written by :meth:`save` (no pickle is used)."""
        with np.load(path, allow_pickle=False) as stored:
            model = cls(GraphLearningConfig(**json.loads(str(stored["config"]))))
            model._mean, model._scale = stored["mean"], stored["scale"]
            model._gnn = [stored[f"gnn_{i}"] for i in range(4)]
            model._gat = [stored[f"gat_{i}"] for i in range(5)]
        return model

    def predict(self, features: np.ndarray, adjacency: np.ndarray) -> GraphProcessPrediction:
        if self._mean is None or self._gnn is None or self._gat is None:
            raise RuntimeError("fit must be called before predict")
        x, graph = _validate(features, adjacency)
        if x.shape[1] != len(self._mean):
            raise ValueError("feature schema differs from training")
        x = self._standardize(x)
        attention, _ = self._attention(x, graph, self._gat[4])
        return GraphProcessPrediction(_softmax(self._logits(self._gnn, x, graph, False)),
                                      _softmax(self._logits(self._gat, x, graph, True)),
                                      attention)
