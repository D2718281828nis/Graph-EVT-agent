"""Small supervised GNN/GAT source models without a deep-learning dependency."""

from dataclasses import dataclass

import numpy as np

from .models import GraphProcessPrediction


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

    def fit(self, episodes: list[GraphEpisode]) -> "GraphProcessModel":
        if not episodes:
            raise ValueError("at least one labelled episode is required")
        checked = [_validate(item.features, item.adjacency, item.source) for item in episodes]
        feature_count = checked[0][0].shape[1]
        if any(x.shape[1] != feature_count for x, _ in checked):
            raise ValueError("all episodes must use the same feature schema")
        stacked = np.vstack([x for x, _ in checked])
        self._mean = stacked.mean(axis=0)
        self._scale = np.maximum(stacked.std(axis=0), 1e-8)
        data = [(self._standardize(x), a, item.source)
                for (x, a), item in zip(checked, episodes)]
        rng = np.random.default_rng(self.config.random_state)
        self._gnn = self._init(rng, feature_count * 3, self.config.hidden_dim)
        # GAT dense parameters plus two feature-space attention vectors.
        self._gat = self._init(rng, feature_count * 2, self.config.hidden_dim)
        self._gat.append(rng.normal(0, 0.1, (feature_count, 2)))
        for parameters, is_gat in ((self._gnn, False), (self._gat, True)):
            moments = [np.zeros_like(value) for value in parameters]
            variances = [np.zeros_like(value) for value in parameters]
            for step in range(1, self.config.epochs + 1):
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
                self._adam(parameters, gradients, moments, variances, step,
                           self.config.learning_rate)
        return self

    def predict(self, features: np.ndarray, adjacency: np.ndarray) -> GraphProcessPrediction:
        if self._mean is None or self._gnn is None or self._gat is None:
            raise RuntimeError("fit must be called before predict")
        x, graph = _validate(features, adjacency)
        if x.shape[1] != len(self._mean):
            raise ValueError("feature schema differs from training")
        x = self._standardize(x)
        _, gnn_logits = self._dense_forward(self._gnn_input(x, graph), self._gnn)
        attention, _ = self._attention(x, graph, self._gat[4])
        _, gat_logits = self._dense_forward(np.column_stack([x, attention @ x]), self._gat)
        return GraphProcessPrediction(_softmax(gnn_logits), _softmax(gat_logits), attention)
