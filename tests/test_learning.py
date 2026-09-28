import numpy as np

from graph_evt_agent import GraphEpisode, GraphLearningConfig, GraphProcessModel


def _chain(nodes: int) -> np.ndarray:
    adjacency = np.eye(nodes, dtype=bool)
    for node in range(nodes - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = True
    return adjacency


def test_gnn_and_gat_learn_source_and_return_process_graph():
    rng = np.random.default_rng(2)
    adjacency = _chain(4)
    episodes = []
    for source in range(4):
        for _ in range(3):
            features = rng.normal(0, 0.1, (4, 5))
            features[source, :3] += 2
            episodes.append(GraphEpisode(features, adjacency, source))
    model = GraphProcessModel(
        GraphLearningConfig(epochs=120, learning_rate=0.03, random_state=2)
    ).fit(episodes)
    features = rng.normal(0, 0.1, (4, 5))
    features[2, :3] += 2
    prediction = model.predict(features, adjacency)
    assert prediction.gnn_source == 2
    assert prediction.gat_source == 2
    assert np.isclose(prediction.gnn_probabilities.sum(), 1)
    assert np.isclose(prediction.gat_probabilities.sum(), 1)
    assert np.allclose(prediction.attention.sum(axis=1), 1)
    assert np.all(prediction.attention[~adjacency] == 0)


def test_graph_learning_requires_fit_before_inference():
    model = GraphProcessModel()
    with np.testing.assert_raises_regex(RuntimeError, "fit must be called"):
        model.predict(np.ones((2, 5)), np.eye(2, dtype=bool))


def test_graph_process_model_round_trips_through_npz(tmp_path):
    rng = np.random.default_rng(0)
    adjacency = _chain(3)
    episodes = [GraphEpisode(rng.normal(size=(3, 5)), adjacency, index % 3) for index in range(6)]
    model = GraphProcessModel(GraphLearningConfig(epochs=5)).fit(episodes)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = GraphProcessModel.load(path)
    features = rng.normal(size=(3, 5))
    np.testing.assert_allclose(model.predict(features, adjacency).gat_probabilities,
                               restored.predict(features, adjacency).gat_probabilities)
    assert restored.config == model.config
