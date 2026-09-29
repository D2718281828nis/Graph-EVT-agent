import numpy as np

from graph_evt_agent import (
    GraphEpisode, GraphLearningConfig, GraphProcessModel, ProcessModelTrainer,
    TemporalGraphConfig, univariate_graph_episode, univariate_temporal_graph,
)


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


def test_univariate_temporal_graph_and_training_method():
    rng = np.random.default_rng(9)
    recordings, onsets = {}, {}
    for index in range(6):
        onset = 120 + index * 3
        values = rng.normal(0, 0.15, 240)
        values[onset:onset + 8] += 4
        recordings[f"rec{index}"] = values
        onsets[f"rec{index}"] = onset
    temporal = TemporalGraphConfig(window_size=16, stride=8, baseline_size=80)
    features, adjacency, starts = univariate_temporal_graph(recordings["rec0"], temporal)
    episode = univariate_graph_episode(recordings["rec0"], onsets["rec0"], temporal)
    assert features.shape == (len(starts), 5)
    assert adjacency.shape == (len(starts), len(starts))
    assert np.array_equal(adjacency, adjacency.T)
    assert starts[episode.source] <= onsets["rec0"] < starts[episode.source] + 16

    report = ProcessModelTrainer(
        GraphLearningConfig(epochs=20, learning_rate=0.03, random_state=3),
        validation_fraction=0.34,
    ).train_1d(recordings, onsets, temporal_config=temporal)
    assert report.split_unit == "group"
    assert report.n_train + report.n_validation == len(recordings)
    assert set(report.validation_metrics) == {"heuristic", "gnn", "gat"}
    prediction = report.model.predict(features, adjacency)
    assert np.isclose(prediction.gnn_probabilities.sum(), 1)


def test_univariate_temporal_graph_rejects_invalid_inputs():
    config = TemporalGraphConfig(baseline_size=4)
    with np.testing.assert_raises_regex(ValueError, "finite 1-D"):
        univariate_temporal_graph([1, np.nan, 2, 3, 4], config)
    with np.testing.assert_raises_regex(ValueError, "event_time"):
        univariate_graph_episode(np.arange(10.0), 10, config)


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
