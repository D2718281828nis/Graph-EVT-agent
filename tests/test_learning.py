import numpy as np

import pytest

from graph_evt_agent import (
    GraphEpisode, GraphLearningConfig, GraphProcessModel, ProcessModelTrainer,
    TemporalGraphConfig, build_temporal_graph, prune_weighted_graph,
    univariate_graph_episode, univariate_temporal_graph,
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


def _spike_series(seed: int = 0, onset: int = 125) -> np.ndarray:
    values = np.random.default_rng(seed).normal(0, 0.15, 240)
    values[onset:onset + 8] += 4
    return values


@pytest.mark.parametrize("edge_method", ["dfa", "wavelet", "ar"])
def test_weighted_temporal_graph_is_fully_connected_before_pruning(edge_method):
    config = TemporalGraphConfig(baseline_size=80, edge_method=edge_method)
    graph = build_temporal_graph(_spike_series(), config)
    nodes = len(graph.window_starts)
    assert graph.adjacency.dtype == float
    assert graph.full_weights.shape == (nodes, nodes)
    np.testing.assert_allclose(graph.adjacency, graph.adjacency.T)
    np.testing.assert_allclose(np.diag(graph.adjacency), 1)
    assert np.all((graph.full_weights >= 0) & (graph.full_weights <= 1))
    assert graph.edge_mask.all() and graph.density == 1
    # Weighted degree is the node strength.
    np.testing.assert_allclose(graph.features[:, 3], graph.adjacency.sum(axis=1) - 1)


def test_wavelet_weights_separate_event_windows_from_background():
    graph = build_temporal_graph(_spike_series(), TemporalGraphConfig(
        baseline_size=80, edge_method="wavelet"))
    event = int(np.argmax(graph.features[:, 0]))
    background = [node for node in range(len(graph.window_starts)) if abs(node - event) > 2]
    to_event = graph.full_weights[event, background].mean()
    within = graph.full_weights[np.ix_(background, background)]
    assert to_event < 0.5 * within[~np.eye(len(background), dtype=bool)].mean()


@pytest.mark.parametrize("prune", ["threshold", "knn", "mst", "disparity"])
def test_pruning_sparsifies_and_keeps_weights(prune):
    config = TemporalGraphConfig(baseline_size=80, edge_method="wavelet", prune=prune,
                                 prune_k=2, prune_threshold=0.6)
    graph = build_temporal_graph(_spike_series(), config)
    kept = np.triu(graph.edge_mask, 1)
    assert graph.density < 1
    np.testing.assert_allclose(graph.adjacency[kept],
                               np.maximum(graph.full_weights[kept], 1e-6))
    np.testing.assert_allclose(graph.adjacency, graph.adjacency.T)


def test_pruning_rules():
    weights = np.array([[1.0, 0.9, 0.1, 0.2],
                        [0.9, 1.0, 0.8, 0.1],
                        [0.1, 0.8, 1.0, 0.3],
                        [0.2, 0.1, 0.3, 1.0]])
    threshold = prune_weighted_graph(weights, "threshold", threshold=0.5) > 0
    assert threshold[0, 1] and threshold[1, 2] and not threshold[2, 3]
    knn = prune_weighted_graph(weights, "knn", k=1) > 0
    assert knn[0, 1] and knn[1, 0] and knn[2, 1] and knn[3, 2] and not knn[0, 3]
    tree = prune_weighted_graph(weights, "mst", k=0) > 0
    assert np.triu(tree, 1).sum() == 3  # spanning tree on 4 nodes
    assert tree[0, 1] and tree[1, 2] and tree[2, 3]
    isolated = prune_weighted_graph(weights, "threshold", threshold=0.95, connected=True) > 0
    assert (isolated.sum(axis=1) > 1).all()
    with pytest.raises(ValueError, match="symmetric"):
        prune_weighted_graph(np.triu(weights), "knn")


def test_temporal_graph_config_validation():
    with pytest.raises(ValueError, match="pruning applies only"):
        TemporalGraphConfig(prune="knn")
    with pytest.raises(ValueError, match="edge_method"):
        TemporalGraphConfig(edge_method="spectral")
    with pytest.raises(ValueError, match="DFA edges"):
        TemporalGraphConfig(window_size=8, edge_method="dfa").resolved_dfa_scales()
    assert TemporalGraphConfig(window_size=32).resolved_dfa_scales() == (4, 8, 16)


def test_weighted_temporal_graph_trains_gnn_and_gat():
    recordings = {f"rec{i}": _spike_series(i, 110 + 5 * i) for i in range(6)}
    onsets = {f"rec{i}": 110 + 5 * i for i in range(6)}
    temporal = TemporalGraphConfig(baseline_size=80, edge_method="ar", prune="mst")
    report = ProcessModelTrainer(
        GraphLearningConfig(epochs=20, learning_rate=0.03, random_state=3),
        validation_fraction=0.34,
    ).train_1d(recordings, onsets, temporal_config=temporal)
    features, adjacency, _ = univariate_temporal_graph(recordings["rec0"], temporal)
    prediction = report.model.predict(features, adjacency)
    assert np.isclose(prediction.gat_probabilities.sum(), 1)
    assert np.all(prediction.attention[adjacency == 0] == 0)


def test_boolean_and_unit_weight_adjacency_give_identical_models():
    rng = np.random.default_rng(4)
    adjacency = _chain(4)
    episodes = [GraphEpisode(rng.normal(size=(4, 5)), adjacency, index % 4) for index in range(8)]
    weighted = [GraphEpisode(e.features, e.adjacency.astype(float), e.source) for e in episodes]
    config = GraphLearningConfig(epochs=10, random_state=1)
    first = GraphProcessModel(config).fit(episodes).predict(episodes[0].features, adjacency)
    second = GraphProcessModel(config).fit(weighted).predict(episodes[0].features,
                                                             adjacency.astype(float))
    np.testing.assert_allclose(first.gat_probabilities, second.gat_probabilities)
    np.testing.assert_allclose(first.gnn_probabilities, second.gnn_probabilities)


def _ramp_series(seed: int, onset: int, length: int = 700) -> np.ndarray:
    """Oscillation + noise with an event that stays below noise for ~40 samples."""
    rng = np.random.default_rng(seed)
    time = np.arange(length)
    values = np.sin(2 * np.pi * time / 97) + 0.3 * rng.normal(size=length)
    duration = 100
    taper = np.expm1(5 * np.linspace(0, 1, duration)) / np.expm1(5)
    values[onset:onset + duration] += 3 * rng.normal(size=duration) * taper
    return values


def test_context_features_extend_schema_and_heuristic_ignores_them():
    config = TemporalGraphConfig(window_size=32, stride=8, baseline_size=200, ar_order=4,
                                 context=("ar", "wavelet", "dfa"), context_horizons=(32, 64))
    graph = build_temporal_graph(_ramp_series(0, 400), config)
    assert graph.features.shape[1] == len(config.feature_names) == 5 + 3 * 2 * 2
    assert graph.feature_names[5:7] == ("ar_future_32", "ar_contrast_32")
    assert np.isfinite(graph.features).all()
    from graph_evt_agent.localization import heuristic_probabilities
    np.testing.assert_allclose(heuristic_probabilities(graph.features),
                               heuristic_probabilities(graph.features[:, :5]))


def test_context_features_localize_a_slowly_growing_onset():
    onsets = [300 + 13 * index for index in range(16)]
    recordings = {f"rec{i}": _ramp_series(i, onset) for i, onset in enumerate(onsets)}
    times = dict(zip(recordings, onsets))
    errors = {}
    for context in ((), ("ar",)):
        temporal = TemporalGraphConfig(window_size=32, stride=8, baseline_size=200,
                                       ar_order=4, context=context)
        report = ProcessModelTrainer(
            GraphLearningConfig(hidden_dim=16, epochs=150, random_state=0),
            validation_fraction=0.25).train_1d(recordings, times, temporal_config=temporal)
        found = []
        for seed, onset in ((100, 330), (101, 410), (102, 470)):
            graph = build_temporal_graph(_ramp_series(seed, onset), temporal)
            node = report.model.predict(graph.features, graph.adjacency).gnn_source
            found.append(abs(graph.window_starts[node] + 16 - onset))
        errors[context] = np.median(found)
    assert errors[("ar",)] <= 16 < errors[()]


def test_context_config_validation():
    with pytest.raises(ValueError, match="context"):
        TemporalGraphConfig(context="ar")
    with pytest.raises(ValueError, match="context"):
        TemporalGraphConfig(context=("spectral",))
    with pytest.raises(ValueError, match="context_horizons"):
        TemporalGraphConfig(context=("ar",), context_horizons=(8,))


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
