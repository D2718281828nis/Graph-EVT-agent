"""Opt-in graph methods: corr, partial, xcorr, pwxcorr, granger, edge rules, pruning, surrogates."""

import dataclasses

import numpy as np
import pytest

from graph_evt_agent import GraphConfig
from graph_evt_agent.graph import build_graph

# Undirected delayed coupling (i, j): lag in samples; both directions with the same lag.
DELAYED_EDGES = {(0, 1): 5, (0, 2): 15, (1, 2): 10, (1, 4): 20, (2, 3): 5, (3, 4): 10}


def delayed_network(n=3000, coupling=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 5))
    noise = rng.normal(size=(n, 5))
    for t in range(n):
        x[t] = noise[t]
        for (i, j), lag in DELAYED_EDGES.items():
            if t >= lag:
                x[t, j] += coupling * x[t - lag, i]
                x[t, i] += coupling * x[t - lag, j]
    return x


def edge_set(adjacency):
    nodes = len(adjacency)
    return {(i, j) for i in range(nodes) for j in range(i + 1, nodes) if adjacency[i, j]}


def test_original_defaults_are_unchanged():
    config = GraphConfig()
    assert (config.method, config.ar_order, config.edge_threshold, config.surrogate_count) == ("auto", 2, 0.35, 99)
    assert (config.edge_rule, config.prune_indirect, config.surrogate_test) == ("threshold", False, False)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conditional_granger_recovers_a_delayed_network_exactly(seed):
    result = build_graph(delayed_network(seed=seed),
                         GraphConfig(method="granger", max_lag=20, lag_step=5, edge_threshold=0.02))
    assert result.method == "conditional_granger"
    assert edge_set(result.adjacency) == set(DELAYED_EDGES)
    true = min(result.dependence[i, j] for i, j in DELAYED_EDGES)
    false = max(result.dependence[i, j] for i in range(5) for j in range(i + 1, 5) if (i, j) not in DELAYED_EDGES)
    assert true > 5 * false


def test_lagged_correlation_sees_indirect_pairs_and_pruning_removes_some():
    data = delayed_network(seed=0)
    plain = build_graph(data, GraphConfig(method="xcorr", max_lag=25, edge_threshold=0.1))
    pruned = build_graph(data, GraphConfig(method="xcorr", max_lag=25, edge_threshold=0.1,
                                           prune_indirect=True, lag_tolerance=2))
    truth = set(DELAYED_EDGES)
    assert truth < edge_set(plain.adjacency)                      # every true edge, plus indirect ones
    assert truth <= edge_set(pruned.adjacency) < edge_set(plain.adjacency)
    assert pruned.diagnostics["pruned"]
    assert abs(plain.diagnostics["lag"][0, 1]) == DELAYED_EDGES[(0, 1)]
    assert plain.diagnostics["lag"][1, 0] == -plain.diagnostics["lag"][0, 1]


def test_partial_correlation_recovers_a_gaussian_chain_where_correlation_does_not():
    precision = np.eye(5) + np.diag([-0.45] * 4, 1) + np.diag([-0.45] * 4, -1)
    data = np.random.default_rng(0).multivariate_normal(np.zeros(5), np.linalg.inv(precision), size=4000)
    chain = {(0, 1), (1, 2), (2, 3), (3, 4)}
    assert edge_set(build_graph(data, GraphConfig(method="partial", edge_threshold=0.15)).adjacency) == chain
    assert edge_set(build_graph(data, GraphConfig(method="corr", edge_threshold=0.15)).adjacency) > chain


@pytest.mark.parametrize("method", ["corr", "xcorr", "pwxcorr"])
def test_surrogate_test_keeps_independent_noise_nearly_empty(method):
    noise = np.random.default_rng(5).normal(size=(1500, 5))
    result = build_graph(noise, GraphConfig(method=method, edge_threshold=0.0, surrogate_test=True,
                                            surrogate_count=49, max_lag=20))
    assert len(edge_set(result.adjacency)) <= 2
    assert result.diagnostics["surrogate_threshold"].shape == (5, 5)


@pytest.mark.parametrize("method", ["ar", "dfa", "corr", "partial", "xcorr", "pwxcorr", "granger"])
def test_topk_rule_gives_every_node_at_least_k_partners(method):
    data = np.random.default_rng(3).normal(size=(800, 6)).cumsum(axis=0) if method == "dfa" else np.random.default_rng(3).normal(size=(800, 6))
    result = build_graph(data, GraphConfig(method=method, edge_rule="topk", top_k_edges=2, max_lag=10, lag_step=5))
    adjacency = result.adjacency
    assert np.array_equal(adjacency, adjacency.T) and adjacency.diagonal().all()
    assert (adjacency.sum(axis=1) - 1 >= 2).all()


@pytest.mark.parametrize("method,name", [
    ("corr", "correlation"), ("partial", "partial_correlation"), ("xcorr", "lagged_cross_correlation"),
    ("pwxcorr", "prewhitened_lagged_cross_correlation"), ("granger", "conditional_granger"),
])
def test_new_methods_return_symmetric_finite_graphs(method, name):
    data = delayed_network(n=800, seed=4)
    result = build_graph(data, GraphConfig(method=method, max_lag=20, lag_step=5))
    assert result.method == name
    assert result.adjacency.dtype == bool and result.adjacency.shape == (5, 5)
    assert np.array_equal(result.adjacency, result.adjacency.T) and result.adjacency.diagonal().all()
    assert np.isfinite(result.dependence).all() and np.allclose(result.dependence, result.dependence.T)


@pytest.mark.parametrize("kwargs", [
    {"method": "kuramoto", "edge_rule": "topk"}, {"method": "ensemble", "edge_rule": "topk"}, {"method": "auto", "edge_rule": "topk"},
    {"method": "corr", "prune_indirect": True}, {"method": "granger", "surrogate_test": True}, {"method": "ar", "surrogate_test": True},
    {"method": "xcorr", "max_lag": 5, "lag_step": 10}, {"method": "corr", "edge_rule": "nearest"}, {"method": "corr", "top_k_edges": 0},
    {"method": "granger", "ridge": -1.0}, {"method": "spectral"},
])
def test_invalid_combinations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GraphConfig(**kwargs)


@pytest.mark.parametrize("method", ["xcorr", "pwxcorr", "granger"])
def test_lagged_methods_reject_a_too_short_baseline(method):
    with pytest.raises(ValueError):
        build_graph(np.random.default_rng(0).normal(size=(60, 4)), GraphConfig(method=method, max_lag=40, lag_step=5))


def test_config_is_still_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        GraphConfig().method = "corr"
