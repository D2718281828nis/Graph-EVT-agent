import numpy as np
import pytest

from graph_evt_agent import (
    EVTConfig,
    GraphEVTPipeline,
    InputConfig,
    InputInspector,
    TimeSeriesInput,
)
from graph_evt_agent.agents import EVTAgentTeam
from graph_evt_agent.config import GraphConfig
from graph_evt_agent.graph import dfa_graph, kuramoto_wavelet_graph
from graph_evt_agent.models import GraphProcessPrediction


def test_directed_graph_configuration_is_rejected_explicitly():
    with pytest.raises(ValueError, match="directed graphs are not supported"):
        GraphConfig(directed=True)


def sample_data() -> np.ndarray:
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, (240, 4))
    data[180:190, 2] += np.linspace(12, 5, 10)
    data[183:193, 1] += np.linspace(8, 3, 10)
    return data


class FakeProcessModel:
    def predict(self, features, adjacency):
        probabilities = np.arange(1, len(features) + 1, dtype=float)
        probabilities /= probabilities.sum()
        attention = adjacency / adjacency.sum(axis=1, keepdims=True)
        return GraphProcessPrediction(probabilities, probabilities, attention)


def test_pipeline_executes_optional_learned_process_model():
    result = GraphEVTPipeline(
        EVTConfig(150, top_k=2, persistence=2), process_model=FakeProcessModel()
    ).run_detailed(sample_data())
    assert result.process.gnn_source == 3
    assert result.process.gat_source == 3
    assert result.process.attention.shape == (4, 4)
    assert result.plan.completed_actions[-1] == "infer_gnn_gat_process"


def test_pipeline_detects_and_ranks_earliest_source():
    pipeline = GraphEVTPipeline(EVTConfig(150, top_k=2, persistence=2))
    detection, graph, ranking = pipeline.run(sample_data())
    assert detection.detected
    assert 180 <= detection.time_index <= 181
    assert graph.adjacency.shape == (4, 4)
    assert ranking.source == 2
    assert np.isclose(ranking.probabilities.sum(), 1)
    detailed = pipeline.run_detailed(sample_data())
    assert detailed.plan.route == "n-d"
    assert detailed.plan.stop_reason is None
    assert detailed.plan.completed_actions[-1] == "rank_source"


def test_one_channel_does_not_claim_localization():
    pipeline = GraphEVTPipeline(EVTConfig(150, persistence=2))
    detection, graph, ranking = pipeline.run(sample_data()[:, 2])
    assert detection.detected
    assert graph is None and ranking is None
    result = pipeline.run_detailed(sample_data()[:, 2])
    assert result.plan.route == "1-d"
    assert result.plan.stop_reason == "single_channel_cannot_localize"


class FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return f"review-{self.calls}"


def test_agent_team_routes_with_input_agent_without_changing_result():
    client = FakeClient()
    team = EVTAgentTeam(GraphEVTPipeline(EVTConfig(150, top_k=2, persistence=2)), client)
    report = team.run(sample_data())
    assert client.calls == 5
    assert report.input_review == "review-1"
    assert report.final_report == "review-5"
    assert report.result["input_profile"]["kind"] == "multivariate"
    assert report.result["ranking"]["node_ids"][0] == 2


def test_cross_dfa_returns_exponents_and_multiscale_profiles():
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(size=512))
    data = np.column_stack([common + rng.normal(0, 0.2, 512),
                            common + rng.normal(0, 0.2, 512),
                            rng.normal(size=512)])
    result = dfa_graph(data, GraphConfig(method="dfa", edge_threshold=0.7))
    assert result.method == "cross_dfa"
    assert result.diagnostics["alpha"].shape == (3,)
    assert result.diagnostics["fluctuation"].shape == (4, 3)
    assert result.dependence[0, 1] > result.dependence[0, 2]


def test_wavelet_kuramoto_detects_phase_locked_pair_reproducibly():
    time = np.arange(512)
    rng = np.random.default_rng(4)
    phase = 2 * np.pi * (time / 20 + 0.00025 * time**2)
    first = np.sin(phase)
    data = np.column_stack([first, np.sin(phase + 0.4),
                            rng.normal(size=len(time))])
    config = GraphConfig(method="kuramoto", edge_threshold=0.65,
                         wavelet_periods=(16.0,), surrogate_count=19, random_state=11)
    result = kuramoto_wavelet_graph(data, config)
    repeated = kuramoto_wavelet_graph(data, config)
    assert result.method == "kuramoto_wavelet_phase_locking"
    assert result.adjacency[0, 1]
    assert not result.adjacency[0, 2]
    assert np.array_equal(result.adjacency, repeated.adjacency)
    assert result.diagnostics["order_parameter_mean"].shape == (1,)


def test_input_agent_identifies_mapping_as_multivariate():
    source = {"left": np.arange(30.0), "right": np.arange(30.0) + 1}
    prepared = InputInspector().inspect(source)
    assert prepared.values.shape == (30, 2)
    assert prepared.profile.kind == "multivariate"
    assert prepared.profile.channel_names == ("left", "right")


def test_timestamp_validation_and_explicit_missing_policy():
    values = np.arange(60.0).reshape(30, 2)
    values[4, 1] = np.nan
    source = TimeSeriesInput(values, timestamps=np.arange(30) * 0.5,
                             channel_names=("a", "b"))
    prepared = InputInspector(InputConfig(missing="interpolate")).inspect(source)
    assert prepared.profile.regular_time
    assert prepared.profile.sampling_interval == 0.5
    assert prepared.profile.missing_per_channel.tolist() == [0, 1]
    assert prepared.profile.actions == ("interpolated_missing_values",)
    assert np.isfinite(prepared.values).all()


def test_univariate_agent_route_skips_graph_agents():
    client = FakeClient()
    team = EVTAgentTeam(GraphEVTPipeline(EVTConfig(150, persistence=2)), client)
    report = team.run(sample_data()[:, 2])
    assert client.calls == 3  # input, EVT, coordinator
    assert report.result["input_profile"]["kind"] == "univariate"
    assert report.graph_review.startswith("Пропущено")
