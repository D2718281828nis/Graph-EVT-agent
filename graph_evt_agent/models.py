from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class EVTDetection:
    detected: bool
    time_index: int | None
    alarm_threshold: float
    indicator: np.ndarray
    location: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class GraphResult:
    adjacency: np.ndarray
    dependence: np.ndarray
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRanking:
    node_ids: np.ndarray
    probabilities: np.ndarray
    features: np.ndarray

    @property
    def source(self) -> int:
        return int(self.node_ids[0])


@dataclass(frozen=True)
class GraphProcessPrediction:
    """Learned source scores and an episode-specific process graph."""

    gnn_probabilities: np.ndarray
    gat_probabilities: np.ndarray
    attention: np.ndarray

    @property
    def gnn_source(self) -> int:
        return int(np.argmax(self.gnn_probabilities))

    @property
    def gat_source(self) -> int:
        return int(np.argmax(self.gat_probabilities))


@dataclass(frozen=True)
class PipelinePlan:
    """Deterministic plan/act/observe trace for one pipeline execution."""

    route: str
    baseline_stationary: bool
    initial_actions: tuple[str, ...]
    completed_actions: tuple[str, ...]
    stop_reason: str | None


@dataclass(frozen=True)
class InputProfile:
    kind: str
    n_time: int
    n_channels: int
    channel_names: tuple[str, ...]
    missing_per_channel: np.ndarray
    has_timestamps: bool
    regular_time: bool | None
    sampling_interval: float | None
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    input_profile: InputProfile
    detection: EVTDetection
    graph: GraphResult | None
    ranking: SourceRanking | None
    plan: PipelinePlan | None = None
    process: GraphProcessPrediction | None = None
