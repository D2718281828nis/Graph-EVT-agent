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
