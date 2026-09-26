from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EVTConfig:
    """Leakage-safe peaks-over-threshold configuration."""

    baseline_size: int
    tail_quantile: float = 0.95
    alarm_probability: float = 0.999
    top_k: int = 3
    persistence: int = 3
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.baseline_size < 20:
            raise ValueError("baseline_size must be at least 20")
        if not 0.5 < self.tail_quantile < 1:
            raise ValueError("tail_quantile must be between 0.5 and 1")
        if not self.tail_quantile < self.alarm_probability < 1:
            raise ValueError("alarm_probability must exceed tail_quantile")
        if self.top_k < 1 or self.persistence < 1:
            raise ValueError("top_k and persistence must be positive")


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for a baseline-only graph hypothesis."""

    method: Literal["auto", "ar", "dfa", "kuramoto", "ensemble"] = "auto"
    ar_order: int = 2
    edge_threshold: float = 0.35
    directed: bool = False
    dfa_scales: tuple[int, ...] = (8, 16, 32, 64)
    wavelet_periods: tuple[float, ...] = (8.0, 16.0, 32.0)
    surrogate_count: int = 99
    surrogate_quantile: float = 0.95
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.method not in {"auto", "ar", "dfa", "kuramoto", "ensemble"}:
            raise ValueError("unknown graph method")
        if self.ar_order < 1:
            raise ValueError("ar_order must be positive")
        if not 0 <= self.edge_threshold <= 1:
            raise ValueError("edge_threshold must be in [0, 1]")
        if len(self.dfa_scales) < 2 or any(scale < 4 for scale in self.dfa_scales):
            raise ValueError("dfa_scales must contain at least two scales >= 4")
        if not self.wavelet_periods or any(period < 2 for period in self.wavelet_periods):
            raise ValueError("wavelet_periods must be >= 2")
        if self.surrogate_count < 1 or not 0 < self.surrogate_quantile < 1:
            raise ValueError("invalid surrogate configuration")
