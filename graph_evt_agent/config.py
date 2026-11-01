from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class InputConfig:
    """Rules applied by the input-inspection stage before EVT."""

    time_axis: int = 0
    missing: Literal["error", "drop_rows", "interpolate"] = "error"
    require_regular_time: bool = True

    def __post_init__(self) -> None:
        if self.time_axis not in {0, 1}:
            raise ValueError("time_axis must be 0 or 1")
        if self.missing not in {"error", "drop_rows", "interpolate"}:
            raise ValueError("unknown missing-value policy")


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


GRAPH_METHODS = ("auto", "ar", "dfa", "kuramoto", "ensemble", "corr", "partial", "xcorr", "pwxcorr", "granger")
# Methods whose edges come from a dependence matrix alone, so the edge rule, lag pruning and the
# channel-shift surrogate test can be applied to them.
EDGE_RULE_METHODS = ("ar", "dfa", "corr", "partial", "xcorr", "pwxcorr", "granger")
LAGGED_METHODS = ("xcorr", "pwxcorr")
SURROGATE_TEST_METHODS = ("corr", "xcorr", "pwxcorr")


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for a baseline-only graph hypothesis.

    ``method`` chooses the dependence measure; the defaults reproduce the original
    ``auto``/``ar``/``dfa``/``kuramoto``/``ensemble`` behaviour exactly. The fields below
    ``random_state`` only affect the newer methods or are opt-in:

    * ``edge_rule`` – ``"threshold"`` (dependence >= ``edge_threshold``) or ``"topk"``
      (the ``top_k_edges`` strongest partners of every node, union);
    * ``max_lag`` / ``lag_step`` / ``ridge`` – lag window of ``xcorr``/``pwxcorr`` and the
      lagged regression of ``granger``;
    * ``prune_indirect`` / ``lag_tolerance`` – drop an ``xcorr``/``pwxcorr`` edge explained by a
      stronger two-step path with matching lag;
    * ``surrogate_test`` – keep an edge only above the ``surrogate_quantile`` of
      ``surrogate_count`` circular channel shifts (``corr``/``xcorr``/``pwxcorr``).
    """

    method: Literal["auto", "ar", "dfa", "kuramoto", "ensemble",
                    "corr", "partial", "xcorr", "pwxcorr", "granger"] = "auto"
    ar_order: int = 2
    edge_threshold: float = 0.35
    directed: bool = False
    dfa_scales: tuple[int, ...] = (8, 16, 32, 64)
    wavelet_periods: tuple[float, ...] = (8.0, 16.0, 32.0)
    surrogate_count: int = 99
    surrogate_quantile: float = 0.95
    random_state: int = 0
    edge_rule: Literal["threshold", "topk"] = "threshold"
    top_k_edges: int = 2
    max_lag: int = 40
    lag_step: int = 5
    ridge: float = 1e-3
    prune_indirect: bool = False
    lag_tolerance: int = 6
    surrogate_test: bool = False

    def __post_init__(self) -> None:
        if self.method not in GRAPH_METHODS:
            raise ValueError("unknown graph method")
        if self.directed:
            raise ValueError("directed graphs are not supported")
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
        if self.edge_rule not in {"threshold", "topk"}:
            raise ValueError("edge_rule must be 'threshold' or 'topk'")
        if self.edge_rule == "topk" and self.method not in EDGE_RULE_METHODS:
            raise ValueError("edge_rule='topk' requires one of " + ", ".join(EDGE_RULE_METHODS))
        if self.top_k_edges < 1:
            raise ValueError("top_k_edges must be positive")
        if self.max_lag < 1 or self.lag_step < 1 or self.lag_step > self.max_lag:
            raise ValueError("need 1 <= lag_step <= max_lag")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.prune_indirect and self.method not in LAGGED_METHODS:
            raise ValueError("prune_indirect requires method 'xcorr' or 'pwxcorr'")
        if self.lag_tolerance < 0:
            raise ValueError("lag_tolerance must be non-negative")
        if self.surrogate_test and self.method not in SURROGATE_TEST_METHODS:
            raise ValueError("surrogate_test requires one of " + ", ".join(SURROGATE_TEST_METHODS))
