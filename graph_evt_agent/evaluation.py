"""Evaluation orchestration that routes each input by its dimensionality.

Push any mix of 1-D and n-D series, optionally with ground truth. The
orchestrator runs the pipeline on each case, follows the planner's route, and
scores only what the route can support:

* **1-D** – detection only: hit/miss, false or early alarm, detection delay;
* **n-D** – detection plus localization of the true source for the
  transparent ranker and, when the pipeline has a fitted process model, the
  GNN and GAT.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .pipeline import GraphEVTPipeline
from .progress import Progress, ProgressCallback
from .training import METHODS

OUTCOMES = ("hit", "miss", "early_alarm", "false_alarm", "correct_rejection", "unlabelled")


@dataclass(frozen=True)
class EvaluationCase:
    """One input with optional truth: onset index and source (index or name)."""

    values: Any
    event_time: int | None = None
    source: int | str | None = None
    case_id: str | None = None
    has_truth: bool = True

    @classmethod
    def from_synthetic(cls, episode: Any, case_id: str | None = None) -> "EvaluationCase":
        """Wrap a :class:`~graph_evt_agent.synthetic.SyntheticEpisode`."""
        return cls(episode.as_input(), episode.event_time, episode.source, case_id)


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    route: str
    stop_reason: str | None
    outcome: str
    detected_time: int | None
    true_event_time: int | None
    delay: int | None
    true_source: int | None
    ranks: dict[str, int] = field(default_factory=dict)
    predicted: dict[str, int] = field(default_factory=dict)
    result: Any = None

    def as_record(self) -> dict[str, Any]:
        record = {
            "case_id": self.case_id, "route": self.route, "outcome": self.outcome,
            "stop_reason": self.stop_reason, "detected_time": self.detected_time,
            "true_event_time": self.true_event_time, "delay": self.delay,
            "true_source": self.true_source,
        }
        for method in METHODS:
            record[f"{method}_source"] = self.predicted.get(method)
            record[f"{method}_rank"] = self.ranks.get(method)
        return record


@dataclass(frozen=True)
class EvaluationReport:
    cases: tuple[CaseEvaluation, ...]
    k: int = 3

    def to_records(self) -> list[dict[str, Any]]:
        return [case.as_record() for case in self.cases]

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-route detection metrics and, for n-D, localization metrics."""
        output: dict[str, dict[str, Any]] = {}
        for route in sorted({case.route for case in self.cases}):
            cases = [case for case in self.cases if case.route == route]
            counts = {outcome: sum(case.outcome == outcome for case in cases)
                      for outcome in OUTCOMES}
            with_event = counts["hit"] + counts["miss"] + counts["early_alarm"]
            without_event = counts["false_alarm"] + counts["correct_rejection"]
            delays = [case.delay for case in cases if case.delay is not None]
            row: dict[str, Any] = {
                "cases": len(cases), **counts,
                "detection_rate": counts["hit"] / with_event if with_event else np.nan,
                "false_alarm_rate": (counts["false_alarm"] / without_event
                                     if without_event else np.nan),
                "mean_delay": float(np.mean(delays)) if delays else np.nan,
                "median_delay": float(np.median(delays)) if delays else np.nan,
            }
            if route == "n-d":
                for method in METHODS:
                    ranks = np.array([case.ranks[method] for case in cases
                                      if method in case.ranks])
                    row[f"{method}_n"] = len(ranks)
                    row[f"{method}_top1"] = float(np.mean(ranks == 1)) if len(ranks) else np.nan
                    row[f"{method}_top{self.k}"] = (float(np.mean(ranks <= self.k))
                                                    if len(ranks) else np.nan)
                    row[f"{method}_mrr"] = float(np.mean(1 / ranks)) if len(ranks) else np.nan
            output[route] = row
        return output


def _rank(weights: np.ndarray, source: int) -> int:
    return 1 + int(np.sum(weights > weights[source]))


class EvaluationOrchestrator:
    """Run and score a batch of 1-D/n-D cases with one frozen pipeline.

    ``early_tolerance`` samples of alarm before the true onset still count as
    a hit (annotation jitter); earlier alarms are ``early_alarm``.
    """

    def __init__(self, pipeline: GraphEVTPipeline, early_tolerance: int = 0, k: int = 3,
                 verbose: bool | int = False,
                 progress_callback: ProgressCallback | None = None):
        self.pipeline = pipeline
        self.early_tolerance = early_tolerance
        self.k = k
        self.verbose = verbose
        self.progress_callback = progress_callback

    def evaluate_case(self, case: EvaluationCase | Any, case_id: str = "case_0",
                      depth: int = 0) -> CaseEvaluation:
        if not isinstance(case, EvaluationCase):
            case = EvaluationCase(case, case_id=case_id, has_truth=False)
        case_id = case.case_id or case_id
        result = self.pipeline._run_detailed(case.values, self.verbose,
                                             self.progress_callback, depth)
        route = result.plan.route
        detection = result.detection
        detected = detection.time_index if detection.detected else None
        delay = None
        if not case.has_truth:
            outcome = "unlabelled"
        elif case.event_time is None:
            outcome = "false_alarm" if detected is not None else "correct_rejection"
        elif detected is None:
            outcome = "miss"
        elif detected < case.event_time - self.early_tolerance:
            outcome = "early_alarm"
        else:
            outcome, delay = "hit", int(detected - case.event_time)

        true_source = None
        if case.source is not None:
            names = result.input_profile.channel_names
            true_source = names.index(case.source) if isinstance(case.source, str) \
                else int(case.source)
        ranks: dict[str, int] = {}
        predicted: dict[str, int] = {}
        if result.ranking is not None:
            weights = {"heuristic": np.empty(len(result.ranking.node_ids))}
            weights["heuristic"][result.ranking.node_ids] = result.ranking.probabilities
            if result.process is not None:
                weights["gnn"] = result.process.gnn_probabilities
                weights["gat"] = result.process.gat_probabilities
            predicted = {method: int(np.argmax(w)) for method, w in weights.items()}
            if outcome == "hit" and true_source is not None:
                ranks = {method: _rank(w, true_source) for method, w in weights.items()}
        return CaseEvaluation(case_id, route, result.plan.stop_reason, outcome, detected,
                              case.event_time, delay, true_source, ranks, predicted, result)

    def evaluate(self, cases: Iterable[EvaluationCase | Any]) -> EvaluationReport:
        """Evaluate every case; raw series without truth are run and described."""
        items = list(cases)
        evaluations = []
        with Progress(len(items), "Evaluating cases", self.verbose,
                      self.progress_callback) as progress:
            for index, case in enumerate(items):
                evaluation = self.evaluate_case(case, f"case_{index}", depth=1)
                evaluations.append(evaluation)
                progress.advance(f"{evaluation.case_id}: {evaluation.route} {evaluation.outcome}")
        return EvaluationReport(tuple(evaluations), self.k)
