"""Deterministic plan/act/observe routing for the numerical pipeline."""

import numpy as np

from .graph import _approximately_stationary
from .models import PipelinePlan


class PipelinePlanner:
    """Plan computation from observable input state, without LLM guesswork."""

    def plan(self, values: np.ndarray, baseline_size: int,
             has_process_model: bool) -> PipelinePlan:
        route = "1-d" if values.shape[1] == 1 else "n-d"
        stationary = _approximately_stationary(values[:baseline_size])
        actions = ["inspect_input", "check_stationarity", "detect_evt"]
        if route == "n-d":
            actions.extend(["build_baseline_graph", "rank_source"])
            if has_process_model:
                actions.append("infer_gnn_gat_process")
        return PipelinePlan(route, stationary, tuple(actions),
                            ("inspect_input", "check_stationarity"), None)

    @staticmethod
    def observe(plan: PipelinePlan, detected: bool,
                has_process_prediction: bool = False) -> PipelinePlan:
        completed = list(plan.completed_actions) + ["detect_evt"]
        if not detected:
            return PipelinePlan(plan.route, plan.baseline_stationary,
                                plan.initial_actions, tuple(completed), "no_evt_detected")
        if plan.route == "1-d":
            return PipelinePlan(plan.route, plan.baseline_stationary,
                                plan.initial_actions, tuple(completed),
                                "single_channel_cannot_localize")
        completed.extend(["build_baseline_graph", "rank_source"])
        if has_process_prediction:
            completed.append("infer_gnn_gat_process")
        return PipelinePlan(plan.route, plan.baseline_stationary,
                            plan.initial_actions, tuple(completed), None)
