from typing import Any, Protocol
from os import PathLike

from .config import EVTConfig, GraphConfig, InputConfig
from .evt import detect
from .graph import build_graph
from .input import InputInspector
from .io import load_timeseries
from .localization import rank_sources
from .models import EVTDetection, GraphResult, PipelineResult, SourceRanking
from .planner import PipelinePlanner
from .progress import Progress, ProgressCallback


class ProcessModel(Protocol):
    def predict(self, features: Any, adjacency: Any) -> Any: ...


class GraphEVTPipeline:
    """Deterministic EVT → frozen graph → source-ranking pipeline."""

    def __init__(self, evt: EVTConfig, graph: GraphConfig | None = None,
                 input_config: InputConfig | None = None,
                 process_model: ProcessModel | None = None,
                 verbose: bool | int = False,
                 progress_callback: ProgressCallback | None = None):
        self.evt_config = evt
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.graph_config = graph or GraphConfig()
        self.inspector = InputInspector(input_config)
        self.process_model = process_model
        self.planner = PipelinePlanner()

    def run_detailed(self, values: Any, verbose: bool | int | None = None) -> PipelineResult:
        """Inspect input, select the 1-D/n-D route, and return full provenance.

        ``verbose`` overrides the instance setting for this call and shows one
        progress step per executed planner action.
        """
        return self._run_detailed(values, self.verbose if verbose is None else verbose,
                                  self.progress_callback)

    def _run_detailed(self, values: Any, verbose: bool | int,
                      callback: ProgressCallback | None, depth: int = 0) -> PipelineResult:
        if isinstance(values, (str, PathLike)):
            values = load_timeseries(values)
        prepared = self.inspector.inspect(values)
        data = prepared.values
        plan = self.planner.plan(data, self.evt_config.baseline_size,
                                 self.process_model is not None)
        with Progress(len(plan.initial_actions), "Graph-EVT pipeline", verbose,
                      callback, depth) as progress:
            progress.advance(f"input {prepared.profile.kind} {data.shape}")
            progress.advance("stationary baseline" if plan.baseline_stationary
                             else "non-stationary baseline")
            detection = detect(data, self.evt_config)
            progress.advance(f"event at {detection.time_index}" if detection.detected
                             else "no event")
            if data.shape[1] == 1 or not detection.detected:
                plan = self.planner.observe(plan, detection.detected)
                progress.advance(f"stop: {plan.stop_reason}", progress.total - progress.step)
                return PipelineResult(prepared.profile, detection, None, None, plan)
            graph = build_graph(data[: self.evt_config.baseline_size], self.graph_config)
            edges = int((graph.adjacency.sum() - len(graph.adjacency)) // 2)
            progress.advance(f"{graph.method}: {edges} edges")
            ranking = rank_sources(data, detection.time_index, graph.adjacency,
                                   detection.location, detection.scale)
            progress.advance(f"source {prepared.profile.channel_names[ranking.source]}")
            process = None
            if self.process_model is not None:
                process = self.process_model.predict(ranking.features, graph.adjacency)
                progress.advance(f"GNN {process.gnn_source} / GAT {process.gat_source}")
        plan = self.planner.observe(plan, True, process is not None)
        return PipelineResult(prepared.profile, detection, graph, ranking, plan, process)

    def run(self, values: Any) -> tuple[EVTDetection, GraphResult | None, SourceRanking | None]:
        """Compatibility API returning the original three-result tuple."""
        result = self.run_detailed(values)
        return result.detection, result.graph, result.ranking
