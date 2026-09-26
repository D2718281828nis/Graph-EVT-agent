from typing import Any
from os import PathLike

from .config import EVTConfig, GraphConfig, InputConfig
from .evt import detect
from .graph import build_graph
from .input import InputInspector
from .io import load_timeseries
from .localization import rank_sources
from .models import EVTDetection, GraphResult, PipelineResult, SourceRanking


class GraphEVTPipeline:
    """Deterministic EVT → frozen graph → source-ranking pipeline."""

    def __init__(self, evt: EVTConfig, graph: GraphConfig | None = None,
                 input_config: InputConfig | None = None):
        self.evt_config = evt
        self.graph_config = graph or GraphConfig()
        self.inspector = InputInspector(input_config)

    def run_detailed(self, values: Any) -> PipelineResult:
        """Inspect input, select the 1-D/n-D route, and return full provenance."""
        if isinstance(values, (str, PathLike)):
            values = load_timeseries(values)
        prepared = self.inspector.inspect(values)
        data = prepared.values
        detection = detect(data, self.evt_config)
        if data.shape[1] == 1 or not detection.detected:
            return PipelineResult(prepared.profile, detection, None, None)
        graph = build_graph(data[: self.evt_config.baseline_size], self.graph_config)
        ranking = rank_sources(data, detection.time_index, graph.adjacency,
                               detection.location, detection.scale)
        return PipelineResult(prepared.profile, detection, graph, ranking)

    def run(self, values: Any) -> tuple[EVTDetection, GraphResult | None, SourceRanking | None]:
        """Compatibility API returning the original three-result tuple."""
        result = self.run_detailed(values)
        return result.detection, result.graph, result.ranking
