import numpy as np

from .config import EVTConfig, GraphConfig
from .evt import detect
from .graph import build_graph
from .localization import rank_sources
from .models import EVTDetection, GraphResult, SourceRanking


class GraphEVTPipeline:
    """Deterministic EVT → frozen graph → source-ranking pipeline."""

    def __init__(self, evt: EVTConfig, graph: GraphConfig | None = None):
        self.evt_config = evt
        self.graph_config = graph or GraphConfig()

    def run(self, values: np.ndarray) -> tuple[EVTDetection, GraphResult | None, SourceRanking | None]:
        data = np.asarray(values, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        detection = detect(data, self.evt_config)
        if data.shape[1] == 1 or not detection.detected:
            return detection, None, None
        graph = build_graph(data[: self.evt_config.baseline_size], self.graph_config)
        ranking = rank_sources(data, detection.time_index, graph.adjacency,
                               detection.location, detection.scale)
        return detection, graph, ranking
