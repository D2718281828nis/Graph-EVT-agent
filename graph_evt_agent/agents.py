"""Mistral-backed specialist agents for explaining and auditing EVT runs.

Numerical results are produced by :class:`GraphEVTPipeline`; the LLM never
silently changes thresholds, edges, or rankings.
"""

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np

from .pipeline import GraphEVTPipeline


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class MistralClient:
    """Small dependency-free client for Mistral's chat-completions API."""

    def __init__(self, api_key: str | None = None, model: str = "mistral-large-latest",
                 base_url: str = "https://api.mistral.ai/v1"):
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set; inject it through the process environment "
                "or a secrets manager"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:  # noqa: S310 - configured API endpoint
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]


@dataclass(frozen=True)
class AgentReport:
    input_review: str
    detection_review: str
    graph_review: str
    localization_review: str
    final_report: str
    result: dict[str, Any]


class EVTAgentTeam:
    """Input router plus specialist reviewers over a deterministic pipeline."""

    def __init__(self, pipeline: GraphEVTPipeline, client: ChatClient):
        self.pipeline = pipeline
        self.client = client

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): EVTAgentTeam._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [EVTAgentTeam._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "__dataclass_fields__"):
            return {k: EVTAgentTeam._jsonable(v) for k, v in asdict(value).items()}
        return value

    @staticmethod
    def _evidence(run: Any) -> dict[str, Any]:
        """Bound prompt size while retaining the decisions agents must audit."""
        detection = run.detection
        payload: dict[str, Any] = {
            "input_profile": EVTAgentTeam._jsonable(run.input_profile),
            "detection": {
                "detected": detection.detected,
                "time_index": detection.time_index,
                "alarm_threshold": detection.alarm_threshold,
                "indicator_min": float(detection.indicator.min()),
                "indicator_max": float(detection.indicator.max()),
                "indicator_length": len(detection.indicator),
            },
            "graph": None,
            "ranking": None,
        }
        if run.graph is not None:
            payload["graph"] = {
                "method": run.graph.method,
                "node_count": len(run.graph.adjacency),
                "undirected_edge_count": int(np.triu(run.graph.adjacency, 1).sum()),
                "dependence_min": float(run.graph.dependence.min()),
                "dependence_max": float(run.graph.dependence.max()),
            }
        if run.ranking is not None:
            limit = min(10, len(run.ranking.node_ids))
            payload["ranking"] = {
                "node_ids": run.ranking.node_ids[:limit].tolist(),
                "probabilities": run.ranking.probabilities[:limit].tolist(),
            }
        return payload

    def run(self, values: Any, task: str = "Найти источник события") -> AgentReport:
        run = self.pipeline.run_detailed(values)
        result = self._jsonable(run)
        evidence_payload = self._evidence(run)
        evidence = json.dumps(evidence_payload, ensure_ascii=False)
        guard = (
            "Опирайся только на JSON. Не изменяй численные результаты. "
            "Отделяй статистическую редкость от причинности и явно отмечай ограничения."
        )
        input_review = self.client.complete(
            f"Ты агент проверки входных данных. {guard}",
            f"Определи 1-D или n-D маршрут, проверь временную ось, пропуски и объём: "
            f"{json.dumps(evidence_payload['input_profile'], ensure_ascii=False)}",
        )
        detection_review = self.client.complete(
            f"Ты агент EVT-детекции. {guard}", f"Задача: {task}\nРезультат: {evidence}"
        )
        if run.input_profile.kind == "univariate" or not run.detection.detected:
            reason = ("одномерный вход" if run.input_profile.kind == "univariate"
                      else "экстремальное событие не обнаружено")
            graph_review = f"Пропущено: {reason}."
            localization_review = f"Пропущено: {reason}."
        else:
            graph_review = self.client.complete(
                f"Ты агент графовых зависимостей. {guard}",
                f"Проверь граф и утечку данных: {evidence}",
            )
            localization_review = self.client.complete(
                f"Ты агент локализации. {guard}", f"Объясни ранжирование узлов: {evidence}"
            )
        reviews = json.dumps(
            {"input": input_review, "EVT": detection_review, "graph": graph_review,
             "localization": localization_review},
            ensure_ascii=False,
        )
        final_report = self.client.complete(
            f"Ты координатор команды. {guard}",
            f"Составь краткий итог для пользователя. Численные данные: {evidence}\nОтзывы: {reviews}",
        )
        return AgentReport(input_review, detection_review, graph_review,
                           localization_review, final_report, result)
