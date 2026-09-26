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
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Pass api_key or set MISTRAL_API_KEY")
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
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:  # noqa: S310 - configured API endpoint
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]


@dataclass(frozen=True)
class AgentReport:
    detection_review: str
    graph_review: str
    localization_review: str
    final_report: str
    result: dict[str, Any]


class EVTAgentTeam:
    """Four cooperating roles over one reproducible numerical pipeline."""

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

    def run(self, values: np.ndarray, task: str = "Найти источник события") -> AgentReport:
        detection, graph, ranking = self.pipeline.run(values)
        result = self._jsonable({"detection": detection, "graph": graph, "ranking": ranking})
        evidence = json.dumps(result, ensure_ascii=False)
        guard = (
            "Опирайся только на JSON. Не изменяй численные результаты. "
            "Отделяй статистическую редкость от причинности и явно отмечай ограничения."
        )
        detection_review = self.client.complete(
            f"Ты агент EVT-детекции. {guard}", f"Задача: {task}\nРезультат: {evidence}"
        )
        graph_review = self.client.complete(
            f"Ты агент графовых зависимостей. {guard}", f"Проверь граф и утечку данных: {evidence}"
        )
        localization_review = self.client.complete(
            f"Ты агент локализации. {guard}", f"Объясни ранжирование узлов: {evidence}"
        )
        reviews = json.dumps(
            {"EVT": detection_review, "graph": graph_review, "localization": localization_review},
            ensure_ascii=False,
        )
        final_report = self.client.complete(
            f"Ты координатор команды. {guard}",
            f"Составь краткий итог для пользователя. Численные данные: {evidence}\nОтзывы: {reviews}",
        )
        return AgentReport(detection_review, graph_review, localization_review, final_report, result)
