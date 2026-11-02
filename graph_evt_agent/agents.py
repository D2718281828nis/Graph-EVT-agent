"""Mistral-backed specialist agents for explaining and auditing EVT runs.

Numerical results are produced by :class:`GraphEVTPipeline`; the LLM never
silently changes thresholds, edges, or rankings.
"""

from dataclasses import asdict, dataclass, field
from importlib import import_module
import json
import math
import os
import re
import time
from typing import Any, Protocol

import numpy as np

from .pipeline import GraphEVTPipeline


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class MistralAPIError(RuntimeError):
    """A safe, actionable error returned by the Mistral HTTP client."""


def _new_mistral_client(api_key: str, server_url: str | None) -> Any:
    """Load the network client only when an LLM client is requested."""
    try:
        mistral = import_module("mistralai")
    except ImportError:
        raise ImportError("MistralClient needs the optional dependency: pip install 'graph-evt-agent[llm]'") from None
    if server_url is None:
        return mistral.Mistral(api_key=api_key)
    return mistral.Mistral(api_key=api_key, server_url=server_url)


class MistralClient:
    """Adapter around Mistral's supported Python SDK."""

    def __init__(self, api_key: str | None = None, model: str = "mistral-small-latest",
                 base_url: str | None = None, max_retries: int = 2):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._api_key = (api_key or os.getenv("MISTRAL_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set; inject it through the process environment "
                "or a secrets manager"
            )
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else None
        self.max_retries = max_retries
        self._client = _new_mistral_client(self._api_key, self.base_url)

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        """Extract an HTTP status from the exception shapes used by the SDK."""
        status = getattr(error, "status_code", None)
        if status is None:
            status = getattr(error, "code", None)
        if status is None and getattr(error, "response", None) is not None:
            status = getattr(error.response, "status_code", None)
        return status if isinstance(status, int) else None

    def complete(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.complete(
                    model=self.model,
                    temperature=0.1,
                    messages=messages,
                )
                break
            except Exception as error:
                status = self._status_code(error)
                if status in (401, 403):
                    raise MistralAPIError(
                        f"Mistral API rejected the credentials (HTTP {status}). "
                        "Check that MISTRAL_API_KEY contains an active API key (not its name, "
                        "a workspace ID, or the .env.example placeholder), then create a new "
                        "key in the Mistral console if necessary."
                    ) from None
                if status is not None and status < 500 and status != 429:
                    raise MistralAPIError(
                        f"Mistral chat-completions request failed with HTTP {status}."
                    ) from None
                if attempt < self.max_retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise MistralAPIError(
                    f"Could not reach the Mistral API after {attempt + 1} attempts: "
                    f"{error}. Check connectivity, the selected model, and Mistral service "
                    "availability."
                ) from None
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise MistralAPIError("Mistral returned a chat response without text content.")
        return content


_NUMBER = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> list[tuple[str, float, int]]:
    """Numbers of a text as (spelling, value, decimals); "13,3" and "13.3" are the same number."""
    out = []
    for raw in _NUMBER.findall(text):
        s = raw.replace("−", "-").replace(",", ".")
        mant = s.split("e")[0].split("E")[0]
        out.append((raw, float(s), len(mant.split(".")[1]) if "." in mant else 0))
    return out


def _evidence_values(obj: Any) -> list[float]:
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)] if math.isfinite(obj) else []
    if isinstance(obj, str):
        return [v for _, v, _ in _numbers(obj)]
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _evidence_values(x)]
    if isinstance(obj, (list, tuple)):
        return [v for x in obj for v in _evidence_values(x)]
    return []


def verify_numbers(text: str, evidence: Any, rel_tol: float = 1e-9, skip_small_int: int = 10) -> list[str]:
    """Numbers of ``text`` that are not values of ``evidence`` (eq:fidelity).

    A number with d decimals is verified when some evidence value e satisfies |a − e| <= 0.5·10^-d (a is e rounded
    to the precision written in the text), also for percentages (a = 100·e). Integers below ``skip_small_int`` are
    not checked (list numbering, "top-3"). Returns the unverified spellings in order of appearance.
    """
    known = _evidence_values(evidence)
    bad = []
    for raw, a, d in _numbers(text):
        if d == 0 and abs(a) < skip_small_int and float(a).is_integer():
            continue
        tol = 0.5 * 10.0 ** -d + rel_tol * abs(a)
        if not any(abs(a - e) <= tol or abs(a - 100 * e) <= tol for e in known):
            bad.append(raw)
    return bad


_VERDICT = re.compile(r"\{[^{}]*\"event_detected\"[^{}]*\}", re.S)


def parse_verdict(text: str) -> dict[str, Any] | None:
    """The coordinator's structured verdict: the last JSON object with "event_detected" (and optionally "source")."""
    found = _VERDICT.findall(text)
    if not found:
        return None
    try:
        verdict = json.loads(found[-1])
    except json.JSONDecodeError:
        return None
    return verdict if isinstance(verdict.get("event_detected"), bool) else None


@dataclass(frozen=True)
class AgentReport:
    """Agent texts next to the unchanged numerical result.

    ``pipeline_verdict`` is the verdict of the numerical pipeline, ``agent_verdict`` — the coordinator's structured
    verdict (None if it gave none), ``disagreement`` — which fields differ (eq:disagree); the pipeline result is never
    replaced. ``unverified_numbers`` lists, per text, numbers that are not in the evidence JSON (eq:fidelity).
    """
    input_review: str
    detection_review: str
    graph_review: str
    localization_review: str
    final_report: str
    result: dict[str, Any]
    pipeline_verdict: dict[str, Any] = field(default_factory=dict)
    agent_verdict: dict[str, Any] | None = None
    disagreement: dict[str, bool] | None = None
    unverified_numbers: dict[str, list[str]] = field(default_factory=dict)
    final_report_rewritten: bool = False


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
            "execution_plan": EVTAgentTeam._jsonable(run.plan),
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
        if run.process is not None:
            strongest = np.dstack(np.unravel_index(
                np.argsort(run.process.attention.ravel())[::-1],
                run.process.attention.shape,
            ))[0][:10]
            payload["learned_process"] = {
                "gnn_source": run.process.gnn_source,
                "gat_source": run.process.gat_source,
                "gnn_probabilities": run.process.gnn_probabilities.tolist(),
                "gat_probabilities": run.process.gat_probabilities.tolist(),
                "strongest_attention_edges": strongest.tolist(),
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
        coordinator = (f"Ты координатор команды. {guard} В последней строке дай вердикт одним JSON-объектом: "
                       '{"event_detected": true|false, "source": "<узел или null>"}.')
        final_report = self.client.complete(
            coordinator,
            f"Составь краткий итог для пользователя. Численные данные: {evidence}\nОтзывы: {reviews}",
        )
        rewritten = False
        bad = verify_numbers(final_report, evidence_payload)
        if bad:                                       # одна переписка; оставшиеся расхождения помечаются
            final_report = self.client.complete(
                coordinator,
                f"В итоге есть числа, которых нет в численных данных: {', '.join(bad)}. Перепиши итог, используя только "
                f"числа из JSON.\nЧисленные данные: {evidence}\nПрежний итог: {final_report}",
            )
            rewritten = True
        texts = {"input": input_review, "detection": detection_review, "graph": graph_review,
                 "localization": localization_review, "final": final_report}
        unverified = {k: v for k, v in ((k, verify_numbers(x, evidence_payload)) for k, x in texts.items()) if v}
        ranking = evidence_payload.get("ranking")
        pipeline_verdict = {"event_detected": bool(run.detection.detected),
                            "source": ranking["node_ids"][0] if ranking else None}
        agent_verdict = parse_verdict(final_report)
        disagreement = None if agent_verdict is None else {
            k: agent_verdict.get(k) != pipeline_verdict[k] for k in pipeline_verdict if k in agent_verdict}
        return AgentReport(input_review, detection_review, graph_review,
                           localization_review, final_report, result, pipeline_verdict, agent_verdict,
                           disagreement, unverified, rewritten)
