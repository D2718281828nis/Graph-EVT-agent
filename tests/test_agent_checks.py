"""Проверка чисел текста агентов (eq:fidelity) и вердикт рядом с результатом конвейера (eq:disagree)."""
import json

from graph_evt_agent import EVTConfig, GraphEVTPipeline
from graph_evt_agent.agents import EVTAgentTeam, parse_verdict, verify_numbers
from test_pipeline import sample_data


def test_verify_numbers_accepts_rounding_and_percent_and_flags_the_rest():
    evidence = {"detection": {"time_index": 214, "alarm_threshold": 3.14159}, "ranking": {"probabilities": [0.4712, 0.2]}}
    text = "Тревога на отсчёте 214, порог 3,14 (точнее 3.1416), вероятность 47 % и 0.47; ещё 3.15, 215 и 5 шагов."
    assert verify_numbers(text, evidence) == ["3.15", "215"]          # 5 — малое целое, не проверяется
    assert verify_numbers("всё по JSON: 0.2", evidence) == []


def test_parse_verdict_takes_the_last_json_object():
    text = 'Итог…\n{"event_detected": false}\nПоправка:\n{"event_detected": true, "source": 2}'
    assert parse_verdict(text) == {"event_detected": True, "source": 2}
    assert parse_verdict("без вердикта") is None
    assert parse_verdict('{"event_detected": "yes"}') is None


class WrongNumberClient:
    """Координатор сначала пишет неверное число и не тот источник, после просьбы переписать — только числа JSON."""

    def __init__(self):
        self.calls, self.prompts = 0, []

    def complete(self, system, user):
        self.calls += 1; self.prompts.append(user)
        if "координатор" not in system:
            return f"review-{self.calls}"
        if "Перепиши итог" in user:
            return 'Событие найдено, источник — узел 2.\n{"event_detected": true, "source": 0}'
        return 'Событие найдено на отсчёте 99999, источник — узел 0.\n{"event_detected": true, "source": 0}'


def test_wrong_number_triggers_one_rewrite_and_disagreement_is_kept_next_to_the_pipeline():
    client = WrongNumberClient()
    team = EVTAgentTeam(GraphEVTPipeline(EVTConfig(150, top_k=2, persistence=2)), client)
    report = team.run(sample_data())
    assert client.calls == 6 and report.final_report_rewritten                 # 5 обычных вызовов + одна переписка
    assert "99999" in client.prompts[-1]
    assert report.unverified_numbers == {}
    assert report.pipeline_verdict == {"event_detected": True, "source": 2}
    assert report.agent_verdict == {"event_detected": True, "source": 0}
    assert report.disagreement == {"event_detected": False, "source": True}    # вердикт агентов расходится с конвейером
    assert report.result["ranking"]["node_ids"][0] == 2                        # численный результат не изменился


def test_plain_reviews_need_no_rewrite_and_have_no_verdict():
    from test_pipeline import FakeClient
    report = EVTAgentTeam(GraphEVTPipeline(EVTConfig(150, top_k=2, persistence=2)), FakeClient()).run(sample_data())
    assert not report.final_report_rewritten and report.agent_verdict is None and report.disagreement is None
    assert json.loads(json.dumps(report.pipeline_verdict)) == report.pipeline_verdict
