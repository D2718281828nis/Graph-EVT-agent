from types import SimpleNamespace

import pytest

from graph_evt_agent.agents import MistralAPIError, MistralClient


class FakeMistral:
    instances = []

    def __init__(self, *, api_key, server_url=None):
        self.api_key = api_key
        self.server_url = server_url
        self.chat = SimpleNamespace(complete=lambda **_kwargs: None)
        self.instances.append(self)


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    FakeMistral.instances.clear()
    monkeypatch.setattr(
        "graph_evt_agent.agents._new_mistral_client",
        lambda api_key, server_url: FakeMistral(api_key=api_key, server_url=server_url),
    )


def test_mistral_client_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-placeholder-not-a-real-key")
    client = MistralClient()
    assert client._api_key == "test-placeholder-not-a-real-key"
    assert client.model == "mistral-small-latest"
    assert not hasattr(client, "api_key")
    assert FakeMistral.instances[0].api_key == "test-placeholder-not-a-real-key"
    assert FakeMistral.instances[0].server_url is None


def test_mistral_client_passes_custom_server_url_to_sdk():
    client = MistralClient(api_key="secret", base_url="https://example.test/")

    assert client.base_url == "https://example.test"
    assert FakeMistral.instances[0].server_url == "https://example.test"


def test_mistral_client_fails_closed_without_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY is not set"):
        MistralClient()


def test_mistral_client_strips_key_whitespace():
    client = MistralClient(api_key="  secret-value\n")
    assert client._api_key == "secret-value"


def test_mistral_client_uses_sdk_chat_complete():
    client = MistralClient(api_key="secret", model="mistral-small-latest")
    client._client.chat.complete = lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    ) if kwargs == {
        "model": "mistral-small-latest",
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    } else None

    assert client.complete("system", "user") == "ok"


def test_mistral_client_explains_unauthorized_without_exposing_key():
    class Unauthorized(Exception):
        status_code = 401

    client = MistralClient(api_key="do-not-print-this-secret")
    client._client.chat.complete = lambda **_kwargs: (_ for _ in ()).throw(Unauthorized())

    with pytest.raises(MistralAPIError, match="active API key") as caught:
        client.complete("system", "user")
    assert "do-not-print-this-secret" not in str(caught.value)


def test_mistral_client_retries_transient_sdk_errors(monkeypatch):
    attempts = 0

    def fail(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise ConnectionError("temporary DNS failure")

    client = MistralClient(api_key="secret")
    client._client.chat.complete = fail
    monkeypatch.setattr("graph_evt_agent.agents.time.sleep", lambda _delay: None)

    with pytest.raises(MistralAPIError, match="after 3 attempts.*temporary DNS failure"):
        client.complete("system", "user")
    assert attempts == 3


def test_mistral_client_retries_server_errors(monkeypatch):
    attempts = 0

    class ServerError(Exception):
        status_code = 503

    def complete_after_errors(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ServerError("unavailable")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    client = MistralClient(api_key="secret")
    client._client.chat.complete = complete_after_errors
    monkeypatch.setattr("graph_evt_agent.agents.time.sleep", lambda _delay: None)

    assert client.complete("system", "user") == "ok"
    assert attempts == 3


def test_mistral_client_rejects_non_text_response():
    client = MistralClient(api_key="secret")
    client._client.chat.complete = lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    with pytest.raises(MistralAPIError, match="without text content"):
        client.complete("system", "user")
