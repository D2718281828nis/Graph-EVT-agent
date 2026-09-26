from io import BytesIO
import ssl
from urllib.error import HTTPError, URLError

import pytest

from graph_evt_agent.agents import MistralAPIError, MistralClient


def test_mistral_client_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-placeholder-not-a-real-key")
    client = MistralClient()
    assert client._api_key == "test-placeholder-not-a-real-key"
    assert client.model == "ministral-3b-2512"
    assert not hasattr(client, "api_key")


def test_mistral_client_fails_closed_without_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY is not set"):
        MistralClient()


def test_mistral_client_strips_key_whitespace():
    client = MistralClient(api_key="  secret-value\n")
    assert client._api_key == "secret-value"


def test_mistral_client_explains_unauthorized_without_exposing_key(monkeypatch):
    def reject(_request, timeout):
        assert timeout == 60
        raise HTTPError("https://api.mistral.ai/v1/chat/completions", 401,
                        "Unauthorized", {}, BytesIO(b'{"message":"bad key"}'))

    monkeypatch.setattr("graph_evt_agent.agents.urlopen", reject)
    client = MistralClient(api_key="do-not-print-this-secret")

    with pytest.raises(MistralAPIError, match="active API key") as caught:
        client.complete("system", "user")
    assert "do-not-print-this-secret" not in str(caught.value)


def test_mistral_client_explains_network_errors(monkeypatch):
    attempts = 0

    def fail(_request, timeout):
        nonlocal attempts
        attempts += 1
        raise URLError("temporary DNS failure")

    monkeypatch.setattr("graph_evt_agent.agents.urlopen", fail)
    monkeypatch.setattr("graph_evt_agent.agents.time.sleep", lambda _delay: None)
    with pytest.raises(MistralAPIError, match="after 3 attempts.*temporary DNS failure"):
        MistralClient(api_key="secret").complete("system", "user")
    assert attempts == 3


def test_mistral_client_retries_transient_ssl_eof(monkeypatch):
    attempts = 0

    def complete_after_eof(_request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")
        return BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr("graph_evt_agent.agents.urlopen", complete_after_eof)
    monkeypatch.setattr("graph_evt_agent.agents.time.sleep", lambda _delay: None)

    result = MistralClient(api_key="secret").complete("system", "user")

    assert result == "ok"
    assert attempts == 3
