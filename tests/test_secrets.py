from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from graph_evt_agent.agents import MistralAPIError, MistralClient


def test_mistral_client_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-placeholder-not-a-real-key")
    client = MistralClient()
    assert client._api_key == "test-placeholder-not-a-real-key"
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
    def fail(_request, timeout):
        raise URLError("temporary DNS failure")

    monkeypatch.setattr("graph_evt_agent.agents.urlopen", fail)
    with pytest.raises(MistralAPIError, match="temporary DNS failure"):
        MistralClient(api_key="secret").complete("system", "user")
