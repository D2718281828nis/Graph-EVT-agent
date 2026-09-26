import pytest

from graph_evt_agent.agents import MistralClient


def test_mistral_client_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-placeholder-not-a-real-key")
    client = MistralClient()
    assert client._api_key == "test-placeholder-not-a-real-key"
    assert not hasattr(client, "api_key")


def test_mistral_client_fails_closed_without_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY is not set"):
        MistralClient()
