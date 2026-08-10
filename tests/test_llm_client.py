"""Mocked Gemini/Groq call tests (Task B5): assert failover triggers on a simulated
timeout/error/empty/truncated response, and that a successful Gemini call never
touches Groq. Everything here mocks llm.client._call_gemini/_call_groq directly, so
no real API keys or network access are needed -- this is standard practice for unit
tests and is not the same thing as the app faking a real LLM response.
"""

import pytest
from google.genai import errors as genai_errors

from llm import client


def _make_gemini_api_error() -> genai_errors.APIError:
    return genai_errors.APIError(code=429, response_json={"error": {"message": "rate limited"}}, response=None)


def test_generate_success_gemini_only(monkeypatch):
    """A successful Gemini call must never touch Groq."""
    groq_called = False

    def fake_gemini(*args, **kwargs):
        return "gemini response text"

    def fake_groq(*args, **kwargs):
        nonlocal groq_called
        groq_called = True
        return "groq response text"

    monkeypatch.setattr(client, "_call_gemini", fake_gemini)
    monkeypatch.setattr(client, "_call_groq", fake_groq)

    result = client.generate("system", "user")

    assert result.provider == "gemini"
    assert result.text == "gemini response text"
    assert groq_called is False


@pytest.mark.parametrize(
    "gemini_exception",
    [
        TimeoutError("simulated timeout"),
        _make_gemini_api_error(),
        client.EmptyResponseError("empty"),
        client.TruncatedResponseError("truncated"),
    ],
    ids=["timeout", "api_error_rate_limit", "empty_response", "truncated_response"],
)
def test_generate_falls_back_to_groq_on_gemini_failure(monkeypatch, gemini_exception):
    def fake_gemini(*args, **kwargs):
        raise gemini_exception

    def fake_groq(*args, **kwargs):
        return "groq response text"

    monkeypatch.setattr(client, "_call_gemini", fake_gemini)
    monkeypatch.setattr(client, "_call_groq", fake_groq)

    result = client.generate("system", "user")

    assert result.provider == "groq"
    assert result.text == "groq response text"


def test_generate_raises_llm_error_when_both_providers_fail(monkeypatch):
    def fake_gemini(*args, **kwargs):
        raise TimeoutError("gemini down")

    def fake_groq(*args, **kwargs):
        raise TimeoutError("groq down too")

    monkeypatch.setattr(client, "_call_gemini", fake_gemini)
    monkeypatch.setattr(client, "_call_groq", fake_groq)

    with pytest.raises(client.LLMError):
        client.generate("system", "user")


def test_generate_returns_elapsed_ms(monkeypatch):
    monkeypatch.setattr(client, "_call_gemini", lambda *a, **k: "ok")
    result = client.generate("system", "user")
    assert isinstance(result.elapsed_ms, float)
    assert result.elapsed_ms >= 0


def test_generate_unrecoverable_gemini_error_is_not_swallowed(monkeypatch):
    """A bug in our own code (e.g. a KeyError) must propagate, not be silently
    treated as a provider failure and masked by a fallback attempt.
    """
    def fake_gemini(*args, **kwargs):
        raise KeyError("GEMINI_API_KEY")

    monkeypatch.setattr(client, "_call_gemini", fake_gemini)

    with pytest.raises(KeyError):
        client.generate("system", "user")
