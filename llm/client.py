"""LLM client with Gemini->Groq failover (Task B5).

generate() is the one entrypoint the rest of llm/ calls. It tries Gemini first and
falls back to Groq transparently on API error, rate limit, timeout, empty response,
or truncated response -- logging which provider actually served each request. No
responses are ever faked or hardcoded: every call in this module reaches a real
provider API (PROJECT_SPEC.md's "no faked LLM responses" rule).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import groq
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

# Loaded at import time so GEMINI_API_KEY/GROQ_API_KEY are available regardless of
# whether backend.ingest (which also loads .env, for PII_HASH_SALT) has been
# imported first -- this module must not depend on that import-order side effect.
load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_S = 30.0

# Exceptions from each provider's own SDK that should trigger falling back to the
# other provider rather than being treated as a bug in our own code. Both SDKs'
# rate-limit/status errors subclass their respective base APIError. httpx.HTTPError
# is also needed: confirmed live (not just in theory) that the google-genai SDK's
# internal retry logic (tenacity) can exhaust its retries and re-raise a raw
# httpx.ReadTimeout instead of wrapping it in genai_errors.APIError -- and
# httpx.ReadTimeout does NOT subclass the builtin TimeoutError, so it would
# otherwise crash the whole request instead of falling back to Groq.
_GEMINI_SDK_ERRORS = (genai_errors.APIError, TimeoutError, httpx.HTTPError)
_GROQ_SDK_ERRORS = (groq.APIError, TimeoutError, httpx.HTTPError)


class LLMError(Exception):
    """Raised when both Gemini and Groq fail to produce a usable response."""


class EmptyResponseError(Exception):
    """A provider returned an empty response body."""


class TruncatedResponseError(Exception):
    """A provider's response was cut off by a token limit before finishing."""


@dataclass
class LLMResult:
    """A successful LLM call: the raw text, which provider served it, and timing."""

    text: str
    provider: str
    elapsed_ms: float


def _call_gemini(system_prompt: str, user_prompt: str, json_mode: bool, timeout_s: float) -> str:
    """One call to Gemini. Raises EmptyResponseError/TruncatedResponseError on a bad
    response, or lets the SDK's own exceptions (APIError, timeouts) propagate.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json" if json_mode else "text/plain",
        http_options=genai_types.HttpOptions(timeout=int(timeout_s * 1000)),
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=user_prompt, config=config)

    candidates = response.candidates or []
    if candidates and candidates[0].finish_reason == genai_types.FinishReason.MAX_TOKENS:
        raise TruncatedResponseError("Gemini response was truncated (finish_reason=MAX_TOKENS)")

    text = response.text
    if not text or not text.strip():
        raise EmptyResponseError("Gemini returned an empty response")
    return text


def _call_groq(system_prompt: str, user_prompt: str, json_mode: bool, timeout_s: float) -> str:
    """One call to Groq. Same contract as _call_gemini."""
    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"], timeout=timeout_s)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise TruncatedResponseError("Groq response was truncated (finish_reason=length)")

    text = choice.message.content
    if not text or not text.strip():
        raise EmptyResponseError("Groq returned an empty response")
    return text


def generate(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> LLMResult:
    """Try Gemini first; on API error, rate limit, timeout, empty response, or
    truncated response, fall back to Groq transparently. Raises LLMError only if
    both providers fail. Always logs which provider actually served the request.
    """
    start = time.perf_counter()
    try:
        text = _call_gemini(system_prompt, user_prompt, json_mode, timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("LLM request served by gemini in %.0fms", elapsed_ms)
        return LLMResult(text=text, provider="gemini", elapsed_ms=elapsed_ms)
    except (*_GEMINI_SDK_ERRORS, EmptyResponseError, TruncatedResponseError) as gemini_error:
        logger.warning(
            "Gemini failed (%s: %s); falling back to Groq",
            type(gemini_error).__name__, gemini_error,
        )

    try:
        text = _call_groq(system_prompt, user_prompt, json_mode, timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("LLM request served by groq (Gemini fallback) in %.0fms", elapsed_ms)
        return LLMResult(text=text, provider="groq", elapsed_ms=elapsed_ms)
    except (*_GROQ_SDK_ERRORS, EmptyResponseError, TruncatedResponseError) as groq_error:
        elapsed_ms = (time.perf_counter() - start) * 1000
        raise LLMError(
            f"Both providers failed after {elapsed_ms:.0f}ms. "
            f"Groq error: {type(groq_error).__name__}: {groq_error}"
        ) from groq_error
