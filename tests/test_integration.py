"""End-to-end pipeline test(s): question -> SQL -> result -> narrative (Task B2),
against the small fixture dataset, asserting output shape rather than exact LLM
wording (PROJECT_SPEC.md Section 8). llm.client.generate is monkeypatched to return
controlled responses so this suite runs deterministically without live network calls
or API keys -- the same reasoning test_llm_client.py mocks the SDK calls directly,
not a relaxation of "no faked LLM responses in the shipped app" (which is about the
running application, not test doubles for a dependency it doesn't control).
"""

import json

import pandas as pd
import pytest

from llm import client, pipeline

TABLE_NAME = "superstore"


def _fake_generate(*responses):
    """A fake llm.client.generate() that yields each of `responses` in order,
    wrapped as a successful LLMResult from "gemini".
    """
    it = iter(responses)

    def fake(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        return client.LLMResult(text=next(it), provider="gemini", elapsed_ms=1.0)

    return fake


def test_answer_question_happy_path(mini_con, dataset_config, monkeypatch):
    """A full Phase 1 -> 2 -> 3 run against the fixture."""
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({
            "sql": "SELECT region, SUM(sales) AS total_sales FROM superstore GROUP BY region",
            "reasoning": "group by region, sum sales",
        }),
        "Sales are led by **West** at 6180, followed by East at 597.",
    ))

    result = pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "What is total revenue by region?")

    assert isinstance(result.result, pd.DataFrame)
    assert set(result.result.columns) == {"region", "total_sales"}
    assert len(result.result) == 2
    assert result.sql.strip().lower().startswith("select")
    assert result.narrative
    assert result.retried is False
    assert result.sql_provider == "gemini"
    assert result.narrative_provider == "gemini"


def test_answer_question_retries_once_on_bad_sql_then_succeeds(mini_con, dataset_config, monkeypatch):
    """First Phase 1 attempt references a table that doesn't exist (sandbox rejects
    it); the pipeline must retry exactly once and succeed on the corrected SQL.
    """
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({"sql": "SELECT * FROM not_a_real_table", "reasoning": "oops"}),
        json.dumps({"sql": "SELECT COUNT(*) AS n FROM superstore", "reasoning": "corrected"}),
        "There are 16 rows in total.",
    ))

    result = pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "How many rows are there?")

    assert result.retried is True
    assert result.result["n"].iloc[0] == 16


def test_answer_question_raises_pipeline_error_after_exhausting_retry(mini_con, dataset_config, monkeypatch):
    """Both the original and the single retry reference an unknown table -- the
    pipeline must not retry indefinitely, and must surface a clean PipelineError.
    """
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({"sql": "SELECT * FROM not_a_real_table", "reasoning": "oops"}),
        json.dumps({"sql": "SELECT * FROM still_not_real", "reasoning": "still wrong"}),
    ))

    with pytest.raises(pipeline.PipelineError):
        pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "How many rows are there?")


def test_answer_question_handles_declared_unanswerable(mini_con, dataset_config, monkeypatch):
    """When Phase 1 sets sql="" (the system prompt's documented "can't answer this"
    signal), the pipeline must skip Phase 2/3 and surface the LLM's own reasoning.
    """
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({"sql": "", "reasoning": "This dataset has no forecasting capability."}),
    ))

    result = pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "What will next quarter's sales be?")

    assert result.sql == ""
    assert result.result.empty
    assert "forecasting" in result.narrative


def test_answer_question_uses_conversation_history(mini_con, dataset_config, monkeypatch):
    """History is threaded into the Phase 1 user prompt so follow-up questions can
    resolve references like "that" -- assert the history text actually reaches the
    prompt, not just that the call succeeds.
    """
    captured_prompts = []

    def fake_generate(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured_prompts.append(user_prompt)
        if len(captured_prompts) == 1:
            return client.LLMResult(
                text=json.dumps({"sql": "SELECT COUNT(*) AS n FROM superstore", "reasoning": "count rows"}),
                provider="gemini", elapsed_ms=1.0,
            )
        return client.LLMResult(text="There are 16 rows.", provider="gemini", elapsed_ms=1.0)

    monkeypatch.setattr(pipeline.client, "generate", fake_generate)

    history = [{"question": "What is total sales?", "sql": "SELECT SUM(sales) FROM superstore", "answer": "6777"}]
    pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "And how many rows is that?", history=history)

    assert "What is total sales?" in captured_prompts[0]
