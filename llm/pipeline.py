"""Phase 1/2/3 orchestration + single auto-retry (Task B2).

The critical invariant: the LLM never executes anything. It only ever produces text
-- SQL in Phase 1, a narrative in Phase 3. Phase 2 is pure, deterministic Python
(llm/sandbox.py) running validated SQL against a connection kept logically separate
from the one ingest.py/quality.py use. On a Phase 2 failure, the error is sent back
to the LLM once for a single retry (llm/prompts.py's retry prompt); if the retry also
fails, a clean PipelineError is raised rather than retrying indefinitely.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

from backend import schema as schema_module
from llm import client, prompts
from llm.sandbox import SandboxViolation, execute_safe

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """A clean, user-facing failure after Phase 1/2's single retry is exhausted."""


@dataclass
class PipelineResult:
    """Everything about one question -> SQL -> result -> narrative run."""

    question: str
    sql: str
    reasoning: str
    result: pd.DataFrame
    narrative: str
    sql_provider: str
    narrative_provider: str
    retried: bool


def _strip_code_fences(text: str) -> str:
    """Defensively strip a ```json ... ``` (or bare ``` ... ```) fence if the LLM
    added one despite JSON-mode instructions not to.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_sql_response(text: str) -> tuple[str, str]:
    """Parse Phase 1's structured JSON response into (sql, reasoning). A malformed
    response is itself the kind of failure the single retry exists to recover from.
    """
    try:
        data = json.loads(_strip_code_fences(text))
    except json.JSONDecodeError as e:
        raise PipelineError(f"LLM response was not valid JSON: {e}") from e
    if not isinstance(data, dict) or "sql" not in data or "reasoning" not in data:
        raise PipelineError(f'LLM response missing required "sql"/"reasoning" keys: {text!r}')
    return data["sql"], data["reasoning"]


def _generate_sql(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    config: dict[str, Any],
    question: str,
    history_text: str,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    """Phase 1: ask the LLM for SQL. Returns (sql, reasoning, provider, live_schema)
    -- the schema is returned too so a retry doesn't need to re-introspect it.
    """
    live_schema = schema_module.get_schema(con, table_name)
    system_prompt = prompts.build_sql_system_prompt(live_schema, table_name, config)
    user_prompt = f"{history_text}\n\nCurrent question: {question}" if history_text else question

    result = client.generate(system_prompt, user_prompt, json_mode=True)
    sql, reasoning = _parse_sql_response(result.text)
    return sql, reasoning, result.provider, live_schema


def _generate_sql_retry(
    table_name: str,
    config: dict[str, Any],
    question: str,
    failed_sql: str,
    error_message: str,
    live_schema: list[dict[str, Any]],
) -> tuple[str, str, str]:
    """The single Phase 1 retry, after a Phase 2 failure."""
    system_prompt = prompts.build_sql_system_prompt(live_schema, table_name, config)
    retry_prompt = prompts.build_sql_retry_prompt(question, failed_sql, error_message)
    result = client.generate(system_prompt, retry_prompt, json_mode=True)
    sql, reasoning = _parse_sql_response(result.text)
    return sql, reasoning, result.provider


def _generate_narrative(question: str, sql: str, result_df: pd.DataFrame) -> tuple[str, str]:
    """Phase 3: ask the LLM to narrate the result. Returns (narrative, provider)."""
    system_prompt = prompts.build_narrative_system_prompt()
    user_prompt = prompts.build_narrative_user_prompt(question, sql, result_df)
    result = client.generate(system_prompt, user_prompt, json_mode=False)
    return result.text, result.provider


def _unanswerable_result(question: str, sql: str, reasoning: str, provider: str, retried: bool) -> PipelineResult:
    """Phase 1 declared the question unanswerable (sql == ""). Its reasoning already
    explains why -- don't force a retry loop or an empty-result narrative call.
    """
    return PipelineResult(
        question=question, sql=sql, reasoning=reasoning,
        result=pd.DataFrame(), narrative=reasoning,
        sql_provider=provider, narrative_provider=provider, retried=retried,
    )


def answer_question(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    config: dict[str, Any],
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> PipelineResult:
    """Run the full Phase 1 -> Phase 2 -> Phase 3 pipeline for one natural-language
    `question`. `con` should be the connection/cursor dedicated to LLM-generated SQL
    (see llm/sandbox.py). `history` is the last-5-turn conversational memory (Task
    B4), newest-last.
    """
    history_text = prompts.build_conversation_context(history or [])

    sql, reasoning, sql_provider, live_schema = _generate_sql(con, table_name, config, question, history_text)
    if not sql or not sql.strip():
        return _unanswerable_result(question, sql, reasoning, sql_provider, retried=False)

    retried = False
    try:
        result_df = execute_safe(con, sql, table_name)
    except (SandboxViolation, duckdb.Error) as first_error:
        logger.warning("Phase 2 failed for %r (%s); retrying once", sql, first_error)
        retried = True
        sql, reasoning, sql_provider = _generate_sql_retry(
            table_name, config, question, sql, str(first_error), live_schema
        )
        if not sql or not sql.strip():
            return _unanswerable_result(question, sql, reasoning, sql_provider, retried=True)
        try:
            result_df = execute_safe(con, sql, table_name)
        except (SandboxViolation, duckdb.Error) as second_error:
            raise PipelineError(
                "I couldn't produce a valid query for that question, even after "
                f"one retry. Last error: {second_error}"
            ) from second_error

    narrative, narrative_provider = _generate_narrative(question, sql, result_df)

    return PipelineResult(
        question=question, sql=sql, reasoning=reasoning, result=result_df,
        narrative=narrative, sql_provider=sql_provider,
        narrative_provider=narrative_provider, retried=retried,
    )
