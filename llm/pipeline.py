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

from backend import query_engine, quality as quality_module
from backend import schema as schema_module
from llm import client, prompts
from llm.sandbox import SandboxViolation, execute_safe

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """A clean, user-facing failure after Phase 1/2's single retry is exhausted."""


@dataclass
class InsightResult:
    """One preset insight (Task B3): the curated data that fed it and the
    LLM-generated narrative."""

    insight_type: str
    data: dict[str, Any]
    narrative: str
    narrative_provider: str


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


# ---------------------------------------------------------------------------
# Task B3 -- preset insight generation
# ---------------------------------------------------------------------------


def generate_dataset_overview(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    config: dict[str, Any],
) -> InsightResult:
    """Preset 1: dataset shape + backend/quality.py's profiling summary, narrated
    in plain language. Fully generic -- profile_report() and get_schema() already
    discover columns dynamically, so no config entry is needed for this preset.
    """
    profile = quality_module.profile_report(con, table_name, id_column=config["dataset"].get("id_column"))
    live_schema = schema_module.get_schema(con, table_name)

    system_prompt = (
        "You are a data analyst writing a short 'dataset overview' for a business "
        "dashboard. You will be given a data-quality profile (row count, duplicate "
        "count, and per-column missing % / IQR outlier counts) and the column "
        "list. Write 3-5 sentences in Markdown describing the dataset's size and "
        "shape and anything notable in the quality profile. Use only the numbers "
        "given -- never invent a figure."
    )
    user_prompt = json.dumps(
        {"profile": profile, "columns": [c["name"] for c in live_schema]}, indent=2, default=str
    )
    result = client.generate(system_prompt, user_prompt, json_mode=False)

    return InsightResult(
        insight_type="dataset_overview", data={"profile": profile},
        narrative=result.text, narrative_provider=result.provider,
    )


def generate_trend_comparison(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    config: dict[str, Any],
) -> InsightResult:
    """Preset 2: the curated aggregation declared in configs/dataset_config.yaml's
    insights.trend_comparison, narrated as a trend/comparison summary. The dims/
    metrics/aggs live in config (not here) so this function stays free of
    Superstore column-name literals.
    """
    spec = config["insights"]["trend_comparison"]
    df, _ = query_engine.groupby_agg(con, table_name, spec["dims"], spec["metrics"], spec["aggs"])
    if spec["dims"]:
        df = df.sort_values(by=spec["dims"]).reset_index(drop=True)

    system_prompt = (
        "You are a data analyst writing a short trend/comparison summary for a "
        "business dashboard. You will be given a small aggregated table. Write 3-5 "
        "sentences in Markdown identifying the main trend or the biggest "
        "differences between groups, using only numbers present in the table."
    )
    user_prompt = json.dumps(
        {"columns": list(df.columns), "rows": df.to_dict(orient="records")}, indent=2, default=str
    )
    result = client.generate(system_prompt, user_prompt, json_mode=False)

    return InsightResult(
        insight_type="trend_comparison", data={"aggregation": df},
        narrative=result.text, narrative_provider=result.provider,
    )


def generate_anomaly_report(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    config: dict[str, Any],
) -> InsightResult:
    """Preset 3: reuses A3's IQR outlier profiling, picks the numeric column with
    the most outliers -- chosen at runtime from the data, never hardcoded -- pulls a
    small sample of the actual outlier rows via A2's filtered_query(), and narrates
    a plain-language anomaly report.
    """
    profile = quality_module.profile_report(con, table_name, id_column=config["dataset"].get("id_column"))
    outlier_columns = [c for c in profile["columns"] if (c["n_outliers_iqr"] or 0) > 0]

    if not outlier_columns:
        narrative = "No IQR-based outliers were detected in any numeric column."
        return InsightResult(
            insight_type="anomaly_report", data={"profile": profile},
            narrative=narrative, narrative_provider="none",
        )

    worst = max(outlier_columns, key=lambda c: c["n_outliers_iqr"])
    bounds = worst["iqr_bounds"]
    id_column = config["dataset"].get("id_column")
    dims = [id_column] if id_column else []

    below, _ = query_engine.filtered_query(
        con, table_name,
        filters=[{"column": worst["name"], "op": "<", "value": bounds["lower"]}],
        dims=dims, metrics=[worst["name"]],
    )
    above, _ = query_engine.filtered_query(
        con, table_name,
        filters=[{"column": worst["name"], "op": ">", "value": bounds["upper"]}],
        dims=dims, metrics=[worst["name"]],
    )
    sample_df = pd.concat([below, above], ignore_index=True).head(10)

    system_prompt = (
        "You are a data analyst writing a short anomaly/outlier report for a "
        "business dashboard. You will be given the numeric column with the most "
        "IQR-based outliers, its normal (IQR) bounds, the total outlier count, and "
        "a small sample of the actual outlier rows. Write 2-4 sentences in "
        "Markdown plainly describing the anomaly, using only the numbers given."
    )
    user_prompt = json.dumps({
        "column": worst["name"],
        "iqr_bounds": bounds,
        "n_outliers_total": worst["n_outliers_iqr"],
        "sample_outlier_rows": sample_df.to_dict(orient="records"),
    }, indent=2, default=str)
    result = client.generate(system_prompt, user_prompt, json_mode=False)

    return InsightResult(
        insight_type="anomaly_report", data={"profile": profile, "outlier_samples": sample_df},
        narrative=result.text, narrative_provider=result.provider,
    )
