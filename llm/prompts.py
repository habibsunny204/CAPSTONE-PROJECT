"""System prompt templates for SQL generation and narrative generation (Task B1).

Schema is always introspected live (backend/schema.py) and every dataset-specific
detail (synonym dictionary, few-shot examples) is read from
configs/dataset_config.yaml rather than hardcoded here -- this module builds prompt
*text*, it never contains a dataset column name as a Python string literal.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

SQL_RESPONSE_INSTRUCTIONS = (
    'Respond with a single JSON object of exactly this shape: '
    '{"sql": "<a single read-only SELECT statement, or an empty string if unanswerable>", '
    '"reasoning": "<one or two sentences explaining your approach>"}. '
    "Do not include markdown code fences, explanations outside the JSON, or any text "
    "before/after the JSON object."
)


def build_sql_system_prompt(
    schema: list[dict[str, Any]],
    table_name: str,
    config: dict[str, Any],
    include_synonyms: bool = True,
    include_few_shot: bool = True,
    scope_description: str = "",
) -> str:
    """Assemble the Phase 1 (NL -> SQL) system prompt: table name, introspected
    schema JSON, synonym dictionary, and few-shot examples.

    The two `include_*` flags default to on, so normal callers get the full
    prompt. They exist so eval/run_ablation.py can build the same prompt with
    individual components removed and measure what each one actually contributes
    to accuracy (PROJECT_SPEC.md Section 10) -- without that, "the synonym
    dictionary helps" would be an assumption rather than a measurement.

    `scope_description` describes the dashboard filters currently in force. The rows
    are *already* restricted before this SQL runs (the connection points at a filtered
    view -- see backend/scope.py), so this text is not what makes the answer correct;
    it exists so the model doesn't silently re-apply a filter that is already applied,
    and so it can say when a question reaches outside the current selection. Empty by
    default, and when empty nothing is added at all -- which keeps the prompt
    byte-identical to the one the benchmark and ablation runs measured.
    """
    synonyms = config.get("synonyms", {}) if include_synonyms else {}
    few_shot = config.get("few_shot_examples", []) if include_few_shot else []

    parts = [
        "You are a SQL generation assistant for a read-only analytics database. "
        f'There is exactly one table, "{table_name}", described by this schema '
        "(one JSON object per column: name, dtype, n_unique, n_null, up to 5 sample_values):",
        json.dumps(schema, indent=2, default=str),
        "\nRules:",
        f'- Only ever query the "{table_name}" table. Never reference any other '
        "table, file, or function.",
        "- Generate exactly one read-only SELECT statement (CTEs and UNION/"
        "INTERSECT/EXCEPT of SELECTs are fine). Never DDL, DML, ATTACH, PRAGMA, "
        "COPY, or multiple statements.",
        '- If the question cannot be answered from this schema (it needs data or '
        'a capability -- like forecasting -- this table does not have), set "sql" '
        'to an empty string and explain why in "reasoning" instead of guessing or '
        "inventing columns.",
    ]

    if scope_description:
        parts.append(
            f'\nThe user has dashboard filters applied, and "{table_name}" already '
            f"contains only the matching rows: {scope_description}. Do not add these "
            "conditions to your WHERE clause -- they are already in effect. If the "
            "question asks about data outside this selection, say so in "
            '"reasoning" rather than implying the full dataset was searched.'
        )

    if synonyms:
        parts.append(
            "\nUsers often use informal terms that don't match column names "
            "exactly. Known synonyms (informal term -> actual column):"
        )
        parts.append(json.dumps(synonyms, indent=2))

    if few_shot:
        parts.append("\nExamples of well-formed responses:")
        for example in few_shot:
            parts.append(json.dumps({
                "question": example["question"],
                "response": {"sql": example["sql"], "reasoning": example["reasoning"]},
            }, indent=2))

    parts.append(f"\n{SQL_RESPONSE_INSTRUCTIONS}")
    return "\n".join(parts)


def build_sql_retry_prompt(question: str, failed_sql: str, error_message: str) -> str:
    """Phase 1 single-retry prompt, sent back to the LLM once after Phase 2
    (sandbox validation/execution) rejects the first attempt (PROJECT_SPEC.md
    Task B2: retry once, then surface a clean failure message -- no unbounded retry).
    """
    return (
        f'Your previous SQL for the question "{question}" failed validation or '
        f"execution.\nSQL you generated:\n{failed_sql}\n\nError:\n{error_message}\n\n"
        f"Fix the SQL and respond again. {SQL_RESPONSE_INSTRUCTIONS}"
    )


def build_narrative_system_prompt(scope_description: str = "") -> str:
    """Phase 3 (result -> narrative) system prompt.

    `scope_description` is appended only when dashboard filters are active, so an
    answer describes its own scope instead of implying it covered everything.
    """
    prompt = (
        "You are a data analyst writing a short answer for a business dashboard. "
        "You will be given the user's original question, the SQL query that was "
        "run, and the resulting data. Write a concise Markdown-formatted narrative "
        "(2-5 sentences, plus a short bullet list if there are multiple notable "
        "figures) that directly answers the question using only numbers present in "
        "the result. Never invent or estimate a number that isn't in the result. If "
        "the result is empty, say so plainly instead of fabricating an answer."
    )
    if scope_description:
        prompt += (
            " These figures cover only the currently filtered selection "
            f"({scope_description}), not the whole dataset -- make that clear so the "
            "reader does not mistake them for dataset-wide totals."
        )
    return prompt


def build_narrative_user_prompt(question: str, sql: str, result_df: pd.DataFrame) -> str:
    """Phase 3 user-turn content: the question, the SQL that answered it, and the
    result rendered as compact JSON records, capped so a very large result doesn't
    blow the prompt budget -- the narrative only needs representative rows.
    """
    max_rows = 50
    records = result_df.head(max_rows).to_dict(orient="records")
    payload = {
        "question": question,
        "sql": sql,
        "row_count": len(result_df),
        "result_rows": records,
        "truncated": len(result_df) > max_rows,
    }
    return json.dumps(payload, indent=2, default=str)


def build_chart_caption_system_prompt() -> str:
    """System prompt for the one-sentence caption printed under an AI-selected chart
    (Task C3).

    Deliberately distinct from the Phase 3 narrative: that one answers the question in
    2-5 sentences, this one describes what the *chart* shows. The rubric asks for a
    caption "describing what the chart shows", so the prompt forbids the two failure
    modes that would defeat that -- restating the question back, and describing the
    chart type ("this is a bar chart") instead of its content, which is exactly what
    the hardcoded string this replaces used to do.
    """
    return (
        "You write one-sentence captions for charts on a business dashboard. "
        "You will be given the user's question, the chart type, which columns are on "
        "which axis, and the plotted rows. Reply with exactly one sentence, in plain "
        "text, describing what the chart shows about the data -- the shape, the "
        "standout value, or the comparison a reader should take away. "
        "Use only numbers present in the rows; never invent or estimate one. "
        "Do not restate the question, do not say what kind of chart it is, do not use "
        "Markdown, and do not add a second sentence."
    )


def build_chart_caption_user_prompt(
    question: str,
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    series_column: str | None,
    result_df: pd.DataFrame,
) -> str:
    """User-turn content for the chart caption: the question, how the chart is bound
    to columns, and the plotted rows as compact JSON.

    Floats are rounded before they enter the prompt. Task D hit this exact bug -- raw
    values reached the model and came back echoed verbatim as
    "-6.434059163572309 percent" -- and a caption is one sentence, so a 16-digit float
    in it is proportionally far more damaging than in a narrative.

    Capped at 30 rows: a caption summarises, so representative rows are enough, and a
    500-row result would otherwise dominate the prompt budget for one sentence.
    """
    max_rows = 30
    frame = result_df.head(max_rows)
    # Round float columns only, by name. A bare frame.round(2) warns on datetime
    # columns ("obj.round has no effect with datetime ... dtypes"), and a datetime
    # x-axis is one of the two most common shapes reaching this function.
    float_columns = [c for c in frame.columns if pd.api.types.is_float_dtype(frame[c])]
    if float_columns:
        frame = frame.round({c: 2 for c in float_columns})
    records = frame.to_dict(orient="records")
    payload = {
        "question": question,
        "chart_type": chart_type,
        "x_axis": x_column,
        "y_axis": y_column,
        "series": series_column,
        "row_count": len(result_df),
        "plotted_rows": records,
        "truncated": len(result_df) > max_rows,
    }
    return json.dumps(payload, indent=2, default=str)


def build_conversation_context(history: list[dict[str, Any]]) -> str:
    """Render up to the last 5 (question, sql, answer) turns as prompt text for
    follow-up questions (Task B4). `history` is newest-last; memory.py owns
    storage/state, this function only owns formatting -- returns "" for no history
    so callers can unconditionally append it without a branch.
    """
    if not history:
        return ""
    lines = [
        "Recent conversation history (most recent last), for resolving follow-up "
        "references like 'that' or 'those':"
    ]
    for turn in history[-5:]:
        lines.append(f"- Q: {turn['question']}")
        lines.append(f"  SQL: {turn['sql']}")
        lines.append(f"  A: {turn['answer']}")
    return "\n".join(lines)
