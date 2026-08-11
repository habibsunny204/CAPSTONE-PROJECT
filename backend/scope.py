"""Filter scope: one predicate spec, rendered two ways (pandas and SQL).

The dashboard's sidebar filters have to constrain two very different consumers: the
chart tabs, which hold a pandas DataFrame, and every LLM/feature path, which holds a
DuckDB connection and a table name and never sees a DataFrame at all. Rendering the
same predicate twice invites the two from drifting apart, so both renderings are driven
from one canonical spec built once in the sidebar:

    [{"column": "region", "op": "in", "value": ["Europe", "Asia"]}, ...]

That is deliberately the exact shape backend/query_engine.py already accepts, so a
scope spec can also be passed straight to groupby_agg()/filtered_query() as filters.
tests/test_scope.py asserts the two renderings select the same rows.

HOW THE SQL SIDE REACHES THE LLM
The LLM path is scoped by pointing it at a *view* that carries the predicate, created
by build_scoped_view() in a per-session schema and named identically to the base table.
Keeping the name identical is what makes this cheap: `table_name` never changes, so the
prompts, the config few-shot examples (which contain literal `FROM <table>` SQL), and
llm/sandbox.py's single-table allowlist all keep working untouched. The model writes an
ordinary unqualified `FROM <table>` and DuckDB's search_path resolves it to the scoped
view.

That resolution is also why llm/sandbox.py rejects schema-qualified table references:
a bare name is scoped, but `main.<table>` would reach around the view to the full table.

Generic by construction -- no dataset column names appear here; everything is driven by
the caller-supplied spec.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd
from sqlglot import exp

# Kept deliberately narrower than query_engine._ALLOWED_OPS: these are the only two
# operators the sidebar widgets can produce (a date range and a multi-select), and a
# scope predicate is built by this app, never by a user or a model. Widening this set
# means widening what build_scoped_view() renders into DDL, so it should be a conscious
# change rather than an inherited one.
_SUPPORTED_OPS = {"in", "between"}


def is_active(spec: list[dict[str, Any]] | None) -> bool:
    """Whether `spec` narrows anything at all."""
    return bool(spec)


def apply_to_frame(df: pd.DataFrame, spec: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Apply a scope spec to a DataFrame -- the rendering the chart tabs consume.

    Columns absent from `df` are skipped rather than raising, matching how the sidebar
    already skips filters whose column isn't present.
    """
    if not spec:
        return df

    scoped = df
    for item in spec:
        column, op = item["column"], item["op"]
        if column not in scoped.columns:
            continue
        if op == "in":
            scoped = scoped[scoped[column].isin(item["value"])]
        elif op == "between":
            low, high = item["value"]
            scoped = scoped[(scoped[column] >= low) & (scoped[column] <= high)]
        else:
            raise ValueError(f"Unsupported scope op: {op!r}")
    return scoped


def _literal(value: Any) -> str:
    """Render a Python value as a SQL literal.

    Rendered through sqlglot -- the same parser llm/sandbox.py already trusts to
    validate model output -- rather than f-string interpolation. A view body cannot
    take bound parameters (DuckDB rejects `?` in DDL), so these values do get inlined,
    and a category value containing an apostrophe must not be able to terminate the
    string early. sqlglot handles the escaping; tests/test_scope.py proves it with an
    adversarial value.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return exp.Literal.number(value).sql(dialect="duckdb")
    if isinstance(value, pd.Timestamp):
        value = value.isoformat(sep=" ")
    return exp.Literal.string(str(value)).sql(dialect="duckdb")


def _quote_ident(name: str) -> str:
    """Double-quote a DuckDB identifier, escaping embedded quotes."""
    return '"' + name.replace('"', '""') + '"'


def predicate_sql(spec: list[dict[str, Any]] | None) -> str:
    """Render a scope spec as a SQL boolean expression (no leading WHERE).

    Returns "" when the spec is empty, so callers can build an unfiltered statement
    without special-casing.
    """
    if not spec:
        return ""

    clauses = []
    for item in spec:
        column, op = item["column"], item["op"]
        if op not in _SUPPORTED_OPS:
            raise ValueError(f"Unsupported scope op: {op!r}")

        quoted = _quote_ident(column)
        if op == "in":
            values = ", ".join(_literal(v) for v in item["value"])
            clauses.append(f"{quoted} IN ({values})")
        else:
            low, high = item["value"]
            clauses.append(f"{quoted} BETWEEN {_literal(low)} AND {_literal(high)}")

    return " AND ".join(clauses)


def describe(spec: list[dict[str, Any]] | None, labels: dict[str, str] | None = None) -> str:
    """One-line human summary of the scope, for prompts and UI captions.

    Returns "" when nothing is filtered, which is what lets callers omit the line
    entirely rather than telling an LLM "no filters are active" -- with no filters the
    prompt is then byte-identical to the un-scoped one the benchmark measured.
    """
    if not spec:
        return ""

    labels = labels or {}
    parts = []
    for item in spec:
        name = labels.get(item["column"], item["column"].replace("_", " "))
        if item["op"] == "in":
            parts.append(f"{name} is one of [{', '.join(_readable(v) for v in item['value'])}]")
        else:
            low, high = item["value"]
            parts.append(f"{name} is between {_readable(low)} and {_readable(high)}")
    return "; ".join(parts)


def _readable(value: Any) -> str:
    """Format a filter value for human display -- midnight timestamps render as plain
    dates, since a date-range filter showing "00:00:00" reads as spurious precision.
    """
    if isinstance(value, pd.Timestamp) and (value.hour, value.minute, value.second) == (0, 0, 0):
        return value.date().isoformat()
    return str(value)


def build_scoped_view(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    schema_name: str,
    spec: list[dict[str, Any]] | None,
    source_schema: str = "main",
) -> None:
    """Create/replace a view of `table_name` carrying the scope predicate, inside
    `schema_name` and under the *same name* as the base table.

    `schema_name` should be unique per browser session: the DuckDB connection is cached
    per server process and shared across sessions, so a fixed schema name would let one
    user's filters silently rewrite another user's results.

    DDL here is intentional and safe: this runs from application code with an
    app-authored predicate, never from model output. llm/sandbox.py's DDL ban applies to
    LLM-generated SQL, which is a different execution path (PROJECT_SPEC.md Section 7).
    """
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_ident(schema_name)}")

    predicate = predicate_sql(spec)
    where = f" WHERE {predicate}" if predicate else ""
    con.execute(
        f"CREATE OR REPLACE VIEW {_quote_ident(schema_name)}.{_quote_ident(table_name)} AS "
        f"SELECT * FROM {_quote_ident(source_schema)}.{_quote_ident(table_name)}{where}"
    )


def scoped_cursor(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    schema_name: str,
    spec: list[dict[str, Any]] | None,
) -> duckdb.DuckDBPyConnection:
    """Build the scoped view and return a cursor whose bare-name lookups resolve to it.

    `search_path` is set per-cursor (verified: a sibling cursor on the same connection
    still sees the full table), so this scopes the LLM/feature path without touching the
    connection the charts and the quality profile read from.
    """
    build_scoped_view(con, table_name, schema_name, spec)
    cursor = con.cursor()
    cursor.execute(f"SET search_path={_literal(schema_name)}")
    return cursor
