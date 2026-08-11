"""Generic, schema-driven query engine: groupby_agg() and filtered_query() (Task A2).

Every identifier (table name, dims, metrics, filter columns) is caller-supplied data,
validated against the table's live schema at call time and safely quoted -- there are
no dataset column names as literals anywhere in this file, per PROJECT_SPEC.md
Section 2's generic/specific boundary. Filter values are always bound as DuckDB query
parameters, never string-interpolated.
"""

from __future__ import annotations

import time
from typing import Any

import duckdb
import pandas as pd

_ALLOWED_OPS = {"=", "!=", "<", ">", "<=", ">=", "in", "between", "like", "is_null", "is_not_null"}

_AGG_TEMPLATES = {
    "sum": "SUM({col})",
    "avg": "AVG({col})",
    "min": "MIN({col})",
    "max": "MAX({col})",
    "count": "COUNT({col})",
    "count_distinct": "COUNT(DISTINCT {col})",
    "median": "MEDIAN({col})",
    "stddev": "STDDEV({col})",
}


def _quote_ident(name: str) -> str:
    """Double-quote a DuckDB identifier, escaping embedded quotes."""
    return '"' + name.replace('"', '""') + '"'


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    """The live set of column names on `table_name`, used to validate caller-supplied
    dims/metrics/filter columns before any SQL is built.
    """
    rows = con.execute(f"PRAGMA table_info({_quote_ident(table_name)})").fetchall()
    return {row[1] for row in rows}


def _validate_columns(columns: list[str], known_columns: set[str]) -> None:
    """Raise ValueError on the first caller-supplied column that isn't on the table."""
    for col in columns:
        if col not in known_columns:
            raise ValueError(f"Unknown column: {col!r}")


def _agg_expr(agg: str, column: str) -> str:
    """Build a single `AGG("column")` SQL expression, validating `agg` against the
    allow-list first.
    """
    if agg not in _AGG_TEMPLATES:
        raise ValueError(f"Unsupported aggregation: {agg!r}")
    return _AGG_TEMPLATES[agg].format(col=_quote_ident(column))


def _filter_clause(
    filters: list[dict[str, Any]],
    known_columns: set[str],
    params: list[Any],
) -> str:
    """Build a ` WHERE ...` clause (implicit AND) from a list of
    {"column", "op", "value"} filter dicts, appending bound values to `params`.
    Returns an empty string if `filters` is empty.
    """
    if not filters:
        return ""

    clauses = []
    for f in filters:
        col, op = f["column"], f["op"]
        if col not in known_columns:
            raise ValueError(f"Unknown column in filter: {col!r}")
        if op not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported filter op: {op!r}")

        quoted = _quote_ident(col)
        if op == "is_null":
            clauses.append(f"{quoted} IS NULL")
        elif op == "is_not_null":
            clauses.append(f"{quoted} IS NOT NULL")
        elif op == "between":
            lo, hi = f["value"]
            clauses.append(f"{quoted} BETWEEN ? AND ?")
            params.extend([lo, hi])
        elif op == "in":
            values = f["value"]
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"{quoted} IN ({placeholders})")
            params.extend(values)
        else:
            clauses.append(f"{quoted} {op} ?")
            params.append(f["value"])

    return " WHERE " + " AND ".join(clauses)


def groupby_agg(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    dims: list[str],
    metrics: list[str],
    aggs: list[str],
    filters: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Group `table_name` by `dims`, aggregating each of `metrics` with each of
    `aggs` pairwise (output column `"{metric}_{agg}"`), optionally filtered by
    `filters` (a list of {"column", "op", "value"} dicts, combined with AND). This is
    what makes "filtered aggregation" (PROJECT_SPEC.md Task A4) a single reusable
    call rather than composing two separately-spec'd functions.

    Returns (result DataFrame, elapsed_ms), where elapsed_ms times only the DuckDB
    execution itself, not SQL construction/validation.
    """
    known_columns = _table_columns(con, table_name)
    _validate_columns([*dims, *metrics], known_columns)

    select_parts = [_quote_ident(d) for d in dims]
    for metric in metrics:
        for agg in aggs:
            alias = _quote_ident(f"{metric}_{agg}")
            select_parts.append(f"{_agg_expr(agg, metric)} AS {alias}")

    params: list[Any] = []
    where_clause = _filter_clause(filters or [], known_columns, params)
    group_by_clause = f" GROUP BY {', '.join(_quote_ident(d) for d in dims)}" if dims else ""

    sql = (
        f"SELECT {', '.join(select_parts)} FROM {_quote_ident(table_name)}"
        f"{where_clause}{group_by_clause}"
    )

    start = time.perf_counter()
    df = con.execute(sql, params).df()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return df, elapsed_ms


def filtered_query(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    filters: list[dict[str, Any]],
    dims: list[str],
    metrics: list[str],
) -> tuple[pd.DataFrame, float]:
    """Row-level SELECT of `dims + metrics` from `table_name` matching `filters` --
    no aggregation. Returns (result DataFrame, elapsed_ms).
    """
    known_columns = _table_columns(con, table_name)
    columns = [*dims, *metrics]
    _validate_columns(columns, known_columns)

    params: list[Any] = []
    where_clause = _filter_clause(filters or [], known_columns, params)
    select_clause = ", ".join(_quote_ident(c) for c in columns)

    sql = f"SELECT {select_clause} FROM {_quote_ident(table_name)}{where_clause}"

    start = time.perf_counter()
    df = con.execute(sql, params).df()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return df, elapsed_ms
