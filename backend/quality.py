"""Data quality profiling and cleaning: missing %, duplicate count, IQR-based outlier
detection, and real cleaning steps (Task A3).

Both public functions are pure(-ish) -- profile_report() only reads, clean() mutates
`table_name` in place and returns a report of what changed -- with JSON-serializable
output and no Streamlit import, so Task C's UI can call them directly and render the
results. Column names driving both functions come from configs/dataset_config.yaml
(the `quality` section: date_columns, categorical_casing_columns,
drop_after_profiling), never as literals here, per PROJECT_SPEC.md Section 2's
generic/specific boundary.

Startup order this module assumes: ingest.load() -> quality.clean() -> the table is
analysis-ready -> schema.get_schema() reflects that cleaned state (parsed dates,
degenerate columns gone). That cleaned state is what Task B's prompts actually see.

clean() runs direct UPDATE/ALTER statements against the table. This is fine: Section
7's DDL/DML ban applies specifically to llm/sandbox.py's LLM-generated-SQL execution
path in Task B, not to this internal, non-LLM-driven data-prep code.
"""

from __future__ import annotations

import datetime
from typing import Any

import duckdb

_NUMERIC_DTYPE_PREFIXES = (
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "DOUBLE", "DECIMAL", "REAL",
)


def _q(name: str) -> str:
    """Double-quote a DuckDB identifier, escaping embedded quotes."""
    return '"' + name.replace('"', '""') + '"'


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """Ordered column names for `table_name`, via DuckDB's own introspection."""
    rows = con.execute(f"PRAGMA table_info({_q(table_name)})").fetchall()
    return [row[1] for row in rows]


def _is_numeric_dtype(dtype: str) -> bool:
    return dtype.upper().startswith(_NUMERIC_DTYPE_PREFIXES)


def profile_report(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Profile `table_name`: row/duplicate counts and, per column, missing % plus
    (for numeric columns only) an IQR-based outlier count and bounds. `id_column`,
    if given, is excluded from duplicate-row detection -- a surrogate key is always
    unique by construction, so including it would make that metric trivially 0.
    """
    total_rows = con.execute(f"SELECT COUNT(*) FROM {_q(table_name)}").fetchone()[0]
    all_columns = _table_columns(con, table_name)

    dedup_columns = [c for c in all_columns if c != id_column]
    if dedup_columns and total_rows:
        distinct_rows = con.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT "
            f"{', '.join(_q(c) for c in dedup_columns)} FROM {_q(table_name)})"
        ).fetchone()[0]
        n_duplicate_rows = total_rows - distinct_rows
    else:
        n_duplicate_rows = 0

    columns_report = []
    columns_info = con.execute(f"PRAGMA table_info({_q(table_name)})").fetchall()
    for _, name, dtype, *_rest in columns_info:
        n_non_null = con.execute(f"SELECT COUNT({_q(name)}) FROM {_q(table_name)}").fetchone()[0]
        missing_pct = round((total_rows - n_non_null) / total_rows * 100, 2) if total_rows else 0.0

        n_outliers = None
        iqr_bounds = None
        if _is_numeric_dtype(dtype):
            q1, q3 = con.execute(
                f"SELECT quantile_cont({_q(name)}, 0.25), quantile_cont({_q(name)}, 0.75) "
                f"FROM {_q(table_name)}"
            ).fetchone()
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_outliers = con.execute(
                    f"SELECT COUNT(*) FROM {_q(table_name)} "
                    f"WHERE {_q(name)} < ? OR {_q(name)} > ?",
                    [lower, upper],
                ).fetchone()[0]
                iqr_bounds = {"q1": q1, "q3": q3, "lower": lower, "upper": upper}

        columns_report.append({
            "name": name,
            "dtype": dtype,
            "missing_pct": missing_pct,
            "n_outliers_iqr": n_outliers,
            "iqr_bounds": iqr_bounds,
        })

    return {
        "table": table_name,
        "n_rows": total_rows,
        "n_duplicate_rows": n_duplicate_rows,
        "columns": columns_report,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def clean(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply real cleaning steps to `table_name` in place, driven by `config`'s
    `quality` section: (1) parse date columns from string to TIMESTAMP, (2)
    standardize categorical casing (trim + title-case) on configured low-cardinality
    string columns, (3) drop columns flagged as degenerate during profiling. Returns
    a report of what changed.
    """
    quality_cfg = config["quality"]
    total_rows = con.execute(f"SELECT COUNT(*) FROM {_q(table_name)}").fetchone()[0]
    steps: list[dict[str, Any]] = []

    # 1. Parse date columns to TIMESTAMP -- needed for correct date-range SQL later.
    existing = set(_table_columns(con, table_name))
    parsed_columns = []
    for col in quality_cfg.get("date_columns", []):
        if col in existing:
            con.execute(
                f"ALTER TABLE {_q(table_name)} ALTER COLUMN {_q(col)} "
                f"TYPE TIMESTAMP USING CAST({_q(col)} AS TIMESTAMP)"
            )
            parsed_columns.append(col)
    if parsed_columns:
        steps.append({"name": "parse_dates", "columns": parsed_columns, "rows_affected": total_rows})

    # 2. Standardize categorical casing (trim + title-case). DuckDB has no built-in
    # title-case function, so the canonical form is computed in Python per distinct
    # value and written back with a targeted UPDATE. On the real dataset this is a
    # defensive no-op -- a direct scan found the checked columns already
    # consistently cased -- but it's a genuine fix for any future/messier import.
    casing_changed: dict[str, int] = {}
    for col in quality_cfg.get("categorical_casing_columns", []):
        if col not in existing:
            continue
        distinct_values = con.execute(
            f"SELECT {_q(col)}, COUNT(*) FROM {_q(table_name)} "
            f"WHERE {_q(col)} IS NOT NULL GROUP BY {_q(col)}"
        ).fetchall()
        n_changed = 0
        for raw_value, count in distinct_values:
            canonical = raw_value.strip().title()
            if canonical != raw_value:
                con.execute(
                    f"UPDATE {_q(table_name)} SET {_q(col)} = ? WHERE {_q(col)} = ?",
                    [canonical, raw_value],
                )
                n_changed += count
        if n_changed:
            casing_changed[col] = n_changed
    steps.append({
        "name": "standardize_categorical_casing",
        "columns": list(casing_changed.keys()),
        "rows_changed": sum(casing_changed.values()),
    })

    # 3. Drop degenerate/junk columns identified during A3 profiling (e.g. the
    # constant-value record_count column) -- a documented structural cleaning
    # decision, not silently dropped.
    existing = set(_table_columns(con, table_name))
    dropped = [c for c in quality_cfg.get("drop_after_profiling", []) if c in existing]
    for col in dropped:
        con.execute(f"ALTER TABLE {_q(table_name)} DROP COLUMN {_q(col)}")
    if dropped:
        steps.append({"name": "drop_degenerate_columns", "columns_dropped": dropped})

    return {"table": table_name, "steps_applied": steps}
