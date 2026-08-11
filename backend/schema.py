"""Schema introspection: produces the per-column JSON contract consumed by every LLM
prompt in Task B (Task A1).

Fully generic: iterates over whatever columns the target table currently has via
DuckDB's own introspection (`PRAGMA table_info`), never a hardcoded column list. This
means the output naturally reflects post-cleaning state once backend/quality.py's
clean() has run (parsed date dtypes, derived calendar-part columns present, any
degenerate columns dropped) — that cleaned state is what Task B's prompts actually see.

Note on the PII gate: this module only controls what appears in the *JSON contract*.
The actual hard gate — raw PII never being queryable at all — is enforced earlier, in
backend/ingest.py, by never persisting dropped/unhashed columns into the DuckDB table
in the first place. A column merely excluded from this JSON but still physically
present in the table would still be reachable by a hallucinated raw SQL SELECT, since
Task B's sandbox.py validates SQL structure, not a column allowlist.
"""

from __future__ import annotations

import datetime
import decimal
from typing import Any

import duckdb


def _json_safe(value: Any) -> Any:
    """Convert a DuckDB-returned value into something JSON-serializable, since the
    schema dict is a public contract sent straight into an LLM prompt.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    return str(value)


def get_schema(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    sample_limit: int = 5,
) -> list[dict[str, Any]]:
    """Introspect `table_name` and return one JSON-serializable dict per column:
    `{name, dtype, n_unique, n_null, sample_values}` (sample_values capped at
    `sample_limit`). This exact shape is the public contract Task B injects into
    every LLM prompt — do not change field names without updating Task B.
    """
    columns_info = con.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    total_rows = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

    schema: list[dict[str, Any]] = []
    for col in columns_info:
        name, dtype = col[1], col[2]

        n_unique = con.execute(
            f'SELECT COUNT(DISTINCT "{name}") FROM "{table_name}"'
        ).fetchone()[0]
        n_non_null = con.execute(
            f'SELECT COUNT("{name}") FROM "{table_name}"'
        ).fetchone()[0]
        n_null = total_rows - n_non_null

        sample_rows = con.execute(
            f'SELECT DISTINCT "{name}" FROM "{table_name}" '
            f'WHERE "{name}" IS NOT NULL LIMIT {sample_limit}'
        ).fetchall()
        sample_values = [_json_safe(row[0]) for row in sample_rows]

        schema.append(
            {
                "name": name,
                "dtype": dtype,
                "n_unique": n_unique,
                "n_null": n_null,
                "sample_values": sample_values,
            }
        )

    return schema
