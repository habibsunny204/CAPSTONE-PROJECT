"""SQL validation and safe execution sandbox (Task B2).

This is what satisfies PROJECT_SPEC.md's sandbox requirement by construction rather
than by trying to police an arbitrary exec(): the LLM only ever produces a SQL
string, and that string is parsed with sqlglot and validated before anything touches
a real connection.

Two layers of defense, both required:
  1. Top-level statement type is an *allowlist* (SELECT, or a UNION/INTERSECT/EXCEPT
     of SELECTs -- still pure reads, no mutation, so treated as "a single SELECT" in
     spirit), not a denylist of forbidden statement types. Any statement type not on
     the allowlist is rejected by default, including ones this module's author never
     thought to enumerate (EXPLAIN, SET, CALL, MERGE, VACUUM, ...) and anything
     sqlglot can't parse and falls back to a generic "Command" node for (e.g. `LOAD`
     in this sqlglot version).
  2. Every table reference anywhere in the query (including inside subqueries) must
     resolve to exactly the one loaded table or a CTE defined within the query
     itself. This is also an allowlist, not a blocklist of dangerous function names
     (`read_csv`, `read_parquet`, `pragma_table_info`, ...) -- DuckDB can add new
     table functions in a future version, and a blocklist would need updating for
     each one; an allowlist of "the one table we loaded" doesn't.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

_ALLOWED_TOP_LEVEL = (exp.Select, exp.Union, exp.Intersect, exp.Except)


class SandboxViolation(Exception):
    """Raised when LLM-generated SQL fails a sandbox safety check."""


def validate_sql(sql: str, table_name: str, dialect: str = "duckdb") -> exp.Expression:
    """Parse and validate `sql` against PROJECT_SPEC.md Section 7's rejection list.
    Returns the parsed expression on success. Raises SandboxViolation on any
    violation: not exactly one statement, not a SELECT-family statement, an
    unrecognized/vendor statement fragment (ATTACH/DETACH/PRAGMA/INSTALL/LOAD/COPY/
    EXPORT and any DDL/DML all fall into this bucket), or any table reference other
    than `table_name` itself or a CTE the query defines.
    """
    if not sql or not sql.strip():
        raise SandboxViolation("Empty SQL.")

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SandboxViolation(f"SQL failed to parse: {e}") from e

    if len(statements) != 1:
        raise SandboxViolation(f"Expected exactly one statement, found {len(statements)}.")

    stmt = statements[0]
    if not isinstance(stmt, _ALLOWED_TOP_LEVEL):
        raise SandboxViolation(f"Only SELECT statements are allowed, got {type(stmt).__name__}.")

    allowed_tables = {table_name.lower()}
    for cte in stmt.find_all(exp.CTE):
        if cte.alias:
            allowed_tables.add(cte.alias.lower())

    for node in stmt.walk():
        if isinstance(node, exp.Command):
            raise SandboxViolation(f"Disallowed/unrecognized statement fragment: {node.sql()!r}")
        if isinstance(node, exp.Table):
            ref_name = (node.name or "").lower()
            if ref_name not in allowed_tables:
                raise SandboxViolation(f"Query references a disallowed table or function: {node.sql()!r}")

    return stmt


def execute_safe(con: duckdb.DuckDBPyConnection, sql: str, table_name: str) -> pd.DataFrame:
    """Validate `sql` against `table_name`, then execute the *validated, re-rendered*
    statement (never the original raw string, so what was validated is guaranteed to
    be what runs) and return the result as a DataFrame.

    `con` should be a connection/cursor used exclusively for LLM-generated SQL,
    logically separate from the connection ingest.py/quality.py use to build and
    clean the table. DuckDB has no read-only mode for in-memory databases to enforce
    that structurally, so validate_sql()'s allowlists -- which reject every DDL/DML/
    multi-statement/file-read path -- are what actually make this "read-only" in
    effect.
    """
    validated = validate_sql(sql, table_name)
    return con.execute(validated.sql(dialect="duckdb")).df()
