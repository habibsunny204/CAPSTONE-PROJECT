"""Exploit-attempt rejection tests for llm/sandbox.py (Task B2,
PROJECT_SPEC.md Section 7). Every rejection case in the spec is covered, plus a few
additional attempts (comment-hidden statements, disallowed table references, unknown
table functions) that the allowlist design should catch even though they aren't
listed verbatim in the spec.
"""

import duckdb
import pytest

from llm.sandbox import SandboxViolation, execute_safe, validate_sql

TABLE_NAME = "superstore"

# (label, sql) -- every one of these must be rejected.
EXPLOIT_ATTEMPTS = [
    ("create_table", "CREATE TABLE evil (a INT)"),
    ("drop_table", "DROP TABLE superstore"),
    ("alter_table", "ALTER TABLE superstore ADD COLUMN evil INT"),
    ("insert", "INSERT INTO superstore (row_id) VALUES (999999)"),
    ("update", "UPDATE superstore SET sales = 0"),
    ("delete", "DELETE FROM superstore"),
    ("attach", "ATTACH 'evil.db' AS evil"),
    ("detach", "DETACH superstore"),
    ("pragma", "PRAGMA table_info('superstore')"),
    ("install", "INSTALL httpfs"),
    ("load", "LOAD httpfs"),
    ("copy_to", "COPY superstore TO 'exfiltrated.csv'"),
    ("copy_from", "COPY superstore FROM 'evil.csv'"),
    ("export_database", "EXPORT DATABASE 'out'"),
    ("multiple_statements", "SELECT * FROM superstore; DROP TABLE superstore"),
    ("multiple_statements_select_only", "SELECT * FROM superstore; SELECT * FROM superstore"),
    ("read_csv_outside_table", "SELECT * FROM read_csv('/etc/passwd')"),
    ("read_csv_auto_outside_table", "SELECT * FROM read_csv_auto('/etc/passwd')"),
    ("read_parquet_outside_table", "SELECT * FROM read_parquet('secrets.parquet')"),
    ("read_json_outside_table", "SELECT * FROM read_json_auto('secrets.json')"),
    ("bare_file_path", "SELECT * FROM 'secrets.csv'"),
    ("unknown_table", "SELECT * FROM some_other_table"),
    ("joined_unknown_table", "SELECT s.* FROM superstore s JOIN secrets sec ON s.row_id = sec.id"),
    ("subquery_unknown_table", "SELECT * FROM superstore WHERE row_id IN (SELECT id FROM secrets)"),
    ("pragma_table_function", "SELECT * FROM pragma_table_info('superstore')"),
    ("information_schema", "SELECT * FROM information_schema.tables"),
    ("empty_string", ""),
    ("whitespace_only", "   \n\t  "),
    ("garbage", "not even sql at all !!! ???"),
    ("call_statement", "CALL some_procedure()"),
    ("vacuum", "VACUUM superstore"),
]


@pytest.mark.parametrize("label,sql", EXPLOIT_ATTEMPTS, ids=[a[0] for a in EXPLOIT_ATTEMPTS])
def test_validate_sql_rejects_exploit_attempts(label, sql):
    with pytest.raises(SandboxViolation):
        validate_sql(sql, TABLE_NAME)


def test_validate_sql_accepts_plain_select():
    stmt = validate_sql("SELECT region, sales FROM superstore WHERE region = 'West'", TABLE_NAME)
    assert stmt is not None


def test_validate_sql_accepts_select_with_cte():
    sql = (
        "WITH regional AS (SELECT region, SUM(sales) AS total FROM superstore GROUP BY region) "
        "SELECT * FROM regional WHERE total > 0"
    )
    stmt = validate_sql(sql, TABLE_NAME)
    assert stmt is not None


def test_validate_sql_accepts_union_of_selects():
    sql = "SELECT region FROM superstore UNION SELECT region FROM superstore"
    stmt = validate_sql(sql, TABLE_NAME)
    assert stmt is not None


def test_validate_sql_treats_sql_inside_comments_as_inert():
    """A DROP hidden inside a /* comment */ is not a bypass -- comments are never
    parsed as executable statements, so this remains a plain, safe SELECT. This
    confirms comment-based statement smuggling doesn't work against validate_sql().
    """
    sql = "SELECT * FROM superstore /* ; DROP TABLE superstore; */ WHERE 1=1"
    stmt = validate_sql(sql, TABLE_NAME)
    assert stmt is not None


def test_validate_sql_case_insensitive_table_match():
    """The loaded table name comparison is case-insensitive (SQL identifiers
    normalize this way in DuckDB by default).
    """
    stmt = validate_sql("SELECT * FROM SUPERSTORE", TABLE_NAME)
    assert stmt is not None


def test_execute_safe_runs_a_valid_query(mini_con):
    df = execute_safe(mini_con, 'SELECT COUNT(*) AS n FROM "superstore"', TABLE_NAME)
    assert df["n"].iloc[0] == 16


def test_execute_safe_blocks_exploit_and_leaves_table_untouched(mini_con):
    before = mini_con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]

    with pytest.raises(SandboxViolation):
        execute_safe(mini_con, "DELETE FROM superstore", TABLE_NAME)

    after = mini_con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
    assert after == before
