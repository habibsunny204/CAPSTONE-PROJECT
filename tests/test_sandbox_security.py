"""Exploit-attempt rejection tests for llm/sandbox.py (Task B2,
PROJECT_SPEC.md Section 7). Every rejection case in the spec is covered, plus a few
additional attempts (comment-hidden statements, disallowed table references, unknown
table functions) that the allowlist design should catch even though they aren't
listed verbatim in the spec.
"""

import duckdb
import pytest

from llm.sandbox import SandboxViolation, execute_safe, validate_sql

TABLE_NAME = "ecommerce_sales"

# (label, sql) -- every one of these must be rejected.
EXPLOIT_ATTEMPTS = [
    ("create_table", "CREATE TABLE evil (a INT)"),
    ("drop_table", "DROP TABLE ecommerce_sales"),
    ("alter_table", "ALTER TABLE ecommerce_sales ADD COLUMN evil INT"),
    ("insert", "INSERT INTO ecommerce_sales (product) VALUES (999999)"),
    ("update", "UPDATE ecommerce_sales SET total_revenue = 0"),
    ("delete", "DELETE FROM ecommerce_sales"),
    ("attach", "ATTACH 'evil.db' AS evil"),
    ("detach", "DETACH ecommerce_sales"),
    ("pragma", "PRAGMA table_info('ecommerce_sales')"),
    ("install", "INSTALL httpfs"),
    ("load", "LOAD httpfs"),
    ("copy_to", "COPY ecommerce_sales TO 'exfiltrated.csv'"),
    ("copy_from", "COPY ecommerce_sales FROM 'evil.csv'"),
    ("export_database", "EXPORT DATABASE 'out'"),
    ("multiple_statements", "SELECT * FROM ecommerce_sales; DROP TABLE ecommerce_sales"),
    ("multiple_statements_select_only", "SELECT * FROM ecommerce_sales; SELECT * FROM ecommerce_sales"),
    ("read_csv_outside_table", "SELECT * FROM read_csv('/etc/passwd')"),
    ("read_csv_auto_outside_table", "SELECT * FROM read_csv_auto('/etc/passwd')"),
    ("read_parquet_outside_table", "SELECT * FROM read_parquet('secrets.parquet')"),
    ("read_json_outside_table", "SELECT * FROM read_json_auto('secrets.json')"),
    ("bare_file_path", "SELECT * FROM 'secrets.csv'"),
    ("unknown_table", "SELECT * FROM some_other_table"),
    ("joined_unknown_table", "SELECT s.* FROM ecommerce_sales s JOIN secrets sec ON s.product = sec.id"),
    ("subquery_unknown_table", "SELECT * FROM ecommerce_sales WHERE product IN (SELECT id FROM secrets)"),
    ("pragma_table_function", "SELECT * FROM pragma_table_info('ecommerce_sales')"),
    ("information_schema", "SELECT * FROM information_schema.tables"),
    # Qualified references to the real table. These are not merely redundant spellings
    # -- the dashboard scopes the LLM's connection to the sidebar filters by resolving
    # the bare name through search_path to a filtered view (backend/scope.py), so a
    # qualified name would read around the filter to every row in the table.
    ("schema_qualified_table", "SELECT * FROM main.ecommerce_sales"),
    ("catalog_qualified_table", "SELECT * FROM memory.main.ecommerce_sales"),
    ("qualified_in_subquery",
     "SELECT * FROM ecommerce_sales WHERE region IN (SELECT region FROM main.ecommerce_sales)"),
    ("qualified_in_cte",
     "WITH r AS (SELECT * FROM main.ecommerce_sales) SELECT COUNT(*) FROM r"),
    ("qualified_in_join",
     "SELECT a.* FROM ecommerce_sales a JOIN main.ecommerce_sales b ON a.product = b.product"),
    ("empty_string", ""),
    ("whitespace_only", "   \n\t  "),
    ("garbage", "not even sql at all !!! ???"),
    ("call_statement", "CALL some_procedure()"),
    ("vacuum", "VACUUM ecommerce_sales"),
]


@pytest.mark.parametrize("label,sql", EXPLOIT_ATTEMPTS, ids=[a[0] for a in EXPLOIT_ATTEMPTS])
def test_validate_sql_rejects_exploit_attempts(label, sql):
    with pytest.raises(SandboxViolation):
        validate_sql(sql, TABLE_NAME)


def test_validate_sql_accepts_plain_select():
    stmt = validate_sql("SELECT region, total_revenue FROM ecommerce_sales WHERE region = 'Asia'", TABLE_NAME)
    assert stmt is not None


def test_validate_sql_accepts_select_with_cte():
    sql = (
        "WITH regional AS (SELECT region, SUM(total_revenue) AS total FROM ecommerce_sales GROUP BY region) "
        "SELECT * FROM regional WHERE total > 0"
    )
    stmt = validate_sql(sql, TABLE_NAME)
    assert stmt is not None


def test_validate_sql_accepts_union_of_selects():
    sql = "SELECT region FROM ecommerce_sales UNION SELECT region FROM ecommerce_sales"
    stmt = validate_sql(sql, TABLE_NAME)
    assert stmt is not None


def test_validate_sql_treats_sql_inside_comments_as_inert():
    """A DROP hidden inside a /* comment */ is not a bypass -- comments are never
    parsed as executable statements, so this remains a plain, safe SELECT. This
    confirms comment-based statement smuggling doesn't work against validate_sql().
    """
    sql = "SELECT * FROM ecommerce_sales /* ; DROP TABLE ecommerce_sales; */ WHERE 1=1"
    stmt = validate_sql(sql, TABLE_NAME)
    assert stmt is not None


def test_validate_sql_case_insensitive_table_match():
    """The loaded table name comparison is case-insensitive (SQL identifiers
    normalize this way in DuckDB by default).
    """
    stmt = validate_sql("SELECT * FROM ECOMMERCE_SALES", TABLE_NAME)
    assert stmt is not None


def test_execute_safe_runs_a_valid_query(mini_con):
    df = execute_safe(mini_con, 'SELECT COUNT(*) AS n FROM "ecommerce_sales"', TABLE_NAME)
    assert df["n"].iloc[0] == 16


def test_execute_safe_blocks_exploit_and_leaves_table_untouched(mini_con):
    before = mini_con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]

    with pytest.raises(SandboxViolation):
        execute_safe(mini_con, "DELETE FROM ecommerce_sales", TABLE_NAME)

    after = mini_con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
    assert after == before
