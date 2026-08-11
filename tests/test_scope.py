"""Tests for backend/scope.py -- the sidebar filters' two renderings.

The central risk this module addresses is divergence: the chart tabs filter a pandas
DataFrame while every LLM/Task-D path filters through a DuckDB view, and if those two
ever describe different row sets the dashboard silently contradicts itself. The
equivalence test below is the one that matters most; the rest guard the pieces it
depends on.
"""

import pandas as pd
import pytest

from backend import ingest, quality, scope
from llm.sandbox import SandboxViolation, execute_safe, validate_sql

TABLE_NAME = "ecommerce_sales"


# ---------------------------------------------------------------------------
# Spec rendering
# ---------------------------------------------------------------------------


def test_empty_spec_is_inactive_and_renders_nothing():
    """An empty scope must produce no predicate and no description, so callers can
    build unfiltered SQL and un-caveated prompts without special-casing.
    """
    for empty in ([], None):
        assert scope.is_active(empty) is False
        assert scope.predicate_sql(empty) == ""
        assert scope.describe(empty) == ""


def test_predicate_sql_renders_both_widget_kinds():
    spec = [
        {"column": "region", "op": "in", "value": ["Europe", "Asia"]},
        {"column": "quantity", "op": "between", "value": [2, 4]},
    ]
    assert scope.predicate_sql(spec) == (
        '"region" IN (\'Europe\', \'Asia\') AND "quantity" BETWEEN 2 AND 4'
    )


def test_predicate_sql_escapes_quotes_in_values():
    """A view body cannot take bound parameters, so values are inlined -- an
    apostrophe in a category name must not be able to terminate the string literal.
    """
    spec = [{"column": "category", "op": "in", "value": ["O'Brien's", 'say "hi"']}]
    assert scope.predicate_sql(spec) == (
        '"category" IN (\'O\'\'Brien\'\'s\', \'say "hi"\')'
    )


def test_predicate_sql_rejects_unsupported_op():
    with pytest.raises(ValueError):
        scope.predicate_sql([{"column": "region", "op": "like", "value": "%Eu%"}])


def test_describe_uses_sidebar_labels_and_plain_dates():
    """Descriptions reach both the UI and the LLM prompt, so they read in the user's
    terms -- and a midnight timestamp shows as a date, not "00:00:00".
    """
    spec = [
        {"column": "region", "op": "in", "value": ["Europe"]},
        {"column": "transaction_date", "op": "between",
         "value": [pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")]},
    ]
    text = scope.describe(spec, {"region": "Region", "transaction_date": "Transaction date"})
    assert text == (
        "Region is one of [Europe]; Transaction date is between 2022-01-01 and 2023-01-01"
    )


def test_apply_to_frame_skips_columns_absent_from_the_frame():
    """Mirrors how the sidebar already skips filters whose column isn't present."""
    df = pd.DataFrame({"region": ["Europe", "Asia"]})
    spec = [{"column": "not_a_column", "op": "in", "value": ["x"]}]
    assert len(scope.apply_to_frame(df, spec)) == 2


# ---------------------------------------------------------------------------
# The invariant: pandas and SQL must select the same rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [
    [],
    [{"column": "region", "op": "in", "value": ["Europe"]}],
    [{"column": "region", "op": "in", "value": ["Europe", "North America"]}],
    [{"column": "category", "op": "in", "value": ["Books"]},
     {"column": "payment_method", "op": "in", "value": ["Cash", "PayPal"]}],
    [{"column": "transaction_date", "op": "between",
      "value": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-02-28")]}],
    [{"column": "region", "op": "in", "value": ["Asia"]},
     {"column": "transaction_date", "op": "between",
      "value": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-03-01")]}],
    [{"column": "region", "op": "in", "value": ["Nowhere"]}],
])
def test_pandas_and_sql_renderings_select_the_same_rows(mini_con_clean, mini_df, spec):
    """The load-bearing test. The charts read the pandas rendering and every SQL path
    reads the view rendering; if these ever disagree, the dashboard shows one row set
    and answers questions about another.
    """
    expected = len(scope.apply_to_frame(mini_df, spec))
    cursor = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_equiv", spec)
    actual = cursor.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
    assert actual == expected


# ---------------------------------------------------------------------------
# Scoped cursor behaviour
# ---------------------------------------------------------------------------


def test_scoped_cursor_leaves_sibling_cursors_unscoped(mini_con_clean, mini_df):
    """search_path is set per-cursor, which is what lets the LLM path be filtered
    while the charts and the data-quality profile still see the whole table.
    """
    spec = [{"column": "region", "op": "in", "value": ["Europe"]}]
    scoped = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_sibling", spec)

    assert scoped.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0] < len(mini_df)
    sibling = mini_con_clean.cursor()
    assert sibling.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0] == len(mini_df)


def test_separate_schemas_do_not_share_scope(mini_con_clean):
    """Two browser sessions must not rescope each other: the DuckDB connection is
    cached per server process and shared, so the isolation has to come from the
    per-session schema name.
    """
    left = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_session_a",
                               [{"column": "region", "op": "in", "value": ["Europe"]}])
    right = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_session_b",
                                [{"column": "region", "op": "in", "value": ["Asia"]}])

    left_rows = left.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
    right_rows = right.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
    assert (left_rows, right_rows) == (5, 8)
    # And re-reading the first cursor still gives the first session's scope.
    assert left.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0] == 5


def test_scope_survives_replacement_on_filter_change(mini_con_clean):
    """Changing filters rebuilds the view in place; a cursor taken afterwards must
    see the new scope, not a stale one.
    """
    scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_change",
                        [{"column": "region", "op": "in", "value": ["Europe"]}])
    widened = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_change",
                                  [{"column": "region", "op": "in", "value": ["Europe", "Asia"]}])
    assert widened.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0] == 13


def test_empty_scope_yields_the_full_table(mini_con_clean, mini_df):
    cursor = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_none", [])
    assert cursor.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0] == len(mini_df)


# ---------------------------------------------------------------------------
# The scope must hold against LLM-generated SQL
# ---------------------------------------------------------------------------


def test_generated_sql_cannot_escape_the_scope(mini_con_clean, mini_df):
    """The whole mechanism rests on a bare table name resolving through search_path.
    A model that qualifies the name would otherwise read around the filter to the
    full table -- so the sandbox must refuse qualified references.
    """
    spec = [{"column": "region", "op": "in", "value": ["Europe"]}]
    cursor = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_escape", spec)

    scoped_rows = execute_safe(cursor, f"SELECT COUNT(*) AS n FROM {TABLE_NAME}", TABLE_NAME).iloc[0, 0]
    assert scoped_rows == 5
    assert scoped_rows < len(mini_df)

    with pytest.raises(SandboxViolation):
        validate_sql(f"SELECT COUNT(*) AS n FROM main.{TABLE_NAME}", TABLE_NAME)


def test_aggregates_reflect_the_scope_not_the_table(mini_con_clean, mini_df):
    """A regression for the reported bug: an aggregate answered through the scoped
    connection must total the filtered rows, not every row in the table.
    """
    spec = [{"column": "region", "op": "in", "value": ["Asia"]}]
    cursor = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_agg", spec)

    scoped_total = execute_safe(
        cursor, f"SELECT SUM(total_revenue) AS t FROM {TABLE_NAME}", TABLE_NAME
    ).iloc[0, 0]
    assert scoped_total == pytest.approx(7594)
    assert scoped_total != pytest.approx(mini_df["total_revenue"].sum())


def test_schema_introspection_reflects_the_scope(mini_con_clean):
    """The schema JSON handed to the SQL-generation prompt is introspected from the
    connection, so under a scope the model sees the filtered distinct values -- it is
    never shown categories that its query could not return.
    """
    from backend import schema as schema_module

    spec = [{"column": "region", "op": "in", "value": ["Europe"]}]
    cursor = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_schema", spec)

    columns = {c["name"]: c for c in schema_module.get_schema(cursor, TABLE_NAME)}
    assert columns["region"]["n_unique"] == 1
    assert columns["region"]["sample_values"] == ["Europe"]


def test_query_engine_works_through_a_scoped_cursor(mini_con_clean):
    """Task D's features and the presets all reach the data via query_engine, which
    validates columns with PRAGMA table_info -- that has to resolve through
    search_path too, or those paths break under a filter.
    """
    from backend import query_engine

    spec = [{"column": "region", "op": "in", "value": ["Asia"]}]
    cursor = scope.scoped_cursor(mini_con_clean, TABLE_NAME, "scope_engine", spec)

    df, _ = query_engine.groupby_agg(cursor, TABLE_NAME, ["region"], ["total_revenue"], ["sum"])
    assert dict(zip(df["region"], df["total_revenue_sum"])) == {"Asia": 7594}


def test_scoped_view_is_built_from_a_config_free_spec(dataset_config, tmp_path):
    """scope.py must stay generic: it is driven entirely by the caller's spec and
    never reads dataset config, so it survives a dataset swap untouched.
    """
    source = "\n".join([
        "Transaction Date,Customer ID,Region,Product,Category,Price,Quantity,"
        "Discount (%),Total Revenue,Payment Method",
        "2022-01-01,CUST_1,Asia,Product_1,Books,10.0,1,0.0,10.0,Cash",
        "2022-01-02,CUST_2,Europe,Product_2,Books,20.0,1,0.0,20.0,Cash",
    ])
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(source, encoding="utf-8")

    con = ingest.load(csv_path, dataset_config)
    quality.clean(con, TABLE_NAME, dataset_config)
    cursor = scope.scoped_cursor(con, TABLE_NAME, "scope_generic",
                                 [{"column": "region", "op": "in", "value": ["Asia"]}])
    assert cursor.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0] == 1
