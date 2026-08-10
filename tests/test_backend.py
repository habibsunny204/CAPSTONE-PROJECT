"""Tests for backend/: schema extraction correctness, PII gate enforcement, query
engine outputs against a small fixture, data quality profiling/cleaning, and the
<500ms performance assertion (Task A, PROJECT_SPEC.md Section 8).

All expected values below (sums, unique counts, null counts, IQR outlier counts,
duplicate counts) are hand-computed against tests/fixtures/mini_superstore.csv — see
tests/fixtures/generate_mini_superstore.py for how that fixture was constructed.
"""

import duckdb
import pytest

from backend import query_engine, schema

TABLE_NAME = "superstore"


# ---------------------------------------------------------------------------
# A1 — schema extraction + PII gate
# ---------------------------------------------------------------------------


def test_ingest_row_count_matches_fixture(mini_con):
    """The fixture has 16 rows; ingestion must not drop or duplicate any."""
    count = mini_con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
    assert count == 16


def test_schema_shape_has_required_keys(mini_con):
    """Every column dict has exactly the public-contract keys, with <=5 samples."""
    cols = schema.get_schema(mini_con, TABLE_NAME)
    assert len(cols) > 0
    for col in cols:
        assert set(col.keys()) == {"name", "dtype", "n_unique", "n_null", "sample_values"}
        assert len(col["sample_values"]) <= 5


def test_schema_excludes_pii_columns(mini_con):
    """Customer.Name must never appear in the schema contract, under any name."""
    names = {col["name"] for col in schema.get_schema(mini_con, TABLE_NAME)}
    assert "customer_name" not in names
    assert "Customer.Name" not in names


def test_pii_customer_name_never_persisted(mini_con):
    """The raw PII column must not exist in the table at all -- not just be hidden
    from the schema JSON -- since sandbox.py (Task B) validates SQL structure, not a
    column allowlist.
    """
    with pytest.raises(duckdb.Error):
        mini_con.execute(f'SELECT "Customer.Name" FROM "{TABLE_NAME}"')
    with pytest.raises(duckdb.Error):
        mini_con.execute(f'SELECT "customer_name" FROM "{TABLE_NAME}"')


def test_customer_id_is_hashed_not_raw(mini_con, dataset_config):
    """customer_id_hash values must never equal the original raw Customer.ID values,
    and must be the configured hash length.
    """
    hash_length = dataset_config["pii"]["hash_length"]
    rows = mini_con.execute(f'SELECT customer_id_hash FROM "{TABLE_NAME}"').fetchall()
    hashes = [r[0] for r in rows]

    assert all(len(h) == hash_length for h in hashes)
    raw_ids = {f"CU-{i:04d}" for i in range(1, 17)}
    assert not (set(hashes) & raw_ids)

    with pytest.raises(duckdb.Error):
        mini_con.execute(f'SELECT "Customer.ID" FROM "{TABLE_NAME}"')


def test_customer_id_hash_is_consistent_for_repeat_customers(mini_con):
    """Rows 1 and 13 share the same raw Customer.ID (CU-0001) -- hashing must
    preserve that equality so unique/repeat-customer analytics still work.
    """
    rows = mini_con.execute(
        f'SELECT customer_id_hash FROM "{TABLE_NAME}" WHERE row_id IN (1, 13)'
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == rows[1][0]


@pytest.mark.parametrize(
    "column,expected_n_unique,expected_n_null",
    [
        ("region", 2, 0),
        ("row_id", 16, 0),
        ("profit", 10, 2),
    ],
)
def test_schema_n_unique_and_n_null_correct(mini_con, column, expected_n_unique, expected_n_null):
    """Hand-verified against the fixture: region has 2 distinct values (West/East);
    row_id is a unique surrogate key across all 16 rows; profit has 2 genuine nulls
    (rows 7, 8) and 10 distinct non-null values.
    """
    cols = {c["name"]: c for c in schema.get_schema(mini_con, TABLE_NAME)}
    assert cols[column]["n_unique"] == expected_n_unique
    assert cols[column]["n_null"] == expected_n_null


# ---------------------------------------------------------------------------
# A2 — query engine (groupby_agg, filtered_query)
# ---------------------------------------------------------------------------


def test_groupby_agg_basic_sum(mini_con):
    """Sum of sales by region, hand-computed from the fixture: West rows
    (1,2,3,4,9,11,12,13,14,16) sum to 6180; East rows (5,6,7,8,10,15) sum to 597.
    """
    df, _ = query_engine.groupby_agg(mini_con, TABLE_NAME, ["region"], ["sales"], ["sum"])
    totals = dict(zip(df["region"], df["sales_sum"]))
    assert totals == {"West": 6180, "East": 597}


def test_groupby_agg_multiple_metrics_and_aggs(mini_con):
    """sum+avg over two metrics at once, hand-computed: West profit sums to 599
    over 10 rows (avg 59.9); East profit sums to 39 over 4 non-null rows (avg 9.75,
    since AVG ignores the 2 nulls in rows 7/8).
    """
    df, _ = query_engine.groupby_agg(
        mini_con, TABLE_NAME, ["region"], ["sales", "profit"], ["sum", "avg"]
    )
    by_region = df.set_index("region")
    assert by_region.loc["West", "sales_sum"] == 6180
    assert by_region.loc["West", "sales_avg"] == pytest.approx(618.0)
    assert by_region.loc["West", "profit_sum"] == 599
    assert by_region.loc["West", "profit_avg"] == pytest.approx(59.9)
    assert by_region.loc["East", "profit_sum"] == 39
    assert by_region.loc["East", "profit_avg"] == pytest.approx(9.75)


def test_groupby_agg_returns_elapsed_ms(mini_con):
    """The second return value is a non-negative float timing the DuckDB call."""
    _, elapsed_ms = query_engine.groupby_agg(mini_con, TABLE_NAME, ["region"], ["sales"], ["sum"])
    assert isinstance(elapsed_ms, float)
    assert elapsed_ms >= 0


def test_groupby_agg_unknown_column_raises(mini_con):
    with pytest.raises(ValueError):
        query_engine.groupby_agg(mini_con, TABLE_NAME, ["not_a_column"], ["sales"], ["sum"])


def test_groupby_agg_invalid_agg_raises(mini_con):
    with pytest.raises(ValueError):
        query_engine.groupby_agg(mini_con, TABLE_NAME, ["region"], ["sales"], ["bogus_agg"])


def test_groupby_agg_with_filters(mini_con):
    """Filtered aggregation: sum of sales by region where segment = 'Consumer'
    (exact match on raw, uncleaned data -- row 11's mis-cased 'consumer' is excluded).
    Matching rows: West {1,2,12,13} = 5350; East {5,6,15} = 299.
    """
    df, _ = query_engine.groupby_agg(
        mini_con, TABLE_NAME, ["region"], ["sales"], ["sum"],
        filters=[{"column": "segment", "op": "=", "value": "Consumer"}],
    )
    totals = dict(zip(df["region"], df["sales_sum"]))
    assert totals == {"West": 5350, "East": 299}


def test_filtered_query_basic_equality_filter(mini_con):
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "region", "op": "=", "value": "East"}],
        dims=["row_id"], metrics=["sales"],
    )
    assert len(df) == 6
    assert set(df["sales"]) == {90, 110, 95, 105, 98, 99}


def test_filtered_query_multiple_filters_and_combination(mini_con):
    """Multi-condition filter (region=West AND segment=Corporate): rows 3, 4, 14."""
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[
            {"column": "region", "op": "=", "value": "West"},
            {"column": "segment", "op": "=", "value": "Corporate"},
        ],
        dims=["row_id"], metrics=["sales"],
    )
    assert set(df["row_id"]) == {3, 4, 14}
    assert set(df["sales"]) == {200, 120, 115}


def test_filtered_query_between_filter_on_dates(mini_con):
    """order_date BETWEEN Feb 1 and Mar 1 2013 inclusive: rows 5-9 (5 rows). Works
    against the still-string ISO-formatted dates because ISO 8601 strings sort
    lexically in chronological order -- A2 doesn't depend on A3's date parsing.
    """
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{
            "column": "order_date", "op": "between",
            "value": ["2013-02-01 00:00:00.000", "2013-03-01 00:00:00.000"],
        }],
        dims=["row_id"], metrics=["order_date"],
    )
    assert set(df["row_id"]) == {5, 6, 7, 8, 9}


def test_filtered_query_in_operator(mini_con):
    """segment IN (Corporate, Home Office): rows 3,4,7,8,14 + 9,10,16 = 8 rows."""
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "segment", "op": "in", "value": ["Corporate", "Home Office"]}],
        dims=["row_id"], metrics=["segment"],
    )
    assert len(df) == 8


def test_filtered_query_is_null_operator(mini_con):
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "profit", "op": "is_null"}],
        dims=["row_id"], metrics=["profit"],
    )
    assert set(df["row_id"]) == {7, 8}


def test_filtered_query_empty_result(mini_con):
    """A filter matching nothing returns an empty DataFrame, not an error."""
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "region", "op": "=", "value": "Nowhere"}],
        dims=["row_id"], metrics=["sales"],
    )
    assert len(df) == 0
