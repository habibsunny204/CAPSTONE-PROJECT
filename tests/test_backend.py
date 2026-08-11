"""Tests for backend/: schema extraction correctness, PII gate enforcement, query
engine outputs against a small fixture, data quality profiling/cleaning, and the
<500ms performance assertion (Task A, PROJECT_SPEC.md Section 8).

All expected values below (sums, unique counts, null counts, IQR outlier counts,
duplicate counts) are hand-computed against tests/fixtures/mini_ecommerce.csv — see
tests/fixtures/generate_mini_ecommerce.py for how that fixture was constructed and
which scenarios it deliberately engineers in.
"""

import duckdb
import pandas as pd
import pytest

from backend import ingest, query_engine, quality, schema
from backend.benchmark_perf import PASS_THRESHOLD_MS

TABLE_NAME = "ecommerce_sales"


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


def test_schema_excludes_raw_pii_column(mini_con):
    """The raw Customer ID must never appear in the schema contract, under any
    spelling -- only the hashed replacement may.
    """
    names = {col["name"] for col in schema.get_schema(mini_con, TABLE_NAME)}
    assert "Customer ID" not in names
    assert "customer_id" not in names
    assert "customer_id_hash" in names


def test_pii_raw_customer_id_never_persisted(mini_con):
    """The raw PII column must not exist in the table at all -- not just be hidden
    from the schema JSON -- since sandbox.py (Task B) validates SQL structure, not a
    column allowlist.
    """
    with pytest.raises(duckdb.Error):
        mini_con.execute(f'SELECT "Customer ID" FROM "{TABLE_NAME}"')
    with pytest.raises(duckdb.Error):
        mini_con.execute(f'SELECT "customer_id" FROM "{TABLE_NAME}"')


def test_customer_id_is_hashed_not_raw(mini_con, dataset_config):
    """customer_id_hash values must never equal the original raw Customer ID values,
    and must be the configured hash length.
    """
    hash_length = dataset_config["pii"]["hash_length"]
    rows = mini_con.execute(f'SELECT customer_id_hash FROM "{TABLE_NAME}"').fetchall()
    hashes = [r[0] for r in rows]

    assert all(len(h) == hash_length for h in hashes)
    raw_ids = {f"CUST_{i:05d}" for i in range(1, 17)}
    assert not (set(hashes) & raw_ids)


def test_customer_id_hash_is_consistent_for_repeat_customers(mini_con):
    """Rows 1 and 13 share the same raw Customer ID (CUST_00001) -- hashing must
    preserve that equality so unique/repeat-customer analytics still work. The
    fixture's 16 rows therefore come from exactly 15 distinct customers.
    """
    total, distinct = mini_con.execute(
        f'SELECT COUNT(*), COUNT(DISTINCT customer_id_hash) FROM "{TABLE_NAME}"'
    ).fetchone()
    assert (total, distinct) == (16, 15)

    # Both rows dated 2022-01-07 are that repeat customer, so they collapse to one hash.
    same_day = mini_con.execute(
        f"SELECT DISTINCT customer_id_hash FROM \"{TABLE_NAME}\" "
        f"WHERE transaction_date = '2022-01-07'"
    ).fetchall()
    assert len(same_day) == 1


def test_pii_hash_skips_columns_absent_from_the_csv(dataset_config, tmp_path):
    """A configured hash column that isn't in the CSV is skipped, not fatal.

    Regression test: the hash path used to index df[raw_col] unguarded (unlike the
    drop path), so any CSV lacking a configured PII column raised KeyError during
    ingestion rather than degrading gracefully.
    """
    csv_path = tmp_path / "no_pii.csv"
    pd.DataFrame({
        "Transaction Date": ["2022-01-01"], "Region": ["Asia"], "Product": ["Product_0001"],
        "Category": ["Books"], "Price": [10.0], "Quantity": [1],
        "Discount (%)": [0.0], "Total Revenue": [10.0], "Payment Method": ["Cash"],
    }).to_csv(csv_path, index=False)

    con = ingest.load(csv_path, dataset_config)
    names = {c["name"] for c in schema.get_schema(con, TABLE_NAME)}
    assert "customer_id_hash" not in names
    assert con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0] == 1


@pytest.mark.parametrize(
    "column,expected_n_unique,expected_n_null",
    [
        ("region", 3, 0),
        ("customer_id_hash", 15, 0),
        ("total_revenue", 12, 2),
    ],
)
def test_schema_n_unique_and_n_null_correct(mini_con, column, expected_n_unique, expected_n_null):
    """Hand-verified against the fixture: region has 3 distinct values (Asia/Europe/
    North America); customer_id_hash has 15 distinct values across 16 rows (row 13
    repeats row 1's customer); total_revenue has 2 genuine nulls (rows 7, 8) and 12
    distinct non-null values (300 recurs three times).
    """
    cols = {c["name"]: c for c in schema.get_schema(mini_con, TABLE_NAME)}
    assert cols[column]["n_unique"] == expected_n_unique
    assert cols[column]["n_null"] == expected_n_null


# ---------------------------------------------------------------------------
# A2 — query engine (groupby_agg, filtered_query)
# ---------------------------------------------------------------------------


def test_groupby_agg_basic_sum(mini_con):
    """Sum of total_revenue by region, hand-computed from the fixture: Asia rows
    (1,2,3,4,9,11,12,13) sum to 7594; Europe rows (5,6,10) sum to 706 (rows 7 and 8
    are null); North America rows (14,15,16) sum to 803.
    """
    df, _ = query_engine.groupby_agg(
        mini_con, TABLE_NAME, ["region"], ["total_revenue"], ["sum"]
    )
    totals = dict(zip(df["region"], df["total_revenue_sum"]))
    assert totals == {"Asia": 7594, "Europe": 706, "North America": 803}


def test_groupby_agg_multiple_metrics_and_aggs(mini_con):
    """sum+avg over two metrics at once, hand-computed. Asia: revenue 7594 over 8
    rows (avg 949.25), price 1440 over 8 rows (avg 180). Europe: revenue 706 but avg
    235.33, because AVG divides by the 3 non-null rows, not all 5 -- while price
    averages over all 5 rows (498/5 = 99.6). That divergence is the point of the test.
    """
    df, _ = query_engine.groupby_agg(
        mini_con, TABLE_NAME, ["region"], ["total_revenue", "price"], ["sum", "avg"]
    )
    by_region = df.set_index("region")
    assert by_region.loc["Asia", "total_revenue_sum"] == 7594
    assert by_region.loc["Asia", "total_revenue_avg"] == pytest.approx(949.25)
    assert by_region.loc["Asia", "price_avg"] == pytest.approx(180.0)
    assert by_region.loc["Europe", "total_revenue_sum"] == 706
    assert by_region.loc["Europe", "total_revenue_avg"] == pytest.approx(235.333333, rel=1e-5)
    assert by_region.loc["Europe", "price_avg"] == pytest.approx(99.6)


def test_groupby_agg_returns_elapsed_ms(mini_con):
    """The second return value is a non-negative float timing the DuckDB call."""
    _, elapsed_ms = query_engine.groupby_agg(
        mini_con, TABLE_NAME, ["region"], ["total_revenue"], ["sum"]
    )
    assert isinstance(elapsed_ms, float)
    assert elapsed_ms >= 0


def test_groupby_agg_unknown_column_raises(mini_con):
    with pytest.raises(ValueError):
        query_engine.groupby_agg(
            mini_con, TABLE_NAME, ["not_a_column"], ["total_revenue"], ["sum"]
        )


def test_groupby_agg_invalid_agg_raises(mini_con):
    with pytest.raises(ValueError):
        query_engine.groupby_agg(
            mini_con, TABLE_NAME, ["region"], ["total_revenue"], ["bogus_agg"]
        )


def test_groupby_agg_with_filters(mini_con):
    """Filtered aggregation: revenue by region where category = 'Fashion' (exact
    match on raw, uncleaned data -- row 11's mis-cased 'fashion' is excluded, which
    is why Asia reads 390 here and not 810).
    """
    df, _ = query_engine.groupby_agg(
        mini_con, TABLE_NAME, ["region"], ["total_revenue"], ["sum"],
        filters=[{"column": "category", "op": "=", "value": "Fashion"}],
    )
    totals = dict(zip(df["region"], df["total_revenue_sum"]))
    assert totals == {"Asia": 390, "Europe": 196, "North America": 375}


def test_filtered_query_basic_equality_filter(mini_con):
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "region", "op": "=", "value": "Europe"}],
        dims=["transaction_date"], metrics=["price"],
    )
    assert len(df) == 5
    assert set(df["price"]) == {90, 110, 95, 105, 98}


def test_filtered_query_multiple_filters_and_combination(mini_con):
    """Multi-condition filter (region=Asia AND category=Books): rows 3 and 4."""
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[
            {"column": "region", "op": "=", "value": "Asia"},
            {"column": "category", "op": "=", "value": "Books"},
        ],
        dims=["transaction_date"], metrics=["price", "total_revenue"],
    )
    assert set(df["price"]) == {200, 120}
    assert set(df["total_revenue"]) == {800, 84}


def test_filtered_query_between_filter_on_dates(mini_con):
    """transaction_date BETWEEN Feb 1 and Mar 1 2022 inclusive: rows 5-9 (5 rows).
    Works against the still-string ISO-formatted dates because ISO 8601 strings sort
    lexically in chronological order -- A2 doesn't depend on A3's date parsing.
    """
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{
            "column": "transaction_date", "op": "between",
            "value": ["2022-02-01", "2022-03-01"],
        }],
        dims=["transaction_date"], metrics=["price"],
    )
    assert len(df) == 5


def test_filtered_query_in_operator(mini_con):
    """payment_method IN (Credit Card, Cash): rows 1,7,12,13,16 + 3,8,15 = 8 rows."""
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "payment_method", "op": "in", "value": ["Credit Card", "Cash"]}],
        dims=["transaction_date"], metrics=["payment_method"],
    )
    assert len(df) == 8


def test_filtered_query_is_null_operator(mini_con):
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "total_revenue", "op": "is_null"}],
        dims=["transaction_date"], metrics=["total_revenue"],
    )
    assert set(df["transaction_date"]) == {"2022-02-10", "2022-02-14"}


def test_filtered_query_empty_result(mini_con):
    """A filter matching nothing returns an empty DataFrame, not an error."""
    df, _ = query_engine.filtered_query(
        mini_con, TABLE_NAME,
        filters=[{"column": "region", "op": "=", "value": "Nowhere"}],
        dims=["transaction_date"], metrics=["total_revenue"],
    )
    assert len(df) == 0


# ---------------------------------------------------------------------------
# A3 — data quality (profiling, cleaning)
# ---------------------------------------------------------------------------


def test_profile_missing_pct(mini_con, dataset_config):
    """total_revenue has 2 nulls (rows 7, 8) out of 16 rows -> 12.5% missing."""
    report = quality.profile_report(
        mini_con, TABLE_NAME, id_column=dataset_config["dataset"].get("id_column")
    )
    cols = {c["name"]: c for c in report["columns"]}
    assert cols["total_revenue"]["missing_pct"] == pytest.approx(12.5)


def test_profile_duplicate_count(mini_con, dataset_config):
    """Row 13 is an exact duplicate of row 1 -- exactly 1 extra duplicate row, out of
    16 total. This dataset has no surrogate key to exclude, so the comparison is over
    every column.
    """
    report = quality.profile_report(
        mini_con, TABLE_NAME, id_column=dataset_config["dataset"].get("id_column")
    )
    assert report["n_rows"] == 16
    assert report["n_duplicate_rows"] == 1


def test_profile_iqr_outliers(mini_con, dataset_config):
    """total_revenue IQR bounds hand-computed from the sorted non-null fixture values
    (linear interpolation, matching DuckDB's quantile_cont): Q1=206, Q3=386.25, so the
    upper fence is 656.625 -- only 800 and 5000 exceed it (2 outliers).
    """
    report = quality.profile_report(
        mini_con, TABLE_NAME, id_column=dataset_config["dataset"].get("id_column")
    )
    cols = {c["name"]: c for c in report["columns"]}
    revenue = cols["total_revenue"]
    assert revenue["n_outliers_iqr"] == 2
    assert revenue["iqr_bounds"]["q1"] == pytest.approx(206.0)
    assert revenue["iqr_bounds"]["q3"] == pytest.approx(386.25)
    assert revenue["iqr_bounds"]["lower"] == pytest.approx(-64.375)
    assert revenue["iqr_bounds"]["upper"] == pytest.approx(656.625)


def test_profile_non_numeric_column_has_no_iqr(mini_con, dataset_config):
    report = quality.profile_report(
        mini_con, TABLE_NAME, id_column=dataset_config["dataset"].get("id_column")
    )
    cols = {c["name"]: c for c in report["columns"]}
    assert cols["region"]["n_outliers_iqr"] is None
    assert cols["region"]["iqr_bounds"] is None


def test_clean_parses_dates(mini_con_clean):
    cols = {c["name"]: c for c in schema.get_schema(mini_con_clean, TABLE_NAME)}
    assert cols["transaction_date"]["dtype"] == "TIMESTAMP"


def test_clean_derives_date_parts(mini_con_clean):
    """year/month are derived from transaction_date during cleaning -- neither exists
    in the raw CSV. The fixture spans Jan-Apr 2022 with 5/4/4/3 rows per month.
    """
    cols = {c["name"]: c for c in schema.get_schema(mini_con_clean, TABLE_NAME)}
    assert cols["year"]["dtype"] == "BIGINT"
    assert cols["month"]["dtype"] == "BIGINT"

    rows = mini_con_clean.execute(
        f'SELECT year, month, COUNT(*) FROM "{TABLE_NAME}" GROUP BY 1, 2 ORDER BY 1, 2'
    ).fetchall()
    assert rows == [(2022, 1, 5), (2022, 2, 4), (2022, 3, 4), (2022, 4, 3)]


def test_clean_standardizes_casing(mini_con_clean):
    """Row 11's deliberately mis-cased 'fashion' becomes 'Fashion' after cleaning,
    collapsing the raw 4 distinct categories back to the real 3.
    """
    categories = mini_con_clean.execute(
        f'SELECT DISTINCT category FROM "{TABLE_NAME}" ORDER BY 1'
    ).fetchall()
    assert [c[0] for c in categories] == ["Books", "Electronics", "Fashion"]


def test_clean_casing_does_not_corrupt_payment_method(mini_con_clean):
    """payment_method is deliberately excluded from categorical_casing_columns because
    title-casing would mangle 'PayPal' into 'Paypal'. This asserts that decision holds
    -- if someone adds payment_method to that config list, this test catches it.
    """
    methods = {
        r[0] for r in
        mini_con_clean.execute(f'SELECT DISTINCT payment_method FROM "{TABLE_NAME}"').fetchall()
    }
    assert "PayPal" in methods
    assert "Paypal" not in methods


def test_clean_report_shape(mini_con_fresh, dataset_config):
    """clean()'s own report documents exactly what it changed. No drop step appears
    here because this dataset configures no degenerate columns to drop (unlike the
    previous dataset's constant-value row-counter artifact) -- the empty
    drop_after_profiling list is a recorded decision, not an oversight.
    """
    report = quality.clean(mini_con_fresh, TABLE_NAME, dataset_config)
    step_names = {s["name"] for s in report["steps_applied"]}
    assert step_names == {"parse_dates", "derive_date_parts", "standardize_categorical_casing"}

    casing_step = next(
        s for s in report["steps_applied"] if s["name"] == "standardize_categorical_casing"
    )
    assert "category" in casing_step["columns"]
    assert casing_step["rows_changed"] == 1

    derive_step = next(s for s in report["steps_applied"] if s["name"] == "derive_date_parts")
    assert derive_step["columns_added"] == ["year", "month"]


# ---------------------------------------------------------------------------
# A4 — performance (full real dataset; skipped if data/raw/ is absent)
# ---------------------------------------------------------------------------


def test_filtered_aggregation_under_500ms(real_con_clean, dataset_config):
    """A representative filtered aggregation on the full 500K-row dataset must
    complete in well under the 500ms bar (PROJECT_SPEC.md Task A4). See
    eval/results/perf_benchmark_*.json for the full scenario sweep and evidence.
    """
    table_name = dataset_config["dataset"]["table_name"]
    _, elapsed_ms = query_engine.groupby_agg(
        real_con_clean, table_name, ["region"], ["total_revenue"], ["sum"],
        filters=[{"column": "payment_method", "op": "=", "value": "Credit Card"}],
    )
    assert elapsed_ms < PASS_THRESHOLD_MS
