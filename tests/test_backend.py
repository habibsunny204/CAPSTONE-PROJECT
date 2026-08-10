"""Tests for backend/: schema extraction correctness, PII gate enforcement, query
engine outputs against a small fixture, data quality profiling/cleaning, and the
<500ms performance assertion (Task A, PROJECT_SPEC.md Section 8).

All expected values below (sums, unique counts, null counts, IQR outlier counts,
duplicate counts) are hand-computed against tests/fixtures/mini_superstore.csv — see
tests/fixtures/generate_mini_superstore.py for how that fixture was constructed.
"""

import duckdb
import pytest

from backend import schema

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
