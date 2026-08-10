"""Shared pytest fixtures for the Task A test suite.

Two dataset fixtures are provided: `mini_con`, built from the small hand-verifiable
CSV fixture (tests/fixtures/mini_superstore.csv) via ingest.load() only, and
`mini_con_clean`, the same fixture additionally passed through quality.clean(). Tests
that only exercise ingestion/schema/query mechanics use `mini_con`; tests that
specifically verify cleaning behavior use `mini_con_clean`. `real_con_clean` loads the
full production dataset and is skipped when data/raw/ isn't present (it's gitignored).

A fixed, non-secret salt is used for PII hashing in tests regardless of the
developer's local .env, so hashing tests are deterministic and don't depend on
machine-specific secrets.
"""

import os
from pathlib import Path

import pytest

from backend import ingest, quality

TESTS_DIR = Path(__file__).resolve().parent
MINI_CSV_PATH = TESTS_DIR / "fixtures" / "mini_superstore.csv"
REAL_CSV_PATH = TESTS_DIR.parent / "data" / "raw" / "global_superstore.csv"

TEST_SALT = "test-salt-not-for-production-use"


@pytest.fixture(autouse=True, scope="session")
def _pii_hash_salt():
    """Force a fixed PII_HASH_SALT for the whole test session."""
    os.environ["PII_HASH_SALT"] = TEST_SALT


@pytest.fixture(scope="session")
def dataset_config():
    """The real configs/dataset_config.yaml — the fixture CSV reuses the same raw
    column names/format as the real dataset, so the same config applies to both.
    """
    return ingest.load_config()


@pytest.fixture(scope="session")
def mini_con(dataset_config, _pii_hash_salt):
    """An in-memory DuckDB connection over the mini fixture, ingested but not cleaned."""
    return ingest.load(MINI_CSV_PATH, dataset_config)


@pytest.fixture(scope="session")
def mini_con_clean(dataset_config, _pii_hash_salt):
    """An in-memory DuckDB connection over the mini fixture, ingested and cleaned."""
    con = ingest.load(MINI_CSV_PATH, dataset_config)
    quality.clean(con, dataset_config["dataset"]["table_name"], dataset_config)
    return con


@pytest.fixture()
def mini_con_fresh(dataset_config, _pii_hash_salt):
    """A function-scoped, uncleaned connection -- for tests that call clean()
    themselves and need to inspect its returned report without disturbing the
    session-scoped mini_con_clean state shared by other tests.
    """
    return ingest.load(MINI_CSV_PATH, dataset_config)


@pytest.fixture(scope="session")
def mini_df(mini_con_clean, dataset_config):
    """The cleaned mini fixture as a DataFrame -- what viz/ and export/ consume
    (they take data, not a connection).
    """
    table_name = dataset_config["dataset"]["table_name"]
    return mini_con_clean.execute(f'SELECT * FROM "{table_name}"').df()


@pytest.fixture(scope="session")
def real_con_clean(dataset_config, _pii_hash_salt):
    """An in-memory DuckDB connection over the full real dataset, ingested and
    cleaned. Skipped if data/raw/global_superstore.csv isn't present locally.
    """
    if not REAL_CSV_PATH.exists():
        pytest.skip("real dataset not present locally (data/raw/ is gitignored)")
    con = ingest.load(REAL_CSV_PATH, dataset_config)
    quality.clean(con, dataset_config["dataset"]["table_name"], dataset_config)
    return con
