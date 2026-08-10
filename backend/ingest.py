"""Loads the raw Global Superstore CSV into an in-memory DuckDB table (Task A1).

Purely mechanical by design: read the CSV, apply the PII gate, rename raw
dot-separated columns to snake_case, and create the DuckDB table. Judgment-based
cleaning (date parsing, casing standardization, dropping degenerate columns) is
deliberately left to backend/quality.py (Task A3) so this module stays a thin,
predictable ingestion step.

Column names and PII rules are never hardcoded here — they are read from
configs/dataset_config.yaml, keeping this module free of Superstore-specific column
name literals per PROJECT_SPEC.md Section 2's generic/specific boundary.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "dataset_config.yaml"

# Loaded once at import time so PII_HASH_SALT (and future LLM API keys) are available
# from .env without every caller having to remember to load it themselves.
load_dotenv()


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read and parse the dataset config YAML into a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hash_value(raw_value: str, salt: str, length: int) -> str:
    """Hash a single raw PII value with the configured salt, truncated to `length`
    hex characters. A privacy control against accidental exposure/re-identification,
    not a cryptographic security control (no per-value random nonce, so equal inputs
    hash identically — this is required to preserve unique/repeat-customer analytics).
    """
    digest = hashlib.sha256((salt + str(raw_value)).encode("utf-8")).hexdigest()
    return digest[:length]


def _apply_pii_gate(df: pd.DataFrame, pii_config: dict[str, Any]) -> pd.DataFrame:
    """Drop and/or hash PII columns per `pii_config`. Runs before the table is ever
    created, so raw PII is never persisted into DuckDB — this is what makes the gate
    a hard gate rather than something merely hidden from the schema JSON later
    (see PROJECT_SPEC.md Section 2 and the note in schema.py).
    """
    drop_columns = pii_config.get("drop_columns", [])
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    hash_columns: dict[str, str] = pii_config.get("hash_columns", {})
    if hash_columns:
        salt_env_var = pii_config["hash_salt_env_var"]
        salt = os.environ.get(salt_env_var)
        if not salt:
            raise RuntimeError(
                f"Environment variable {salt_env_var} is not set. PII hashing requires "
                "a salt before ingestion can run — set it in .env (see .env.example) "
                "rather than falling back to a default."
            )
        hash_length = pii_config.get("hash_length", 16)
        for raw_col, target_col in hash_columns.items():
            df[raw_col] = df[raw_col].astype(str).map(
                lambda v: _hash_value(v, salt, hash_length)
            )
        df = df.rename(columns=hash_columns)

    return df


def load(
    csv_path: str | Path,
    config: dict[str, Any],
    con: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load `csv_path` into an in-memory DuckDB table, applying the PII gate and the
    raw->normalized column rename map from `config`. Creates a new in-memory
    connection if `con` is not supplied; returns the connection that owns the table.
    """
    if con is None:
        con = duckdb.connect(":memory:")

    encoding = config["dataset"].get("encoding", "utf-8")
    table_name = config["dataset"]["table_name"]

    df = pd.read_csv(csv_path, encoding=encoding)
    df = _apply_pii_gate(df, config["pii"])

    rename_map = config["columns"]["rename_map"]
    df = df.rename(columns=rename_map)

    con.register("_ingest_tmp", df)
    con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _ingest_tmp')
    con.unregister("_ingest_tmp")
    return con


def load_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    con: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Convenience wrapper: load the dataset config and ingest the CSV path it
    declares. This is the entrypoint the app/tests use for the real dataset; unit
    tests exercise `load()` directly against a small fixture CSV instead.
    """
    config = load_config(config_path)
    repo_root = Path(config_path).resolve().parent.parent
    csv_path = repo_root / config["dataset"]["raw_csv_path"]
    return load(csv_path, config, con=con)
