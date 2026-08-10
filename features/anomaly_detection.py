"""Anomaly detection: flags specific outlier rows and has the LLM explain each one
in plain language (Task D1).

Reuses backend/quality.py's IQR profiling rather than reimplementing outlier
detection -- that logic already exists from Task A3, which is exactly why this
feature was chosen. This module adds what A3 doesn't do: identifying the specific
*rows* responsible, ranking them by how far past the fence they sit, and turning
them into an explanation a non-technical reader can act on.

Kept column-agnostic: the column to examine is either passed in or picked from the
live profile at runtime, never hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd

from backend import query_engine, quality
from llm import client


@dataclass
class Anomaly:
    """One flagged row: its values, and how far outside the IQR fence it sits."""

    values: dict[str, Any]
    value: float
    direction: str
    severity: float


@dataclass
class AnomalyReport:
    """Everything the UI needs to present one column's anomalies."""

    column: str
    bounds: dict[str, float]
    n_total_outliers: int
    anomalies: list[Anomaly] = field(default_factory=list)
    narrative: str = ""
    provider: str = ""

    def to_frame(self) -> pd.DataFrame:
        """The flagged rows as a DataFrame, for display and export."""
        if not self.anomalies:
            return pd.DataFrame()
        rows = []
        for anomaly in self.anomalies:
            row = dict(anomaly.values)
            row["_direction"] = anomaly.direction
            row["_severity_iqrs"] = round(anomaly.severity, 2)
            rows.append(row)
        return pd.DataFrame(rows)


def outlier_columns(con: duckdb.DuckDBPyConnection, table_name: str,
                    config: dict[str, Any]) -> list[dict[str, Any]]:
    """Numeric columns that actually have IQR outliers, worst first. Discovered
    from the live profile, so it reflects whatever the table currently holds.
    """
    profile = quality.profile_report(con, table_name, id_column=config["dataset"].get("id_column"))
    with_outliers = [c for c in profile["columns"] if (c["n_outliers_iqr"] or 0) > 0]
    return sorted(with_outliers, key=lambda c: c["n_outliers_iqr"], reverse=True)


def detect_anomalies(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    config: dict[str, Any],
    column: str,
    context_columns: list[str] | None = None,
    limit: int = 10,
) -> AnomalyReport:
    """Flag the `limit` most extreme rows for `column`, ranked by how many IQRs
    past the fence they sit. `context_columns` are the extra columns carried along
    so a reader can see *which* records are anomalous, not just that some are.

    Deterministic and LLM-free -- explain_anomalies() adds the narrative
    separately, so detection can be tested without any API call.
    """
    profile_columns = {
        c["name"]: c
        for c in quality.profile_report(
            con, table_name, id_column=config["dataset"].get("id_column")
        )["columns"]
    }
    if column not in profile_columns:
        raise ValueError(f"Unknown column: {column!r}")

    stats = profile_columns[column]
    bounds = stats["iqr_bounds"]
    if not bounds:
        return AnomalyReport(column=column, bounds={}, n_total_outliers=0)

    lower, upper = bounds["lower"], bounds["upper"]
    iqr = bounds["q3"] - bounds["q1"]

    dims = [c for c in (context_columns or []) if c != column]
    below, _ = query_engine.filtered_query(
        con, table_name, filters=[{"column": column, "op": "<", "value": lower}],
        dims=dims, metrics=[column],
    )
    above, _ = query_engine.filtered_query(
        con, table_name, filters=[{"column": column, "op": ">", "value": upper}],
        dims=dims, metrics=[column],
    )

    anomalies: list[Anomaly] = []
    for frame, direction in ((below, "below"), (above, "above")):
        for record in frame.to_dict(orient="records"):
            value = record[column]
            # How far past the fence, measured in IQRs -- comparable across
            # columns with very different scales.
            distance = (lower - value) if direction == "below" else (value - upper)
            severity = distance / iqr if iqr else 0.0
            anomalies.append(Anomaly(values=record, value=value,
                                     direction=direction, severity=severity))

    anomalies.sort(key=lambda a: a.severity, reverse=True)

    return AnomalyReport(
        column=column, bounds=bounds,
        n_total_outliers=stats["n_outliers_iqr"],
        anomalies=anomalies[:limit],
    )


def explain_anomalies(report: AnomalyReport) -> AnomalyReport:
    """Ask the LLM for a plain-language explanation of the flagged rows, and
    return the report with `narrative`/`provider` filled in.

    The prompt carries the actual flagged values and the IQR bounds, and forbids
    inventing figures -- the LLM is describing rows the deterministic detector
    already chose, not deciding what counts as an anomaly.
    """
    if not report.anomalies:
        report.narrative = f"No IQR-based outliers were detected in `{report.column}`."
        report.provider = "none"
        return report

    system_prompt = (
        "You are a data analyst explaining flagged anomalies to a business "
        "audience. You will be given a numeric column, its normal (IQR) range, "
        "the total number of outliers, and the most extreme flagged rows with "
        "their context. Write a short Markdown explanation: one or two sentences "
        "framing the overall picture, then one bullet per flagged row saying what "
        "is unusual about it in plain language, referring to the row's own "
        "context values. Use only the numbers given -- never invent a figure, and "
        "never speculate about causes the data cannot show."
    )
    # Values are rounded before they reach the prompt: handed raw floats, the model
    # echoes them verbatim, and a full-precision figure reads as a bug in prose.
    def _round_values(values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in values.items()
        }

    user_prompt = json.dumps({
        "column": report.column,
        "normal_range": {"lower": round(report.bounds["lower"], 2),
                         "upper": round(report.bounds["upper"], 2)},
        "n_total_outliers": report.n_total_outliers,
        "flagged_rows": [
            {"values": _round_values(a.values), "direction": a.direction,
             "iqrs_beyond_fence": round(a.severity, 2)}
            for a in report.anomalies
        ],
    }, indent=2, default=str)

    result = client.generate(system_prompt, user_prompt, json_mode=False)
    report.narrative = result.text
    report.provider = result.provider
    return report
