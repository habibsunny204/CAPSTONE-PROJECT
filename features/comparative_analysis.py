"""Comparative analysis: two slices of the data side by side, with an LLM-written
comparison narrative and a paired visualization (Task D2).

Reuses backend/query_engine.py's filtered_query()/groupby_agg() for both sides
rather than writing new query code, and viz/charts.py's shared theme for the paired
chart. Which dimension and which two values to compare are chosen by the caller at
runtime, so nothing here is tied to particular dataset columns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd
import plotly.graph_objects as go

from backend import query_engine
from llm import client
from viz import charts


@dataclass
class ComparisonSide:
    """One side of a comparison: its label, filters, and aggregated totals."""

    label: str
    filters: list[dict[str, Any]]
    totals: dict[str, float]
    n_rows: int


@dataclass
class ComparisonResult:
    """Both sides, their differences, the narrative, and the paired chart."""

    dimension: str
    metrics: list[str]
    left: ComparisonSide
    right: ComparisonSide
    deltas: dict[str, dict[str, float]]
    narrative: str = ""
    provider: str = ""

    def to_frame(self) -> pd.DataFrame:
        """A tidy side-by-side table, for display and export."""
        rows = []
        for metric in self.metrics:
            delta = self.deltas[metric]
            rows.append({
                "Metric": metric.replace("_", " ").title(),
                self.left.label: self.left.totals[metric],
                self.right.label: self.right.totals[metric],
                "Difference": delta["absolute"],
                "% change": delta["percent"],
            })
        return pd.DataFrame(rows)


def _side_totals(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    label: str,
    filters: list[dict[str, Any]],
    metrics: list[str],
) -> ComparisonSide:
    """Aggregate `metrics` for one filtered slice, via the existing query engine."""
    totals_df, _ = query_engine.groupby_agg(
        con, table_name, dims=[], metrics=metrics, aggs=["sum"], filters=filters
    )
    counts_df, _ = query_engine.groupby_agg(
        con, table_name, dims=[], metrics=metrics[:1], aggs=["count"], filters=filters
    )

    def _total(metric: str) -> float:
        """SUM() over zero matching rows returns NULL, which pandas surfaces as
        NaN. That must become 0.0 -- note `NaN or 0` does NOT work here, since NaN
        is truthy in Python and would pass straight through.
        """
        if totals_df.empty:
            return 0.0
        value = totals_df[f"{metric}_sum"].iloc[0]
        return 0.0 if pd.isna(value) else float(value)

    totals = {metric: _total(metric) for metric in metrics}
    n_rows = int(counts_df.iloc[0, 0]) if not counts_df.empty else 0
    return ComparisonSide(label=label, filters=filters, totals=totals, n_rows=n_rows)


def compare_dimension_values(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    dimension: str,
    left_value: str,
    right_value: str,
    metrics: list[str],
) -> ComparisonResult:
    """Compare two values of a categorical dimension (e.g. two regions)."""
    left = _side_totals(con, table_name, str(left_value),
                        [{"column": dimension, "op": "=", "value": left_value}], metrics)
    right = _side_totals(con, table_name, str(right_value),
                         [{"column": dimension, "op": "=", "value": right_value}], metrics)
    return _finalize(dimension, metrics, left, right)


def compare_date_ranges(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    date_column: str,
    left_range: tuple[Any, Any],
    right_range: tuple[Any, Any],
    metrics: list[str],
) -> ComparisonResult:
    """Compare two date ranges of the same metrics (e.g. this year vs last)."""
    left = _side_totals(
        con, table_name, f"{left_range[0]} to {left_range[1]}",
        [{"column": date_column, "op": "between", "value": [left_range[0], left_range[1]]}], metrics,
    )
    right = _side_totals(
        con, table_name, f"{right_range[0]} to {right_range[1]}",
        [{"column": date_column, "op": "between", "value": [right_range[0], right_range[1]]}], metrics,
    )
    return _finalize(date_column, metrics, left, right)


def _finalize(dimension: str, metrics: list[str],
              left: ComparisonSide, right: ComparisonSide) -> ComparisonResult:
    """Compute per-metric absolute and percentage differences (right vs left)."""
    deltas: dict[str, dict[str, float]] = {}
    for metric in metrics:
        left_value, right_value = left.totals[metric], right.totals[metric]
        absolute = right_value - left_value
        # Guard against a zero baseline -- a percentage change from nothing is
        # undefined, and reporting it as a huge number would be misleading.
        percent = (absolute / abs(left_value) * 100) if left_value else float("nan")
        deltas[metric] = {"absolute": absolute, "percent": percent}

    return ComparisonResult(
        dimension=dimension, metrics=metrics, left=left, right=right, deltas=deltas
    )


def build_comparison_chart(result: ComparisonResult, config: dict[str, Any]) -> go.Figure:
    """A grouped bar chart pairing both sides across every metric.

    Metrics are normalised to percentages of the larger side so measures on very
    different scales stay legible together; the real figures are in the
    side-by-side table and the hover text, so nothing is lost.
    """
    palette = config["charts"]["theme"]["categorical_colors"]
    labels = [m.replace("_", " ").title() for m in result.metrics]

    figure = go.Figure()
    for index, (side, color) in enumerate(((result.left, palette[0]), (result.right, palette[1]))):
        scaled, raw = [], []
        for metric in result.metrics:
            largest = max(abs(result.left.totals[metric]), abs(result.right.totals[metric])) or 1
            scaled.append(side.totals[metric] / largest * 100)
            raw.append(side.totals[metric])
        figure.add_trace(go.Bar(
            name=side.label, x=labels, y=scaled, customdata=raw,
            marker=dict(color=color, line=dict(width=2, color="white")),
            hovertemplate=f"{side.label}<br>%{{x}}: %{{customdata:,.0f}}<extra></extra>",
        ))

    figure.update_layout(barmode="group")
    return charts.apply_theme(
        figure, config,
        f"{result.left.label} vs {result.right.label}",
        x_title="Metric", y_title="% of larger side",
    )


def explain_comparison(result: ComparisonResult) -> ComparisonResult:
    """Ask the LLM for a structured side-by-side narrative, and return the result
    with `narrative`/`provider` filled in. All arithmetic is done in Python above;
    the LLM only describes figures it is handed.
    """
    system_prompt = (
        "You are a data analyst writing a side-by-side comparison for a business "
        "dashboard. You will be given two slices of the same dataset, their "
        "totals for each metric, and the differences between them. Write a short "
        "Markdown comparison: one sentence summarising the headline difference, "
        "then one bullet per metric giving both figures and the change. Use only "
        "the numbers provided -- they are already computed and rounded, so never "
        "recalculate, estimate, or add decimal places. Write figures with "
        "thousands separators. If a percentage change is null, say the baseline "
        "was zero rather than reporting a percentage."
    )

    # Round before the values ever reach the prompt. Handing over raw floats makes
    # the model echo them verbatim, and "108418.44890000022" in a business
    # narrative reads as a bug even though the arithmetic is right.
    def _round(values: dict[str, float]) -> dict[str, float]:
        return {key: round(value, 2) for key, value in values.items()}

    user_prompt = json.dumps({
        "compared_by": result.dimension,
        "left": {"label": result.left.label, "row_count": result.left.n_rows,
                 "totals": _round(result.left.totals)},
        "right": {"label": result.right.label, "row_count": result.right.n_rows,
                  "totals": _round(result.right.totals)},
        "differences_right_minus_left": {
            metric: {"absolute": round(delta["absolute"], 2),
                     "percent": None if math.isnan(delta["percent"]) else round(delta["percent"], 1)}
            for metric, delta in result.deltas.items()
        },
    }, indent=2, default=str)

    response = client.generate(system_prompt, user_prompt, json_mode=False)
    result.narrative = response.text
    result.provider = response.provider
    return result
