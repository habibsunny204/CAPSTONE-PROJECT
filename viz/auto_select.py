"""Result-shape -> chart-type rule table for AI-driven visualization (Task C3).

Fully generic by construction: every rule keys off the *shape* of the result
DataFrame -- how many columns, and whether each is datetime, numeric, or
categorical -- never off column identity. There are no dataset column names in
this file, and swapping in a different dataset would not change a line of it
(PROJECT_SPEC.md Section 2's generic/specific boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CHART_LINE = "line"
CHART_BAR = "bar"
CHART_SCATTER = "scatter"
CHART_TABLE = "table"
CHART_METRIC = "metric"

# Offered in the UI's override dropdown (Task C3 lets the user override the
# auto-selection).
ALL_CHART_TYPES = [CHART_LINE, CHART_BAR, CHART_SCATTER, CHART_TABLE, CHART_METRIC]


@dataclass
class ChartSelection:
    """The chosen chart type, which columns map to which axis, and why."""

    chart_type: str
    x: str | None
    y: str | None
    color: str | None
    reason: str


def _classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split columns into (datetime, numeric, categorical) by dtype alone."""
    datetime_cols, numeric_cols, categorical_cols = [], [], []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return datetime_cols, numeric_cols, categorical_cols


def select_chart(df: pd.DataFrame) -> ChartSelection:
    """Pick a chart type from `df`'s shape, per the spec's rule table: datetime +
    numeric -> line; categorical + numeric -> bar; numeric + numeric -> scatter;
    >=3 columns -> table. A single scalar cell renders as a headline metric rather
    than a one-point chart, and an empty result falls through to a table.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return ChartSelection(CHART_TABLE, None, None, None, "Empty result -- nothing to plot.")

    datetime_cols, numeric_cols, categorical_cols = _classify_columns(df)
    n_cols = len(df.columns)

    if n_cols == 1:
        col = df.columns[0]
        if len(df) == 1 and col in numeric_cols:
            return ChartSelection(CHART_METRIC, None, col, None,
                                  "Single numeric value -- shown as a headline figure.")
        if col in numeric_cols:
            return ChartSelection(CHART_TABLE, None, None, None,
                                  "One numeric column with no dimension to plot against.")
        return ChartSelection(CHART_TABLE, None, None, None, "Single non-numeric column.")

    if n_cols == 2:
        if datetime_cols and numeric_cols:
            return ChartSelection(CHART_LINE, datetime_cols[0], numeric_cols[0], None,
                                  "A date column and a numeric column -- shown as a time series.")
        if categorical_cols and numeric_cols:
            return ChartSelection(CHART_BAR, categorical_cols[0], numeric_cols[0], None,
                                  "A categorical column and a numeric column -- shown as a bar chart.")
        if len(numeric_cols) == 2:
            return ChartSelection(CHART_SCATTER, numeric_cols[0], numeric_cols[1], None,
                                  "Two numeric columns -- shown as a scatter plot.")
        return ChartSelection(CHART_TABLE, None, None, None,
                              "Two columns that don't map to a standard chart shape.")

    # >=3 columns: the spec's default is a table, but a date/categorical plus one
    # numeric plus one extra categorical is a legible multi-series chart, so prefer
    # that when the extra dimension is low-cardinality enough to read.
    if len(numeric_cols) == 1 and len(categorical_cols) + len(datetime_cols) == 2:
        metric = numeric_cols[0]
        series_col = min(categorical_cols, key=lambda c: df[c].nunique()) if categorical_cols else None
        if series_col is not None and df[series_col].nunique() <= 8:
            axis_candidates = datetime_cols + [c for c in categorical_cols if c != series_col]
            if axis_candidates:
                axis = axis_candidates[0]
                chart = CHART_LINE if axis in datetime_cols else CHART_BAR
                return ChartSelection(chart, axis, metric, series_col,
                                      f"Grouped by {series_col} -- shown as a multi-series {chart} chart.")

    return ChartSelection(CHART_TABLE, None, None, None,
                          f"{n_cols} columns -- shown as a table.")


def resolve_override(df: pd.DataFrame, chart_type: str) -> ChartSelection | None:
    """Axes for a user-chosen chart type, re-derived from `df`'s shape.

    select_chart() only ever proposes a chart the data can actually support, but the
    UI lets the user override that choice, and an override cannot reuse the
    auto-selection's axes: when select_chart() picked `table` it left x and y as None,
    and handing those to Plotly raises rather than degrading.

    Returns None when `df` genuinely cannot be drawn that way (asking for a scatter
    with only one numeric column, or a metric from a multi-row result), so the caller
    can say so instead of crashing. Same generic rule: dtype and column count only,
    never column identity.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return ChartSelection(CHART_TABLE, None, None, None, "Empty result -- nothing to plot.")

    if chart_type == CHART_TABLE:
        return ChartSelection(CHART_TABLE, None, None, None, "Shown as a table.")

    datetime_cols, numeric_cols, categorical_cols = _classify_columns(df)

    if chart_type == CHART_METRIC:
        if df.shape == (1, 1):
            return ChartSelection(CHART_METRIC, None, df.columns[0], None,
                                  "Single value -- shown as a headline figure.")
        return None

    if chart_type == CHART_SCATTER:
        if len(numeric_cols) >= 2:
            return ChartSelection(CHART_SCATTER, numeric_cols[0], numeric_cols[1], None,
                                  "Manually overridden to a scatter plot.")
        return None

    # Line and bar both want something to plot along x and a numeric y.
    if not numeric_cols:
        return None
    metric = numeric_cols[-1] if len(numeric_cols) > 1 else numeric_cols[0]
    axis_candidates = (datetime_cols + categorical_cols) if chart_type == CHART_LINE \
        else (categorical_cols + datetime_cols)
    axis = next((c for c in axis_candidates if c != metric), None)
    if axis is None:
        # Only numeric columns: one can still serve as the axis for the other.
        axis = next((c for c in numeric_cols if c != metric), None)
    if axis is None:
        return None

    return ChartSelection(chart_type, axis, metric, None,
                          f"Manually overridden to a {chart_type} chart.")
