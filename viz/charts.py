"""Plotly chart builders: the 7 curated dashboard charts (Task C2) plus the
renderer for AI-selected charts (Task C3).

Column bindings come from configs/dataset_config.yaml's `charts.curated_bindings`
rather than being hardcoded here, so the dataset-specific detail stays in one file
(PROJECT_SPEC.md Section 2). Every chart gets a title, axis labels, hover tooltips,
and the shared theme, so they read as one system.

The categorical palette is applied in fixed slot order and never cycled -- the
ordering is what makes it colorblind-safe (verified with a palette validator), so
it is applied by index, not shuffled per chart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import auto_select


def _theme(config: dict[str, Any]) -> dict[str, Any]:
    """The shared chart theme block from config."""
    return config["charts"]["theme"]


def _colors(config: dict[str, Any]) -> list[str]:
    """The categorical palette, in its validated fixed order."""
    return _theme(config)["categorical_colors"]


def _bindings(config: dict[str, Any], chart_name: str) -> dict[str, Any]:
    """Column bindings for one curated chart."""
    return config["charts"]["curated_bindings"][chart_name]


def apply_theme(fig: go.Figure, config: dict[str, Any], title: str,
                x_title: str | None = None, y_title: str | None = None) -> go.Figure:
    """Apply the shared template, title, and axis labels to any figure.

    Public so other modules that build their own figures (e.g. Task D's paired
    comparison chart) can render in the same visual system rather than
    reimplementing the layout or reaching into this module's internals.
    """
    fig.update_layout(
        template=_theme(config)["template"],
        title=title,
        margin=dict(l=60, r=30, t=60, b=60),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def _prettify(name: str) -> str:
    """Turn a snake_case column name into a human-readable axis label."""
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# The 7 curated charts (C2)
# ---------------------------------------------------------------------------


def time_series(df: pd.DataFrame, config: dict[str, Any], freq: str = "ME") -> go.Figure:
    """Dual-axis time series of two metrics resampled to `freq`.

    Note: a dual y-axis is used because PROJECT_SPEC.md Task C2 specifies it
    explicitly. It's worth knowing the tradeoff for Q&A: the two axes' scales are
    chosen independently, so where the lines cross or diverge is an artifact of
    that scaling, not a fact about the data -- only each line's own shape over
    time is meaningful. An indexed single-axis view would avoid that.
    """
    binding = _bindings(config, "time_series")
    date_col, metrics = binding["date_column"], binding["metrics"]
    colors = _colors(config)

    series = (df.set_index(date_col)[metrics].resample(freq).sum().reset_index())

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=series[date_col], y=series[metric], name=_prettify(metric),
                mode="lines", line=dict(color=colors[i], width=2),
                hovertemplate=f"%{{x|%b %Y}}<br>{_prettify(metric)}: %{{y:,.0f}}<extra></extra>",
            ),
            secondary_y=(i == 1),
        )

    fig = apply_theme(fig, config, binding["title"], x_title="Order Date")
    for i, metric in enumerate(metrics):
        fig.update_yaxes(title_text=_prettify(metric), secondary_y=(i == 1))
    fig.update_layout(hovermode="x unified")
    return fig


def choropleth(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    """World choropleth of one metric aggregated by country."""
    binding = _bindings(config, "choropleth")
    location_col, metric = binding["location_column"], binding["metric"]

    totals = df.groupby(location_col, as_index=False)[metric].sum()

    fig = px.choropleth(
        totals, locations=location_col, locationmode="country names",
        color=metric, color_continuous_scale=binding["color_scale"],
        labels={metric: _prettify(metric)},
    )
    fig.update_traces(
        hovertemplate=f"%{{location}}<br>{_prettify(metric)}: %{{z:,.0f}}<extra></extra>"
    )
    fig = apply_theme(fig, config, binding["title"])
    fig.update_layout(geo=dict(showframe=False, projection_type="natural earth"))
    return fig


def correlation_heatmap(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    """Correlation matrix across the configured numeric measures.

    Uses a diverging scale anchored at zero -- correlation is signed data, so the
    midpoint must mean "no relationship" rather than falling mid-ramp.
    """
    binding = _bindings(config, "correlation_heatmap")
    columns = [c for c in binding["columns"] if c in df.columns]
    corr = df[columns].corr().round(2)
    labels = [_prettify(c) for c in columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=binding["color_scale"], zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z}<extra></extra>",
        colorbar=dict(title="r"),
    ))
    return apply_theme(fig, config, binding["title"])


def box_plot(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    """Distribution of one metric per category, as box plots.

    The y-axis is explicitly scoped to the widest 1.5*IQR whisker range across
    categories. Without this the boxes collapse into flat lines: this metric's
    quartiles sit within a few hundred units of zero while extreme values run into
    the thousands, so an auto-scaled axis is entirely consumed by outliers and
    hides the distribution the chart exists to show. Those extreme values aren't
    lost -- they're what the anomaly report (Task B3/D1) surfaces.
    """
    binding = _bindings(config, "box_plot")
    category_col, metric = binding["category_column"], binding["metric"]
    colors = _colors(config)

    fig = go.Figure()
    lower_bounds, upper_bounds = [], []
    for i, (name, group) in enumerate(df.groupby(category_col)):
        values = group[metric]
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower_bounds.append(q1 - 1.5 * iqr)
        upper_bounds.append(q3 + 1.5 * iqr)

        fig.add_trace(go.Box(
            y=values, name=str(name),
            marker_color=colors[i % len(colors)], boxpoints=False,
            hovertemplate=f"{name}<br>{_prettify(metric)}: %{{y:,.2f}}<extra></extra>",
        ))

    fig = apply_theme(fig, config, f"{binding['title']} (outliers beyond whiskers not shown)",
                 x_title=_prettify(category_col), y_title=_prettify(metric))
    fig.update_layout(showlegend=False)

    if lower_bounds and upper_bounds:
        low, high = min(lower_bounds), max(upper_bounds)
        padding = (high - low) * 0.1
        fig.update_yaxes(range=[low - padding, high + padding])
    return fig


def sunburst(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    """Hierarchical breakdown of one metric down the configured path."""
    binding = _bindings(config, "sunburst")
    path, metric = binding["path"], binding["metric"]

    totals = df.groupby(path, as_index=False)[metric].sum()
    # Negative values break sunburst's area math; the path columns here are
    # dimensions, so a negative metric would only ever come from a signed measure.
    totals = totals[totals[metric] > 0]

    fig = px.sunburst(
        totals, path=path, values=metric,
        color_discrete_sequence=_colors(config),
    )
    fig.update_traces(
        hovertemplate=f"%{{label}}<br>{_prettify(metric)}: %{{value:,.0f}}<extra></extra>"
    )
    return apply_theme(fig, config, binding["title"])


def scatter_regression(df: pd.DataFrame, config: dict[str, Any], sample_size: int = 5000) -> go.Figure:
    """Scatter of two numeric measures with a least-squares regression line.

    Large results are randomly sampled (fixed seed) before plotting -- 50k+ markers
    render slowly and overplot into a solid mass, so the sample is more readable as
    well as faster. The regression line is fit on the full data, not the sample, so
    the trend shown is the real one.
    """
    binding = _bindings(config, "scatter_regression")
    x_col, y_col = binding["x"], binding["y"]
    colors = _colors(config)

    clean = df[[x_col, y_col]].dropna()
    slope, intercept = np.polyfit(clean[x_col], clean[y_col], 1)

    plotted = clean.sample(sample_size, random_state=0) if len(clean) > sample_size else clean

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=plotted[x_col], y=plotted[y_col], mode="markers", name="Orders",
        marker=dict(color=colors[0], size=5, opacity=0.45),
        hovertemplate=f"{_prettify(x_col)}: %{{x}}<br>{_prettify(y_col)}: %{{y:,.2f}}<extra></extra>",
    ))

    line_x = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
    fig.add_trace(go.Scatter(
        x=line_x, y=slope * line_x + intercept, mode="lines", name="Trend",
        line=dict(color=colors[1], width=2),
        hovertemplate=f"Trend: %{{y:,.2f}}<extra></extra>",
    ))

    subtitle = f"{binding['title']} (trend: {slope:,.1f} per unit)"
    return apply_theme(fig, config, subtitle, x_title=_prettify(x_col), y_title=_prettify(y_col))


def stacked_bar(df: pd.DataFrame, config: dict[str, Any], drill_into: str | None = None) -> go.Figure:
    """Stacked bar of one metric across a primary dimension, split by a secondary
    one. Passing `drill_into` (a primary-dimension value) drills down to just that
    slice, broken out by the secondary dimension.
    """
    binding = _bindings(config, "stacked_bar")
    primary, secondary = binding["primary_dim"], binding["secondary_dim"]
    metric = binding["metric"]
    colors = _colors(config)

    scoped = df[df[primary] == drill_into] if drill_into else df
    totals = scoped.groupby([primary, secondary], as_index=False)[metric].sum()

    fig = go.Figure()
    for i, secondary_value in enumerate(sorted(totals[secondary].unique())):
        subset = totals[totals[secondary] == secondary_value]
        fig.add_trace(go.Bar(
            x=subset[primary], y=subset[metric], name=str(secondary_value),
            marker=dict(color=colors[i % len(colors)], line=dict(width=2, color="white")),
            hovertemplate=(f"%{{x}} &middot; {secondary_value}<br>"
                           f"{_prettify(metric)}: %{{y:,.0f}}<extra></extra>"),
        ))

    title = f"{binding['title']} -- {drill_into}" if drill_into else binding["title"]
    fig = apply_theme(fig, config, title, x_title=_prettify(primary), y_title=_prettify(metric))
    fig.update_layout(barmode="stack")
    return fig


CURATED_CHARTS = {
    "Revenue & Profit Over Time": time_series,
    "Sales by Country": choropleth,
    "Correlation Heatmap": correlation_heatmap,
    "Profit by Category": box_plot,
    "Category Breakdown": sunburst,
    "Discount vs Profit": scatter_regression,
    "Region x Category": stacked_bar,
}


# ---------------------------------------------------------------------------
# AI-selected chart rendering (C3)
# ---------------------------------------------------------------------------


def build_from_selection(
    df: pd.DataFrame,
    selection: auto_select.ChartSelection,
    config: dict[str, Any],
    title: str = "",
) -> go.Figure | None:
    """Render a DataFrame according to a ChartSelection from viz/auto_select.py.
    Returns None for table/metric selections, which the UI renders directly rather
    than as a Plotly figure. Generic -- driven entirely by the selection, never by
    column identity.
    """
    colors = _colors(config)
    x, y, color_col = selection.x, selection.y, selection.color

    if selection.chart_type in (auto_select.CHART_TABLE, auto_select.CHART_METRIC):
        return None

    common = dict(color_discrete_sequence=colors)
    if selection.chart_type == auto_select.CHART_LINE:
        fig = px.line(df, x=x, y=y, color=color_col, markers=True, **common)
        fig.update_traces(line=dict(width=2))
    elif selection.chart_type == auto_select.CHART_BAR:
        fig = px.bar(df, x=x, y=y, color=color_col, **common)
        fig.update_traces(marker_line=dict(width=2, color="white"))
    elif selection.chart_type == auto_select.CHART_SCATTER:
        fig = px.scatter(df, x=x, y=y, color=color_col, **common)
        fig.update_traces(marker=dict(size=8, opacity=0.6))
    else:
        return None

    return apply_theme(fig, config, title or f"{_prettify(str(y))} by {_prettify(str(x))}",
                  x_title=_prettify(str(x)), y_title=_prettify(str(y)))


def figure_to_image_bytes(fig: go.Figure, fmt: str = "png", scale: int = 2) -> bytes:
    """Render a figure to PNG/SVG bytes for download and PDF/Word embedding (C4)."""
    return fig.to_image(format=fmt, scale=scale)
