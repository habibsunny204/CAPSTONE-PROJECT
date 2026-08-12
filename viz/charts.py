"""Plotly chart builders: the 8 curated dashboard charts (Task C2) plus the
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

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
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


def _drop_incomplete_edge_periods(series: pd.DataFrame, date_col: str,
                                  raw_dates: pd.Series, freq: str) -> pd.DataFrame:
    """Drop the first/last resampled bucket if the underlying data doesn't span
    that whole period.

    A sidebar date-range filter routinely cuts a month off partway through (e.g.
    "...-2024/01/01" leaves a single day in the January bucket). `resample` still
    labels that bucket at the full calendar month, so its sum is a fraction of a
    normal month's -- the line plunges toward zero right at the edge, which reads
    as a real drop in revenue rather than the filter boundary it actually is.
    Interior buckets are never touched: a real drop mid-series is real data.
    """
    if len(series) < 2:
        return series
    period_alias = freq[:-1] if freq.endswith("E") and len(freq) > 1 else freq
    periods = series[date_col].dt.to_period(period_alias)
    data_min, data_max = raw_dates.min(), raw_dates.max()
    keep = pd.Series(True, index=series.index)
    if data_min > periods.iloc[0].start_time:
        keep.iloc[0] = False
    if data_max < periods.iloc[-1].end_time:
        keep.iloc[-1] = False
    if not keep.any():
        return series
    return series[keep].reset_index(drop=True)


def _prettify(name: str) -> str:
    """Turn a snake_case column name into a human-readable axis label."""
    return name.replace("_", " ").title()


def format_scalar(value: Any) -> str:
    """Render a single result cell as display text.

    Thousands-separated to 2dp when the value is numeric, left as-is when it isn't.
    The obvious f"{value:,.2f}" raises ValueError on any non-numeric cell, which is
    how the headline-metric view took the whole dashboard down when the chart-type
    override was pointed at a text result.
    """
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass  # Not a scalar pandas can test -- fall through to str().
    if not isinstance(value, bool) and pd.api.types.is_number(value):
        return f"{value:,.2f}"
    return str(value)


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
    series = _drop_incomplete_edge_periods(series, date_col, df[date_col], freq)

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

    fig = apply_theme(fig, config, binding["title"],
                      x_title=binding.get("x_title", _prettify(date_col)))
    for i, metric in enumerate(metrics):
        fig.update_yaxes(title_text=_prettify(metric), secondary_y=(i == 1))
    fig.update_layout(hovermode="x unified")
    return fig


def animated_bar(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    """Slider-driven animated bar chart: one metric by one dimension, stepping
    through time (Task C2's "animated or transitioning chart").

    Three things here are deliberate, and each of them is a bug if left out:

    - **Pre-aggregated in pandas.** Handing Plotly the raw frame with an
      `animation_frame` ships every one of the 500k rows to the browser for every
      frame. Grouping to (period x dimension) first leaves a few hundred rows total.
      This is the same fix box_plot needed for the same reason.
    - **Every (period, dimension) pair is filled, with 0 for the missing ones.**
      Plotly builds its frames from whatever rows exist, so a dimension absent in one
      period would vanish and reappear mid-animation, and the remaining bars would
      slide sideways to close the gap -- which reads as movement in the data that
      didn't happen.
    - **The y-axis range is pinned across frames.** Plotly rescales per frame
      otherwise, so every period looks equally tall and the growth the animation
      exists to show disappears.
    """
    binding = _bindings(config, "animated_bar")
    date_col, dimension = binding["date_column"], binding["dimension"]
    metric, period = binding["metric"], binding.get("period", "M")

    frame = df[[date_col, dimension, metric]].copy()
    # Group on the Period values and stringify the ~25 aggregated labels afterwards,
    # not the 500k source rows: .astype(str) before the groupby costs ~220ms of the
    # ~300ms, on every rerun of the tab, for an identical result.
    frame["_period"] = frame[date_col].dt.to_period(period)

    totals = frame.groupby(["_period", dimension], as_index=False, observed=True)[metric].sum()
    totals["_period"] = totals["_period"].astype(str)

    # Complete grid, so no category can pop in or out between frames.
    periods = sorted(totals["_period"].unique())
    categories = sorted(totals[dimension].unique())
    grid = pd.MultiIndex.from_product([periods, categories], names=["_period", dimension])
    totals = (totals.set_index(["_period", dimension])
                    .reindex(grid, fill_value=0)
                    .reset_index())

    colors = _colors(config)
    bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
    hover = f"%{{x}}<br>{_prettify(metric)}: %{{y:,.0f}}<extra></extra>"

    def _bar(period_label: str) -> go.Bar:
        """The single bar trace for one frame, categories in fixed order."""
        values = totals.loc[totals["_period"] == period_label, metric]
        return go.Bar(x=categories, y=values.tolist(), marker=dict(color=bar_colors),
                      hovertemplate=hover)

    # Built with graph_objects rather than px.bar(animation_frame=...). Plotly Express
    # spent ~1.1s assembling these frames from an already-aggregated 200-row table --
    # a fixed cost, unrelated to data volume, paid on every rerun of this tab. Frames
    # are cheap to construct directly, and the rest of this module uses go.* anyway.
    fig = go.Figure(
        data=[_bar(periods[0])],
        frames=[go.Frame(data=[_bar(p)], name=p) for p in periods],
    )

    y_max = totals[metric].max()
    fig.update_yaxes(range=[0, y_max * 1.1 if y_max else 1])

    transition = dict(duration=250, easing="cubic-in-out")
    fig.update_layout(
        showlegend=False,
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.0, y=-0.28, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(
                    frame=dict(duration=600, redraw=True), fromcurrent=True,
                    transition=transition)]),
                dict(label="Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.12, y=-0.22, len=0.88,
            currentvalue=dict(prefix=f"{_prettify(date_col)}: "),
            steps=[dict(label=p, method="animate", args=[[p], dict(
                frame=dict(duration=300, redraw=True), mode="immediate",
                transition=transition)]) for p in periods],
        )],
    )

    fig = apply_theme(fig, config, binding["title"],
                      x_title=_prettify(dimension), y_title=_prettify(metric))
    # The play button and slider sit below the plot, so the default bottom margin
    # would clip them into the x-axis title.
    fig.update_layout(margin=dict(l=60, r=30, t=60, b=140))
    return fig


def _load_region_countries(path: str) -> dict[str, list[str]]:
    """Read a region -> member-countries mapping from a YAML file at `path`
    (relative to the repo root).
    """
    full_path = Path(__file__).resolve().parent.parent / path
    with open(full_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def choropleth(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
    """World choropleth of one metric aggregated by the configured location column.

    Handles two cases, both driven by config rather than by column identity:

    1. The location column already holds country names -- plotted directly.
    2. The location column holds coarser regions (this dataset's `region` is
       continent-level, and there is no country column). Setting
       `region_countries_path` on the binding expands each region to its member
       countries so the map has shapes to draw, filling every country in a region with
       that region's total.

    In case 2 the hover text stays bound to the REGION name and the REGION total, so a
    reader is never shown a country-level number the data does not contain. That
    distinction is the whole reason this branch is explicit rather than incidental.
    """
    binding = _bindings(config, "choropleth")
    location_col, metric = binding["location_column"], binding["metric"]
    locationmode = binding.get("locationmode", "country names")

    totals = df.groupby(location_col, as_index=False)[metric].sum()

    region_countries_path = binding.get("region_countries_path")
    if region_countries_path:
        region_countries = _load_region_countries(region_countries_path)
        expanded = totals.assign(
            _country=totals[location_col].map(lambda r: region_countries.get(r, []))
        ).explode("_country").dropna(subset=["_country"])
        plot_locations, custom_regions = expanded["_country"], expanded[location_col]
        plot_values = expanded[metric]
    else:
        plot_locations, custom_regions = totals[location_col], totals[location_col]
        plot_values = totals[metric]

    fig = go.Figure(go.Choropleth(
        locations=plot_locations, z=plot_values, locationmode=locationmode,
        colorscale=binding["color_scale"], customdata=custom_regions,
        colorbar=dict(title=_prettify(metric)),
        # Bound to customdata (the region), never %{location} (the country shape).
        hovertemplate=f"%{{customdata}}<br>{_prettify(metric)}: %{{z:,.0f}}<extra></extra>",
    ))
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
        q1, median, q3 = values.quantile(0.25), values.quantile(0.5), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lower_bounds.append(lower)
        upper_bounds.append(upper)

        # Quartiles are computed here and passed as precomputed statistics rather than
        # handing Plotly the raw y-values. At this dataset's scale that is the
        # difference between shipping ~500k floats to the browser per render and
        # shipping five numbers per box; the drawn result is identical because
        # boxpoints=False means the individual points were never rendered anyway.
        whisker_low = max(lower, values.min())
        whisker_high = min(upper, values.max())
        fig.add_trace(go.Box(
            q1=[q1], median=[median], q3=[q3],
            lowerfence=[whisker_low], upperfence=[whisker_high],
            name=str(name), marker_color=colors[i % len(colors)], boxpoints=False,
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
    """Scatter of two numeric measures with a least-squares regression line and a 95%
    confidence band (C2's "scatter plot with regression line and confidence interval").

    Large results are randomly sampled (fixed seed) before plotting -- 50k+ markers
    render slowly and overplot into a solid mass, so the sample is more readable as
    well as faster. The regression line and the band are computed on the full data,
    not the sample, so the trend shown is the real one.

    **What the band is, precisely** -- worth being able to state in Q&A, because at
    this sample size it draws as a very thin ribbon and "why is your CI invisible?" is
    a fair question. It is a confidence interval on the *mean* response: the range in
    which the fitted line itself plausibly sits, at +/-1.96 standard errors of the
    fit. It is NOT a prediction interval, which would describe where individual points
    fall and would be far wider (it would include the residual scatter, which here is
    enormous). The band narrows as n grows -- with 500k rows the position of the line
    is pinned down very precisely even though individual points are not.

    Uses the normal critical value 1.96 rather than a t distribution: they differ by
    under 0.1% past a few hundred degrees of freedom, far below a pixel here, and
    reaching for scipy to get a t-value would add a dependency for nothing.
    """
    binding = _bindings(config, "scatter_regression")
    x_col, y_col = binding["x"], binding["y"]
    colors = _colors(config)

    clean = df[[x_col, y_col]].dropna()
    x, y = clean[x_col].to_numpy(dtype=float), clean[y_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    plotted = clean.sample(sample_size, random_state=0) if len(clean) > sample_size else clean

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=plotted[x_col], y=plotted[y_col], mode="markers",
        name=binding.get("point_label", "Records"),
        marker=dict(color=colors[0], size=5, opacity=0.45),
        hovertemplate=f"{_prettify(x_col)}: %{{x}}<br>{_prettify(y_col)}: %{{y:,.2f}}<extra></extra>",
    ))

    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    lower, upper = _regression_confidence_band(x, y, line_x, slope, intercept)

    # Upper edge first, then the lower edge filling back to it: Plotly's "tonexty"
    # fills against the *previous* trace, so the order is load-bearing. Both edges are
    # hoverinfo="skip" so the band never steals the tooltip from the points.
    fig.add_trace(go.Scatter(
        x=line_x, y=upper, mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_translucent(colors[1], 0.25),
        name="95% confidence interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines", name="Trend",
        line=dict(color=colors[1], width=2),
        hovertemplate=f"Trend: %{{y:,.2f}}<extra></extra>",
    ))

    subtitle = f"{binding['title']} (trend: {slope:,.1f} per unit, 95% CI shaded)"
    return apply_theme(fig, config, subtitle, x_title=_prettify(x_col), y_title=_prettify(y_col))


def _translucent(hex_color: str, alpha: float) -> str:
    """A #rrggbb colour as an rgba() string, so a fill can reuse a palette entry at
    partial opacity without introducing a second, unvalidated colour.
    """
    raw = hex_color.lstrip("#")
    r, g, b = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _regression_confidence_band(
    x: np.ndarray, y: np.ndarray, line_x: np.ndarray, slope: float, intercept: float,
    z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    """95% confidence band for the fitted line, evaluated at `line_x`.

    Standard textbook form: the standard error of the fitted mean at a point x0 is
    s * sqrt(1/n + (x0 - xbar)^2 / Sxx), where s is the residual standard error with
    n-2 degrees of freedom. That (x0 - xbar)^2 term is why the band is narrowest at
    the mean of x and flares towards the extremes -- the slope's uncertainty pivots
    the line about its centroid.

    Degenerate inputs (fewer than 3 points, or every x identical, which makes Sxx
    zero and the slope undefined) return the line itself as both edges: a zero-width
    band, rather than a divide-by-zero or a NaN ribbon that would blank the chart.
    """
    n = len(x)
    x_mean = x.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if n < 3 or sxx == 0:
        fitted = slope * line_x + intercept
        return fitted, fitted

    residuals = y - (slope * x + intercept)
    s = np.sqrt(float((residuals ** 2).sum()) / (n - 2))
    se_fit = s * np.sqrt(1.0 / n + (line_x - x_mean) ** 2 / sxx)

    fitted = slope * line_x + intercept
    return fitted - z * se_fit, fitted + z * se_fit


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


# Binding name (the key under charts.curated_bindings in config) -> builder function.
# Keyed by binding name rather than by display title: titles are dataset-specific and
# live in config, while these keys are the stable structural names the code refers to.
CURATED_BUILDERS = {
    "time_series": time_series,
    "choropleth": choropleth,
    "correlation_heatmap": correlation_heatmap,
    "box_plot": box_plot,
    "sunburst": sunburst,
    "scatter_regression": scatter_regression,
    "stacked_bar": stacked_bar,
    "animated_bar": animated_bar,
}


def anomaly_scatter(
    all_rows: pd.DataFrame,
    flagged: pd.DataFrame,
    bounds: dict[str, float],
    column: str,
    config: dict[str, Any],
    sample_size: int = 5000,
) -> go.Figure:
    """Overlay the rows Task D3 flagged against everything else, with the IQR fences
    drawn in.

    Answers "highlight anomalies in the dashboard" (Task D3), which the flagged-rows
    table alone did not: a table lists what was flagged, this shows where those points
    sit relative to the fence and the bulk of normal values, in one picture.

    `all_rows` is the background -- sampled (fixed seed) the same way
    scatter_regression samples, since this dataset scale would otherwise overplot into
    a solid mass and slow the browser for no benefit. `flagged` (from
    AnomalyReport.to_frame()) is the small, already-curated set the whole chart exists
    to point at, so it is never sampled -- dropping even one flagged row here would
    make the chart disagree with the table sitting next to it.

    x-axis comes from config (`charts.anomaly_scatter.x_column`) rather than being
    guessed, so the same axis is used every time regardless of which column the user
    is currently examining -- `column` itself is chosen at runtime via the UI's
    selectbox, which is why (unlike the other curated charts) it is a parameter here
    rather than a config binding.
    """
    x_column = config["charts"]["anomaly_scatter"]["x_column"]
    alert_color = config["charts"]["theme"]["alert_color"]
    base_color = _colors(config)[0]

    background = all_rows[[x_column, column]].dropna()
    if len(background) > sample_size:
        background = background.sample(sample_size, random_state=0)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=background[x_column], y=background[column], mode="markers",
        name="Normal",
        marker=dict(color=base_color, size=5, opacity=0.35),
        hovertemplate=f"{_prettify(column)}: %{{y:,.2f}}<extra></extra>",
    ))

    if not flagged.empty:
        # Marker area scales with severity so the most extreme points read as more
        # prominent, not just differently coloured; capped so one huge outlier can't
        # swallow the plot.
        sizes = 8 + (flagged["_severity_iqrs"].clip(upper=10) * 2.5)
        symbols = flagged["_direction"].map({"above": "triangle-up", "below": "triangle-down"})
        fig.add_trace(go.Scatter(
            x=flagged[x_column], y=flagged[column], mode="markers", name="Flagged",
            marker=dict(color=alert_color, size=sizes, symbol=symbols,
                       line=dict(color="white", width=1)),
            customdata=flagged["_severity_iqrs"],
            hovertemplate=(
                f"{_prettify(column)}: %{{y:,.2f}}<br>"
                "%{customdata:.1f} IQRs past the fence<extra></extra>"
            ),
        ))

    for bound_name, bound_value in (("lower", bounds.get("lower")), ("upper", bounds.get("upper"))):
        if bound_value is not None:
            fig.add_hline(
                y=bound_value, line=dict(color="#666666", width=1, dash="dash"),
                annotation_text=f"{bound_name.title()} fence", annotation_position="right",
            )

    title = f"{_prettify(column)} Anomalies Over Time"
    return apply_theme(fig, config, title, x_title=_prettify(x_column), y_title=_prettify(column))


def curated_charts(config: dict[str, Any]) -> dict[str, Any]:
    """Display title -> builder function, for every configured curated chart.

    Titles are read from each binding rather than duplicated here, so the menu labels
    can never drift out of sync with the titles the charts actually render (they
    previously did), and retargeting the dataset stays a config-only change.
    """
    bindings = config["charts"]["curated_bindings"]
    return {
        bindings[name]["title"]: builder
        for name, builder in CURATED_BUILDERS.items()
        if name in bindings
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
