"""Tests for viz/: shape-based chart auto-selection (C3) and the curated chart
builders (C2).

auto_select tests use hand-built DataFrames rather than the dataset fixture on
purpose -- the module is supposed to key off result *shape*, never column identity,
so testing it with generically-named columns is what actually proves that.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from viz import auto_select, charts

TABLE_NAME = "ecommerce_sales"


# ---------------------------------------------------------------------------
# C3 -- auto_select
# ---------------------------------------------------------------------------


def test_select_chart_datetime_plus_numeric_is_line():
    df = pd.DataFrame({"when": pd.to_datetime(["2013-01-01", "2013-02-01"]), "amount": [1.0, 2.0]})
    selection = auto_select.select_chart(df)
    assert selection.chart_type == auto_select.CHART_LINE
    assert selection.x == "when"
    assert selection.y == "amount"


def test_select_chart_categorical_plus_numeric_is_bar():
    df = pd.DataFrame({"grouping": ["a", "b"], "amount": [1.0, 2.0]})
    selection = auto_select.select_chart(df)
    assert selection.chart_type == auto_select.CHART_BAR
    assert selection.x == "grouping"
    assert selection.y == "amount"


def test_select_chart_numeric_plus_numeric_is_scatter():
    df = pd.DataFrame({"first": [1.0, 2.0], "second": [3.0, 4.0]})
    selection = auto_select.select_chart(df)
    assert selection.chart_type == auto_select.CHART_SCATTER


def test_select_chart_single_scalar_is_metric():
    df = pd.DataFrame({"total": [42.0]})
    selection = auto_select.select_chart(df)
    assert selection.chart_type == auto_select.CHART_METRIC
    assert selection.y == "total"


def test_select_chart_many_columns_is_table():
    df = pd.DataFrame({"a": ["x"], "b": ["y"], "c": ["z"], "d": [1], "e": [2]})
    selection = auto_select.select_chart(df)
    assert selection.chart_type == auto_select.CHART_TABLE


def test_select_chart_empty_result_is_table():
    selection = auto_select.select_chart(pd.DataFrame())
    assert selection.chart_type == auto_select.CHART_TABLE


def test_select_chart_three_columns_with_low_cardinality_series_is_multi_series():
    """A date + metric + small grouping column is more useful as a multi-series
    line than a raw table, even though it has 3 columns.
    """
    df = pd.DataFrame({
        "when": pd.to_datetime(["2013-01-01", "2013-01-01", "2013-02-01", "2013-02-01"]),
        "grouping": ["a", "b", "a", "b"],
        "amount": [1.0, 2.0, 3.0, 4.0],
    })
    selection = auto_select.select_chart(df)
    assert selection.chart_type == auto_select.CHART_LINE
    assert selection.color == "grouping"


def test_select_chart_three_columns_with_high_cardinality_falls_back_to_table():
    """Too many distinct series would be unreadable -- fall back to the table."""
    df = pd.DataFrame({
        "when": pd.to_datetime(["2013-01-01"] * 12),
        "grouping": [f"g{i}" for i in range(12)],
        "amount": list(range(12)),
    })
    selection = auto_select.select_chart(df)
    assert selection.chart_type == auto_select.CHART_TABLE


def test_select_chart_never_inspects_column_names():
    """The same shape must select the same chart type regardless of what the
    columns are called -- this is the generic/specific boundary, tested directly.
    """
    shape_a = pd.DataFrame({"region": ["a", "b"], "sales": [1.0, 2.0]})
    shape_b = pd.DataFrame({"zzz": ["a", "b"], "qqq": [1.0, 2.0]})
    assert auto_select.select_chart(shape_a).chart_type == auto_select.select_chart(shape_b).chart_type


# ---------------------------------------------------------------------------
# C2 -- curated charts, built against the real cleaned fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chart_name", list(charts.CURATED_BUILDERS))
def test_curated_charts_build_without_error(chart_name, mini_df, dataset_config):
    """Every curated chart builds a real Plotly figure from the fixture data."""
    fig = charts.CURATED_BUILDERS[chart_name](mini_df, dataset_config)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("chart_name", list(charts.CURATED_BUILDERS))
def test_curated_charts_have_a_title(chart_name, mini_df, dataset_config):
    """Titles are a spec requirement (C2: 'titles ... on every chart')."""
    fig = charts.CURATED_BUILDERS[chart_name](mini_df, dataset_config)
    assert fig.layout.title.text


def test_curated_chart_menu_titles_come_from_config(dataset_config):
    """The chart menu is built from each binding's configured title, so a menu label
    can never drift from the title the chart actually renders.
    """
    bindings = dataset_config["charts"]["curated_bindings"]
    menu = charts.curated_charts(dataset_config)
    assert set(menu) == {b["title"] for b in bindings.values()}
    assert len(menu) == len(charts.CURATED_BUILDERS)


def test_choropleth_hover_reports_the_region_not_the_country(mini_df, dataset_config):
    """This dataset has no country column -- the map shades member countries of each
    region, so the hover text must be bound to the region (customdata), never to the
    country shape (%{location}), or a reader would see a country-level figure that
    does not exist in the data.
    """
    binding = dataset_config["charts"]["curated_bindings"]["choropleth"]
    trace = charts.choropleth(mini_df, dataset_config).data[0]

    assert "%{customdata}" in trace.hovertemplate
    assert "%{location}" not in trace.hovertemplate

    # Every plotted country carries its region, and each region's value is that
    # region's total -- not a per-country split of it.
    regions = set(trace.customdata)
    assert regions == set(mini_df[binding["location_column"]].unique())

    totals = mini_df.groupby(binding["location_column"])[binding["metric"]].sum()
    for region, country_value in zip(trace.customdata, trace.z):
        assert country_value == pytest.approx(totals[region])


def test_stacked_bar_drilldown_narrows_to_one_primary_value(mini_df, dataset_config):
    binding = dataset_config["charts"]["curated_bindings"]["stacked_bar"]
    primary = binding["primary_dim"]
    target = mini_df[primary].iloc[0]

    fig = charts.stacked_bar(mini_df, dataset_config, drill_into=target)

    plotted_x = {x for trace in fig.data for x in trace.x}
    assert plotted_x == {target}


def test_box_plot_scopes_y_axis_to_whisker_range(dataset_config):
    """Regression test: with extreme outliers present, an auto-scaled y-axis
    collapses every box into a flat line. The axis must stay near the quartiles
    (verified visually against the real dataset, where revenue quartiles sit between
    ~470 and ~1890 while extremes reach nearly 5000).
    """
    binding = dataset_config["charts"]["curated_bindings"]["box_plot"]
    category_col, metric = binding["category_column"], binding["metric"]
    df = pd.DataFrame({
        category_col: ["A"] * 20 + ["B"] * 20,
        metric: ([10.0] * 19 + [9000.0]) + ([20.0] * 19 + [-9000.0]),
    })

    fig = charts.box_plot(df, dataset_config)
    low, high = fig.layout.yaxis.range

    assert low > -1000 and high < 1000


def test_charts_use_the_configured_palette_in_order(mini_df, dataset_config):
    """Colors are assigned by fixed slot order (the CVD-safety mechanism), not
    cycled arbitrarily -- assert the first traces use the first palette slots.
    """
    palette = dataset_config["charts"]["theme"]["categorical_colors"]
    fig = charts.time_series(mini_df, dataset_config)
    assert fig.data[0].line.color == palette[0]
    assert fig.data[1].line.color == palette[1]


def test_build_from_selection_returns_none_for_table(dataset_config):
    df = pd.DataFrame({"a": ["x"], "b": ["y"], "c": ["z"], "d": [1], "e": [2]})
    selection = auto_select.select_chart(df)
    assert charts.build_from_selection(df, selection, dataset_config) is None


def test_build_from_selection_builds_a_figure_for_bar(dataset_config):
    df = pd.DataFrame({"grouping": ["a", "b"], "amount": [1.0, 2.0]})
    selection = auto_select.select_chart(df)
    fig = charts.build_from_selection(df, selection, dataset_config)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text


# ---------------------------------------------------------------------------
# Chart-type override safety (C3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (42.0, "42.00"),
    (1234567.891, "1,234,567.89"),
    (0, "0.00"),
    (-71.9, "-71.90"),
    ("Product_8109", "Product_8109"),
    (None, "n/a"),
    (float("nan"), "n/a"),
    (True, "True"),
])
def test_format_scalar_handles_every_cell_type(value, expected):
    """Regression: the headline-metric view formatted with f"{v:,.2f}", which raises
    ValueError on a non-numeric cell and took the entire dashboard down with it.
    """
    assert charts.format_scalar(value) == expected


OVERRIDE_FRAMES = {
    "single_numeric": pd.DataFrame({"n": [42.0]}),
    "single_string": pd.DataFrame({"name": ["Product_1"]}),
    "categorical_numeric": pd.DataFrame({"region": ["A", "B"], "revenue": [1.0, 2.0]}),
    "datetime_numeric": pd.DataFrame({"d": pd.to_datetime(["2022-01-01", "2022-02-01"]),
                                      "v": [1.0, 2.0]}),
    "four_column_listing": pd.DataFrame({"product": ["P1", "P2"], "region": ["Asia", "Europe"],
                                         "category": ["Fashion", "Books"], "price": [612.2, 981.2]}),
    "all_strings": pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}),
    "empty": pd.DataFrame(),
}


@pytest.mark.parametrize("frame_name", list(OVERRIDE_FRAMES))
@pytest.mark.parametrize("chart_type", auto_select.ALL_CHART_TYPES)
def test_every_chart_override_is_safe_on_every_result_shape(frame_name, chart_type, dataset_config):
    """The override dropdown offers all five chart types for any result, so every
    combination has to either build a figure or decline cleanly -- never raise.

    Two combinations used to raise: `metric` on a text cell, and line/bar/scatter on a
    3+-column result (auto-select had chosen `table` and left x/y as None, which Plotly
    rejects). Both crashed the whole app.
    """
    df = OVERRIDE_FRAMES[frame_name]
    selection = auto_select.resolve_override(df, chart_type)
    if selection is None:
        return  # Declined -- the UI falls back to a table and says so.
    figure = charts.build_from_selection(df, selection, dataset_config)
    assert figure is None or isinstance(figure, go.Figure)


def test_resolve_override_declines_impossible_combinations():
    """Declining is a real outcome, not just an absence of crashing -- the UI shows an
    explanation and the table, so these must return None rather than something bogus.
    """
    text_only = OVERRIDE_FRAMES["all_strings"]
    assert auto_select.resolve_override(text_only, auto_select.CHART_BAR) is None
    assert auto_select.resolve_override(text_only, auto_select.CHART_SCATTER) is None

    multi_row = OVERRIDE_FRAMES["categorical_numeric"]
    assert auto_select.resolve_override(multi_row, auto_select.CHART_METRIC) is None

    one_numeric = OVERRIDE_FRAMES["categorical_numeric"]
    assert auto_select.resolve_override(one_numeric, auto_select.CHART_SCATTER) is None


def test_resolve_override_finds_axes_the_auto_selection_left_empty():
    """The bug's mechanism: auto-select returns x=None/y=None for a table, so an
    override must re-derive axes from the frame rather than reuse them.
    """
    df = OVERRIDE_FRAMES["four_column_listing"]
    assert auto_select.select_chart(df).x is None

    override = auto_select.resolve_override(df, auto_select.CHART_BAR)
    assert override is not None
    assert override.x in df.columns and override.y in df.columns
    assert pd.api.types.is_numeric_dtype(df[override.y])


# ---------------------------------------------------------------------------
# C2 -- animated chart
# ---------------------------------------------------------------------------


def test_animated_bar_has_frames_and_a_slider(mini_df, dataset_config):
    """C2's "animated or transitioning chart (slider-driven time animation)" needs
    both halves: real animation frames, and a slider to drive them.
    """
    fig = charts.animated_bar(mini_df, dataset_config)

    assert len(fig.frames) > 1, "an animation needs more than one frame"
    assert len(fig.layout.sliders) == 1
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)
    labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert "Play" in labels and "Pause" in labels


def test_animated_bar_pins_the_y_axis_across_frames(mini_df, dataset_config):
    """Plotly rescales the y-axis per frame by default, which makes every period look
    equally tall and hides the growth the animation exists to show. The range must be
    fixed, and it must actually contain the largest value in any frame.
    """
    fig = charts.animated_bar(mini_df, dataset_config)

    low, high = fig.layout.yaxis.range
    assert low == 0
    tallest = max(max(frame.data[0].y) for frame in fig.frames)
    assert high >= tallest


def test_animated_bar_keeps_every_category_in_every_frame(mini_df, dataset_config):
    """A category missing from one period would vanish mid-animation and the other
    bars would slide across to close the gap -- movement that isn't in the data. The
    grid is completed with zeros so the axis is stable.
    """
    fig = charts.animated_bar(mini_df, dataset_config)

    category_sets = {tuple(frame.data[0].x) for frame in fig.frames}
    assert len(category_sets) == 1, "categories must not change between frames"
    assert len({len(frame.data[0].y) for frame in fig.frames}) == 1


def test_animated_bar_preaggregates_rather_than_shipping_raw_rows(mini_df, dataset_config):
    """The perf invariant, as a test.

    An animation_frame over the raw frame sends every source row to the browser for
    every frame. Each frame should carry one value per category -- a few hundred
    numbers in total, not len(df) * n_frames.
    """
    fig = charts.animated_bar(mini_df, dataset_config)

    dimension = dataset_config["charts"]["curated_bindings"]["animated_bar"]["dimension"]
    n_categories = mini_df[dimension].nunique()
    for frame in fig.frames:
        assert len(frame.data[0].y) == n_categories

    total_points = sum(len(frame.data[0].y) for frame in fig.frames)
    assert total_points == n_categories * len(fig.frames)
    assert total_points < len(mini_df) * len(fig.frames)


def test_animated_bar_frame_values_match_the_underlying_totals(mini_df, dataset_config):
    """The animation must show the real numbers -- assert one frame against a
    hand-computed groupby rather than trusting the builder's own arithmetic.
    """
    binding = dataset_config["charts"]["curated_bindings"]["animated_bar"]
    date_col, dimension = binding["date_column"], binding["dimension"]
    metric, period = binding["metric"], binding.get("period", "M")

    fig = charts.animated_bar(mini_df, dataset_config)
    frame = fig.frames[0]

    expected = (mini_df[mini_df[date_col].dt.to_period(period).astype(str) == frame.name]
                .groupby(dimension)[metric].sum())
    for category, value in zip(frame.data[0].x, frame.data[0].y):
        assert value == pytest.approx(expected.get(category, 0.0))


# ---------------------------------------------------------------------------
# C2 -- regression confidence interval
# ---------------------------------------------------------------------------


def test_scatter_regression_draws_a_confidence_band(mini_df, dataset_config):
    """C2 asks for "regression line AND confidence interval" -- the chart had only
    the line. Assert a filled band trace exists alongside the trend.
    """
    fig = charts.scatter_regression(mini_df, dataset_config)

    filled = [t for t in fig.data if getattr(t, "fill", None) == "tonexty"]
    assert len(filled) == 1, "expected exactly one filled confidence band"
    assert "confidence" in filled[0].name.lower()
    assert any(t.name == "Trend" for t in fig.data)


def test_confidence_band_encloses_the_fitted_line(mini_df, dataset_config):
    """A band that doesn't contain its own line is drawn wrong (usually the two
    'tonexty' edges in the wrong order, which fills against the scatter instead).
    """
    import numpy as np

    fig = charts.scatter_regression(mini_df, dataset_config)
    upper = np.array(fig.data[1].y, dtype=float)
    lower = np.array(fig.data[2].y, dtype=float)
    trend = np.array(fig.data[3].y, dtype=float)

    assert np.all(lower <= trend + 1e-9)
    assert np.all(trend <= upper + 1e-9)


def test_confidence_band_is_narrowest_at_the_mean_of_x(mini_df, dataset_config):
    """The (x0 - xbar)^2 term is the whole character of a regression CI: the band
    pivots about the centroid, so it is tightest at the mean of x and flares towards
    both extremes. A band of constant width means that term was dropped.

    The narrowest point is located at xbar, NOT at the middle of the plotted range --
    the fixture's x is skewed (mean sits ~6% along min..max), so testing the midpoint
    would assert something false about correct code.
    """
    import numpy as np

    binding = dataset_config["charts"]["curated_bindings"]["scatter_regression"]
    # Dropped pairwise, exactly as the builder does. The fixture is deliberately
    # dirty and carries a null in the y column, so dropping on x alone gives a
    # different mean than the one the fit actually used.
    x = mini_df[[binding["x"], binding["y"]]].dropna()[binding["x"]].to_numpy(dtype=float)

    fig = charts.scatter_regression(mini_df, dataset_config)
    grid = np.array(fig.data[1].x, dtype=float)
    width = np.array(fig.data[1].y, dtype=float) - np.array(fig.data[2].y, dtype=float)

    assert np.all(width >= 0)
    assert width.argmin() == np.abs(grid - x.mean()).argmin()
    # Strictly widening as it moves away from the mean, in both directions.
    at_mean = width.argmin()
    assert np.all(np.diff(width[at_mean:]) > 0)
    assert np.all(np.diff(width[:at_mean + 1]) < 0)


def test_confidence_band_half_width_matches_the_closed_form():
    """Pin the arithmetic against the analytic value rather than the implementation.

    At x = xbar the standard error of the fit collapses to s/sqrt(n), so the half
    width there must be exactly 1.96 * s / sqrt(n). This is the assertion that would
    catch a wrong degrees-of-freedom term (n instead of n-2) or a dropped 1/n.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    y = 3.0 * x + 2.0 + rng.normal(0, 1.5, 200)
    slope, intercept = np.polyfit(x, y, 1)

    lower, upper = charts._regression_confidence_band(
        x, y, np.array([x.mean()]), slope, intercept
    )

    residuals = y - (slope * x + intercept)
    s = np.sqrt((residuals ** 2).sum() / (len(x) - 2))
    assert (upper[0] - lower[0]) / 2 == pytest.approx(1.96 * s / np.sqrt(len(x)))


def test_confidence_band_degenerate_inputs_collapse_to_the_line():
    """Two points, or every x identical, makes Sxx zero and the standard error
    undefined. Return a zero-width band rather than dividing by zero and blanking the
    chart with a NaN ribbon.
    """
    import numpy as np

    for x, y, slope, intercept in (
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 0.0),
        (np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0]), 0.0, 2.0),
    ):
        lower, upper = charts._regression_confidence_band(
            x, y, np.array([x.mean()]), slope, intercept
        )
        assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
        assert lower == pytest.approx(upper)


def test_confidence_band_covers_the_true_mean_response_at_the_nominal_rate():
    """The property that makes it a 95% interval at all: across repeated samples from
    a known model, the band should contain the true mean response about 95% of the
    time. Loose bounds -- this catches a band that is wildly mis-scaled (a missing
    1.96, or a standard deviation used where a standard error belongs), not sampling
    noise.
    """
    import numpy as np

    inside = 0
    trials = 300
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 10, 120)
        y = 3.0 * x + 2.0 + rng.normal(0, 1.5, 120)
        slope, intercept = np.polyfit(x, y, 1)
        lower, upper = charts._regression_confidence_band(
            x, y, np.array([7.0]), slope, intercept
        )
        inside += lower[0] <= (3.0 * 7.0 + 2.0) <= upper[0]

    assert 0.88 <= inside / trials <= 0.99, f"coverage {inside / trials:.1%}, expected ~95%"
