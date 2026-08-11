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
