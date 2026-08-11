"""Streamlit UI smoke tests via AppTest (PROJECT_SPEC.md Section 8).

Deliberately shallow: does the app boot, do the three tabs render, do the sidebar
filters exist, does filtering actually narrow the data. It does NOT make live LLM
calls -- asserting on live model output would make the suite slow, costly, and
non-deterministic. Pipeline behaviour is covered by tests/test_integration.py instead.

Most LLM work only fires on a button press, which these tests never make. The one
exception is the C3 chart caption, which is generated when a stored answer *renders*,
so the `no_live_captions` autouse fixture below stubs it -- otherwise seeding a turn
would quietly spend real API quota.

These run against the real dataset, so they skip when data/raw/ is absent.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from llm import pipeline

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "app.py"
REAL_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "global_ecommerce_sales.csv"

STUB_CAPTION = "A stubbed caption."

pytestmark = pytest.mark.skipif(
    not REAL_CSV_PATH.exists(), reason="real dataset not present locally (data/raw/ is gitignored)"
)


@pytest.fixture(autouse=True)
def no_live_captions(monkeypatch):
    """Stub the C3 caption call for every test in this module.

    AppTest executes the app in this process, so app.py's `pipeline` is this module
    object and patching it here is enough. Autouse rather than opt-in because the call
    fires on *render* of any answer whose result is chartable -- it is easy to add a
    test that seeds a turn without realising it now costs an API call.
    """
    monkeypatch.setattr(
        pipeline, "generate_chart_caption",
        lambda *args, **kwargs: (STUB_CAPTION, "gemini"),
    )


@pytest.fixture(scope="module")
def app():
    """Boot the app once for the module -- ingesting 500k rows per test is wasteful."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    return at


def test_app_boots_without_exception(app):
    """The single most valuable UI assertion: the script runs top to bottom."""
    assert not app.exception, [str(e) for e in app.exception]


def test_app_renders_title(app):
    assert any("E-Commerce AI Analytics" in t.value for t in app.title)


def test_app_renders_the_expected_tabs(app):
    """C1 requires st.tabs() for Overview / Exploration / AI Assistant; Advanced
    is added on top of those for Task D's two features, which nest two more tabs
    inside it (so app.tabs, which is flat, sees six in total).
    """
    labels = [t.label for t in app.tabs]
    for expected in ("Overview", "Exploration", "AI Assistant", "Advanced"):
        assert expected in labels
    assert "Anomaly detection" in labels
    assert "Comparative analysis" in labels


def test_advanced_tab_has_both_task_d_controls(app):
    """Task D's two features are both reachable in the UI."""
    button_labels = [b.label for b in app.button]
    assert "Detect" in button_labels
    assert "Compare" in button_labels


def test_sidebar_has_the_configured_filters(app):
    """Filters are config-driven; assert the widgets actually materialized."""
    widget_labels = (
        [w.label for w in app.sidebar.multiselect] + [w.label for w in app.sidebar.date_input]
    )
    assert "Region" in widget_labels
    assert "Category" in widget_labels
    assert "Payment method" in widget_labels
    assert "Transaction date range" in widget_labels


def test_filters_actually_narrow_the_data(app):
    """Selecting a region must reduce the row count -- proves the filter is wired
    to the data, not just rendered.
    """
    before = app.sidebar.metric[0].value

    app.sidebar.multiselect(key="filter_region").select("Asia").run()

    after = app.sidebar.metric[0].value
    assert not app.exception, [str(e) for e in app.exception]
    assert after != before


def test_filters_scope_the_sql_path_not_just_the_charts(app):
    """The reported bug, as a regression test.

    With a filter applied, the sidebar showed a reduced row count while the AI
    Assistant and Task D kept answering from all 500,000 rows: the filters produced a
    pandas DataFrame that only the chart tabs consumed, and every DuckDB path queried
    the base table.

    The probe is the anomaly picker's outlier count. That number comes from
    quality.profile_report() executed through the same scoped cursor the AI Assistant
    uses, so it is genuine SQL against the scope -- if it moves when the sidebar
    moves, the SQL path is scoped. (backend/scope.py's own tests prove the scope
    matches the pandas rendering row-for-row; this proves the app is wired to it.)
    """
    def outlier_label():
        return app.selectbox(key="anomaly_column").options[0]

    app.sidebar.multiselect(key="filter_region").set_value([]).run()
    unfiltered = outlier_label()

    app.sidebar.multiselect(key="filter_region").select("Asia").run()
    assert not app.exception, [str(e) for e in app.exception]
    filtered = outlier_label()

    assert filtered != unfiltered, (
        f"outlier count unchanged under a filter ({unfiltered!r}) -- the SQL path is "
        "still reading the unfiltered table"
    )
    assert int(app.sidebar.metric[0].value.replace(",", "")) < 500_000


def test_task_d_option_lists_respect_the_filters(app):
    """Comparative analysis must not offer dimension values the sidebar excluded --
    otherwise a user filtered to one region can pick a comparison whose other side is
    necessarily empty.
    """
    app.sidebar.multiselect(key="filter_region").set_value([]).run()
    assert len(app.selectbox(key="comparison_left").options) > 1

    app.sidebar.multiselect(key="filter_region").select("Asia").run()
    assert app.selectbox(key="comparison_left").options == ["Asia"]


def test_scope_notice_appears_only_when_filtered(app):
    """An answer must never be read without its scope visible -- but an unfiltered
    dashboard shouldn't nag about filters that aren't applied.
    """
    app.sidebar.multiselect(key="filter_region").set_value([]).run()
    assert not any("Filtered view" in i.value for i in app.info)

    app.sidebar.multiselect(key="filter_region").select("Europe").run()
    assert any("Filtered view" in i.value for i in app.info)


def test_data_quality_panel_stays_dataset_wide(app):
    """The one deliberate exception: the quality profile documents what ingestion
    (Task A3) did to the source data, which is a property of the dataset rather than
    of a slice. It must not follow the filters even when everything else does.
    """
    app.sidebar.multiselect(key="filter_region").select("Asia").run()

    profiled = next(m for m in app.metric if m.label == "Rows profiled")
    assert profiled.value == "500,000"
    assert int(app.sidebar.metric[0].value.replace(",", "")) < 500_000


def test_ai_tab_has_chat_input_and_preset_buttons(app):
    """The AI Assistant tab's controls exist without any LLM call being made."""
    assert any(w.key == "chat_input" for w in app.chat_input)
    button_labels = [b.label for b in app.button]
    assert "Clear chat" in button_labels
    assert "Dataset overview" in button_labels
    assert "Anomaly report" in button_labels


def test_ai_tab_shows_an_empty_state_prompt(app):
    """With no conversation yet, the transcript invites a first question rather
    than rendering blank.
    """
    assert any("Ask me anything" in m.value for m in app.markdown)


def test_exports_are_not_generated_on_every_rerun(app):
    """Regression test for a real performance bug: chart images and PDF/Word
    reports were being built eagerly on every script rerun. Each one shells out to
    kaleido (~1s), so a single keystroke cost 10+ seconds with all charts on
    screen. They must now be behind an explicit 'Prepare' button, so a plain rerun
    stays fast.
    """
    import time

    started = time.perf_counter()
    app.run()
    elapsed = time.perf_counter() - started

    assert not app.exception, [str(e) for e in app.exception]
    assert elapsed < 5, f"a plain rerun took {elapsed:.1f}s -- exports are likely eager again"
    assert any(b.label == "Prepare PNG" for b in app.button)


@pytest.mark.live_llm
def test_ask_a_question_end_to_end(app):
    """The full user journey against live providers: type into the chat box, get
    back a narrative and an auto-selected chart in the transcript.

    Not run by default -- it costs real API quota and depends on a provider being
    up. Run explicitly with: pytest -m live_llm
    """
    question = "What is the total revenue by region?"
    app.chat_input(key="chat_input").set_value(question).run()

    assert not app.exception, [str(e) for e in app.exception]

    rendered = [m.value for m in app.markdown]
    assert question in rendered, "the user's own message should appear in the transcript"
    assert any(s.label == "Chart type" for s in app.selectbox)


# ---------------------------------------------------------------------------
# Error containment
# ---------------------------------------------------------------------------


def _seeded_answer(result_df):
    """A stored conversation turn, shaped the way _run_question stores one."""
    return {
        "kind": "question", "question": "list products with a price over 500",
        "sql": "SELECT product, region, category, price FROM ecommerce_sales WHERE price > 500",
        "reasoning": "filter on price", "result": result_df,
        "narrative": "Some products cost more than 500.", "provider": "gemini",
        "retried": False, "elapsed": 18.7,
    }


def test_metric_override_on_a_text_result_does_not_crash_the_page():
    """The reported crash, reproduced through the real app.

    With the chart-type override set to `metric` and a result whose first cell is a
    string, the headline-metric view formatted it with f"{v:,.2f}" and raised
    ValueError -- which Streamlit turns into a full-page traceback, wiping the tabs,
    the sidebar and every earlier answer.
    """
    import pandas as pd

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["turns"] = [_seeded_answer(pd.DataFrame({
        "product": ["Product_6551", "Product_3625"],
        "region": ["Asia", "North America"],
        "category": ["Sports & Outdoors", "Fashion"],
        "price": [612.23, 612.66],
    }))]
    at.session_state["chart_type_0"] = "metric"
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    # The answer still renders, and the page explains why it isn't a metric.
    assert any("can't be shown as a" in i.value for i in at.info)
    assert any("Some products cost more than 500." in m.value for m in at.markdown)


@pytest.mark.parametrize("chart_type", ["line", "bar", "scatter", "table", "metric"])
def test_no_chart_override_can_take_the_page_down(chart_type):
    """Every option the dropdown offers must be survivable on an awkward result --
    line/bar/scatter used to raise here too, for a different reason (no axes).
    """
    import pandas as pd

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    # The same shape as the reported crash: a mixed 3+-column listing. auto-select
    # calls this a table and leaves x/y as None, which is exactly what made every
    # non-table override raise inside Plotly.
    at.session_state["turns"] = [_seeded_answer(pd.DataFrame({
        "product": ["Product_6551", "Product_3625"],
        "region": ["Asia", "North America"],
        "category": ["Sports & Outdoors", "Fashion"],
        "price": [612.23, 612.66],
    }))]
    at.session_state["chart_type_0"] = chart_type
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    # Stricter than "the page survived": the per-turn guard would swallow a raise and
    # still leave at.exception empty, so this would pass even with the override logic
    # broken. Requiring no error box asserts the override actually produced something
    # viewable -- a figure, or a clean "can't be shown as a ..." fallback.
    assert not at.error, [e.value for e in at.error]


def test_a_broken_answer_does_not_hide_the_rest_of_the_transcript():
    """Turns are guarded individually: one answer that cannot be rendered reports
    itself in place, and the conversation around it survives.
    """
    import pandas as pd

    broken = _seeded_answer("not a dataframe at all")  # forces a render failure
    ok = _seeded_answer(pd.DataFrame({"region": ["Asia"], "revenue": [1.0]}))
    ok["narrative"] = "This answer is fine."

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["turns"] = [broken, ok]
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert any("Something went wrong while rendering" in e.value for e in at.error)
    assert any("This answer is fine." in m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# AI chart caption (Task C3)
# ---------------------------------------------------------------------------


def _chartable_answer():
    """A stored turn whose result auto-selects to a bar chart, so a caption renders."""
    import pandas as pd

    answer = _seeded_answer(pd.DataFrame({
        "region": ["Asia", "Europe", "Africa"],
        "total_revenue": [7594.0, 706.0, 803.0],
    }))
    answer["question"] = "What is total revenue by region?"
    return answer


def test_ai_chart_caption_is_llm_generated(monkeypatch):
    """C3 requires the caption to come from the model. The pre-fix code printed
    auto_select's hardcoded `reason` string, so assert the model's text is what
    reaches the page and that the deterministic string is no longer the caption.
    """
    monkeypatch.setattr(
        pipeline, "generate_chart_caption",
        lambda *args, **kwargs: ("Asia contributes over ten times Europe's revenue.", "groq"),
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["turns"] = [_chartable_answer()]
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]

    captions = [c.value for c in at.caption]
    assert any("Asia contributes over ten times Europe's revenue." in c for c in captions)
    assert any("caption by groq" in c for c in captions), "provider should be attributed"
    # The string the hardcoded implementation used to print.
    assert not any(
        "shown as a bar chart" in c for c in captions
    ), "the deterministic selection reason must no longer be the caption"


def test_ai_chart_caption_is_generated_once_across_reruns(monkeypatch):
    """The regression test for the real constraint here.

    Streamlit re-executes the script on every widget interaction, so an uncached call
    would fire a live request on every keystroke -- burning a rate-limited free tier,
    not just time. The caption must survive a rerun from cache.
    """
    calls = []

    def counting(*args, **kwargs):
        calls.append(args)
        return ("A caption.", "gemini")

    monkeypatch.setattr(pipeline, "generate_chart_caption", counting)

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["turns"] = [_chartable_answer()]
    at.run()
    assert len(calls) == 1, "first render should generate the caption"

    at.run()  # a plain rerun, as any widget interaction would cause
    assert len(calls) == 1, f"rerun must reuse the cached caption, got {len(calls)} calls"
    assert not at.exception, [str(e) for e in at.exception]


def test_overriding_the_chart_type_regenerates_the_caption(monkeypatch):
    """The flip side: a caption describing a bar chart is wrong once the user picks a
    line, so the cache is keyed on chart type rather than on the answer alone.
    """
    calls = []

    def counting(question, chart_type, *args, **kwargs):
        calls.append(chart_type)
        return (f"Caption for a {chart_type}.", "gemini")

    monkeypatch.setattr(pipeline, "generate_chart_caption", counting)

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["turns"] = [_chartable_answer()]
    at.run()
    assert calls == ["bar"]

    at.session_state["chart_type_0"] = "line"
    at.run()

    assert calls == ["bar", "line"]
    assert any("Caption for a line." in c.value for c in at.caption)


def test_caption_failure_falls_back_instead_of_losing_the_answer(monkeypatch):
    """A provider outage must not cost the reader the narrative and the chart. The
    per-turn guard would catch the raise, but trading a whole answer for a decorative
    sentence is the wrong outcome -- so this is handled where it happens.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("both providers failed")

    monkeypatch.setattr(pipeline, "generate_chart_caption", boom)

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["turns"] = [_chartable_answer()]
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    # The answer survives intact, and the caption degrades to the deterministic reason.
    assert any("Some products cost more than 500." in m.value for m in at.markdown)
    assert any("shown as a bar chart" in c.value for c in at.caption)


# ---------------------------------------------------------------------------
# Anomaly overlay chart (Task D3)
# ---------------------------------------------------------------------------


def test_anomaly_overlay_chart_renders_from_a_seeded_report():
    """D3 requires anomalies to be highlighted in the dashboard, not just listed in a
    table. Seed a report the way a real 'Detect' click would produce one and confirm
    the overlay chart renders alongside the table, with no crash or error box.
    """
    from features.anomaly_detection import Anomaly, AnomalyReport

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["anomaly_report"] = AnomalyReport(
        column="price",
        bounds={"lower": 5.0, "upper": 500.0, "q1": 50.0, "q3": 300.0},
        n_total_outliers=2,
        anomalies=[
            Anomaly(values={"transaction_date": "2022-03-01", "price": 612.23},
                    value=612.23, direction="above", severity=2.1),
            Anomaly(values={"transaction_date": "2022-05-01", "price": 1.0},
                    value=1.0, direction="below", severity=0.8),
        ],
        narrative="Two transactions sit outside the normal price range.",
        provider="gemini",
    )
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("Two transactions sit outside" in m.value for m in at.markdown)
    # AppTest exposes no accessor for st.plotly_chart, so the chart's own caption --
    # emitted only after charts.anomaly_scatter() returns without raising -- is the
    # available proxy for "the overlay actually built and rendered".
    assert any("Triangles mark the flagged rows" in c.value for c in at.caption)
    assert any(s.value == "Flagged rows" for s in at.subheader)


# ---------------------------------------------------------------------------
# Chart image export coverage (Task C4)
# ---------------------------------------------------------------------------


def test_overview_charts_have_image_export(app):
    """C4 asks for PNG/SVG export of individual charts. It previously covered only
    the Exploration tab -- the two Overview charts (time series, choropleth) had no
    download control at all.
    """
    downloads = [e for e in app.expander if e.label == "Download this chart"]
    # 6 from Exploration (box_plot, scatter_regression, correlation_heatmap, sunburst,
    # animated_bar, stacked_bar) + 2 from Overview (time_series, choropleth).
    assert len(downloads) >= 8


def test_ai_chart_has_image_export():
    """The AI-selected chart (C3) was the second uncovered figure -- assert its
    download control renders alongside a seeded answer.
    """
    import pandas as pd

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["turns"] = [_seeded_answer(pd.DataFrame({
        "region": ["Asia", "Europe"], "total_revenue": [7594.0, 706.0],
    }))]
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any(e.label == "Download this chart" for e in at.expander)


def test_comparison_chart_has_image_export():
    """The D5 comparison chart was the third uncovered figure."""
    from features.comparative_analysis import ComparisonResult, ComparisonSide

    left = ComparisonSide(label="Asia", filters=[], totals={"total_revenue": 7594.0}, n_rows=100)
    right = ComparisonSide(label="Europe", filters=[], totals={"total_revenue": 706.0}, n_rows=80)

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["comparison_result"] = ComparisonResult(
        dimension="region", metrics=["total_revenue"], left=left, right=right,
        deltas={"total_revenue": {"absolute": 6888.0, "percent": 975.5}},
        narrative="Asia outperforms Europe.", provider="gemini",
    )
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any(e.label == "Download this chart" for e in at.expander)
