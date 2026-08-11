"""Streamlit UI smoke tests via AppTest (PROJECT_SPEC.md Section 8).

Deliberately shallow: does the app boot, do the three tabs render, do the sidebar
filters exist, does filtering actually narrow the data. It does NOT touch the AI
Assistant's LLM calls -- those only fire on a button press, and asserting on live
model output would make the suite slow, costly, and non-deterministic. Pipeline
behaviour is covered by tests/test_integration.py instead.

These run against the real dataset, so they skip when data/raw/ is absent.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "app.py"
REAL_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "global_ecommerce_sales.csv"

pytestmark = pytest.mark.skipif(
    not REAL_CSV_PATH.exists(), reason="real dataset not present locally (data/raw/ is gitignored)"
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
