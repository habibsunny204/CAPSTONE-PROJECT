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
