"""End-to-end pipeline test(s): question -> SQL -> result -> narrative (Task B2),
against the small fixture dataset, asserting output shape rather than exact LLM
wording (PROJECT_SPEC.md Section 8). llm.client.generate is monkeypatched to return
controlled responses so this suite runs deterministically without live network calls
or API keys -- the same reasoning test_llm_client.py mocks the SDK calls directly,
not a relaxation of "no faked LLM responses in the shipped app" (which is about the
running application, not test doubles for a dependency it doesn't control).
"""

import json
import warnings

import pandas as pd
import pytest

from backend import scope
from llm import client, pipeline
from llm.memory import MAX_TURNS, ConversationMemory

TABLE_NAME = "ecommerce_sales"


def _fake_generate(*responses):
    """A fake llm.client.generate() that yields each of `responses` in order,
    wrapped as a successful LLMResult from "gemini".
    """
    it = iter(responses)

    def fake(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        return client.LLMResult(text=next(it), provider="gemini", elapsed_ms=1.0)

    return fake


def test_answer_question_happy_path(mini_con, dataset_config, monkeypatch):
    """A full Phase 1 -> 2 -> 3 run against the fixture."""
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({
            "sql": "SELECT region, SUM(total_revenue) AS total_revenue FROM ecommerce_sales GROUP BY region",
            "reasoning": "group by region, sum total_revenue",
        }),
        "Revenue is led by **Asia** at 7594, ahead of Europe at 706.",
    ))

    result = pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "What is total revenue by region?")

    assert isinstance(result.result, pd.DataFrame)
    assert set(result.result.columns) == {"region", "total_revenue"}
    assert len(result.result) == 3
    assert result.sql.strip().lower().startswith("select")
    assert result.narrative
    assert result.retried is False
    assert result.sql_provider == "gemini"
    assert result.narrative_provider == "gemini"


def test_answer_question_retries_once_on_bad_sql_then_succeeds(mini_con, dataset_config, monkeypatch):
    """First Phase 1 attempt references a table that doesn't exist (sandbox rejects
    it); the pipeline must retry exactly once and succeed on the corrected SQL.
    """
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({"sql": "SELECT * FROM not_a_real_table", "reasoning": "oops"}),
        json.dumps({"sql": "SELECT COUNT(*) AS n FROM ecommerce_sales", "reasoning": "corrected"}),
        "There are 16 rows in total.",
    ))

    result = pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "How many rows are there?")

    assert result.retried is True
    assert result.result["n"].iloc[0] == 16


def test_answer_question_raises_pipeline_error_after_exhausting_retry(mini_con, dataset_config, monkeypatch):
    """Both the original and the single retry reference an unknown table -- the
    pipeline must not retry indefinitely, and must surface a clean PipelineError.
    """
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({"sql": "SELECT * FROM not_a_real_table", "reasoning": "oops"}),
        json.dumps({"sql": "SELECT * FROM still_not_real", "reasoning": "still wrong"}),
    ))

    with pytest.raises(pipeline.PipelineError):
        pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "How many rows are there?")


def test_answer_question_handles_declared_unanswerable(mini_con, dataset_config, monkeypatch):
    """When Phase 1 sets sql="" (the system prompt's documented "can't answer this"
    signal), the pipeline must skip Phase 2/3 and surface the LLM's own reasoning.
    """
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({"sql": "", "reasoning": "This dataset has no forecasting capability."}),
    ))

    result = pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "What will next quarter's sales be?")

    assert result.sql == ""
    assert result.result.empty
    assert "forecasting" in result.narrative


def test_answer_question_uses_conversation_history(mini_con, dataset_config, monkeypatch):
    """History is threaded into the Phase 1 user prompt so follow-up questions can
    resolve references like "that" -- assert the history text actually reaches the
    prompt, not just that the call succeeds.
    """
    captured_prompts = []

    def fake_generate(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured_prompts.append(user_prompt)
        if len(captured_prompts) == 1:
            return client.LLMResult(
                text=json.dumps({"sql": "SELECT COUNT(*) AS n FROM ecommerce_sales", "reasoning": "count rows"}),
                provider="gemini", elapsed_ms=1.0,
            )
        return client.LLMResult(text="There are 16 rows.", provider="gemini", elapsed_ms=1.0)

    monkeypatch.setattr(pipeline.client, "generate", fake_generate)

    history = [{"question": "What is total revenue?", "sql": "SELECT SUM(total_revenue) FROM ecommerce_sales", "answer": "6777"}]
    pipeline.answer_question(mini_con, TABLE_NAME, dataset_config, "And how many rows is that?", history=history)

    assert "What is total revenue?" in captured_prompts[0]


# ---------------------------------------------------------------------------
# B3 -- preset insight generation
# ---------------------------------------------------------------------------


def test_generate_dataset_overview(mini_con_clean, dataset_config, monkeypatch):
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        "This dataset has 16 rows across West and East regions.",
    ))

    insight = pipeline.generate_dataset_overview(mini_con_clean, TABLE_NAME, dataset_config)

    assert insight.insight_type == "dataset_overview"
    assert insight.data["profile"]["n_rows"] == 16
    assert insight.narrative
    assert insight.narrative_provider == "gemini"


def test_generate_trend_comparison(mini_con_clean, dataset_config, monkeypatch):
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        "All fixture transactions fall in 2022, so there is a single year of data.",
    ))

    insight = pipeline.generate_trend_comparison(mini_con_clean, TABLE_NAME, dataset_config)

    assert insight.insight_type == "trend_comparison"
    df = insight.data["aggregation"]
    assert set(df.columns) == {"year", "total_revenue_sum", "quantity_sum"}
    assert insight.narrative


def test_generate_anomaly_report(mini_con_clean, dataset_config, monkeypatch):
    """The fixture's deliberate revenue outlier (row 12, total_revenue=5000) must surface."""
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        "One transaction has an unusually high revenue value compared to the rest.",
    ))

    insight = pipeline.generate_anomaly_report(mini_con_clean, TABLE_NAME, dataset_config)

    assert insight.insight_type == "anomaly_report"
    assert insight.data["profile"]["columns"]
    samples = insight.data["outlier_samples"]
    assert len(samples) >= 1
    assert insight.narrative


# ---------------------------------------------------------------------------
# B4 -- conversational memory
# ---------------------------------------------------------------------------


def test_conversation_memory_add_and_get_history():
    memory = ConversationMemory()
    memory.add("Q1", "SELECT 1", "A1")
    memory.add("Q2", "SELECT 2", "A2")

    history = memory.get_history()
    assert history == [
        {"question": "Q1", "sql": "SELECT 1", "answer": "A1"},
        {"question": "Q2", "sql": "SELECT 2", "answer": "A2"},
    ]


def test_conversation_memory_evicts_oldest_beyond_max_turns():
    memory = ConversationMemory()
    for i in range(MAX_TURNS + 3):
        memory.add(f"Q{i}", f"SELECT {i}", f"A{i}")

    history = memory.get_history()
    assert len(history) == MAX_TURNS
    assert [turn["question"] for turn in history] == [f"Q{i}" for i in range(3, MAX_TURNS + 3)]


def test_conversation_memory_reset_clears_history():
    memory = ConversationMemory()
    memory.add("Q1", "SELECT 1", "A1")
    memory.reset()
    assert memory.get_history() == []


def test_conversation_memory_get_history_is_a_copy():
    """Mutating the returned list must not affect internal state."""
    memory = ConversationMemory()
    memory.add("Q1", "SELECT 1", "A1")
    history = memory.get_history()
    history.append({"question": "injected", "sql": "", "answer": ""})
    assert len(memory.get_history()) == 1


# ---------------------------------------------------------------------------
# The pipeline under a sidebar filter scope
# ---------------------------------------------------------------------------


def test_answer_question_runs_against_the_scoped_view(mini_con_clean, dataset_config, monkeypatch):
    """The reported bug at pipeline level.

    The model emits ordinary unqualified SQL over the base table name; when the
    connection is scoped, that same SQL must total only the rows in scope. Nothing
    about the generated SQL changes -- only what the table name resolves to.
    """
    monkeypatch.setattr(pipeline.client, "generate", _fake_generate(
        json.dumps({"sql": f"SELECT SUM(total_revenue) AS t FROM {TABLE_NAME}",
                    "reasoning": "sum revenue"}),
        "Revenue totals are summarized above.",
    ))

    scoped = scope.scoped_cursor(
        mini_con_clean, TABLE_NAME, "pipeline_scope",
        [{"column": "region", "op": "in", "value": ["Asia"]}],
    )
    result = pipeline.answer_question(
        scoped, TABLE_NAME, dataset_config, "What is total revenue?",
        scope_description="Region is one of [Asia]",
    )

    # Asia's rows total 7594; the whole fixture totals 9103.
    assert result.result.iloc[0, 0] == pytest.approx(7594)
    assert result.retried is False


def test_scope_description_reaches_the_prompt_only_when_filtered(mini_con_clean, dataset_config,
                                                                monkeypatch):
    """With no filters the prompt must be byte-identical to the unscoped one -- that
    is what keeps the benchmark and ablation numbers comparable across this change.
    """
    captured = []

    def capture(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured.append(system_prompt)
        return client.LLMResult(
            text=json.dumps({"sql": f"SELECT COUNT(*) AS n FROM {TABLE_NAME}", "reasoning": "count"}),
            provider="gemini", elapsed_ms=1.0,
        )

    monkeypatch.setattr(pipeline.client, "generate", capture)

    pipeline.answer_question_sql_only(mini_con_clean, TABLE_NAME, dataset_config, "how many rows?")
    pipeline.answer_question_sql_only(
        mini_con_clean, TABLE_NAME, dataset_config, "how many rows?",
        scope_description="Region is one of [Asia]",
    )

    unscoped_prompt, scoped_prompt = captured
    assert "dashboard filters" not in unscoped_prompt
    assert "Region is one of [Asia]" in scoped_prompt
    assert "Do not add these conditions" in scoped_prompt


# ---------------------------------------------------------------------------
# Chart captions (Task C3)
# ---------------------------------------------------------------------------


def test_generate_chart_caption_returns_one_sentence_and_provider(monkeypatch):
    """The caption is a live LLM call, not a template -- assert the model's text is
    what comes back, and that the provider is reported alongside it.
    """
    monkeypatch.setattr(
        pipeline.client, "generate",
        _fake_generate("Asia leads revenue at 7,594, roughly ten times Europe's 706."),
    )

    caption, provider = pipeline.generate_chart_caption(
        "What is total revenue by region?", "bar",
        pd.DataFrame({"region": ["Asia", "Europe"], "total_revenue": [7594.0, 706.0]}),
        x_column="region", y_column="total_revenue",
    )

    assert caption == "Asia leads revenue at 7,594, roughly ten times Europe's 706."
    assert provider == "gemini"


def test_chart_caption_prompt_carries_the_chart_binding_and_rows(monkeypatch):
    """The prompt must describe what is actually plotted -- chart type, axes, and the
    rows -- otherwise the model can only restate the question, which is the failure
    mode the system prompt forbids.
    """
    captured = {}

    def capture(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return client.LLMResult(text="A caption.", provider="groq", elapsed_ms=1.0)

    monkeypatch.setattr(pipeline.client, "generate", capture)

    pipeline.generate_chart_caption(
        "Revenue by region?", "bar",
        pd.DataFrame({"region": ["Asia"], "total_revenue": [7594.0]}),
        x_column="region", y_column="total_revenue",
    )

    payload = json.loads(captured["user"])
    assert payload["chart_type"] == "bar"
    assert payload["x_axis"] == "region"
    assert payload["y_axis"] == "total_revenue"
    assert payload["plotted_rows"] == [{"region": "Asia", "total_revenue": 7594.0}]
    assert "exactly one sentence" in captured["system"]
    # The two failure modes the caption must avoid.
    assert "do not say what kind of chart it is" in captured["system"].lower()
    assert "do not restate the question" in captured["system"].lower()


def test_chart_caption_prompt_rounds_floats(monkeypatch):
    """Task D shipped "-6.434059163572309 percent" into a narrative before values were
    rounded at the prompt boundary. A caption is one sentence, so an unrounded float
    dominates it -- assert nothing longer than 2dp reaches the model.
    """
    captured = {}

    def capture(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured["user"] = user_prompt
        return client.LLMResult(text="A caption.", provider="groq", elapsed_ms=1.0)

    monkeypatch.setattr(pipeline.client, "generate", capture)

    pipeline.generate_chart_caption(
        "Average discount by region?", "bar",
        pd.DataFrame({"region": ["Asia"], "avg_discount": [14.293847562819374]}),
        x_column="region", y_column="avg_discount",
    )

    assert json.loads(captured["user"])["plotted_rows"] == [
        {"region": "Asia", "avg_discount": 14.29}
    ]


def test_chart_caption_prompt_handles_a_datetime_axis(monkeypatch):
    """A datetime x-axis is one of the two most common shapes reaching the caption.
    Rounding must not warn or fail on it (a bare DataFrame.round() warns on datetime
    dtypes), and the dates must survive into the payload.
    """
    captured = {}

    def capture(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured["user"] = user_prompt
        return client.LLMResult(text="A caption.", provider="groq", elapsed_ms=1.0)

    monkeypatch.setattr(pipeline.client, "generate", capture)

    frame = pd.DataFrame({
        "transaction_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "total_revenue": [1.23456, 2.34567],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline.generate_chart_caption(
            "Revenue over time?", "line", frame,
            x_column="transaction_date", y_column="total_revenue",
        )

    rows = json.loads(captured["user"])["plotted_rows"]
    assert [r["total_revenue"] for r in rows] == [1.23, 2.35]
    assert "2024-01-01" in rows[0]["transaction_date"]


def test_chart_caption_prompt_truncates_large_results(monkeypatch):
    """A caption summarises, so it does not need every row -- and a 500-row result
    would otherwise dominate the prompt budget for one sentence.
    """
    captured = {}

    def capture(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured["user"] = user_prompt
        return client.LLMResult(text="A caption.", provider="groq", elapsed_ms=1.0)

    monkeypatch.setattr(pipeline.client, "generate", capture)

    pipeline.generate_chart_caption(
        "Revenue by product?", "bar",
        pd.DataFrame({"product": [f"P{i}" for i in range(500)],
                      "total_revenue": [float(i) for i in range(500)]}),
        x_column="product", y_column="total_revenue",
    )

    payload = json.loads(captured["user"])
    assert len(payload["plotted_rows"]) == 30
    assert payload["row_count"] == 500
    assert payload["truncated"] is True
