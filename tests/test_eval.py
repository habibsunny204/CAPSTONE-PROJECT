"""Tests for eval/: shared scoring and the prompt-ablation harness.

The scoring rules are what turn raw pipeline output into the accuracy figures the
written report quotes, so they're worth testing directly -- a scorer that's too
lenient would silently inflate every number in the report.
"""

import pandas as pd
import pytest

from eval import run_ablation, scoring
from llm import pipeline, prompts


def _result(df: pd.DataFrame, sql: str = "SELECT 1") -> pipeline.PipelineResult:
    return pipeline.PipelineResult(
        question="q", sql=sql, reasoning="r", result=df, narrative="",
        sql_provider="gemini", narrative_provider="", retried=False,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_scalar_ground_truth_passes_on_close_match():
    question = {"category": "simple_aggregation", "ground_truth": 12642905}
    result = _result(pd.DataFrame({"total": [12642905.0]}))
    assert scoring.score_question(question, result, None)["passed"]


def test_scalar_ground_truth_fails_on_wrong_value():
    question = {"category": "simple_aggregation", "ground_truth": 12642905}
    result = _result(pd.DataFrame({"total": [999.0]}))
    assert not scoring.score_question(question, result, None)["passed"]


def test_scalar_tolerance_accepts_rounding_but_not_real_differences():
    question = {"category": "simple_aggregation", "ground_truth": 100.0}
    assert scoring.score_question(question, _result(pd.DataFrame({"v": [100.5]})), None)["passed"]
    assert not scoring.score_question(question, _result(pd.DataFrame({"v": [120.0]})), None)["passed"]


def test_dict_ground_truth_requires_every_value_present():
    question = {"category": "synonym_resolution",
                "ground_truth": {"Technology": 4744691, "Furniture": 4110884}}

    complete = _result(pd.DataFrame({"cat": ["a", "b"], "v": [4744691, 4110884]}))
    assert scoring.score_question(question, complete, None)["passed"]

    partial = _result(pd.DataFrame({"cat": ["a"], "v": [4744691]}))
    assert not scoring.score_question(question, partial, None)["passed"]


def test_out_of_scope_passes_when_declined():
    question = {"category": "out_of_scope", "ground_truth": "unanswerable"}
    declined = _result(pd.DataFrame(), sql="")
    assert scoring.score_question(question, declined, None)["passed"]


def test_out_of_scope_passes_on_pipeline_error():
    question = {"category": "out_of_scope", "ground_truth": "unanswerable"}
    error = pipeline.PipelineError("could not answer")
    assert scoring.score_question(question, None, error)["passed"]


def test_out_of_scope_fails_when_the_model_hallucinates_an_answer():
    """The whole point of the out-of-scope category: answering confidently is a
    failure, not a pass.
    """
    question = {"category": "out_of_scope", "ground_truth": "unanswerable"}
    hallucinated = _result(pd.DataFrame({"forecast": [123456]}), sql="SELECT 123456")
    assert not scoring.score_question(question, hallucinated, None)["passed"]


def test_empty_result_ground_truth_requires_an_empty_result():
    question = {"category": "out_of_scope", "ground_truth": "empty result (0 rows)"}
    assert scoring.score_question(question, _result(pd.DataFrame()), None)["passed"]
    assert not scoring.score_question(
        question, _result(pd.DataFrame({"v": [1]})), None
    )["passed"]


def test_pipeline_error_on_an_answerable_question_fails():
    question = {"category": "simple_aggregation", "ground_truth": 100}
    assert not scoring.score_question(question, None, pipeline.PipelineError("boom"))["passed"]


# ---------------------------------------------------------------------------
# Ablation harness
# ---------------------------------------------------------------------------


# Section markers, rather than individual words: a synonym key like "revenue"
# legitimately appears inside a few-shot example question too, so searching for
# the bare word would conflate the two components.
SYNONYM_SECTION = "Known synonyms"
FEW_SHOT_SECTION = "Examples of well-formed responses"


def test_prompt_flags_actually_remove_components(dataset_config):
    """The ablation is only meaningful if the flags change the prompt -- assert
    the synonym dictionary and few-shot examples really do drop out.
    """
    schema = [{"name": "total_revenue", "dtype": "BIGINT", "n_unique": 10, "n_null": 0,
               "sample_values": [1, 2, 3]}]

    full = prompts.build_sql_system_prompt(schema, "ecommerce_sales", dataset_config)
    schema_only = prompts.build_sql_system_prompt(
        schema, "ecommerce_sales", dataset_config, include_synonyms=False, include_few_shot=False
    )

    assert SYNONYM_SECTION in full and SYNONYM_SECTION not in schema_only
    assert FEW_SHOT_SECTION in full and FEW_SHOT_SECTION not in schema_only
    # The schema itself must survive in both -- that's the baseline, not a component.
    assert "total_revenue" in schema_only


def test_prompt_flags_are_independent(dataset_config):
    schema = [{"name": "total_revenue", "dtype": "BIGINT", "n_unique": 1, "n_null": 0, "sample_values": [1]}]

    synonyms_only = prompts.build_sql_system_prompt(
        schema, "ecommerce_sales", dataset_config, include_synonyms=True, include_few_shot=False
    )
    few_shot_only = prompts.build_sql_system_prompt(
        schema, "ecommerce_sales", dataset_config, include_synonyms=False, include_few_shot=True
    )

    assert SYNONYM_SECTION in synonyms_only and FEW_SHOT_SECTION not in synonyms_only
    assert FEW_SHOT_SECTION in few_shot_only and SYNONYM_SECTION not in few_shot_only


def test_ablation_configurations_isolate_one_component_each():
    """Each configuration should differ from the baseline by a single component,
    otherwise the sweep can't attribute a change to anything.
    """
    configurations = run_ablation.CONFIGURATIONS
    assert configurations["schema_only"] == {"include_synonyms": False, "include_few_shot": False}
    assert configurations["full_prompt"] == {"include_synonyms": True, "include_few_shot": True}

    for name, options in configurations.items():
        assert set(options) == {"include_synonyms", "include_few_shot"}, name


def test_accuracy_by_category():
    per_question = [
        {"category": "simple_aggregation", "passed": True},
        {"category": "simple_aggregation", "passed": False},
        {"category": "out_of_scope", "passed": True},
    ]
    by_category = run_ablation.accuracy_by_category(per_question)
    assert by_category["simple_aggregation"] == pytest.approx(0.5)
    assert by_category["out_of_scope"] == pytest.approx(1.0)


def test_format_markdown_table():
    summary = {
        "configurations": {
            "schema_only": {
                "accuracy": 0.6, "n_passed": 9, "n_questions": 15,
                "accuracy_by_category": {"simple_aggregation": 1.0, "synonym_resolution": 0.25},
            },
            "full_prompt": {
                "accuracy": 1.0, "n_passed": 15, "n_questions": 15,
                "accuracy_by_category": {"simple_aggregation": 1.0, "synonym_resolution": 1.0},
            },
        }
    }
    table = run_ablation.format_markdown_table(summary)
    assert "schema_only" in table and "full_prompt" in table
    assert "60%" in table and "100%" in table
    assert table.count("\n") == 3  # header + divider + two rows


@pytest.mark.parametrize("message,expected", [
    ("Both providers failed after 4272ms. Groq error: RateLimitError: 429", True),
    ("429 RESOURCE_EXHAUSTED quota exceeded", True),
    ("Read timeout", True),
    ("Binder Error: Referenced column 'revenue' not found", False),
    ("I couldn't produce a valid query for that question, even after one retry.", False),
])
def test_provider_failure_detection(message, expected):
    """Only provider-availability failures should abort a sweep. Bad SQL is a
    genuine accuracy signal and must count as a normal failed question.
    """
    assert run_ablation._is_provider_failure(RuntimeError(message)) is expected


def test_ablation_aborts_when_providers_are_exhausted(monkeypatch, dataset_config):
    """Regression test for a real incident: a sweep run against an exhausted free
    tier produced a complete-looking accuracy table (schema_only 20%, full_prompt
    13%) that was purely an artifact of run order -- the configuration that ran
    first spent the remaining quota. It must abort instead.
    """
    def always_rate_limited(*args, **kwargs):
        raise RuntimeError("Both providers failed after 4272ms. Groq error: 429 rate limit")

    monkeypatch.setattr(run_ablation.pipeline, "answer_question_sql_only", always_rate_limited)

    questions = [
        {"id": f"q{i}", "category": "simple_aggregation", "question": "?", "ground_truth": 1}
        for i in range(15)
    ]

    with pytest.raises(run_ablation.QuotaExhausted):
        run_ablation.run_configuration(
            llm_con=None, table_name="ecommerce_sales", config=dataset_config,
            questions=questions, prompt_options={},
        )


def test_ablation_tolerates_a_few_provider_failures(monkeypatch, dataset_config):
    """A single transient blip shouldn't abort an otherwise-valid sweep."""
    calls = {"n": 0}

    def fail_once_then_succeed(con, table_name, config, question, prompt_options=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Both providers failed: 429 rate limit")
        return _result(pd.DataFrame({"v": [1]}))

    monkeypatch.setattr(run_ablation.pipeline, "answer_question_sql_only", fail_once_then_succeed)
    monkeypatch.setattr(run_ablation.time, "sleep", lambda _: None)

    questions = [
        {"id": f"q{i}", "category": "simple_aggregation", "question": "?", "ground_truth": 1}
        for i in range(15)
    ]

    outcome = run_ablation.run_configuration(
        llm_con=None, table_name="ecommerce_sales", config=dataset_config,
        questions=questions, prompt_options={},
    )
    assert outcome["n_provider_failures"] == 1
    assert outcome["n_passed"] == 14


def test_answer_question_sql_only_skips_the_narrative(mini_con, dataset_config, monkeypatch):
    """The ablation's cost saving depends on Phase 3 never running -- assert it."""
    import json as json_module

    calls = []

    def fake_generate(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        calls.append(json_mode)
        from llm import client
        return client.LLMResult(
            text=json_module.dumps({"sql": "SELECT COUNT(*) AS n FROM ecommerce_sales",
                                    "reasoning": "count"}),
            provider="gemini", elapsed_ms=1.0,
        )

    monkeypatch.setattr(pipeline.client, "generate", fake_generate)
    result = pipeline.answer_question_sql_only(
        mini_con, "ecommerce_sales", dataset_config, "how many rows?"
    )

    assert len(calls) == 1, "only Phase 1 should call the LLM"
    assert result.narrative == ""
    assert result.result["n"].iloc[0] == 16
