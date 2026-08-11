"""Tests for features/: anomaly detection (D1) and comparative analysis (D2).

Detection and comparison arithmetic are deterministic and tested without any LLM
call -- that separation is deliberate, since it means the parts that must be
*correct* (which rows are flagged, what the differences are) are provable, and the
LLM only ever describes figures Python already computed.
"""

import json
import math

import pandas as pd
import plotly.graph_objects as go
import pytest

from features import anomaly_detection, comparative_analysis
from llm import client

TABLE_NAME = "ecommerce_sales"


def _fake_generate(text="A narrative."):
    def fake(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        return client.LLMResult(text=text, provider="gemini", elapsed_ms=1.0)
    return fake


# ---------------------------------------------------------------------------
# D1 -- anomaly detection
# ---------------------------------------------------------------------------


def test_outlier_columns_finds_the_seeded_outlier(mini_con_clean, dataset_config):
    """The fixture's row 12 has total_revenue=5000 among values of 84-800."""
    columns = anomaly_detection.outlier_columns(mini_con_clean, TABLE_NAME, dataset_config)
    names = [c["name"] for c in columns]
    assert "total_revenue" in names


def test_outlier_columns_sorted_worst_first(mini_con_clean, dataset_config):
    columns = anomaly_detection.outlier_columns(mini_con_clean, TABLE_NAME, dataset_config)
    counts = [c["n_outliers_iqr"] for c in columns]
    assert counts == sorted(counts, reverse=True)


def test_detect_anomalies_flags_the_extreme_row(mini_con_clean, dataset_config):
    """The 5000 total_revenue value must be flagged, above the fence."""
    report = anomaly_detection.detect_anomalies(
        mini_con_clean, TABLE_NAME, dataset_config, column="total_revenue",
        context_columns=["transaction_date", "region"],
    )
    values = [a.value for a in report.anomalies]
    assert 5000 in values

    extreme = next(a for a in report.anomalies if a.value == 5000)
    assert extreme.direction == "above"
    assert extreme.severity > 0


def test_detect_anomalies_ranks_by_severity(mini_con_clean, dataset_config):
    report = anomaly_detection.detect_anomalies(
        mini_con_clean, TABLE_NAME, dataset_config, column="total_revenue", context_columns=["transaction_date"],
    )
    severities = [a.severity for a in report.anomalies]
    assert severities == sorted(severities, reverse=True)
    # The most extreme value in the fixture must rank first.
    assert report.anomalies[0].value == 5000


def test_detect_anomalies_respects_limit(mini_con_clean, dataset_config):
    report = anomaly_detection.detect_anomalies(
        mini_con_clean, TABLE_NAME, dataset_config, column="total_revenue",
        context_columns=["transaction_date"], limit=1,
    )
    assert len(report.anomalies) == 1


def test_detect_anomalies_unknown_column_raises(mini_con_clean, dataset_config):
    with pytest.raises(ValueError):
        anomaly_detection.detect_anomalies(
            mini_con_clean, TABLE_NAME, dataset_config, column="not_a_column"
        )


def test_detect_anomalies_makes_no_llm_call(mini_con_clean, dataset_config, monkeypatch):
    """Detection must be deterministic -- if it reached for the LLM, this fails."""
    def explode(*args, **kwargs):
        raise AssertionError("detection should not call the LLM")

    monkeypatch.setattr(anomaly_detection.client, "generate", explode)
    anomaly_detection.detect_anomalies(
        mini_con_clean, TABLE_NAME, dataset_config, column="total_revenue", context_columns=["transaction_date"]
    )


def test_report_to_frame_includes_direction_and_severity(mini_con_clean, dataset_config):
    report = anomaly_detection.detect_anomalies(
        mini_con_clean, TABLE_NAME, dataset_config, column="total_revenue", context_columns=["transaction_date"],
    )
    frame = report.to_frame()
    assert "_direction" in frame.columns
    assert "_severity_iqrs" in frame.columns
    assert len(frame) == len(report.anomalies)


def test_explain_anomalies_fills_in_narrative(mini_con_clean, dataset_config, monkeypatch):
    monkeypatch.setattr(anomaly_detection.client, "generate", _fake_generate("The 5000 transaction is unusual."))
    report = anomaly_detection.detect_anomalies(
        mini_con_clean, TABLE_NAME, dataset_config, column="total_revenue", context_columns=["transaction_date"],
    )
    explained = anomaly_detection.explain_anomalies(report)
    assert explained.narrative == "The 5000 transaction is unusual."
    assert explained.provider == "gemini"


def test_explain_anomalies_handles_no_anomalies():
    """An empty report must not call the LLM or crash."""
    empty = anomaly_detection.AnomalyReport(column="total_revenue", bounds={}, n_total_outliers=0)
    explained = anomaly_detection.explain_anomalies(empty)
    assert "No IQR-based outliers" in explained.narrative
    assert explained.provider == "none"


# ---------------------------------------------------------------------------
# D2 -- comparative analysis
# ---------------------------------------------------------------------------


def test_compare_dimension_values_totals_match_the_fixture(mini_con_clean, dataset_config):
    """Hand-computed from the fixture: Asia total_revenue 7594, Europe 706."""
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Asia", right_value="Europe", metrics=["total_revenue"],
    )
    assert result.left.totals["total_revenue"] == pytest.approx(7594)
    assert result.right.totals["total_revenue"] == pytest.approx(706)


def test_compare_computes_absolute_and_percent_deltas(mini_con_clean, dataset_config):
    """Deltas are right-minus-left: 706 - 7594 = -6888, i.e. -90.70%."""
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Asia", right_value="Europe", metrics=["total_revenue"],
    )
    delta = result.deltas["total_revenue"]
    assert delta["absolute"] == pytest.approx(-6888)
    assert delta["percent"] == pytest.approx(-90.70, abs=0.01)


def test_compare_handles_zero_baseline_without_dividing_by_zero(mini_con_clean, dataset_config):
    """A side with no matching rows gives a zero baseline -- percent change is
    undefined and must be NaN, not an exception or a misleading huge number.
    """
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Nowhere", right_value="Asia", metrics=["total_revenue"],
    )
    assert result.left.totals["total_revenue"] == 0
    assert math.isnan(result.deltas["total_revenue"]["percent"])


def test_compare_counts_rows_per_side(mini_con_clean, dataset_config):
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Asia", right_value="Europe", metrics=["total_revenue"],
    )
    assert result.left.n_rows == 8
    # Europe has 5 rows in the fixture, but rows 7 and 8 have a null total_revenue and
    # so contribute nothing to the metric being compared. Counting 3 here (not 5) is
    # what keeps n_rows consistent with the total it accompanies.
    assert result.right.n_rows == 3


def test_compare_date_ranges(mini_con_clean, dataset_config):
    """Both fixture months are in 2022; January vs February should both be non-empty."""
    result = comparative_analysis.compare_date_ranges(
        mini_con_clean, TABLE_NAME, date_column="transaction_date",
        left_range=("2022-01-01", "2022-01-31"),
        right_range=("2022-02-01", "2022-02-28"),
        metrics=["total_revenue"],
    )
    assert result.left.n_rows > 0
    assert result.right.n_rows > 0


def test_comparison_to_frame_is_tidy(mini_con_clean, dataset_config):
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Asia", right_value="Europe", metrics=["total_revenue", "price"],
    )
    frame = result.to_frame()
    assert list(frame["Metric"]) == ["Total Revenue", "Price"]
    assert "Asia" in frame.columns and "Europe" in frame.columns
    assert "Difference" in frame.columns and "% change" in frame.columns


def test_build_comparison_chart(mini_con_clean, dataset_config):
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Asia", right_value="Europe", metrics=["total_revenue", "price"],
    )
    figure = comparative_analysis.build_comparison_chart(result, dataset_config)
    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text
    assert len(figure.data) == 2


def test_explain_comparison_fills_in_narrative(mini_con_clean, dataset_config, monkeypatch):
    monkeypatch.setattr(comparative_analysis.client, "generate", _fake_generate("Asia leads."))
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Asia", right_value="Europe", metrics=["total_revenue"],
    )
    explained = comparative_analysis.explain_comparison(result)
    assert explained.narrative == "Asia leads."
    assert explained.provider == "gemini"


def test_comparison_prompt_rounds_figures(mini_con_clean, dataset_config, monkeypatch):
    """Regression test: handed raw floats, the model echoed values like
    '-6.434059163572309 percent' straight into the narrative. Everything in the
    prompt payload must be pre-rounded.
    """
    captured = {}

    def capture(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured["user_prompt"] = user_prompt
        return client.LLMResult(text="ok", provider="gemini", elapsed_ms=1.0)

    monkeypatch.setattr(comparative_analysis.client, "generate", capture)
    result = comparative_analysis.compare_dimension_values(
        mini_con_clean, TABLE_NAME, dimension="region",
        left_value="Asia", right_value="Europe", metrics=["total_revenue", "price"],
    )
    comparative_analysis.explain_comparison(result)

    payload = json.loads(captured["user_prompt"])
    numbers = [
        value
        for section in ("left", "right")
        for value in payload[section]["totals"].values()
    ] + [
        value
        for delta in payload["differences_right_minus_left"].values()
        for value in delta.values()
        if value is not None
    ]
    for number in numbers:
        assert round(number, 2) == number, f"{number} reached the prompt unrounded"


def test_anomaly_prompt_rounds_figures(mini_con_clean, dataset_config, monkeypatch):
    """Same rounding guarantee for the anomaly explanation prompt."""
    captured = {}

    def capture(system_prompt, user_prompt, json_mode=False, timeout_s=30.0):
        captured["user_prompt"] = user_prompt
        return client.LLMResult(text="ok", provider="gemini", elapsed_ms=1.0)

    monkeypatch.setattr(anomaly_detection.client, "generate", capture)
    report = anomaly_detection.detect_anomalies(
        mini_con_clean, TABLE_NAME, dataset_config, column="price",
        context_columns=["transaction_date", "total_revenue"],
    )
    anomaly_detection.explain_anomalies(report)

    payload = json.loads(captured["user_prompt"])
    for bound in payload["normal_range"].values():
        assert round(bound, 2) == bound
    for row in payload["flagged_rows"]:
        for value in row["values"].values():
            if isinstance(value, float):
                assert round(value, 2) == value
