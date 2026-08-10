"""Shared scoring for the benchmark and the prompt ablation.

Extracted so both harnesses score identically -- if they drifted apart, the
ablation's per-configuration accuracies wouldn't be comparable to the headline
benchmark number, which is the whole point of running it.

Scoring is deliberately loose: a question passes if the ground-truth figure(s)
appear anywhere in the result within a small tolerance. NL->SQL correctness isn't
perfectly auto-gradable (many different queries produce a right answer), so these
scores are a screening signal, and every run also logs the full SQL and result for
human review.
"""

from __future__ import annotations

import math
from typing import Any

NUMERIC_REL_TOL = 0.02
NUMERIC_ABS_TOL = 0.01


def numeric_match(actual: Any, expected: float) -> bool:
    """Whether `actual` is close enough to `expected` to count as a match,
    tolerating both a relative (2%) and a small absolute difference (rounding).
    """
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=NUMERIC_REL_TOL, abs_tol=NUMERIC_ABS_TOL
        )
    except (TypeError, ValueError):
        return False


def score_out_of_scope(question: dict[str, Any], result: Any, error: Exception | None) -> dict[str, Any]:
    """Out-of-scope questions pass if the system declines gracefully (ground truth
    "unanswerable": sql=="" or a clean PipelineError) or returns a genuinely empty
    result (ground truth "empty result (0 rows)") -- either way, not a hallucinated
    non-empty answer.
    """
    ground_truth = question["ground_truth"]
    if ground_truth == "unanswerable":
        passed = error is not None or (result is not None and result.sql == "")
    else:
        passed = result is not None and result.result.empty
    return {"passed": passed, "method": "out_of_scope_graceful_check"}


def score_answerable(question: dict[str, Any], result: Any, error: Exception | None) -> dict[str, Any]:
    """Scalar ground truth passes if any cell in the result is numerically close.
    Dict ground truth (e.g. a per-category breakdown) passes only if every numeric
    value in it is found somewhere in the result.
    """
    if result is None:
        return {"passed": False, "method": "pipeline_error", "detail": str(error)}

    ground_truth = question["ground_truth"]
    values = result.result.values.flatten().tolist() if not result.result.empty else []

    if isinstance(ground_truth, (int, float)):
        return {
            "passed": any(numeric_match(v, ground_truth) for v in values),
            "method": "scalar_numeric_match",
        }

    if isinstance(ground_truth, dict):
        targets = [v for v in ground_truth.values() if isinstance(v, (int, float))]
        matched = sum(1 for t in targets if any(numeric_match(v, t) for v in values))
        return {
            "passed": bool(targets) and matched == len(targets),
            "method": "dict_numeric_match", "matched": matched, "of": len(targets),
        }

    return {"passed": False, "method": "unrecognized_ground_truth_shape"}


def score_question(question: dict[str, Any], result: Any, error: Exception | None) -> dict[str, Any]:
    """Score one question, dispatching on its category."""
    if question["category"] == "out_of_scope":
        return score_out_of_scope(question, result, error)
    return score_answerable(question, result, error)
