"""Prompt ablation: measures what each prompt component actually contributes to
NL->SQL accuracy (PROJECT_SPEC.md Section 10).

Runs the same fixed question set from eval/benchmark_questions.json under several
prompt configurations -- schema only, +synonyms, +few-shot, and the full prompt --
scoring each with eval/scoring.py so the numbers are directly comparable to the
headline benchmark. Without this, claims like "the synonym dictionary helps" are
assumptions; with it, they're measurements.

Only Phase 1 and Phase 2 run (SQL generation, validation, execution). The Phase 3
narrative is skipped: it doesn't affect whether the SQL was right, and skipping it
halves the API calls a sweep costs, which matters on a free tier.

Usage:
    python -m eval.run_ablation                 # all configurations
    python -m eval.run_ablation --repeats 3     # average over repeated runs

Interpreting the output: LLM sampling is non-deterministic, so a one- or two-point
difference between configurations on a 15-question set is noise, not signal. Treat
only large, consistent gaps as real, and prefer --repeats for anything you intend
to publish in the report.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any

from backend import ingest, quality
from eval import scoring
from llm import pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "benchmark_questions.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Seconds to pause between questions, to stay under free-tier per-minute rate
# limits during a long sweep.
PACING_SECONDS = 1.0

# If more than this fraction of a configuration's questions fail because *both*
# providers were unavailable, the run is measuring quota exhaustion rather than
# prompt quality, and its accuracy figures are meaningless.
PROVIDER_FAILURE_ABORT_RATIO = 0.2


class QuotaExhausted(RuntimeError):
    """Both providers became unavailable partway through a sweep.

    Raised instead of returning partial results, because a half-completed sweep
    still produces a plausible-looking accuracy table -- one where the
    configurations that happened to run first score highest purely by run order.
    That is worse than no result: it invites a confident conclusion from noise.
    """


def _is_provider_failure(error: Exception | None) -> bool:
    """Whether a failure came from the provider being unavailable (rate limit,
    quota, timeout) rather than from the model producing bad SQL. Only the latter
    is a real accuracy signal.
    """
    if error is None:
        return False
    text = str(error).lower()
    return any(marker in text for marker in
               ("both providers failed", "rate limit", "resource_exhausted",
                "quota", "429", "timeout"))

# The prompt configurations to compare, cumulative so each row isolates the
# contribution of one added component.
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "schema_only": {"include_synonyms": False, "include_few_shot": False},
    "schema_plus_synonyms": {"include_synonyms": True, "include_few_shot": False},
    "schema_plus_few_shot": {"include_synonyms": False, "include_few_shot": True},
    "full_prompt": {"include_synonyms": True, "include_few_shot": True},
}


def run_configuration(
    llm_con,
    table_name: str,
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    prompt_options: dict[str, Any],
) -> dict[str, Any]:
    """Run every question under one prompt configuration and score the results."""
    per_question = []
    n_passed = 0
    n_provider_failures = 0

    for question in questions:
        result = None
        error: Exception | None = None
        try:
            result = pipeline.answer_question_sql_only(
                llm_con, table_name, config, question["question"],
                prompt_options=prompt_options,
            )
        except Exception as caught:  # noqa: BLE001 - a failure is a failed question, not a crash
            error = caught

        if _is_provider_failure(error):
            n_provider_failures += 1
            if n_provider_failures > max(1, len(questions) * PROVIDER_FAILURE_ABORT_RATIO):
                raise QuotaExhausted(
                    f"{n_provider_failures} of {len(per_question) + 1} questions failed "
                    f"because both providers were unavailable, so these results would "
                    f"measure API quota, not prompt quality. Last error: {error}"
                )

        score = scoring.score_question(question, result, error)
        n_passed += int(score["passed"])
        per_question.append({
            "id": question["id"],
            "category": question["category"],
            "passed": score["passed"],
            "sql": result.sql if result else None,
            "error": str(error) if error else None,
        })
        time.sleep(PACING_SECONDS)

    return {
        "n_passed": n_passed,
        "n_questions": len(questions),
        "accuracy": n_passed / len(questions) if questions else 0.0,
        "n_provider_failures": n_provider_failures,
        "per_question": per_question,
    }


def accuracy_by_category(per_question: list[dict[str, Any]]) -> dict[str, float]:
    """Per-category accuracy, which is where an ablation usually shows its effect
    -- e.g. synonyms should matter most for the synonym_resolution questions and
    barely at all elsewhere.
    """
    by_category: dict[str, list[bool]] = {}
    for entry in per_question:
        by_category.setdefault(entry["category"], []).append(entry["passed"])
    return {
        category: sum(results) / len(results)
        for category, results in sorted(by_category.items())
    }


def format_markdown_table(summary: dict[str, Any]) -> str:
    """Render the results as a Markdown table, ready to paste into the report."""
    categories = sorted({
        category
        for configuration in summary["configurations"].values()
        for category in configuration["accuracy_by_category"]
    })

    header = "| Prompt configuration | Overall | " + " | ".join(
        c.replace("_", " ") for c in categories
    ) + " |"
    divider = "|---" * (len(categories) + 2) + "|"

    rows = []
    for name, configuration in summary["configurations"].items():
        cells = [
            f"{configuration['accuracy_by_category'].get(category, float('nan')):.0%}"
            for category in categories
        ]
        rows.append(
            f"| `{name}` | **{configuration['accuracy']:.0%}** "
            f"({configuration['n_passed']}/{configuration['n_questions']}) | "
            + " | ".join(cells) + " |"
        )

    return "\n".join([header, divider, *rows])


def run(repeats: int = 1) -> dict[str, Any]:
    """Run the full ablation sweep and write results to eval/results/."""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    config = ingest.load_config()
    table_name = config["dataset"]["table_name"]
    con = ingest.load_from_config()
    quality.clean(con, table_name, config)
    llm_con = con.cursor()

    results: dict[str, Any] = {}
    for name, prompt_options in CONFIGURATIONS.items():
        print(f"\n=== {name} ===")
        runs = []
        for repeat in range(repeats):
            outcome = run_configuration(llm_con, table_name, config, questions, prompt_options)
            runs.append(outcome)
            print(f"  run {repeat + 1}/{repeats}: "
                  f"{outcome['n_passed']}/{outcome['n_questions']} "
                  f"({outcome['accuracy']:.0%})")

        best = max(runs, key=lambda r: r["accuracy"])
        results[name] = {
            "prompt_options": prompt_options,
            "accuracy": sum(r["accuracy"] for r in runs) / len(runs),
            "n_passed": sum(r["n_passed"] for r in runs) // len(runs),
            "n_questions": runs[0]["n_questions"],
            "accuracy_per_run": [r["accuracy"] for r in runs],
            "accuracy_by_category": accuracy_by_category(best["per_question"]),
            "per_question": best["per_question"],
        }

    summary = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repeats": repeats,
        "n_questions": len(questions),
        "configurations": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"ablation_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + format_markdown_table(summary))
    print(f"\nResults written to {output_path.relative_to(REPO_ROOT)}")
    return summary


def main() -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Run the prompt ablation sweep.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per configuration; averages out LLM sampling noise")
    args = parser.parse_args()

    try:
        run(repeats=args.repeats)
    except QuotaExhausted as exhausted:
        print(f"\nAborted: {exhausted}")
        print(
            "\nNo results file was written -- a partial sweep would look like a real\n"
            "finding while actually just ranking configurations by run order.\n"
            "\nA full sweep needs roughly 130k input tokens (4 configurations x 15\n"
            "questions x ~2.1k tokens of schema/synonyms/few-shot), which exceeds\n"
            "Groq's free daily token allowance. Options: wait for the quota window to\n"
            "reset and re-run, spread configurations across days, or trim the question\n"
            "set with a smaller benchmark file."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
