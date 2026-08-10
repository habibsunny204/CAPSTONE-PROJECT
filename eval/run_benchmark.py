"""Runs the fixed eval/benchmark_questions.json set through the live LLM pipeline,
scores accuracy, and logs results (Task B1/B5).

Scoring is a best-effort automatic check (numeric ground truth is matched against
any cell in the result with a small relative tolerance; out-of-scope questions are
checked for graceful handling rather than hallucination) -- full SQL/result/
narrative is logged for every question either way, since NL->SQL correctness isn't
perfectly auto-gradable and the written report's accuracy table should be
cross-checked by a human against this raw output, not taken as ground truth itself.

Re-run this every time a prompt in llm/prompts.py or configs/dataset_config.yaml's
synonyms/few_shot_examples changes, per PROJECT_SPEC.md Section 10.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from backend import ingest, quality
from eval import scoring
from llm import pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "benchmark_questions.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

def run() -> dict[str, Any]:
    """Ingest+clean the full dataset, run every benchmark question through the live
    pipeline (a dedicated cursor for LLM-generated SQL, per llm/sandbox.py's
    design), score each, print a summary, and write full evidence to eval/results/.
    """
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    config = ingest.load_config()
    table_name = config["dataset"]["table_name"]
    con = ingest.load_from_config()
    quality.clean(con, table_name, config)
    llm_con = con.cursor()

    results = []
    n_passed = 0
    for q in questions:
        entry: dict[str, Any] = {"id": q["id"], "category": q["category"], "question": q["question"]}
        pipeline_result: pipeline.PipelineResult | None = None
        error: Exception | None = None

        try:
            pipeline_result = pipeline.answer_question(llm_con, table_name, config, q["question"])
            entry.update(
                sql=pipeline_result.sql,
                reasoning=pipeline_result.reasoning,
                result_preview=pipeline_result.result.head(10).to_dict(orient="records"),
                narrative=pipeline_result.narrative,
                sql_provider=pipeline_result.sql_provider,
                retried=pipeline_result.retried,
            )
        except pipeline.PipelineError as e:
            error = e
            entry["error"] = str(e)

        score = scoring.score_question(q, pipeline_result, error)
        entry["passed"] = score["passed"]
        entry["score_method"] = score.get("method")
        n_passed += int(score["passed"])
        results.append(entry)

        print(f"[{'PASS' if score['passed'] else 'FAIL'}] {q['id']} ({q['category']}): {q['question']}")

    accuracy = n_passed / len(questions) if questions else 0.0
    evidence = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_questions": len(questions),
        "n_passed": n_passed,
        "accuracy": accuracy,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, default=str)

    print(f"\nAccuracy: {n_passed}/{len(questions)} ({accuracy:.0%})")
    print(f"Results written to {out_path.relative_to(REPO_ROOT)}")
    return evidence


if __name__ == "__main__":
    run()
