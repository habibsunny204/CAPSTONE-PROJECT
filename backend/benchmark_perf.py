"""Performance benchmark proving filtered aggregation is <500ms on the full dataset,
with results logged as evidence to eval/results/ (Task A4).

Runs groupby_agg(..., filters=[...]) scenarios declared in configs/dataset_config.yaml
(benchmark.scenarios) against the full, cleaned, real dataset -- config-driven so this
file contains no Superstore column-name literals, matching every other module in
backend/. Each scenario gets a few discarded warmup calls, then N timed iterations;
median/p95/etc. are computed from the timed runs and checked against the 500ms bar.
"""

from __future__ import annotations

import datetime
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import duckdb

from backend import ingest, query_engine, quality

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"

N_WARMUP = 2
N_ITERATIONS = 30
PASS_THRESHOLD_MS = 500.0


def _time_scenario(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    scenario: dict[str, Any],
    n_warmup: int = N_WARMUP,
    n_iterations: int = N_ITERATIONS,
) -> list[float]:
    """Run one scenario's groupby_agg() call n_warmup+n_iterations times, discarding
    the warmup runs, and return the list of timed elapsed_ms values.
    """
    dims, metrics, aggs = scenario["dims"], scenario["metrics"], scenario["aggs"]
    filters = scenario.get("filters")

    for _ in range(n_warmup):
        query_engine.groupby_agg(con, table_name, dims, metrics, aggs, filters=filters)

    timings = []
    for _ in range(n_iterations):
        _, elapsed_ms = query_engine.groupby_agg(con, table_name, dims, metrics, aggs, filters=filters)
        timings.append(elapsed_ms)
    return timings


def _stats(timings: list[float]) -> dict[str, float]:
    """min/median/mean/max/p95 of a list of timings (linear-interpolation-free
    nearest-rank p95, adequate at N=30)."""
    sorted_timings = sorted(timings)
    p95_index = min(len(sorted_timings) - 1, round(0.95 * (len(sorted_timings) - 1)))
    return {
        "min_ms": min(timings),
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "max_ms": max(timings),
        "p95_ms": sorted_timings[p95_index],
    }


def run_benchmark(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run every scenario, returning one result dict per scenario: its params, timing
    stats, and pass/fail against PASS_THRESHOLD_MS.
    """
    results = []
    for scenario in scenarios:
        timings = _time_scenario(con, table_name, scenario)
        stats = _stats(timings)
        results.append({
            "name": scenario["name"],
            "params": {
                "dims": scenario["dims"],
                "metrics": scenario["metrics"],
                "aggs": scenario["aggs"],
                "filters": scenario.get("filters"),
            },
            "n_iterations": N_ITERATIONS,
            "stats_ms": stats,
            "passed": stats["median_ms"] < PASS_THRESHOLD_MS,
        })
    return results


def main() -> int:
    """Ingest+clean the full real dataset, run all configured benchmark scenarios,
    print a summary, and write evidence to eval/results/. Returns a process exit code
    (0 if every scenario's median is under the 500ms bar, 1 otherwise).
    """
    config = ingest.load_config()
    table_name = config["dataset"]["table_name"]

    con = ingest.load_from_config()
    quality.clean(con, table_name, config)
    n_rows = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

    results = run_benchmark(con, table_name, config["benchmark"]["scenarios"])

    evidence = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "duckdb_version": duckdb.__version__,
        "n_rows": n_rows,
        "pass_threshold_ms": PASS_THRESHOLD_MS,
        "n_warmup": N_WARMUP,
        "n_iterations": N_ITERATIONS,
        "scenarios": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"perf_benchmark_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print(f"Benchmarked {n_rows} rows, {N_ITERATIONS} iterations per scenario "
          f"(threshold: {PASS_THRESHOLD_MS}ms median)\n")
    all_passed = True
    for r in results:
        all_passed = all_passed and r["passed"]
        status = "PASS" if r["passed"] else "FAIL"
        s = r["stats_ms"]
        print(f"[{status}] {r['name']}: median={s['median_ms']:.2f}ms "
              f"p95={s['p95_ms']:.2f}ms min={s['min_ms']:.2f}ms max={s['max_ms']:.2f}ms")
    print(f"\nEvidence written to {out_path.relative_to(REPO_ROOT)}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
