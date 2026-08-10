# Progress Checklist

Living checklist for this capstone project, maintained per `PROJECT_SPEC.md` Section 9. Status
values are restricted to `not started` / `in progress` / `done + tested`. Update this file at the
end of every session — a fresh session has no memory of prior ones beyond what's written here.

## Task A — Data layer (2.0 marks)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| A1 | Ingestion & schema (`backend/ingest.py`, `backend/schema.py`) | done + tested | Customer.ID PII decision resolved: hashed (sha256+salt) to `customer_id_hash`, salt from `PII_HASH_SALT` env var. `Customer.Name` dropped entirely. Full raw CSV moved to `data/raw/global_superstore.csv`. 9 tests passing in `tests/test_backend.py`. |
| A2 | Query engine (`backend/query_engine.py`: `groupby_agg`, `filtered_query`) | done + tested | `groupby_agg` gained an optional `filters` param beyond the spec's literal signature so "filtered aggregation" is one reusable call (flagged deviation, see plan doc). Both functions also take an explicit `table_name` param, matching `schema.get_schema`'s pattern. 12 new tests passing. |
| A3 | Data quality (`backend/quality.py`: profiling, cleaning, IQR outliers) | done + tested | `record_count` dropped (constant, zero info); `market_group`/`week_num` kept (documented as derived). Casing standardization scope was narrowed after testing against the real 51K-row dataset found naive `str.title()` mangling real data: abbreviations (`market`/`market_group`/`region` contain "US"/"EU"/"APAC"/"EMEA" -> "Us"/"Eu"/"Apac"/"Emea") and geographic proper nouns (`country`/`state`/`city`, e.g. "Rio de Janeiro" -> "Rio De Janeiro", "Cote d'Ivoire" -> "Cote D'Ivoire"). Casing standardization now only applies to `category`/`segment`/`ship_mode`/`order_priority`/`sub_category` (small closed-vocabulary business categories), which real-data verification confirms is a no-op on production data today. 8 new tests passing. |
| A4 | Performance benchmark (`backend/benchmark_perf.py`, <500ms evidence) | done + tested | 3 scenarios x 30 iterations against the full 51,290-row dataset: medians 2.7-6.7ms, ~75-180x under the 500ms bar. Evidence logged to `eval/results/perf_benchmark_20260810_221003.json`. 1 new test (skipped if `data/raw/` absent). |

## Task B — LLM integration (2.5 marks, heaviest component)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| B1 | Prompt design (schema + synonyms + few-shot) | done + tested | `llm/prompts.py`, 40+ entry synonym dict + 3 few-shot examples, `eval/benchmark_questions.json` (15 questions, all 5 categories), `eval/run_benchmark.py`. **Live run completed 2026-08-10: 15/15 (100%) accuracy** against real Gemini/Groq -- see `eval/results/benchmark_20260810_233113.json`. Gemini's free-tier daily quota (20 req/day) was exhausted almost immediately, so nearly every question actually exercised the live Groq fallback path, not just Gemini -- incidentally a strong real-world proof of B5's failover, not just the mocked tests. |
| B2 | Pipeline (Phase 1/2/3 + sandbox + single retry) | done + tested | `llm/sandbox.py` (allowlist-based sqlglot validation, 38 exploit tests), `llm/client.py` (Gemini->Groq failover, 9 mocked tests), `llm/pipeline.py` (Phase 1/2/3 + single retry + graceful "unanswerable" handling, 5 integration tests). |
| B3 | Insight generation (3 preset prompts) | done + tested | `llm/pipeline.py`: `generate_dataset_overview` (fully generic, no config needed), `generate_trend_comparison` (dims/metrics/aggs from `configs/dataset_config.yaml`'s new `insights.trend_comparison`), `generate_anomaly_report` (reuses A3's IQR profiling, picks the worst-outlier column at runtime -- data-driven, not hardcoded). 3 new tests. |
| B4 | Conversational context (last-5-turn memory) | done + tested | `llm/memory.py`: `ConversationMemory` wraps a plain list (storage-agnostic -- Task C backs an instance with `st.session_state`), `add`/`get_history`/`reset`. 4 new tests (eviction beyond 5 turns, reset, defensive copy on read). |
| B5 | Reliability (validation, loading indicator, Gemini->Groq failover) | in progress | Validation/failover/timing/logging done and **live-proven** (see B1 note above) -- also caught and fixed a real bug this way: google-genai's internal retry (tenacity) can re-raise a raw `httpx.ReadTimeout` that does NOT subclass the builtin `TimeoutError`, which crashed the whole request instead of falling back to Groq until `_GEMINI_SDK_ERRORS` was widened to include `httpx.HTTPError` (regression test added). Still open: the loading-indicator UI itself is Task C's job (`app.py` doesn't exist yet). |

## Task C — Dashboard (2.0 marks)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| C1 | Architecture (`st.tabs()`, sidebar filters, session state) | not started | |
| C2 | Visualization suite (>=6 chart types) | not started | |
| C3 | AI-driven visualization (shape -> chart-type auto-select) | not started | |
| C4 | Export (PDF, Word, PNG/SVG) | not started | |

## Task D — Advanced features (1.5 marks)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| D1 | Anomaly detection (extends A3's IQR logic) | not started | |
| D2 | Comparative analysis (two regions/date ranges) | not started | |

## Session log

- 2026-08-10: Repo scaffolded (empty/stub structure per Section 5). This checklist created.
  Full design plan produced for Task A (A1-A4) and approved; implementation not yet started.
  One blocking open decision recorded above (A1: Customer.ID PII handling).
- 2026-08-10: Task A (A1-A4) fully implemented, tested, and committed (4 commits:
  A1-A4). Customer.ID PII decision resolved (hashed). 30 tests passing in
  `tests/test_backend.py`. Real bug found and fixed during A3: naive `str.title()`
  casing standardization was mangling abbreviations and geographic proper nouns on
  the full dataset -- caught by testing against real data before trusting the fixture
  alone, scope narrowed accordingly (see A3 row above). A4 benchmark run against the
  full 51,290-row dataset: all scenarios ~75-180x under the 500ms bar. Git repo
  initialized locally (no remote). Task A is done + tested; Tasks B/C/D not started.
- 2026-08-10: Repo connected to a GitHub remote (habibsunny204/CAPSTONE-PROJECT,
  branch renamed master->main) via the IDE, not by the assistant. User will push
  manually going forward -- do not run `git push` (see memory: feedback_no_auto_push).
- 2026-08-10: Task B (B1-B5) implemented and tested, 6 commits. B2 (sandbox/client/
  pipeline) built and tested first using only mocked calls, since no API keys were
  available yet. Once the user added real GEMINI_API_KEY/GROQ_API_KEY to `.env`, ran
  a live smoke test of both providers, then the full B1 benchmark: 15/15 (100%)
  accuracy. Two real bugs found via live testing that no mock would have caught: (1)
  `load_dotenv()` was only called in `backend/ingest.py`, so `llm/client.py` failed
  with a KeyError when used standalone -- fixed by also loading .env in client.py;
  (2) google-genai's internal retry can leak a raw `httpx.ReadTimeout` that doesn't
  subclass the builtin `TimeoutError`, crashing the request instead of failing over
  to Groq -- fixed by widening the caught-exception tuple, with a regression test
  added. 90 tests passing overall. B1/B2/B3/B4 done + tested; B5's backend logic is
  done and live-proven, only the UI loading indicator (Task C's job) remains open.
  Tasks C/D not started.
