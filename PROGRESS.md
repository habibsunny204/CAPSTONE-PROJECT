# Progress Checklist

Living checklist for this capstone project, maintained per `PROJECT_SPEC.md` Section 9. Status
values are restricted to `not started` / `in progress` / `done + tested`. Update this file at the
end of every session — a fresh session has no memory of prior ones beyond what's written here.

## Task A — Data layer (2.0 marks)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| A1 | Ingestion & schema (`backend/ingest.py`, `backend/schema.py`) | done + tested | Customer.ID PII decision resolved: hashed (sha256+salt) to `customer_id_hash`, salt from `PII_HASH_SALT` env var. `Customer.Name` dropped entirely. Full raw CSV moved to `data/raw/global_superstore.csv`. 9 tests passing in `tests/test_backend.py`. |
| A2 | Query engine (`backend/query_engine.py`: `groupby_agg`, `filtered_query`) | not started | |
| A3 | Data quality (`backend/quality.py`: profiling, cleaning, IQR outliers) | not started | Owns the deliberate keep/drop call on `record_count`/`market_group`/`week_num`. |
| A4 | Performance benchmark (`backend/benchmark_perf.py`, <500ms evidence) | not started | |

## Task B — LLM integration (2.5 marks, heaviest component)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| B1 | Prompt design (schema + synonyms + few-shot) | not started | |
| B2 | Pipeline (Phase 1/2/3 + sandbox + single retry) | not started | |
| B3 | Insight generation (3 preset prompts) | not started | |
| B4 | Conversational context (last-5-turn memory) | not started | |
| B5 | Reliability (validation, loading indicator, Gemini->Groq failover) | not started | |

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
