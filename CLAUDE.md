# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a Masters-level Deep Learning capstone: a natural-language analytics platform over the
Global Superstore dataset. A user asks a plain-language question, an LLM turns it into
executable SQL, a sandboxed DuckDB layer runs it, and the result comes back as a chart plus an
LLM-written narrative. It also supports preset insight generation, multi-turn conversational
follow-ups, and PDF/Word export.

Grading is out of 13 marks: Task A (data layer) 2.0, Task B (LLM integration) 2.5 — the
heaviest and most likely to be probed in Q&A — Task C (dashboard) 2.0, Task D (advanced
features) 1.5, plus a separately-graded 5-mark presentation/report.

**The authoritative source of truth is [PROJECT_SPEC.md](PROJECT_SPEC.md). Read it in full
before writing any code.** It contains the complete tech stack decisions, architecture, target
repo layout, task-by-task spec, security requirements, and working agreements. This file only
summarizes and highlights; it does not replace the spec. Do not deviate from spec decisions
without flagging the deviation explicitly.

## Current repo state

This repo is **pre-implementation**. As of now it contains only `PROJECT_SPEC.md` and the raw
dataset — no `app/`, `backend/`, `llm/`, `viz/`, tests, `requirements.txt`, `.env.example`, or
`PROGRESS.md` exist yet. There is nothing to build/lint/run yet; the commands below are the
ones the spec defines for once scaffolding lands, not commands that work today.

Build in dependency order: **A before B** (B needs the schema), **B before C's AI-driven chart
selection** (C needs a result DataFrame to shape-detect on), **D last** (it reuses A2 and A3).

## Dataset reality check (verified directly against the raw CSV, not just the spec's prose)

- The raw file currently lives at repo root as `superstore.csv` (51,291 rows), not yet at the
  spec's expected `data/raw/global_superstore.csv` — move/reference it there during Task A1
  ingestion.
- Actual header (dot-separated, not the spaced names used in the spec's prose examples):
  `Category, City, Country, Customer.ID, Customer.Name, Discount, Market, 记录数, Order.Date,
  Order.ID, Order.Priority, Product.ID, Product.Name, Profit, Quantity, Region, Row.ID, Sales,
  Segment, Ship.Date, Ship.Mode, Shipping.Cost, State, Sub.Category, Year, Market2, weeknum`.
  Ingestion needs to normalize these to whatever naming convention the schema JSON uses, and the
  synonym dictionary in `configs/dataset_config.yaml` must map from the *actual* raw names.
- **PII confirmed present**: `Customer.Name` and `Customer.ID` are real columns in this file.
  Per the spec's Section 2 hard gate, these must be dropped or hashed during Task A1 ingestion —
  before any schema JSON derived from this data ever reaches an LLM prompt. Treat this as
  blocking, not a later cleanup step.
- Three columns aren't mentioned anywhere in the spec's column list: `记录数` (a non-English
  "record count" column), `Market2`, and `weeknum`. Decide deliberately what to do with them
  during Task A3 data-quality profiling rather than silently dropping or silently keeping them.

## Non-negotiable rules

- No pre-built dashboard template or existing analytics app as a starting point — built from scratch.
- No faked or hardcoded LLM responses in the shipped app — all AI calls must be live.
- One subtask, one commit, referencing the subtask ID (e.g. `A2: implement filtered_query with
  elapsed-time logging`). No dump commits — single-commit repos are explicitly penalized.
- Every module gets a module-level docstring; every function gets a docstring. This is a grading
  requirement, not a style preference — Q&A will probe whether the team can explain any line.

## Tech stack (do not substitute without discussion — see spec Section 3)

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Data layer | DuckDB, in-process (no client-server DB, no SQLAlchemy ORM) |
| Dashboard | Streamlit (`st.tabs()`, local dev via `streamlit run`) |
| LLM primary | Google Gemini 2.5 Flash (structured/JSON output mode) |
| LLM fallback | Groq (Llama 3.3 70B or similar), triggered only on Gemini error/timeout/rate-limit |
| SQL validation | `sqlglot` |
| Charts | Plotly |
| Word / PDF export | `python-docx` / `reportlab` or `weasyprint` |
| Testing | `pytest` |
| Env/secrets | `python-dotenv` + gitignored `.env` |

No multi-LLM-provider comparison beyond Gemini + Groq — explicitly out of scope.

## Architecture

```
Raw CSV -> DuckDB + schema + query engine (backend/)
        -> Phase 1: LLM generates SQL as structured JSON {"sql", "reasoning"}
        -> Phase 2: sqlglot validates, executes against a READ-ONLY DuckDB connection
        -> Phase 3: LLM generates a Markdown narrative from the result + original question
        -> Streamlit dashboard (tabs, filters, chat) -> PDF/Word/PNG export
```

**Critical invariant: the LLM never executes anything.** It only ever produces text — SQL in
Phase 1, narrative in Phase 3. Phase 2 is pure, deterministic, sandboxed Python running
*validated* SQL against a *read-only* connection. This is how the sandbox requirement (block
imports, file access, os/sys, network calls) is satisfied by construction — SQL has no `import`
statement or socket call to block in the first place. Keep this separation strict: if a feature
seems to need the LLM to "just run some Python," either express it as SQL, or (for Task D)
build a second, much narrower, separately-sandboxed pandas execution path — never loosen Phase 2.

On Phase 2 failure, send the error back to the LLM once for a single retry, then surface a clean
failure message. Don't retry indefinitely.

## Generic vs. dataset-specific boundary

The project targets Global Superstore only, but the specific/generic split must stay deliberate:

- **Must stay generic** (no Superstore column names as literals): `backend/` (ingestion, schema
  introspection, query engine, quality profiling), `llm/pipeline.py`, `llm/sandbox.py`,
  `viz/auto_select.py` (keys off result *shape* — datetime+numeric, categorical+numeric — never
  off column identity).
- **Allowed to be dataset-specific, but centralized**: the synonym dictionary and curated-chart
  column bindings belong in one place, `configs/dataset_config.yaml`. Everything else reads from
  that file rather than hardcoding literal column-name strings.

## Planned repo layout

Not yet built — this is the target structure from spec Section 5, to be filled in task order:

```
configs/dataset_config.yaml   # synonym map, dataset path, curated-chart column bindings
data/raw/                     # global_superstore.csv (gitignored)
app/app.py                    # Streamlit entrypoint
backend/                      # ingest.py, schema.py, query_engine.py, quality.py, benchmark_perf.py
llm/                          # client.py (Gemini->Groq failover), prompts.py, pipeline.py, sandbox.py, memory.py
viz/                          # charts.py, auto_select.py
features/                     # anomaly_detection.py, comparative_analysis.py  (Task D)
export/                       # pdf_export.py, docx_export.py
eval/                         # benchmark_questions.json, run_benchmark.py, results/
tests/                        # test_backend.py, test_sandbox_security.py, test_llm_client.py, test_integration.py
```

## Sandbox security requirements (Task B2, must be test-proven)

The SQL execution path must reject:
- Anything that isn't a single `SELECT`
- `ATTACH`, `DETACH`, `PRAGMA`, `INSTALL`, `LOAD`, `COPY`, `EXPORT`
- Any DDL (`CREATE`, `DROP`, `ALTER`) or DML (`INSERT`, `UPDATE`, `DELETE`)
- Multiple statements separated by `;`
- File-read attempts (`read_csv`/`read_parquet`) pointed outside the loaded table

If Task D or any future feature needs a pandas/`exec()` path, it must be a **separate, narrower**
sandbox: AST-walk generated code, reject `Import`/`ImportFrom`, reject `eval`/`exec`/`open`/
`__import__` calls, reject dunder attribute access, and run with a `builtins` dict that omits
`os`, `sys`, `subprocess`, `socket`, `open`. Write exploit-attempt tests for this path too.

## Testing

- `tests/test_backend.py` — schema extraction, query engine outputs against a small fixture, the
  <500ms performance assertion.
- `tests/test_sandbox_security.py` — every rejection case above, both SQL and any pandas path.
- `tests/test_llm_client.py` — mocked Gemini/Groq calls; assert failover on simulated
  timeout/error, and that a successful Gemini call never touches Groq.
- `tests/test_integration.py` — at least one full pipeline run (question -> SQL -> result ->
  narrative) against a small fixture, asserting output shape rather than exact LLM wording.
- A task is not "done" until its tests exist and the full suite passes.

## Working agreement

- Maintain `PROGRESS.md` at repo root as a running checklist of subtasks (A1, A2, ..., D2) with
  status (`not started` / `in progress` / `done + tested`). It doesn't exist yet — create it with
  the first subtask, and update it at the end of every session.
- `.env` is created locally from `.env.example` and never committed — verify `.gitignore`
  includes it before the first commit.
- Keep deployment-agnostic: no hardcoded deployment target (Streamlit Community Cloud vs.
  Hugging Face Spaces, etc.) — not decided yet.

## Commands (planned — not runnable until the corresponding scaffolding exists)

- Run the app: `streamlit run app/app.py`
- Run all tests: `pytest`
- Run a single test: `pytest tests/test_backend.py::test_name`
- Run the accuracy benchmark: `python eval/run_benchmark.py` (against `eval/benchmark_questions.json`)
- Run the perf benchmark: `python backend/benchmark_perf.py`
