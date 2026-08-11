# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a Masters-level Deep Learning capstone: a natural-language analytics platform over the
Global E-Commerce Sales dataset. A user asks a plain-language question, an LLM turns it into
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

All tasks A1–D2 are **implemented and tested** — see `PROGRESS.md` for per-subtask status. The
full suite passes (`pytest`, 217 tests + 1 opt-in `live_llm` test).

The project was originally built against Global Superstore and has since been **retargeted to
Global E-Commerce Sales**. `PROJECT_SPEC.md` still describes the Superstore dataset in its prose
examples; the spec remains authoritative on architecture, tasks, and security, but this file is
the current truth on the dataset itself.

Build order for any new work: **A before B** (B needs the schema), **B before C's AI-driven chart
selection** (C needs a result DataFrame to shape-detect on), **D last** (it reuses A2 and A3).

## Dataset reality check (verified directly against the raw CSV, not just the spec's prose)

- The raw file lives at `data/raw/global_ecommerce_sales.csv` (500,000 rows, ~44 MB, gitignored).
- Actual header: `Transaction Date, Customer ID, Region, Product, Category, Price, Quantity,
  Discount (%), Total Revenue, Payment Method`. The rename map in `configs/dataset_config.yaml`
  normalizes these to snake_case, and the synonym dictionary maps informal phrases to those
  normalized names.
- **PII confirmed present**: `Customer ID` (98,348 distinct, `CUST_nnnnn`). Hashed to
  `customer_id_hash` during Task A1 ingestion, before any schema JSON reaches an LLM prompt.
  There is no customer-name column, so `pii.drop_columns` is configured but empty.
- **`Discount (%)` is a percentage on a 0–30 scale, not a 0–1 fraction.** This is the single
  most dangerous thing to get wrong: the previous dataset stored discount as a fraction, so
  "more than 20% discount" is now `discount_pct > 20`, not `> 0.2`. The column is deliberately
  renamed with a `_pct` suffix, a few-shot example calls the unit out explicitly, and
  `filtered_agg_1` in the benchmark exists to catch a regression (the right answer averages
  1133.06 over 166,032 rows; the fraction reading gives 1280.80 over 496,574).
- Data is clean on arrival: zero nulls, zero duplicate rows, and `Total Revenue` is exactly
  `Price × Quantity × (1 − Discount/100)`. `drop_after_profiling` is therefore an empty list —
  a recorded decision, not an oversight. The `tests/fixtures/` CSV is deliberately dirtier than
  production so the cleaning and profiling paths are still exercised.
- **Three structural gaps versus the old dataset**, each resolved deliberately:
  - *No profit column.* Charts are built on the four real measures (revenue, price, quantity,
    discount). Do **not** synthesize a profit column — fabricated data is indefensible in Q&A.
  - *No country column.* `Region` is six continents. The choropleth expands each region to its
    member countries (`configs/region_countries.yaml`, ISO-3) and binds hover text to the region,
    never the country shape.
  - *No Row.ID and no Year column.* `dataset.id_column` is omitted (consumers read it with
    `.get()`), and `year`/`month` are derived from `transaction_date` by A3's `derive_date_parts`
    step.
- `payment_method` is deliberately excluded from `categorical_casing_columns`: the cleaner
  title-cases values, which would mangle `PayPal` into `Paypal`. There is a test asserting this.

## Filter scope (added after the dataset migration)

The sidebar filters constrain **every** query path, not just the charts. This matters because
the two halves of the app consume data differently: the chart tabs hold a pandas DataFrame,
while the AI Assistant, the preset insights and both Task D features hold only
`(connection, table_name)`.

- `render_sidebar_filters()` (app.py) builds one canonical spec — `{"column", "op", "value"}`
  dicts, the shape `query_engine` already takes — and `backend/scope.py` renders it either to a
  pandas mask or to a DuckDB view.
- The view sits in a **per-session schema** and is **named identically to the base table**, with
  the LLM cursor's `search_path` pointed at that schema. So `table_name` never changes and the
  prompts, config few-shot SQL, and sandbox allowlist need no edits. Do not "fix" this by
  renaming the view — that breaks the few-shot examples, which contain literal `FROM <table>`.
- **`llm/sandbox.py` rejects schema/catalog-qualified table names.** This is load-bearing, not
  cosmetic: a bare name resolves through `search_path` to the scoped view, but `main.<table>`
  reaches around it to the unfiltered table. Do not relax it.
- The per-session schema name exists because the DuckDB connection is `@st.cache_resource`d and
  shared across browser sessions.
- One deliberate exception: the Overview tab's data-quality panel stays dataset-wide, since it
  documents A3 ingestion cleaning rather than a slice. Its caption says so, and a test pins it.
- `tests/test_scope.py` asserts the pandas and SQL renderings select the same rows. If you touch
  either renderer, that test is the one that catches divergence.

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

The project targets Global E-Commerce Sales only, but the specific/generic split must stay
deliberate — it is what made retargeting from the previous dataset a mostly config-only change:

- **Must stay generic** (no dataset column names as literals): `backend/` (ingestion, schema
  introspection, query engine, quality profiling, scope), `llm/pipeline.py`, `llm/sandbox.py`,
  `viz/auto_select.py` (keys off result *shape* — datetime+numeric, categorical+numeric — never
  off column identity).
- **Allowed to be dataset-specific, but centralized**: the synonym dictionary, curated-chart
  column bindings, sidebar filters, and dashboard chrome (title, KPI tiles, example questions)
  belong in one place, `configs/dataset_config.yaml` — plus `configs/region_countries.yaml`,
  which the choropleth alone reads. Everything else reads from those files rather than
  hardcoding literal column-name strings.

## Repo layout

Built out per spec Section 5:

```
configs/dataset_config.yaml   # synonym map, dataset path, curated-chart bindings, app chrome
configs/region_countries.yaml # region -> ISO-3 country codes (choropleth only)
data/raw/                     # global_ecommerce_sales.csv (gitignored)
app/app.py                    # Streamlit entrypoint
backend/                      # ingest.py, schema.py, query_engine.py, quality.py, scope.py, benchmark_perf.py
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
  status (`not started` / `in progress` / `done + tested`). Update it at the end of every session.
- `.env` is created locally from `.env.example` and never committed — verify `.gitignore`
  includes it before the first commit.
- Keep deployment-agnostic: no hardcoded deployment target (Streamlit Community Cloud vs.
  Hugging Face Spaces, etc.) — not decided yet.

## Commands

- Run the app: `streamlit run app/app.py`
- Run all tests: `pytest`
- Run a single test: `pytest tests/test_backend.py::test_name`
- Run the accuracy benchmark: `python -m eval.run_benchmark` (against `eval/benchmark_questions.json`)
- Run the perf benchmark: `python -m backend.benchmark_perf` (must be run as a module, not by path)
