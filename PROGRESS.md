# Progress Checklist

Living checklist for this capstone project, maintained per `PROJECT_SPEC.md` Section 9. Status
values are restricted to `not started` / `in progress` / `done + tested`. Update this file at the
end of every session — a fresh session has no memory of prior ones beyond what's written here.

## Task A — Data layer (2.0 marks)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| A1 | Ingestion & schema (`backend/ingest.py`, `backend/schema.py`) | done + tested | PII: `Customer ID` hashed (sha256+salt) to `customer_id_hash`, salt from `PII_HASH_SALT` env var; no customer-name column exists in this dataset so `drop_columns` is empty. Raw CSV at `data/raw/global_ecommerce_sales.csv` (500,000 rows). Hash path is presence-guarded (a configured PII column missing from the CSV is skipped, not fatal). 13 tests passing in `tests/test_backend.py`. |
| A2 | Query engine (`backend/query_engine.py`: `groupby_agg`, `filtered_query`) | done + tested | `groupby_agg` gained an optional `filters` param beyond the spec's literal signature so "filtered aggregation" is one reusable call (flagged deviation, see plan doc). Both functions also take an explicit `table_name` param, matching `schema.get_schema`'s pattern. 12 new tests passing. |
| A3 | Data quality (`backend/quality.py`: profiling, cleaning, IQR outliers) | done + tested | Cleaning is config-driven: parse dates, derive calendar parts, standardize casing, drop degenerate columns. `drop_after_profiling` is empty for this dataset -- unlike the previous one it carries no constant-value artifact column, recorded as a decision rather than left implicit. `derive_date_parts` was added to synthesize `year`/`month` from `transaction_date`, since this dataset has no Year column and the Trend Analysis preset needs a discrete time dimension. Casing standardization applies only to `category`/`region`; `payment_method` is deliberately excluded because `str.title()` mangles "PayPal" -> "Paypal", and `product` is excluded as a 10,000-value synthetic identifier. A test asserts the PayPal case directly. |
| A4 | Performance benchmark (`backend/benchmark_perf.py`, <500ms evidence) | done + tested | 3 scenarios x 30 iterations against the full 500,000-row dataset: medians 7.1-13.5ms, ~37-70x under the 500ms bar. Evidence logged to `eval/results/perf_benchmark_20260811_195829.json`. Run with `python -m backend.benchmark_perf`. |

## Task B — LLM integration (2.5 marks, heaviest component)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| B1 | Prompt design (schema + synonyms + few-shot) | done + tested | `llm/prompts.py`, 40+ entry synonym dict + 3 few-shot examples, `eval/benchmark_questions.json` (15 questions, all 5 categories), `eval/run_benchmark.py`. **Live run completed 2026-08-10: 15/15 (100%) accuracy** against real Gemini/Groq -- see `eval/results/benchmark_20260810_233113.json`. Gemini's free-tier daily quota (20 req/day) was exhausted almost immediately, so nearly every question actually exercised the live Groq fallback path, not just Gemini -- incidentally a strong real-world proof of B5's failover, not just the mocked tests. |
| B2 | Pipeline (Phase 1/2/3 + sandbox + single retry) | done + tested | `llm/sandbox.py` (allowlist-based sqlglot validation, 38 exploit tests), `llm/client.py` (Gemini->Groq failover, 9 mocked tests), `llm/pipeline.py` (Phase 1/2/3 + single retry + graceful "unanswerable" handling, 5 integration tests). |
| B3 | Insight generation (3 preset prompts) | done + tested | `llm/pipeline.py`: `generate_dataset_overview` (fully generic, no config needed), `generate_trend_comparison` (dims/metrics/aggs from `configs/dataset_config.yaml`'s new `insights.trend_comparison`), `generate_anomaly_report` (reuses A3's IQR profiling, picks the worst-outlier column at runtime -- data-driven, not hardcoded). 3 new tests. |
| B4 | Conversational context (last-5-turn memory) | done + tested | `llm/memory.py`: `ConversationMemory` wraps a plain list (storage-agnostic -- Task C backs an instance with `st.session_state`), `add`/`get_history`/`reset`. 4 new tests (eviction beyond 5 turns, reset, defensive copy on read). |
| B5 | Reliability (validation, loading indicator, Gemini->Groq failover) | done + tested | Validation/failover/timing/logging done and **live-proven** (see B1 note above) -- also caught and fixed a real bug this way: google-genai's internal retry (tenacity) can re-raise a raw `httpx.ReadTimeout` that does NOT subclass the builtin `TimeoutError`, which crashed the whole request instead of falling back to Groq until `_GEMINI_SDK_ERRORS` was widened to include `httpx.HTTPError` (regression test added). Closed out by C1: the loading indicator is `st.status` in `app/app.py`, showing elapsed time and which provider served the answer. |

## Task C — Dashboard (2.0 marks)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| C1 | Architecture (`st.tabs()`, sidebar filters, session state) | done + tested | `app/app.py`: 3 tabs (Overview / Exploration / AI Assistant), config-driven sidebar filters (date range + 3 multiselects), `st.session_state` for filters/memory/answers, `@st.cache_resource` so the 500k-row ingest runs once per process rather than per rerun. 8 `AppTest` smoke tests. |
| C2 | Visualization suite (>=6 chart types) | done + tested | All 7 built in `viz/charts.py` (time series, choropleth, correlation heatmap, box plot, sunburst, scatter+regression, stacked bar with drill-down). Bindings + a CVD-validated palette live in `configs/dataset_config.yaml`. |
| C3 | AI-driven visualization (shape -> chart-type auto-select) | done + tested | `viz/auto_select.py` keys off result shape only (proven by a test that feeds identical shapes under different column names); UI dropdown allows manual override; each AI chart carries a caption explaining the selection. |
| C4 | Export (PDF, Word, PNG/SVG) | done + tested | `export/pdf_export.py` + `export/docx_export.py` (metadata, filters, narrative, chart image, result table, SQL), plus per-chart PNG/SVG. All generated on demand behind a "Prepare" button -- see the perf note in the session log. |

## Task D — Advanced features (1.5 marks)

| ID | Subtask | Status | Notes |
|----|---------|--------|-------|
| D1 | Anomaly detection (extends A3's IQR logic) | done + tested | `features/anomaly_detection.py` reuses A3's IQR profiling (no reimplementation), flags the specific rows responsible, ranks them by how many IQRs past the fence they sit, and has the LLM explain each. Detection is deterministic and LLM-free -- a test asserts it never calls the model -- so what gets flagged is provable and the LLM only describes it. 10 tests. |
| D2 | Comparative analysis (two regions/date ranges) | done + tested | `features/comparative_analysis.py` runs A2's query engine once per side (dimension values or date ranges), computes deltas in Python, and has the LLM narrate. Paired grouped-bar chart normalises metrics to % of the larger side so different scales stay legible, with real figures in the hover and table. 10 tests. |

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
- 2026-08-11: Task C (C1-C4) implemented and tested, 3 commits. 134 tests passing.
  This also closed out B5 (its loading indicator is `st.status` in `app/app.py`).
  Three real problems found by rendering/running rather than trusting tests:
  (1) the box plot was unreadable -- profit quartiles sit within a few hundred of
  zero while extremes reach +/-8400, so an auto-scaled y-axis collapsed every box
  into a flat line; fixed by scoping the axis to the 1.5*IQR whisker range.
  (2) A serious perf bug: chart PNG/SVG and PDF/Word reports were built eagerly on
  every Streamlit rerun, and each shells out to kaleido (~1s), so one keystroke
  cost 10+ seconds. Now generated on demand behind a "Prepare" button; a plain
  rerun went from 10+s to ~0.5s, with a regression test asserting the bound.
  (3) `use_container_width` is past its removal date in Streamlit 1.61 -- switched
  all 17 uses to `width='stretch'`.
  Note on the spec's dual-axis time series (C2): built as specified, since the spec
  is the grading authority, but the known tradeoff is documented in the function's
  docstring -- where the two lines cross is an artifact of independent axis scaling,
  not a fact about the data. Worth being ready to discuss in Q&A.
  Live end-to-end verified in the running app (question -> Groq fallback -> correct
  narrative -> auto-selected bar chart); that test is marked `live_llm` and
  excluded from the default suite via `pytest.ini` so CI doesn't burn API quota.
  Caveat: verification was via Streamlit's `AppTest` (which really executes the app)
  plus direct chart-image rendering -- no browser automation tool was available, so
  the UI has not been eyeballed in an actual browser. Worth a manual look.
  Task D not started.
- 2026-08-11: Reworked the AI Assistant tab into a chat interface after the user
  ran the app and pointed out the input sat at the top with answers buried below
  the preset insights. Now a scrollable transcript with the input beneath it,
  chronological order, presets appending into the same conversation. Note
  `st.chat_input` only pins to the viewport bottom in the app's main body -- inside
  `st.tabs()` it renders inline (confirmed in the installed Streamlit's docs), so
  the transcript's bounded height is what keeps the input in a stable place.
- 2026-08-11: Task D (D1+D2) implemented and tested, added as a fourth "Advanced"
  tab (an addition to C1's three tabs, not a change to them). 156 tests passing.
  Both features verified live in the running app. Two real bugs caught this way:
  (1) `SUM()` over zero matching rows returns NULL -> NaN, and the guard
  `value or 0` did NOT catch it because NaN is truthy in Python, so an empty
  comparison side showed NaN instead of 0; (2) both features handed raw floats to
  the LLM, which echoed them verbatim -- narratives read
  "-6.434059163572309 percent". Values are now rounded before entering the prompt,
  with regression tests asserting nothing unrounded reaches it.
  **All four tasks (A-D) are now done + tested.** Remaining: the 5-mark
  presentation/report component, and a manual browser pass over the UI.
- 2026-08-11: Built the prompt ablation harness (`eval/run_ablation.py`) that spec
  Section 10 calls the most report-worthy result available. Scoring was extracted
  to `eval/scoring.py` so the ablation and the benchmark score identically (their
  numbers would otherwise not be comparable), and `pipeline.answer_question_sql_only`
  now exposes Phase 1+2 without Phase 3, halving the API calls a sweep costs.
  **The sweep has NOT produced a valid result yet, and this is blocked on API
  quota, not code.** The first run returned a complete-looking table (schema_only
  20%, full_prompt 13%) that was entirely an artifact: both free tiers were
  exhausted (Gemini 20 requests/day, Groq 100k tokens/day), so nearly every call
  429'd, and configurations effectively ranked by run order -- whichever ran first
  spent the remaining quota. Those results were deleted rather than kept.
  A full sweep needs ~130k input tokens (4 configs x 15 questions x ~2.1k tokens),
  which structurally exceeds Groq's 100k/day free allowance, so it cannot complete
  in one day on the current plan. The harness now aborts with `QuotaExhausted` and
  writes no file when provider failures exceed 20% of a configuration, with tests
  covering both the abort and the tolerate-one-blip case.
  To actually get the number: run on a day with fresh quota using
  `--repeats 1`, spread configurations across days, use a trimmed question set, or
  upgrade a provider tier.

## Dataset migration: Global Superstore -> Global E-Commerce Sales

- 2026-08-11: Retargeted the whole platform from Global Superstore (51,291 rows, 27
  columns) to Global E-Commerce Sales (500,000 rows, 10 columns). The generic/specific
  boundary held: `backend/`, all of `llm/`, `viz/auto_select.py`, `features/`, and
  `export/` needed **no column-name changes at all** -- only `configs/`, the chart
  builders' labels, the app's chrome, the eval ground truths, and the test fixtures.
  Full suite green (182 tests), perf re-verified at 10x the row count.

  Four structural gaps in the new data drove real decisions rather than mechanical
  renames:

  1. **No profit column.** Charts were rebuilt on the four real measures (revenue,
     price, quantity, discount). A synthetic profit column was deliberately NOT
     invented -- fabricated data would not survive Q&A.
  2. **No country column** (`region` is six continents). Rather than drop the
     choropleth and fall to the spec's bare minimum of 6 charts, the map now expands
     each region to its member countries via `configs/region_countries.yaml` (ISO-3
     codes) and shades them with the region total. Hover text binds to the region, not
     the country shape, so it cannot be misread as country-level data; a test asserts
     the hover template never references `%{location}`. ISO-3 was chosen over country
     names because Plotly deprecates the name lookup and names are ambiguous across
     Natural Earth vintages.
  3. **No Row.ID.** `dataset.id_column` is now omitted entirely; consumers already read
     it via `.get()`. With no surrogate key and 0 duplicates in the raw file, full-row
     duplicate detection is meaningful as-is.
  4. **No Year column.** Added a generic, config-driven `derive_date_parts` step to
     `quality.py` (allowlisted date parts, quoted identifiers -- config is not an
     injection vector).

  Three genuine bugs were found and fixed in passing, all latent on the old dataset:
  - `ingest.py` hashed PII columns unguarded, so any CSV missing a configured PII
    column raised `KeyError`. Now presence-guarded like the drop path, and the salt is
    only demanded when there is something to hash. Regression test added.
  - `app.py` guarded `date_range` filters on column *existence* but not *dtype*, so a
    date filter on a non-datetime column raised `AttributeError`.
  - `viz/charts.py` hardcoded `locationmode="country names"`, which config could not
    override -- the structural coupling that would have silently produced an empty map.

  One perf fix: `box_plot` passed raw y-values to Plotly, shipping ~500k floats to the
  browser per render at this dataset's scale. It now precomputes quartiles and fences
  in pandas and passes five numbers per box; the drawn result is identical because
  `boxpoints=False` meant individual points were never rendered.

  **The highest-risk change is the discount unit.** Superstore stored discount as a
  0-1 fraction; this dataset stores it as a 0-30 percentage. The column is renamed
  `discount_pct` to carry the unit into the schema the LLM sees, a few-shot example
  states it explicitly, and benchmark question `filtered_agg_1` exists to catch a
  regression: the correct reading (`discount_pct > 20`) averages 1133.06 over 166,032
  rows, while the fraction reading (`> 0.2`) averages 1280.80 over 496,574 -- far
  enough apart that scoring cannot confuse them.

  All 15 benchmark ground truths were recomputed against the cleaned table with DuckDB
  and verified to match their `reference_sql`, preserving the original category mix.
  **The accuracy benchmark has not been re-run live against the new questions** -- that
  needs API quota, same constraint as the ablation sweep below.
