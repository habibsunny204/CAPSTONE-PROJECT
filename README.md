# AI-Powered Data Analytics & Visualization Platform

A natural-language analytics platform over the Global E-Commerce Sales dataset, built for a
Masters-level Deep Learning capstone. You ask a question in plain English; an LLM writes
SQL; a sandbox validates and runs it against a read-only DuckDB connection; the result
comes back as a chart plus a written answer.

The authoritative spec is [PROJECT_SPEC.md](PROJECT_SPEC.md); current build status is in
[PROGRESS.md](PROGRESS.md).

## How it works

```
Raw CSV -> DuckDB + schema + query engine (backend/)
        -> Phase 1: LLM generates SQL as structured JSON {"sql", "reasoning"}
        -> Phase 2: sqlglot validates it, then it runs in a sandbox
        -> Phase 3: LLM writes a narrative from the result
        -> Streamlit dashboard -> PDF / Word / PNG / SVG export
```

**The LLM never executes anything.** It only ever produces text — SQL in Phase 1, prose in
Phase 3. Phase 2 is deterministic Python. That's what makes the sandbox requirement
satisfiable by construction: SQL has no `import`, no filesystem access, and no sockets to
block in the first place. See [`llm/sandbox.py`](llm/sandbox.py) for the two allowlists
that enforce it, and [`tests/test_sandbox_security.py`](tests/test_sandbox_security.py)
for the exploit-attempt tests.

## Setup

Requires **Python 3.10+** (developed and tested on 3.14).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Dataset

The raw CSV is gitignored (it's ~44 MB), so it isn't in this repo. Place it at:

```
data/raw/global_ecommerce_sales.csv
```

500,000 transaction rows spanning 2022-01-01 to 2024-01-01, across 10 columns. Ingestion
normalizes the raw headers (`Transaction Date`, `Customer ID`, `Discount (%)`, …) to
snake_case via the rename map in `configs/dataset_config.yaml`.

Two notes on the data that matter when reading results:

- **`discount_pct` is a percentage on a 0–30 scale, not a 0–1 fraction.** The `_pct` suffix
  is deliberate — it carries the unit into the schema the LLM sees, so "more than 20%
  discount" becomes `discount_pct > 20`. One benchmark question exists purely to check this.
- **There is no profit column and no country column.** Analysis is built on the four real
  measures (revenue, price, quantity, discount), and geography stops at six continent-level
  regions — see the choropleth note under *Layout*.

### Environment variables

Copy `.env.example` to `.env` and fill in real values. `.env` is gitignored and must never
be committed.

| Variable | Required for | Notes |
|---|---|---|
| `PII_HASH_SALT` | Ingestion (Task A1) | Any long random string. Ingestion **fails fast** if unset rather than falling back to a default. |
| `GEMINI_API_KEY` | LLM features | [Google AI Studio](https://aistudio.google.com/apikey), free tier. |
| `GROQ_API_KEY` | LLM fallback | [Groq Console](https://console.groq.com/keys), free tier. |

Generate a salt with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Running

```bash
streamlit run app/app.py
```

Then open http://localhost:8501. The dashboard has four tabs:

- **Overview** — headline metrics, revenue/units over time, a world choropleth, and the
  data-quality summary
- **Exploration** — the remaining curated charts, with drill-down and per-chart PNG/SVG export
- **AI Assistant** — the chat interface, preset insights, and PDF/Word export per answer
- **Advanced** — anomaly detection and comparative analysis (Task D)

Sidebar filters apply to **every** tab and persist across reruns — the charts, the AI
Assistant's answers, the preset insights, and both Task D features all describe the same
filtered rows. See *Filter scope* below for how that works.

The single deliberate exception is the Overview tab's data-quality panel, which reports what
Task A3 ingestion cleaning did to the source data — a property of the dataset rather than of
a slice — and says so in its caption.

## Privacy note

`Customer ID` is replaced by a salted SHA-256 hash — **before** the DuckDB table is
created, so raw PII is never queryable and never reaches an LLM prompt. (This dataset has
no customer-name column; the drop list is configured but empty.) Hiding a column from the schema JSON alone would not be
enough: the sandbox validates SQL *structure*, not a column allowlist, so a hallucinated
`SELECT` on a raw PII column would still succeed if the column were physically present.

The hash is a privacy control against accidental exposure and re-identification against
public copies of this dataset — not a cryptographic guarantee (equal inputs hash
identically, which is required to preserve unique-customer counts: 500,000 transactions
come from 98,348 distinct customers, and that figure has to survive hashing).

## Testing

```bash
pytest                    # full suite
pytest -m live_llm        # opt-in tests that call real LLM APIs (costs quota)
pytest tests/test_backend.py::test_schema_shape_has_required_keys   # a single test
```

Tests that need the raw dataset skip automatically when `data/raw/` is absent. Tests that
would hit live provider APIs are excluded by default via `pytest.ini`, so a routine run is
fast, deterministic, and free.

| File | Covers |
|---|---|
| `tests/test_backend.py` | Schema extraction, the PII gate, query engine, data quality, the <500 ms performance assertion |
| `tests/test_sandbox_security.py` | Every rejection case in spec Section 7, plus extras |
| `tests/test_llm_client.py` | Mocked Gemini→Groq failover behaviour |
| `tests/test_integration.py` | Full pipeline runs, preset insights, conversation memory |
| `tests/test_viz.py` | Chart auto-selection and the curated chart builders |
| `tests/test_export.py` | PDF and Word generation |
| `tests/test_features.py` | Task D's anomaly detection and comparative analysis |
| `tests/test_app_smoke.py` | Streamlit `AppTest` UI smoke tests, including that filters scope the SQL path |
| `tests/test_scope.py` | Filter scope: the pandas and SQL renderings must select the same rows |

## Benchmarks

```bash
python -m eval.run_benchmark      # NL->SQL accuracy over the fixed question set (needs API keys)
python -m backend.benchmark_perf  # query performance against the <500 ms bar
```

Both write timestamped evidence to `eval/results/`. The most recent runs:

- **Accuracy**: the only logged run (`eval/results/benchmark_20260810_233113.json`,
  15/15) predates the Global Superstore → Global E-Commerce Sales migration and was
  scored against `ecommerce_sales`'s predecessor table (its SQL literally reads
  `FROM superstore`). The current 15-question set in `eval/benchmark_questions.json`
  has not yet been run live — both providers' free-tier daily quotas were exhausted
  during the migration session (see `PROGRESS.md`). Re-run
  `python -m eval.run_benchmark` on a day with fresh quota before citing an accuracy
  number for the current dataset.
- **Performance**: filtered aggregation on all 500,000 rows runs in **8.9–17.3 ms
  median** across three scenarios (`eval/results/perf_benchmark_20260811_210226.json`)
  — roughly 29–56× under the 500 ms requirement.

## Layout

```
configs/dataset_config.yaml   # the ONLY place dataset-specific detail lives
configs/region_countries.yaml # region -> ISO-3 country codes, for the choropleth only
backend/                      # ingestion, schema, query engine, quality, perf benchmark
llm/                          # client (failover), prompts, pipeline, sandbox, memory
viz/                          # chart builders, shape-based auto-selection
features/                     # anomaly detection, comparative analysis
export/                       # PDF and Word report builders
eval/                         # benchmark question set, runner, logged results
app/app.py                    # Streamlit entrypoint
```

`backend/`, `llm/pipeline.py`, `llm/sandbox.py`, and `viz/auto_select.py` contain **no
dataset column names** — they're driven by the schema and by
`configs/dataset_config.yaml`. `viz/auto_select.py` in particular keys off the result's
*shape* (datetime+numeric, categorical+numeric, …), never off column identity, which is
asserted by a test that feeds identical shapes under different column names.

### A note on the choropleth

This dataset's geography stops at continent level — `region` holds six values and there is
no country column. Rather than drop the map, each region is expanded to its member
countries (`configs/region_countries.yaml`, ISO-3 codes) and every country in a region is
shaded with that region's total. The hover text is bound to the **region** name and total,
never to the country shape, so the chart cannot be misread as country-level data it does
not have — [`tests/test_viz.py`](tests/test_viz.py) asserts that the hover template never
references `%{location}`.

### Filter scope

The sidebar builds one canonical predicate — a list of `{"column", "op", "value"}` dicts, the
same shape [`backend/query_engine.py`](backend/query_engine.py) already accepts — and renders it
two ways from that single source:

- **pandas**, for the chart tabs, which hold a DataFrame
- **a DuckDB view**, for everything else, because the AI Assistant, the preset insights and both
  Task D features hold a connection and a table name and never see a DataFrame

The view lives in a per-session schema and is named *identically to the base table*, with the
LLM's cursor pointed at that schema via `search_path`. Keeping the name identical is what makes
this cheap: `table_name` never changes, so the prompts, the config few-shot examples (which
contain literal `FROM <table>` SQL) and the sandbox's single-table allowlist all work untouched.
The model writes an ordinary `FROM <table>` and gets filtered rows.

That indirection is also why [`llm/sandbox.py`](llm/sandbox.py) rejects schema-qualified table
references: a bare name resolves to the scoped view, but `main.<table>` would reach around it to
the full table. `tests/test_scope.py` asserts both that the two renderings select the same rows
and that generated SQL cannot escape the scope.

The schema is per browser session because the DuckDB connection is cached per server process and
shared — a fixed name would let one user's filters rescope another user's answers.
