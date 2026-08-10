# AI-Powered Data Analytics & Visualization Platform

A natural-language analytics platform over the Global Superstore dataset, built for a
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

The raw CSV is gitignored (it's ~15 MB), so it isn't in this repo. Place it at:

```
data/raw/global_superstore.csv
```

Ingestion normalizes the raw dot-separated headers (`Order.Date`, `Customer.ID`, …) to
snake_case via the rename map in `configs/dataset_config.yaml`.

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

- **Overview** — headline metrics, revenue/profit over time, a world choropleth, and the
  data-quality summary
- **Exploration** — the remaining curated charts, with drill-down and per-chart PNG/SVG export
- **AI Assistant** — the chat interface, preset insights, and PDF/Word export per answer
- **Advanced** — anomaly detection and comparative analysis (Task D)

Sidebar filters apply across the Overview and Exploration tabs and persist across reruns.

## Privacy note

`Customer.Name` is dropped outright at ingestion and `Customer.ID` is replaced by a salted
SHA-256 hash — **before** the DuckDB table is created, so raw PII is never queryable and
never reaches an LLM prompt. Hiding a column from the schema JSON alone would not be
enough: the sandbox validates SQL *structure*, not a column allowlist, so a hallucinated
`SELECT` on a raw PII column would still succeed if the column were physically present.

The hash is a privacy control against accidental exposure and re-identification against
public copies of this well-known dataset — not a cryptographic guarantee (equal inputs
hash identically, which is required to preserve unique-customer counts).

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
| `tests/test_app_smoke.py` | Streamlit `AppTest` UI smoke tests |

## Benchmarks

```bash
python -m eval.run_benchmark      # NL->SQL accuracy over the fixed question set (needs API keys)
python -m backend.benchmark_perf  # query performance against the <500 ms bar
```

Both write timestamped evidence to `eval/results/`. The most recent runs:

- **Accuracy**: 15/15 (100%) on the fixed benchmark set, spanning simple aggregation,
  filtered aggregation, synonym resolution, multi-condition filters, and out-of-scope
  questions that must be declined rather than hallucinated.
- **Performance**: filtered aggregation on all 51,290 rows runs in **2.7–6.7 ms median**
  across three scenarios — roughly 75–180× under the 500 ms requirement.

## Layout

```
configs/dataset_config.yaml   # the ONLY place dataset-specific detail lives
backend/                      # ingestion, schema, query engine, quality, perf benchmark
llm/                          # client (failover), prompts, pipeline, sandbox, memory
viz/                          # chart builders, shape-based auto-selection
features/                     # anomaly detection, comparative analysis
export/                       # PDF and Word report builders
eval/                         # benchmark question set, runner, logged results
app/app.py                    # Streamlit entrypoint
```

`backend/`, `llm/pipeline.py`, `llm/sandbox.py`, and `viz/auto_select.py` contain **no
Superstore column names** — they're driven by the schema and by
`configs/dataset_config.yaml`. `viz/auto_select.py` in particular keys off the result's
*shape* (datetime+numeric, categorical+numeric, …), never off column identity, which is
asserted by a test that feeds identical shapes under different column names.
