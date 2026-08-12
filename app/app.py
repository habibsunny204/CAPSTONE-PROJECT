"""Streamlit entrypoint: st.tabs() layout for Overview / Exploration / AI Assistant
(Task C1).

Run with `streamlit run app/app.py`.

Uses st.tabs() rather than the multipage pages/ folder deliberately (PROJECT_SPEC.md
C1): pages/ resets widget state across page switches, which would break the
persistent global sidebar filters this dashboard is built around.

Data is loaded and cleaned once per server process via @st.cache_resource, not once
per rerun -- Streamlit re-executes this whole script top-to-bottom on every widget
interaction, so an uncached ingest would re-parse the 500k-row CSV on every click.

Page chrome (title, headline KPIs, example questions, export provenance) is read from
configs/dataset_config.yaml's `app` section rather than hardcoded, so this module holds
no dataset-specific column names or branding.

The sidebar filters constrain every tab, not just the charts. render_sidebar_filters()
builds one canonical scope spec; the chart tabs get it applied to a DataFrame, and
scoped_connection() turns the same spec into a DuckDB view that the AI Assistant, the
preset insights and both Task D features query through. Since those all take only
(connection, table_name), swapping the connection is what scopes them -- see
backend/scope.py for why the view keeps the base table's name.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Allow `streamlit run app/app.py` to import the sibling packages (backend/, llm/,
# viz/, export/) -- Streamlit puts the script's own directory on sys.path, not the
# repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import ingest, quality, schema as schema_module, scope  # noqa: E402
from export.docx_export import build_answer_docx  # noqa: E402
from export.pdf_export import build_answer_pdf  # noqa: E402
from features import anomaly_detection, comparative_analysis  # noqa: E402
from llm import pipeline  # noqa: E402
from llm.memory import ConversationMemory  # noqa: E402
from viz import auto_select, charts  # noqa: E402

st.set_page_config(page_title=ingest.load_config()["app"]["title"], page_icon="B", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached once per server process)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading and cleaning the dataset...")
def load_everything() -> dict[str, Any]:
    """Ingest + clean the dataset once, returning the connection, config, schema,
    quality reports, and the full DataFrame the charts render from.
    """
    config = ingest.load_config()
    table_name = config["dataset"]["table_name"]

    con = ingest.load_from_config()
    profile_before = quality.profile_report(con, table_name, id_column=config["dataset"].get("id_column"))
    clean_report = quality.clean(con, table_name, config)
    profile_after = quality.profile_report(con, table_name, id_column=config["dataset"].get("id_column"))

    return {
        "config": config,
        "table_name": table_name,
        "con": con,
        "schema": schema_module.get_schema(con, table_name),
        "profile_before": profile_before,
        "profile_after": profile_after,
        "clean_report": clean_report,
        "df": con.execute(f'SELECT * FROM "{table_name}"').df(),
    }


def _init_session_state() -> None:
    """Seed session state once per browser session (C1: filters and chat history
    must survive Streamlit's rerun-on-every-interaction model).
    """
    st.session_state.setdefault("memory", ConversationMemory())
    # One chronological list of conversation turns (typed questions, preset
    # insights, and errors alike), so the transcript renders in a single pass.
    st.session_state.setdefault("turns", [])
    # DuckDB schema holding this session's filtered view. Per-session because the
    # connection is @st.cache_resource'd and shared across browser sessions -- a fixed
    # name would let one user's sidebar filters silently rescope another user's answers.
    st.session_state.setdefault("scope_schema", f"scope_{uuid4().hex[:12]}")


# ---------------------------------------------------------------------------
# Sidebar filters (C1)
# ---------------------------------------------------------------------------


def render_sidebar_filters(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Render the configured global filters and return (filtered DataFrame, a
    human-readable summary for display/export, the canonical scope spec). Filter
    definitions come from configs/dataset_config.yaml, so adding one is a config
    change, not a code change.

    The third return value is the machine-readable form -- a list of
    {"column", "op", "value"} dicts, the same shape backend/query_engine.py accepts.
    The DataFrame is only usable by the chart tabs; every LLM and Task-D path holds a
    connection and a table name instead, and needs the predicate itself to build a
    scoped view (backend/scope.py). Both renderings come from this one spec so they
    cannot describe different row sets.
    """
    st.sidebar.header("Filters")
    st.sidebar.caption("Applied across every tab: charts, AI answers, and Task D features.")

    spec_list: list[dict[str, Any]] = []
    applied: dict[str, Any] = {}

    for spec in config.get("filters", []):
        column, label = spec["column"], spec["label"]
        if column not in df.columns:
            continue

        if spec["type"] == "date_range":
            # Dtype-guarded, not just presence-guarded: .min().date() below is only
            # valid on a datetime column, so a date_range filter pointed at a
            # non-datetime column is skipped rather than raising AttributeError.
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                continue
            min_date, max_date = df[column].min().date(), df[column].max().date()
            chosen = st.sidebar.date_input(
                label, value=(min_date, max_date),
                min_value=min_date, max_value=max_date, key=f"filter_{column}",
            )
            # Only recorded when it actually narrows: a range left at the full bounds
            # is a no-op, and emitting it anyway would make the scope read as "active"
            # and attach a filtered-data caveat to answers that cover everything.
            if isinstance(chosen, tuple) and len(chosen) == 2 and tuple(chosen) != (min_date, max_date):
                start, end = chosen
                spec_list.append({
                    "column": column, "op": "between",
                    # Normalized to Timestamps here so the pandas and SQL renderings
                    # compare against an identical value.
                    "value": [pd.Timestamp(start), pd.Timestamp(end)],
                })
                applied[label] = f"{start} to {end}"

        elif spec["type"] == "multiselect":
            options = sorted(df[column].dropna().unique().tolist())
            chosen = st.sidebar.multiselect(label, options, default=[], key=f"filter_{column}")
            if chosen:
                spec_list.append({"column": column, "op": "in", "value": list(chosen)})
                applied[label] = ", ".join(str(c) for c in chosen)

    filtered = scope.apply_to_frame(df, spec_list)

    st.sidebar.metric("Rows after filters", f"{len(filtered):,}", delta=f"{len(filtered) - len(df):,}"
                      if len(filtered) != len(df) else None)
    if not applied:
        applied = {"Filters": "none (full dataset)"}
    return filtered, applied, spec_list


@contextmanager
def guarded_section(label: str):
    """Contain a rendering failure to the section that caused it.

    Streamlit's default for an uncaught exception is to replace the entire page with a
    red traceback -- every tab, chart and prior answer vanishes, and the user is shown
    a stack trace they can do nothing with. For a dashboard that is both alarming and
    unhelpful, and during a live demo it looks like the whole app fell over when in
    fact one chart could not be drawn.

    Wrapping each section means the rest of the page keeps working and the failure is
    reported where it happened. The traceback is kept, but folded away: it is for
    whoever is debugging, not for whoever is reading.
    """
    try:
        yield
    except Exception as error:  # noqa: BLE001 - a UI backstop must catch everything
        logger.exception("Rendering failed in %s", label)
        st.error(
            f"Something went wrong while rendering {label}: "
            f"**{type(error).__name__}: {error}**\n\n"
            "The rest of the dashboard is unaffected -- adjust the filters or your "
            "question and try again."
        )
        with st.expander("Technical details"):
            st.code(traceback.format_exc(), language="text")


def scope_labels(config: dict[str, Any]) -> dict[str, str]:
    """Column -> sidebar label, so scope descriptions read the way the UI does."""
    return {f["column"]: f["label"] for f in config.get("filters", [])}


def scoped_connection(data: dict[str, Any], spec: list[dict[str, Any]]) -> Any:
    """A cursor whose bare table name resolves to the sidebar-filtered rows.

    This is the single point where the sidebar reaches the SQL world. Everything
    downstream -- the NL pipeline, the three preset insights, anomaly detection and
    comparative analysis -- takes only (connection, table_name), so swapping the
    connection scopes all of them at once with no change to their signatures.

    The schema is per browser session because the DuckDB connection is cached per
    server process (@st.cache_resource) and therefore shared: a fixed schema name would
    let one user's filters rewrite another user's results mid-session.
    """
    schema_name = st.session_state["scope_schema"]
    return scope.scoped_cursor(data["con"], data["table_name"], schema_name, spec)


def render_scope_notice(spec: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """State the active scope on tabs whose numbers would otherwise look dataset-wide."""
    if not scope.is_active(spec):
        return
    st.info(
        f"Filtered view: **{scope.describe(spec, scope_labels(config))}**. "
        "Results below cover only these rows.",
        icon=":material/filter_alt:",
    )


# ---------------------------------------------------------------------------
# Tab 1 -- Overview
# ---------------------------------------------------------------------------


def render_overview_tab(data: dict[str, Any], filtered: pd.DataFrame) -> None:
    """Headline metrics, the data-quality summary (A3), and two curated charts."""
    config = data["config"]

    st.subheader("At a glance")
    # Headline metrics come from config's `app.kpis` rather than being hardcoded, so
    # which measures lead the dashboard is a dataset decision, not a code edit.
    kpi_specs = [k for k in config["app"].get("kpis", []) if k["column"] in filtered.columns]
    columns = st.columns(1 + len(kpi_specs))
    columns[0].metric("Rows", f"{len(filtered):,}")
    for slot, spec in zip(columns[1:], kpi_specs):
        series = filtered[spec["column"]]
        value = series.nunique() if spec["agg"] == "nunique" else getattr(series, spec["agg"])()
        slot.metric(spec["label"], format(value, spec.get("format", ",.0f")))

    if filtered.empty:
        st.warning("No rows match the current filters. Widen them in the sidebar to see charts.")
        return

    _chart_with_export(charts.time_series(filtered, config), _binding_stem(config, "time_series"))
    _chart_with_export(charts.choropleth(filtered, config), _binding_stem(config, "choropleth"))

    st.subheader("Data quality")
    st.caption(
        "Profiled and cleaned at load time by `backend/quality.py` (Task A3). "
        "This is the one panel that deliberately ignores the sidebar: it reports what "
        "ingestion did to the source data, which is a property of the dataset rather "
        "than of any filtered slice. Every other number in this app is filtered."
    )

    profile = data["profile_after"]
    quality_columns = st.columns(3)
    quality_columns[0].metric("Rows profiled", f"{profile['n_rows']:,}")
    quality_columns[1].metric("Duplicate rows", f"{profile['n_duplicate_rows']:,}")
    quality_columns[2].metric("Columns", f"{len(profile['columns']):,}")

    with st.expander("Cleaning steps applied"):
        for step in data["clean_report"]["steps_applied"]:
            details = ", ".join(
                f"{key.replace('_', ' ')}: {value}"
                for key, value in step.items()
                if key != "name" and value not in (None, [], {}, 0)
            )
            st.markdown(f"**{step['name'].replace('_', ' ')}** &mdash; {details or 'no changes needed'}")

    with st.expander("Per-column quality profile"):
        st.dataframe(pd.DataFrame(profile["columns"]), width='stretch')


# ---------------------------------------------------------------------------
# Tab 2 -- Exploration
# ---------------------------------------------------------------------------


def _binding_stem(config: dict[str, Any], chart_name: str) -> str:
    """Download filename stem for a curated chart, slugified from its configured
    title so it describes whatever the chart actually shows on the current dataset.

    Module-level (not just Exploration-local) so the Overview tab's charts can share
    it and get the same naming convention rather than inventing a second one.
    """
    title = config["charts"]["curated_bindings"][chart_name]["title"]
    return "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")


def render_exploration_tab(data: dict[str, Any], filtered: pd.DataFrame) -> None:
    """The remaining curated charts, with drill-down and per-chart image export."""
    config = data["config"]

    if filtered.empty:
        st.warning("No rows match the current filters. Widen them in the sidebar to see charts.")
        return

    bindings = config["charts"]["curated_bindings"]

    def _stem(chart_name: str) -> str:
        return _binding_stem(config, chart_name)

    left, right = st.columns(2)
    with left:
        _chart_with_export(charts.box_plot(filtered, config), _stem("box_plot"))
    with right:
        _chart_with_export(charts.scatter_regression(filtered, config), _stem("scatter_regression"))

    _chart_with_export(charts.correlation_heatmap(filtered, config), _stem("correlation_heatmap"))
    _chart_with_export(charts.sunburst(filtered, config), _stem("sunburst"))

    _chart_with_export(charts.animated_bar(filtered, config), _stem("animated_bar"))
    st.caption(
        "Press play, or drag the slider, to step through the months. Image export "
        "captures the frame currently shown."
    )

    binding = bindings["stacked_bar"]
    primary = binding["primary_dim"]
    st.subheader(binding["title"])
    all_label = f"All {primary.replace('_', ' ')}s"
    drill_options = [all_label] + sorted(filtered[primary].dropna().unique().tolist())
    chosen = st.selectbox("Drill into", drill_options, key="stacked_bar_drill")
    drill_into = None if chosen == all_label else chosen
    _chart_with_export(charts.stacked_bar(filtered, config, drill_into=drill_into), _stem("stacked_bar"))

    with st.expander("Browse the filtered rows"):
        st.dataframe(filtered.head(500), width='stretch')
        st.caption(f"Showing up to 500 of {len(filtered):,} filtered rows.")


def _chart_with_export(figure, filename_stem: str) -> None:
    """Render a chart plus on-demand PNG/SVG export (C4's per-chart export).

    The image bytes are generated only when the user asks for them, not on every
    rerun. Streamlit re-executes this script on every widget interaction, and
    rendering a Plotly figure to an image shells out to kaleido (~1s each), so
    generating eagerly for every chart made a single keystroke cost 10+ seconds.
    """
    st.plotly_chart(figure, width='stretch')

    with st.expander("Download this chart"):
        png_column, svg_column = st.columns(2)
        for column, fmt, mime in (
            (png_column, "png", "image/png"),
            (svg_column, "svg", "image/svg+xml"),
        ):
            state_key = f"image_{filename_stem}_{fmt}"
            if column.button(f"Prepare {fmt.upper()}", key=f"prepare_{state_key}",
                             width='stretch'):
                with st.spinner(f"Rendering {fmt.upper()}..."):
                    st.session_state[state_key] = charts.figure_to_image_bytes(figure, fmt)
            if state_key in st.session_state:
                column.download_button(
                    f"Save {fmt.upper()}", st.session_state[state_key],
                    file_name=f"{filename_stem}.{fmt}", mime=mime,
                    key=f"download_{state_key}", width='stretch',
                )


# ---------------------------------------------------------------------------
# Tab 3 -- AI Assistant
# ---------------------------------------------------------------------------


def render_ai_tab(data: dict[str, Any], applied_filters: dict[str, Any],
                  scope_spec: list[dict[str, Any]]) -> None:
    """A chat interface over the natural-language pipeline (B2), preset insights
    (B3), conversational memory (B4), AI-driven chart selection (C3), and
    per-answer export (C4).

    Laid out as a scrollable transcript with the input directly beneath it, so the
    conversation reads oldest-to-newest and you type where you last read -- rather
    than typing at the top and hunting for the answer further down the page.

    st.chat_input renders inline here rather than pinned to the viewport bottom:
    Streamlit only pins it when it's in the app's main body, and nesting it in any
    layout container (st.tabs included) switches it to inline. Bounding the
    transcript's height is what keeps the input in a stable place on screen.
    """
    config, table_name = data["config"], data["table_name"]
    # A dedicated cursor for LLM-generated SQL, kept separate from the connection
    # ingest/quality used to build the table (see llm/sandbox.py), and scoped to the
    # sidebar filters so a generated query cannot answer from rows the user filtered
    # out. The table name is unchanged -- only what it resolves to.
    llm_con = scoped_connection(data, scope_spec)
    scope_text = scope.describe(scope_spec, scope_labels(config))
    # Counted through the scoped cursor rather than taken from the pandas frame, so an
    # exported report states the row count of the scope its SQL actually ran against.
    scope_rows = llm_con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

    render_scope_notice(scope_spec, config)

    header_column, reset_column = st.columns([4, 1])
    header_column.caption(
        "Answers are generated live by Gemini (falling back to Groq), turned into "
        "validated read-only SQL, executed in a sandbox, then narrated."
    )
    if reset_column.button("Clear chat", width='stretch', key="reset_button"):
        st.session_state["memory"].reset()
        st.session_state["turns"] = []
        st.rerun()

    _render_preset_buttons(llm_con, table_name, config, scope_text)

    transcript = st.container(height=560)
    with transcript:
        if not st.session_state["turns"]:
            examples = " or ".join(f"*{q}*" for q in config["app"]["example_questions"][:2])
            st.chat_message("assistant").markdown(
                f"Ask me anything about the {config['app']['dataset_display_name']} data "
                f"&mdash; for example, {examples} You can also use the buttons above "
                "for a ready-made insight."
            )
        for index, turn in enumerate(st.session_state["turns"]):
            # Guarded per turn: one answer that can't be drawn must not take the whole
            # transcript with it, since the conversation above it is the user's history.
            with guarded_section(f"answer {index + 1}"):
                _render_turn(turn, index, config, applied_filters, data, scope_rows)

    question = st.chat_input("Ask a question about the data", key="chat_input")
    if question and question.strip():
        _run_question(llm_con, table_name, config, question.strip(), transcript, scope_text)
        st.rerun()


def _run_question(llm_con, table_name: str, config: dict[str, Any], question: str,
                  transcript, scope_description: str = "") -> None:
    """Run one question through the pipeline and append it to the conversation.

    The loading indicator (Task B5's UI half) is written into `transcript` so
    progress appears in the conversation where the answer will land, not detached
    at the bottom of the page.
    """
    with transcript:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"):
            status = st.status("Thinking...", expanded=True)
            started = time.perf_counter()
            try:
                status.write("Generating SQL...")
                result = pipeline.answer_question(
                    llm_con, table_name, config, question,
                    history=st.session_state["memory"].get_history(),
                    scope_description=scope_description,
                )
                elapsed = time.perf_counter() - started
                status.update(
                    label=f"Answered in {elapsed:.1f}s via {result.sql_provider}", state="complete"
                )

                st.session_state["memory"].add(question, result.sql, result.narrative)
                st.session_state["turns"].append({
                    "kind": "question", "question": question, "sql": result.sql,
                    "reasoning": result.reasoning, "result": result.result,
                    "narrative": result.narrative, "provider": result.sql_provider,
                    "retried": result.retried, "elapsed": elapsed,
                })
            except pipeline.PipelineError as error:
                status.update(label="Couldn't answer that one", state="error")
                st.session_state["turns"].append(
                    {"kind": "error", "question": question, "message": str(error)}
                )
            except Exception as error:  # noqa: BLE001 - surface provider/network failures cleanly
                status.update(label="Something went wrong", state="error")
                st.session_state["turns"].append({
                    "kind": "error", "question": question,
                    "message": f"{type(error).__name__}: {error}",
                })


def _render_preset_buttons(llm_con, table_name: str, config: dict[str, Any],
                           scope_description: str = "") -> None:
    """The three preset insights (Task B3), offered as suggested prompts.

    Their output is appended to the same conversation as a normal assistant turn,
    so presets and typed questions share one transcript instead of accumulating in
    a separate stack of expanders.
    """
    presets = {
        "Dataset overview": pipeline.generate_dataset_overview,
        "Trend analysis": pipeline.generate_trend_comparison,
        "Anomaly report": pipeline.generate_anomaly_report,
    }

    for column, (label, generator) in zip(st.columns(len(presets)), presets.items()):
        if column.button(label, width='stretch', key=f"preset_{label}"):
            with st.spinner(f"Generating {label.lower()}..."):
                try:
                    insight = generator(llm_con, table_name, config, scope_description)
                    st.session_state["turns"].append({
                        "kind": "insight", "question": label,
                        "narrative": insight.narrative,
                        "provider": insight.narrative_provider,
                        "aggregation": insight.data.get("aggregation"),
                        "outlier_samples": insight.data.get("outlier_samples"),
                    })
                except Exception as error:  # noqa: BLE001
                    st.session_state["turns"].append({
                        "kind": "error", "question": label,
                        "message": f"{type(error).__name__}: {error}",
                    })
            st.rerun()


def _render_turn(turn: dict[str, Any], index: int, config: dict[str, Any],
                 applied_filters: dict[str, Any], data: dict[str, Any],
                 scope_rows: int) -> None:
    """Render one conversation turn as a user/assistant message pair."""
    st.chat_message("user").markdown(turn["question"])

    with st.chat_message("assistant"):
        if turn["kind"] == "error":
            st.error(turn["message"])
            return

        if turn["kind"] == "insight":
            st.markdown(turn["narrative"])
            st.caption(f"Preset insight, generated by {turn['provider']}")
            if turn.get("aggregation") is not None:
                st.dataframe(turn["aggregation"], width='stretch')
            samples = turn.get("outlier_samples")
            if samples is not None and not samples.empty:
                st.caption("Sample flagged rows")
                st.dataframe(samples, width='stretch')
            return

        _render_answer(turn, index, config, applied_filters, data, scope_rows)


def _chart_caption(answer: dict[str, Any], index: int,
                   selection: auto_select.ChartSelection) -> str:
    """The one-sentence LLM-generated caption under an AI chart (Task C3), generated
    once per (answer, chart type) and cached in session state.

    The caching is not an optimisation, it is what makes this correct. Streamlit
    re-executes this whole script on every widget interaction, so a bare
    pipeline.generate_chart_caption() call here would fire a live API call on every
    keystroke in a sidebar filter -- the same shape of bug that made chart image export
    cost 10+ seconds per rerun, except this one also burns a rate-limited free-tier
    quota. Keying on chart_type (rather than just the answer) means overriding the
    chart type regenerates the caption, which is the point: a caption describing a bar
    chart is wrong once the user switches to a line.

    A provider failure degrades to the deterministic selection reason rather than
    raising. guarded_section() would catch the raise, but losing the whole answer over
    a decorative sentence is a bad trade, and this failure is foreseeable enough to
    handle where it happens.
    """
    cache_key = f"caption_{index}_{selection.chart_type}"
    if cache_key not in st.session_state:
        try:
            caption, provider = pipeline.generate_chart_caption(
                answer["question"], selection.chart_type, answer["result"],
                x_column=selection.x, y_column=selection.y,
                series_column=selection.color,
            )
            st.session_state[cache_key] = f"{caption} _(caption by {provider})_"
        except Exception:
            logger.exception("Chart caption generation failed for turn %s", index)
            st.session_state[cache_key] = selection.reason
    return st.session_state[cache_key]


def _render_answer(answer: dict[str, Any], index: int, config: dict[str, Any],
                   applied_filters: dict[str, Any], data: dict[str, Any],
                   scope_rows: int) -> None:
    """Render one answered question: narrative, auto-selected chart with a manual
    override, the result table, and PDF/Word export.
    """
    st.markdown(answer["narrative"])
    st.caption(
        f"Answered by {answer['provider']} in {answer['elapsed']:.1f}s"
        + (" (after one retry)" if answer["retried"] else "")
    )

    result_df = answer["result"]
    figure = None

    if not result_df.empty:
        selection = auto_select.select_chart(result_df)
        options = auto_select.ALL_CHART_TYPES
        chosen_type = st.selectbox(
            "Chart type", options, index=options.index(selection.chart_type),
            key=f"chart_type_{index}",
            help=f"Auto-selected: {selection.chart_type}. {selection.reason}",
        )

        # An override cannot reuse the auto-selection's axes -- when select_chart()
        # chose `table` it left x and y as None, and passing those to Plotly raises.
        # resolve_override() re-derives axes from the result's shape, and returns None
        # when the data genuinely cannot be drawn that way.
        effective = selection
        if chosen_type != selection.chart_type:
            effective = auto_select.resolve_override(result_df, chosen_type)

        if effective is None:
            st.info(
                f"This result can't be shown as a **{chosen_type}** chart -- "
                "showing the table instead. Pick another chart type above."
            )
            st.dataframe(result_df, width='stretch')
        else:
            figure = charts.build_from_selection(
                result_df, effective, config, title=answer["question"]
            )
            if figure is not None:
                _chart_with_export(figure, f"ai_answer_{index}_{effective.chart_type}")
                st.caption(_chart_caption(answer, index, effective))
            elif effective.chart_type == auto_select.CHART_METRIC:
                st.metric(str(result_df.columns[0]), charts.format_scalar(result_df.iloc[0, 0]))
            else:
                st.dataframe(result_df, width='stretch')

        with st.expander("Result data and generated SQL"):
            st.dataframe(result_df, width='stretch')
            st.code(answer["sql"], language="sql")

    # Rows in scope, not rows in the dataset: the report already prints the applied
    # filters beside this, so a dataset-wide count here would contradict them.
    total_rows = data["profile_after"]["n_rows"]
    metadata = {
        "Rows": f"{scope_rows:,}" + (f" of {total_rows:,} (filtered)" if scope_rows != total_rows else ""),
        "Source": config["app"]["dataset_display_name"],
        "Provider": answer["provider"],
    }
    # Reports are built on demand for the same reason chart images are: each one
    # renders the figure through kaleido, so building them eagerly on every rerun
    # cost ~5s per stored answer per keystroke.
    with st.expander("Export this answer"):
        builders = {
            "PDF": (build_answer_pdf, "pdf", "application/pdf"),
            "Word": (build_answer_docx, "docx",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        }
        for column, (label, (builder, extension, mime)) in zip(st.columns(2), builders.items()):
            state_key = f"report_{index}_{extension}"
            if column.button(f"Prepare {label}", key=f"prepare_{state_key}",
                             width='stretch'):
                with st.spinner(f"Building {label} report..."):
                    st.session_state[state_key] = builder(
                        question=answer["question"], narrative=answer["narrative"],
                        sql=answer["sql"], result=result_df, figure=figure,
                        dataset_metadata=metadata, applied_filters=applied_filters,
                    )
            if state_key in st.session_state:
                column.download_button(
                    f"Save {label}", st.session_state[state_key],
                    file_name=f"answer_{index}.{extension}", mime=mime,
                    key=f"download_{state_key}", width='stretch',
                )


# ---------------------------------------------------------------------------
# Tab 4 -- Advanced (Task D)
# ---------------------------------------------------------------------------


def render_advanced_tab(data: dict[str, Any], filtered: pd.DataFrame,
                        scope_spec: list[dict[str, Any]]) -> None:
    """Task D's two features: anomaly detection (D1) and comparative analysis (D2).

    These get their own tab rather than being folded into Exploration because
    they're a separately-graded deliverable, and both are interactive workflows
    rather than static charts. That's an addition to the three tabs C1 lists, not
    a change to them.
    """
    config, table_name = data["config"], data["table_name"]
    llm_con = scoped_connection(data, scope_spec)

    render_scope_notice(scope_spec, config)
    if filtered.empty:
        st.warning("No rows match the current filters. Widen them in the sidebar.")
        return

    anomaly_section, comparison_section = st.tabs(["Anomaly detection", "Comparative analysis"])
    with anomaly_section:
        _render_anomaly_detection(llm_con, table_name, config, filtered)
    with comparison_section:
        _render_comparative_analysis(llm_con, table_name, config, filtered)


def _render_anomaly_detection(llm_con, table_name: str, config: dict[str, Any],
                              filtered: pd.DataFrame) -> None:
    """D1: pick a numeric column, flag its most extreme rows, explain them.

    `llm_con` is scoped to the sidebar filters, so the IQR fences are computed from
    the current selection rather than the whole dataset -- "what is unusual within
    this slice", which stays self-consistent with the rows actually shown.
    """
    st.caption(
        "Reuses the IQR outlier logic from the data-quality layer (Task A3) to flag "
        "specific rows, then asks the LLM to explain what makes each one unusual. "
        "Outlier bounds are computed from the currently filtered rows."
    )

    candidates = anomaly_detection.outlier_columns(llm_con, table_name, config)
    if not candidates:
        st.info("No numeric column in this dataset has IQR-based outliers.")
        return

    controls = st.columns([2, 1, 1], vertical_alignment="bottom")
    column = controls[0].selectbox(
        "Column to examine",
        [c["name"] for c in candidates],
        format_func=lambda name: f"{name} ({dict((c['name'], c['n_outliers_iqr']) for c in candidates)[name]:,} outliers)",
        key="anomaly_column",
    )
    limit = controls[1].number_input("Rows to flag", min_value=1, max_value=50, value=10,
                                     key="anomaly_limit")
    run = controls[2].button("Detect", type="primary", width='stretch', key="anomaly_run")

    context_columns = [c for c in config["app"].get("anomaly_context_columns", [])
                       if c in filtered.columns]

    if run:
        # Guarded around the whole detect-and-explain step: on failure nothing is
        # written to session state, so the previous report (if any) stays on screen
        # rather than the tab going blank behind an error.
        with guarded_section("anomaly detection"):
            report = anomaly_detection.detect_anomalies(
                llm_con, table_name, config, column=column,
                context_columns=context_columns, limit=int(limit),
            )
            with st.spinner("Explaining the flagged rows..."):
                try:
                    report = anomaly_detection.explain_anomalies(report)
                except Exception as error:  # noqa: BLE001
                    report.narrative = (
                        f"Could not generate an explanation: {type(error).__name__}: {error}"
                    )
            st.session_state["anomaly_report"] = report

    report = st.session_state.get("anomaly_report")
    if report is None:
        return

    summary = st.columns(3)
    summary[0].metric("Column", report.column)
    summary[1].metric("Total outliers", f"{report.n_total_outliers:,}")
    if report.bounds:
        summary[2].metric(
            "Normal range",
            f"{report.bounds['lower']:,.0f} to {report.bounds['upper']:,.0f}",
        )

    st.markdown(report.narrative)
    if report.provider and report.provider != "none":
        st.caption(f"Explanation generated by {report.provider}")

    frame = report.to_frame()
    if not frame.empty:
        x_column = config["charts"]["anomaly_scatter"]["x_column"]
        if x_column in filtered.columns and x_column in frame.columns:
            st.plotly_chart(
                charts.anomaly_scatter(filtered, frame, report.bounds, report.column, config),
                width='stretch',
            )
            st.caption(
                "Triangles mark the flagged rows -- pointing up for above the fence, "
                "down for below -- sized by how many IQRs past it they sit. Dashed "
                "lines mark the fences themselves."
            )
        st.subheader("Flagged rows")
        st.caption("`_severity_iqrs` is how many IQRs past the fence each row sits.")
        st.dataframe(frame, width='stretch')


def _render_comparative_analysis(llm_con, table_name: str, config: dict[str, Any],
                                 filtered: pd.DataFrame) -> None:
    """D2: compare two dimension values or two date ranges side by side.

    Option lists are built from the *filtered* frame, not the full dataset. That is
    what stops the confusing case where the sidebar is narrowed to one region but the
    dimension dropdown still offers the excluded ones, inviting a comparison whose
    other side is necessarily zero.
    """
    st.caption(
        "Runs the query engine (Task A2) once per side, then asks the LLM for a "
        "structured side-by-side narrative. All arithmetic is done in Python -- the "
        "model only describes figures it is handed."
    )

    df = filtered
    numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                       and c != config["dataset"].get("id_column")]
    categorical_columns = [c for c in df.columns
                           if not pd.api.types.is_numeric_dtype(df[c])
                           and not pd.api.types.is_datetime64_any_dtype(df[c])
                           and df[c].nunique() <= 50]

    mode = st.radio("Compare", ["Two dimension values", "Two date ranges"],
                    horizontal=True, key="comparison_mode")
    metrics = st.multiselect("Metrics", numeric_columns,
                             default=[c for c in config["app"].get("comparison_default_metrics", [])
                                      if c in numeric_columns][:2],
                             key="comparison_metrics")

    result = None
    if mode == "Two dimension values":
        controls = st.columns(3)
        dimension = controls[0].selectbox("Dimension", categorical_columns, key="comparison_dimension")
        options = sorted(df[dimension].dropna().unique().tolist())
        left_value = controls[1].selectbox("Left", options, index=0, key="comparison_left")
        right_value = controls[2].selectbox("Right", options,
                                            index=min(1, len(options) - 1), key="comparison_right")

        if st.button("Compare", type="primary", key="comparison_run") and metrics:
            with guarded_section("the comparison"):
                result = comparative_analysis.compare_dimension_values(
                    llm_con, table_name, dimension, left_value, right_value, metrics
                )
    else:
        date_columns = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        date_column = st.selectbox("Date column", date_columns, key="comparison_date_column")
        bounds = (df[date_column].min().date(), df[date_column].max().date())

        left_column, right_column = st.columns(2)
        left_range = left_column.date_input("Left period", value=bounds,
                                            min_value=bounds[0], max_value=bounds[1],
                                            key="comparison_left_range")
        right_range = right_column.date_input("Right period", value=bounds,
                                              min_value=bounds[0], max_value=bounds[1],
                                              key="comparison_right_range")

        ready = (isinstance(left_range, tuple) and len(left_range) == 2
                 and isinstance(right_range, tuple) and len(right_range) == 2)
        if st.button("Compare", type="primary", key="comparison_run") and metrics and ready:
            with guarded_section("the comparison"):
                result = comparative_analysis.compare_date_ranges(
                    llm_con, table_name, date_column,
                    (str(left_range[0]), str(left_range[1])),
                    (str(right_range[0]), str(right_range[1])),
                    metrics,
                )

    if result is not None:
        with st.spinner("Writing the comparison..."):
            try:
                result = comparative_analysis.explain_comparison(result)
            except Exception as error:  # noqa: BLE001
                result.narrative = f"Could not generate a narrative: {type(error).__name__}: {error}"
        st.session_state["comparison_result"] = result

    result = st.session_state.get("comparison_result")
    if result is None:
        return

    st.markdown(result.narrative)
    if result.provider:
        st.caption(f"Narrative generated by {result.provider}")

    # Stem includes both side labels so a new comparison gets its own download
    # rather than silently reusing session state left over from the previous one.
    comparison_stem = "".join(
        c if c.isalnum() else "_" for c in f"comparison_{result.left.label}_vs_{result.right.label}".lower()
    ).strip("_")
    _chart_with_export(comparative_analysis.build_comparison_chart(result, config), comparison_stem)
    st.dataframe(result.to_frame(), width='stretch')
    st.caption(
        f"{result.left.label}: {result.left.n_rows:,} rows &nbsp;|&nbsp; "
        f"{result.right.label}: {result.right.n_rows:,} rows"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Compose the page: title, sidebar filters, and the tabs."""
    _init_session_state()
    data = load_everything()

    st.title(data["config"]["app"]["title"])
    st.caption(
        "Ask questions in plain language; an LLM writes the SQL, a sandbox runs it, "
        "and the result comes back as a chart and a written answer."
    )

    filtered, applied_filters, scope_spec = render_sidebar_filters(data["df"], data["config"])

    overview_tab, exploration_tab, ai_tab, advanced_tab = st.tabs(
        ["Overview", "Exploration", "AI Assistant", "Advanced"]
    )
    # Each tab is guarded independently so a failure in one -- an undrawable chart, a
    # filter combination a feature can't handle -- reports itself in place instead of
    # replacing the whole dashboard with a traceback.
    tabs = [
        (overview_tab, "the Overview tab", lambda: render_overview_tab(data, filtered)),
        (exploration_tab, "the Exploration tab", lambda: render_exploration_tab(data, filtered)),
        (ai_tab, "the AI Assistant tab", lambda: render_ai_tab(data, applied_filters, scope_spec)),
        (advanced_tab, "the Advanced tab", lambda: render_advanced_tab(data, filtered, scope_spec)),
    ]
    for tab, label, render in tabs:
        with tab, guarded_section(label):
            render()


main()
