"""Streamlit entrypoint: st.tabs() layout for Overview / Exploration / AI Assistant
(Task C1).

Run with `streamlit run app/app.py`.

Uses st.tabs() rather than the multipage pages/ folder deliberately (PROJECT_SPEC.md
C1): pages/ resets widget state across page switches, which would break the
persistent global sidebar filters this dashboard is built around.

Data is loaded and cleaned once per server process via @st.cache_resource, not once
per rerun -- Streamlit re-executes this whole script top-to-bottom on every widget
interaction, so an uncached ingest would re-parse the 51k-row CSV on every click.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow `streamlit run app/app.py` to import the sibling packages (backend/, llm/,
# viz/, export/) -- Streamlit puts the script's own directory on sys.path, not the
# repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import ingest, quality, schema as schema_module  # noqa: E402
from export.docx_export import build_answer_docx  # noqa: E402
from export.pdf_export import build_answer_pdf  # noqa: E402
from llm import pipeline  # noqa: E402
from llm.memory import ConversationMemory  # noqa: E402
from viz import auto_select, charts  # noqa: E402

st.set_page_config(page_title="Superstore AI Analytics", page_icon="B", layout="wide")


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
    profile_before = quality.profile_report(con, table_name, id_column=config["dataset"]["id_column"])
    clean_report = quality.clean(con, table_name, config)
    profile_after = quality.profile_report(con, table_name, id_column=config["dataset"]["id_column"])

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
    st.session_state.setdefault("answers", [])
    st.session_state.setdefault("insights", {})


# ---------------------------------------------------------------------------
# Sidebar filters (C1)
# ---------------------------------------------------------------------------


def render_sidebar_filters(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render the configured global filters and return (filtered DataFrame, a
    human-readable summary of what's applied). Filter definitions come from
    configs/dataset_config.yaml, so adding one is a config change, not a code change.
    """
    st.sidebar.header("Filters")
    st.sidebar.caption("Applied across the Overview and Exploration tabs.")

    filtered = df
    applied: dict[str, Any] = {}

    for spec in config.get("filters", []):
        column, label = spec["column"], spec["label"]
        if column not in df.columns:
            continue

        if spec["type"] == "date_range":
            min_date, max_date = df[column].min().date(), df[column].max().date()
            chosen = st.sidebar.date_input(
                label, value=(min_date, max_date),
                min_value=min_date, max_value=max_date, key=f"filter_{column}",
            )
            if isinstance(chosen, tuple) and len(chosen) == 2:
                start, end = chosen
                filtered = filtered[
                    (filtered[column] >= pd.Timestamp(start)) & (filtered[column] <= pd.Timestamp(end))
                ]
                if (start, end) != (min_date, max_date):
                    applied[label] = f"{start} to {end}"

        elif spec["type"] == "multiselect":
            options = sorted(df[column].dropna().unique().tolist())
            chosen = st.sidebar.multiselect(label, options, default=[], key=f"filter_{column}")
            if chosen:
                filtered = filtered[filtered[column].isin(chosen)]
                applied[label] = ", ".join(str(c) for c in chosen)

    st.sidebar.metric("Rows after filters", f"{len(filtered):,}", delta=f"{len(filtered) - len(df):,}"
                      if len(filtered) != len(df) else None)
    if not applied:
        applied = {"Filters": "none (full dataset)"}
    return filtered, applied


# ---------------------------------------------------------------------------
# Tab 1 -- Overview
# ---------------------------------------------------------------------------


def render_overview_tab(data: dict[str, Any], filtered: pd.DataFrame) -> None:
    """Headline metrics, the data-quality summary (A3), and two curated charts."""
    config = data["config"]

    st.subheader("At a glance")
    columns = st.columns(4)
    columns[0].metric("Rows", f"{len(filtered):,}")
    columns[1].metric("Total sales", f"{filtered['sales'].sum():,.0f}"
                      if "sales" in filtered else "n/a")
    columns[2].metric("Total profit", f"{filtered['profit'].sum():,.0f}"
                      if "profit" in filtered else "n/a")
    columns[3].metric("Countries", f"{filtered['country'].nunique():,}"
                      if "country" in filtered else "n/a")

    if filtered.empty:
        st.warning("No rows match the current filters. Widen them in the sidebar to see charts.")
        return

    st.plotly_chart(charts.time_series(filtered, config), width='stretch')
    st.plotly_chart(charts.choropleth(filtered, config), width='stretch')

    st.subheader("Data quality")
    st.caption(
        "Profiled and cleaned at load time by `backend/quality.py` (Task A3). "
        "These figures describe the full dataset, before sidebar filters."
    )

    profile = data["profile_after"]
    quality_columns = st.columns(3)
    quality_columns[0].metric("Rows profiled", f"{profile['n_rows']:,}")
    quality_columns[1].metric("Duplicate rows", f"{profile['n_duplicate_rows']:,}")
    quality_columns[2].metric("Columns", f"{len(profile['columns']):,}")

    with st.expander("Cleaning steps applied"):
        for step in data["clean_report"]["steps_applied"]:
            st.markdown(f"**{step['name']}** &mdash; {step}")

    with st.expander("Per-column quality profile"):
        st.dataframe(pd.DataFrame(profile["columns"]), width='stretch')


# ---------------------------------------------------------------------------
# Tab 2 -- Exploration
# ---------------------------------------------------------------------------


def render_exploration_tab(data: dict[str, Any], filtered: pd.DataFrame) -> None:
    """The remaining curated charts, with drill-down and per-chart image export."""
    config = data["config"]

    if filtered.empty:
        st.warning("No rows match the current filters. Widen them in the sidebar to see charts.")
        return

    left, right = st.columns(2)
    with left:
        _chart_with_export(charts.box_plot(filtered, config), "profit_by_category")
    with right:
        _chart_with_export(charts.scatter_regression(filtered, config), "discount_vs_profit")

    _chart_with_export(charts.correlation_heatmap(filtered, config), "correlation_heatmap")
    _chart_with_export(charts.sunburst(filtered, config), "category_breakdown")

    st.subheader("Region and category breakdown")
    binding = config["charts"]["curated_bindings"]["stacked_bar"]
    primary = binding["primary_dim"]
    drill_options = ["All regions"] + sorted(filtered[primary].dropna().unique().tolist())
    chosen = st.selectbox("Drill into", drill_options, key="stacked_bar_drill")
    drill_into = None if chosen == "All regions" else chosen
    _chart_with_export(charts.stacked_bar(filtered, config, drill_into=drill_into), "region_category")

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


def render_ai_tab(data: dict[str, Any], applied_filters: dict[str, Any]) -> None:
    """The natural-language pipeline (B2), preset insights (B3), conversational
    memory (B4), AI-driven chart selection (C3), and per-answer export (C4).
    """
    config, table_name = data["config"], data["table_name"]
    # A dedicated cursor for LLM-generated SQL, kept separate from the connection
    # ingest/quality used to build the table (see llm/sandbox.py).
    llm_con = data["con"].cursor()

    st.subheader("Ask a question")
    st.caption(
        "Answers are generated live by Gemini (falling back to Groq), turned into "
        "validated read-only SQL, executed in a sandbox, then narrated."
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the total revenue by region?",
        key="ai_question",
    )
    ask_column, reset_column = st.columns([1, 1])
    asked = ask_column.button("Ask", type="primary", width='stretch', key="ask_button")
    if reset_column.button("Reset conversation", width='stretch', key="reset_button"):
        st.session_state["memory"].reset()
        st.session_state["answers"] = []
        st.rerun()

    if asked and question.strip():
        _run_question(llm_con, table_name, config, question.strip())

    _render_preset_insights(llm_con, table_name, config)

    for index, answer in enumerate(reversed(st.session_state["answers"])):
        _render_answer(answer, index, config, applied_filters, data)


def _run_question(llm_con, table_name: str, config: dict[str, Any], question: str) -> None:
    """Run one question through the pipeline, with a loading indicator and elapsed
    time (Task B5's UI half), and store the result in session state.
    """
    status = st.status("Thinking...", expanded=True)
    started = time.perf_counter()
    try:
        status.write("Generating SQL...")
        result = pipeline.answer_question(
            llm_con, table_name, config, question,
            history=st.session_state["memory"].get_history(),
        )
        elapsed = time.perf_counter() - started
        status.update(label=f"Answered in {elapsed:.1f}s via {result.sql_provider}", state="complete")

        st.session_state["memory"].add(question, result.sql, result.narrative)
        st.session_state["answers"].append({
            "question": question, "sql": result.sql, "reasoning": result.reasoning,
            "result": result.result, "narrative": result.narrative,
            "provider": result.sql_provider, "retried": result.retried, "elapsed": elapsed,
        })
    except pipeline.PipelineError as error:
        status.update(label="Couldn't answer that one", state="error")
        st.error(str(error))
    except Exception as error:  # noqa: BLE001 - surface any provider/network failure cleanly
        status.update(label="Something went wrong", state="error")
        st.error(f"{type(error).__name__}: {error}")


def _render_preset_insights(llm_con, table_name: str, config: dict[str, Any]) -> None:
    """The three preset insight buttons (Task B3)."""
    st.subheader("Preset insights")
    presets = {
        "Dataset overview": pipeline.generate_dataset_overview,
        "Trend analysis": pipeline.generate_trend_comparison,
        "Anomaly report": pipeline.generate_anomaly_report,
    }

    for column, (label, generator) in zip(st.columns(len(presets)), presets.items()):
        if column.button(label, width='stretch', key=f"preset_{label}"):
            with st.spinner(f"Generating {label.lower()}..."):
                try:
                    insight = generator(llm_con, table_name, config)
                    st.session_state["insights"][label] = insight
                except Exception as error:  # noqa: BLE001
                    st.error(f"{label} failed: {type(error).__name__}: {error}")

    for label, insight in st.session_state["insights"].items():
        with st.expander(f"{label} (generated by {insight.narrative_provider})", expanded=True):
            st.markdown(insight.narrative)
            aggregation = insight.data.get("aggregation")
            if aggregation is not None:
                st.dataframe(aggregation, width='stretch')
            samples = insight.data.get("outlier_samples")
            if samples is not None and not samples.empty:
                st.caption("Sample flagged rows")
                st.dataframe(samples, width='stretch')


def _render_answer(answer: dict[str, Any], index: int, config: dict[str, Any],
                   applied_filters: dict[str, Any], data: dict[str, Any]) -> None:
    """Render one stored answer: narrative, auto-selected chart with a manual
    override, the result table, and PDF/Word export.
    """
    st.divider()
    st.markdown(f"#### {answer['question']}")
    st.caption(
        f"Answered by {answer['provider']} in {answer['elapsed']:.1f}s"
        + (" (after one retry)" if answer["retried"] else "")
    )
    st.markdown(answer["narrative"])

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

        effective = selection
        if chosen_type != selection.chart_type:
            effective = auto_select.ChartSelection(
                chart_type=chosen_type, x=selection.x, y=selection.y,
                color=selection.color, reason="Manually overridden.",
            )

        figure = charts.build_from_selection(result_df, effective, config, title=answer["question"])
        if figure is not None:
            st.plotly_chart(figure, width='stretch')
            st.caption(effective.reason)
        elif effective.chart_type == auto_select.CHART_METRIC:
            st.metric(str(result_df.columns[0]), f"{result_df.iloc[0, 0]:,.2f}")
        else:
            st.dataframe(result_df, width='stretch')

        with st.expander("Result data and generated SQL"):
            st.dataframe(result_df, width='stretch')
            st.code(answer["sql"], language="sql")

    metadata = {
        "Rows": f"{data['profile_after']['n_rows']:,}",
        "Source": "Global Superstore",
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
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Compose the page: title, sidebar filters, and the three tabs."""
    _init_session_state()
    data = load_everything()

    st.title("Superstore AI Analytics")
    st.caption(
        "Ask questions in plain language; an LLM writes the SQL, a sandbox runs it, "
        "and the result comes back as a chart and a written answer."
    )

    filtered, applied_filters = render_sidebar_filters(data["df"], data["config"])

    overview_tab, exploration_tab, ai_tab = st.tabs(["Overview", "Exploration", "AI Assistant"])
    with overview_tab:
        render_overview_tab(data, filtered)
    with exploration_tab:
        render_exploration_tab(data, filtered)
    with ai_tab:
        render_ai_tab(data, applied_filters)


main()
