"""One-click PDF export of an AI Assistant answer: dataset metadata, applied
filters, narrative, and chart image (Task C4).

Returns PDF bytes rather than writing to disk, so the caller (Streamlit's
st.download_button) can hand them straight to the browser without the app needing
a writable working directory -- which also keeps this deployment-agnostic per
PROJECT_SPEC.md Section 11.
"""

from __future__ import annotations

import datetime
import io
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from viz.charts import figure_to_image_bytes

MAX_TABLE_ROWS = 15


def _markdown_to_flowable_text(markdown_text: str) -> str:
    """Convert the small subset of Markdown the narrative prompts produce (**bold**,
    *italic*, and `-` bullets) into ReportLab's mini-HTML. Not a general Markdown
    parser -- just enough that bold figures in a narrative don't render as literal
    asterisks.
    """
    import re

    text = markdown_text.strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"^\s*[-*]\s+", "&bull; ", text, flags=re.MULTILINE)
    return text.replace("\n", "<br/>")


def _styles() -> dict[str, ParagraphStyle]:
    """Paragraph styles for the report."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontSize=18, spaceAfter=6),
        "meta": ParagraphStyle("Meta", parent=base["Normal"], fontSize=8,
                               textColor=colors.HexColor("#52514e"), spaceAfter=2),
        "heading": ParagraphStyle("SectionHeading", parent=base["Heading2"], fontSize=12,
                                  spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=10, leading=15,
                               alignment=TA_LEFT, spaceAfter=6),
        "code": ParagraphStyle("Code", parent=base["Code"], fontSize=8, leading=11,
                               textColor=colors.HexColor("#0b0b0b")),
    }


def _result_table(df: pd.DataFrame, styles: dict[str, ParagraphStyle]) -> Table:
    """Render up to MAX_TABLE_ROWS of a result DataFrame as a styled table."""
    shown = df.head(MAX_TABLE_ROWS)
    header = [str(c) for c in shown.columns]
    body = [[str(v) for v in row] for row in shown.itertuples(index=False)]

    table = Table([header] + body, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeec")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0b0b0b")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c8c3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_answer_pdf(
    question: str,
    narrative: str,
    sql: str | None = None,
    result: pd.DataFrame | None = None,
    figure: go.Figure | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    applied_filters: dict[str, Any] | None = None,
) -> bytes:
    """Build a PDF report for one AI Assistant answer and return its bytes.

    Every section is optional except the question and narrative, so the same
    function serves a chart-backed answer, a table-only answer, and a declined
    ("unanswerable") answer without the caller branching.
    """
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="AI Analytics Report",
    )

    story: list[Any] = [
        Paragraph("AI Analytics Report", styles["title"]),
        Paragraph(f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["meta"]),
    ]

    if dataset_metadata:
        details = " &nbsp;|&nbsp; ".join(f"{k}: {v}" for k, v in dataset_metadata.items())
        story.append(Paragraph(details, styles["meta"]))

    if applied_filters:
        active = " &nbsp;|&nbsp; ".join(f"{k}: {v}" for k, v in applied_filters.items())
        story.append(Paragraph(f"Filters &mdash; {active}", styles["meta"]))

    story += [
        Spacer(1, 8),
        Paragraph("Question", styles["heading"]),
        Paragraph(question, styles["body"]),
        Paragraph("Answer", styles["heading"]),
        Paragraph(_markdown_to_flowable_text(narrative), styles["body"]),
    ]

    if figure is not None:
        story.append(Paragraph("Chart", styles["heading"]))
        image_bytes = figure_to_image_bytes(figure, fmt="png", scale=2)
        story.append(Image(io.BytesIO(image_bytes), width=170 * mm, height=98 * mm))

    if result is not None and not result.empty:
        story.append(Paragraph("Result data", styles["heading"]))
        story.append(_result_table(result, styles))
        if len(result) > MAX_TABLE_ROWS:
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"Showing first {MAX_TABLE_ROWS} of {len(result):,} rows.", styles["meta"]))

    if sql:
        story.append(Paragraph("Generated SQL", styles["heading"]))
        story.append(Paragraph(sql.replace("\n", "<br/>"), styles["code"]))

    doc.build(story)
    return buffer.getvalue()
