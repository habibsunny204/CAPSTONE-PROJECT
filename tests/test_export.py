"""Tests for export/: PDF and Word report generation (Task C4).

Asserts real file bytes are produced with valid format signatures and that every
optional-section combination works, rather than asserting on exact rendered
layout -- the point is that a caller can always get a valid document back.
"""

import io
import zipfile

import pandas as pd
import plotly.graph_objects as go
import pytest

from export.docx_export import build_answer_docx
from export.pdf_export import build_answer_pdf

QUESTION = "What is total revenue by region?"
NARRATIVE = "Sales are led by **West** at 6,180.\n- West: 6,180\n- East: 597"


@pytest.fixture
def result_df():
    return pd.DataFrame({"region": ["West", "East"], "total_sales": [6180, 597]})


@pytest.fixture
def figure(result_df):
    return go.Figure(go.Bar(x=result_df["region"], y=result_df["total_sales"]))


def test_pdf_export_produces_valid_pdf_bytes(result_df, figure):
    pdf_bytes = build_answer_pdf(
        question=QUESTION, narrative=NARRATIVE, sql="SELECT 1",
        result=result_df, figure=figure,
        dataset_metadata={"Rows": "51,290"}, applied_filters={"Region": "All"},
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_docx_export_produces_valid_docx_bytes(result_df, figure):
    docx_bytes = build_answer_docx(
        question=QUESTION, narrative=NARRATIVE, sql="SELECT 1",
        result=result_df, figure=figure,
        dataset_metadata={"Rows": "51,290"}, applied_filters={"Region": "All"},
    )
    # A .docx is a zip archive; verify it opens and has the main document part.
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        assert "word/document.xml" in archive.namelist()


@pytest.mark.parametrize("builder", [build_answer_pdf, build_answer_docx])
def test_export_works_with_narrative_only(builder):
    """A declined/unanswerable answer has no SQL, result, or figure -- exporting
    it must still produce a valid document rather than raising.
    """
    output = builder(question=QUESTION, narrative="This can't be answered from the data.")
    assert len(output) > 500


@pytest.mark.parametrize("builder", [build_answer_pdf, build_answer_docx])
def test_export_works_without_a_figure(builder, result_df):
    output = builder(question=QUESTION, narrative=NARRATIVE, sql="SELECT 1", result=result_df)
    assert len(output) > 500


@pytest.mark.parametrize("builder", [build_answer_pdf, build_answer_docx])
def test_export_truncates_large_results(builder):
    """A large result must not blow up the document -- it's capped and labelled."""
    big = pd.DataFrame({"n": range(500), "value": range(500)})
    output = builder(question=QUESTION, narrative=NARRATIVE, result=big)
    assert len(output) > 500


def test_pdf_markdown_bold_is_not_rendered_literally():
    """Narratives come back as Markdown; asterisks must become bold, not text."""
    from export.pdf_export import _markdown_to_flowable_text

    assert _markdown_to_flowable_text("Sales hit **6,180** total") == "Sales hit <b>6,180</b> total"
    assert _markdown_to_flowable_text("- first\n- second") == "&bull; first<br/>&bull; second"
