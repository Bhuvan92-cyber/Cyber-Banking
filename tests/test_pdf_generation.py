"""
tests/test_pdf_generation.py
============================
Validates ReportLab PDF document compilation and byte streaming.
"""
from unittest.mock import patch
from api import services

def test_generate_pdf_report_returns_valid_pdf_bytes(temp_csv_file):
    ai_summary = "Customer risk profiles indicate stable balance distribution across tiers."
    
    pdf_bytes = services.generate_pdf_report(
        csv_path=temp_csv_file,
        ai_summary=ai_summary,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in pdf_bytes[-1024:]

def test_generate_pdf_report_without_ai_summary(temp_csv_file):
    with patch("ml_models.rag_pipeline.generate_executive_summary", return_value="Auto-generated summary."):
        pdf_bytes = services.generate_pdf_report(
            csv_path=temp_csv_file,
            ai_summary="",
        )

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")
