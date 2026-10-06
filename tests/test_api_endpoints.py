"""
tests/test_api_endpoints.py
===========================
End-to-end HTTP and permission tests for all DRF endpoints:
- POST /api/upload/
- POST /api/predict/
- GET  /api/report/
- POST /api/chat/
"""
from pathlib import Path
from unittest.mock import patch
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status

pytestmark = pytest.mark.django_db


# ── 1. POST /api/upload/ ────────────────────────────────────────────────────────

class TestUploadAPI:
    url = "/api/upload/"

    def test_upload_unauthenticated_returns_401(self, api_client, valid_csv_bytes):
        file = SimpleUploadedFile("data.csv", valid_csv_bytes.read(), content_type="text/csv")
        response = api_client.post(self.url, {"file": file}, format="multipart")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

    def test_upload_missing_file_returns_400(self, auth_client):
        response = auth_client.post(self.url, {}, format="multipart")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "file" in response.data

    def test_upload_invalid_file_extension_returns_400(self, auth_client):
        file = SimpleUploadedFile("payload.exe", b"MZbinaryexec", content_type="application/octet-stream")
        response = auth_client.post(self.url, {"file": file}, format="multipart")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_upload_valid_csv_returns_201_with_summary(self, auth_client, valid_csv_bytes, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path)
        file = SimpleUploadedFile("customers.csv", valid_csv_bytes.read(), content_type="text/csv")
        response = auth_client.post(self.url, {"file": file}, format="multipart")

        assert response.status_code == status.HTTP_201_CREATED
        assert response.data["dataset_id"] == "customers.csv"
        assert "summary" in response.data
        assert response.data["summary"]["rows"] == 20
        assert (Path(settings.MEDIA_ROOT) / "datasets" / "customers.csv").exists()


# ── 2. POST /api/predict/ ───────────────────────────────────────────────────────

class TestPredictAPI:
    url = "/api/predict/"

    def test_predict_unauthenticated_returns_401(self, api_client):
        response = api_client.post(self.url, {"dataset_name": "none.csv"}, format="json")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

    def test_predict_empty_payload_returns_400(self, auth_client):
        response = auth_client.post(self.url, {}, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_predict_dataset_not_found_returns_404(self, auth_client, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path)
        response = auth_client.post(self.url, {"dataset_name": "missing.csv"}, format="json")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.data["error"].lower()

    @patch("api.services.run_all_models")
    def test_predict_batch_mode_returns_200(self, mock_run, auth_client, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path)
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        (datasets_dir / "existing.csv").write_text("a,b\n1,2")

        mock_run.return_value = {
            "best_label": "Random Forest",
            "best_accuracy": 0.94,
            "models": [{"key": "rf", "accuracy": 0.94}],
        }

        response = auth_client.post(self.url, {"dataset_name": "existing.csv"}, format="json")
        assert response.status_code == status.HTTP_200_OK
        assert response.data["mode"] == "batch_training"
        assert response.data["best_model"] == "Random Forest"

    @patch("api.services.predict_single_record")
    def test_predict_single_inference_returns_200(self, mock_inference, auth_client):
        mock_inference.return_value = {
            "prediction": "Non-Default",
            "probability": 0.88,
            "feature_importance": {"Balance": 0.42},
        }
        payload = {"input_data": {"Age": "35", "Balance": "15000"}}
        response = auth_client.post(self.url, payload, format="json")

        assert response.status_code == status.HTTP_200_OK
        assert response.data["prediction"] == "Non-Default"


# ── 3. GET /api/report/ ─────────────────────────────────────────────────────────

class TestReportAPI:
    url = "/api/report/"

    def test_report_unauthenticated_returns_401(self, api_client):
        response = api_client.get(self.url + "?dataset_name=test.csv")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

    def test_report_missing_param_returns_400(self, auth_client):
        response = auth_client.get(self.url)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "required" in response.data["error"].lower()

    def test_report_dataset_not_found_returns_404(self, auth_client, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path)
        response = auth_client.get(self.url + "?dataset_name=nonexistent.csv")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("api.services.generate_pdf_report")
    def test_report_valid_dataset_streams_pdf_200(self, mock_pdf, auth_client, tmp_path, settings):
        settings.MEDIA_ROOT = str(tmp_path)
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        (datasets_dir / "valid.csv").write_text("a,b\n1,2")

        mock_pdf.return_value = b"%PDF-1.4 Mock PDF Content"

        response = auth_client.get(self.url + "?dataset_name=valid.csv")
        assert response.status_code == status.HTTP_200_OK
        assert response["Content-Type"] == "application/pdf"
        assert b"%PDF-1.4" in response.getvalue()


# ── 4. POST /api/chat/ ──────────────────────────────────────────────────────────

class TestChatAPI:
    url = "/api/chat/"

    def test_chat_unauthenticated_returns_401(self, api_client):
        response = api_client.post(self.url, {"dataset_id": "cust.csv", "query": "hello"}, format="json")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

    def test_chat_missing_fields_returns_400(self, auth_client):
        response = auth_client.post(self.url, {"query": "only query"}, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch("api.services.chat_with_dataset_service")
    def test_chat_success_returns_200(self, mock_chat, auth_client):
        mock_chat.return_value = {
            "success": True,
            "dataset_name": "cust.csv",
            "query": "Average balance?",
            "answer": "The average balance is $50,699.",
        }
        payload = {"dataset_id": "cust.csv", "query": "Average balance?"}
        response = auth_client.post(self.url, payload, format="json")

        assert response.status_code == status.HTTP_200_OK
        assert response.data["success"] is True
        assert "50,699" in response.data["answer"]

    @patch("api.services.chat_with_dataset_service")
    def test_chat_service_failure_returns_error_status(self, mock_chat, auth_client):
        mock_chat.return_value = {
            "success": False,
            "error": "Ollama service connection refused at http://127.0.0.1:11434",
            "status_code": 503,
        }
        payload = {"dataset_id": "cust.csv", "query": "Summarize."}
        response = auth_client.post(self.url, payload, format="json")
        assert response.status_code in [status.HTTP_503_SERVICE_UNAVAILABLE, status.HTTP_500_INTERNAL_SERVER_ERROR, status.HTTP_400_BAD_REQUEST]
