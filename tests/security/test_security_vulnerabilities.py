"""
tests/security/test_security_vulnerabilities.py
==============================================
DevSecOps Automated Penetration & Vulnerability Tests:
1. Path Traversal & LFI Attacks
2. Malicious File Upload & MIME-Type Spoofing
3. Unhandled Exception Stack Trace & Credential Leakage
4. Endpoint DoS & Rate Throttling
"""
import io
import time
from unittest.mock import patch
import pytest
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework import status

pytestmark = pytest.mark.django_db


# ── 1. Path Traversal & Arbitrary File Access ──────────────────────────────────

class TestPathTraversalSecurity:
    """
    Validates that file access parameters cannot escape the sandbox MEDIA_ROOT/datasets/ directory.
    """
    report_url = "/api/report/"
    predict_url = "/api/predict/"

    @pytest.mark.parametrize("payload", [
        "../../../../etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",
        "....//....//etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "..\\..\\..\\Windows\\win.ini",
        "datasets/../../../settings.py",
    ])
    def test_report_path_traversal_blocked(self, auth_client, payload):
        response = auth_client.get(f"{self.report_url}?dataset_name={payload}")
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND]
        assert b"root:x:0:0" not in response.getvalue()
        assert b"[fonts]" not in response.getvalue()

    def test_predict_path_traversal_blocked(self, auth_client):
        payload = {"dataset_name": "../../../etc/passwd"}
        response = auth_client.post(self.predict_url, payload, format="json")
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND]


# ── 2. Malicious File Upload Attacks ──────────────────────────────────────────

class TestFileUploadSecurity:
    """
    Validates that executable binaries, web shells, and scripts disguised
    as CSV files are rejected during ingestion.
    """
    upload_url = "/api/upload/"

    def test_reject_windows_pe_executable_disguised_as_csv(self, auth_client):
        exe_content = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00malicious_payload"
        spoofed_file = SimpleUploadedFile("dataset.csv", exe_content, content_type="text/csv")
        response = auth_client.post(self.upload_url, {"file": spoofed_file}, format="multipart")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "error" in response.data

    def test_reject_svg_xss_disguised_as_csv(self, auth_client):
        xss_content = b"<svg xmlns=\"http://www.w3.org/2000/svg\"><script>alert(document.cookie)</script></svg>"
        spoofed_file = SimpleUploadedFile("customers.csv", xss_content, content_type="text/csv")
        response = auth_client.post(self.upload_url, {"file": spoofed_file}, format="multipart")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_reject_executable_extensions(self, auth_client):
        dangerous_names = ["payload.sh", "backdoor.py", "exploit.exe", "test.html"]
        for fname in dangerous_names:
            dummy_file = SimpleUploadedFile(fname, b"dummy data", content_type="application/octet-stream")
            response = auth_client.post(self.upload_url, {"file": dummy_file}, format="multipart")
            assert response.status_code == status.HTTP_400_BAD_REQUEST


# ── 3. Information Leakage in Production ───────────────────────────────────────

class TestInformationLeakage:
    """
    Ensures debug stack traces, SQL queries, and environment secrets
    never leak to API clients on unexpected errors when DEBUG=False.
    """
    @override_settings(DEBUG=False)
    def test_500_error_suppresses_stack_trace_when_debug_false(self, auth_client):
        with patch("api.services.run_all_models", side_effect=Exception("Database password=SuperSecretPassword123!")):
            with patch("api.views._dataset_path") as mock_path:
                mock_path.return_value.exists.return_value = True
                
                response = auth_client.post("/api/predict/", {"dataset_name": "data.csv"}, format="json")
                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                response_str = str(response.data)
                assert "SuperSecretPassword123!" not in response_str
                assert "Traceback" not in response_str
                assert "File \"" not in response_str


# ── 4. Rate Limiting / API Throttling ──────────────────────────────────────────

class TestRateLimitingResilience:
    """
    Validates API burst resilience and throttle enforcement on heavy AI endpoints.
    """
    def test_rapid_burst_calls_do_not_crash_worker(self, auth_client):
        status_codes = []
        for _ in range(25):
            res = auth_client.post("/api/chat/", {"dataset_id": "test.csv"}, format="json")
            status_codes.append(res.status_code)
        
        assert 500 not in status_codes
        assert 502 not in status_codes
        assert all(sc in [status.HTTP_400_BAD_REQUEST, status.HTTP_429_TOO_MANY_REQUESTS] for sc in status_codes)
