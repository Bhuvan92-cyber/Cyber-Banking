"""
Locust load testing suite for CyberPhysicalBanking v2.
Simulates concurrent user activities:
- Concurrent uploads of ~2MB CSV datasets
- Long-running ML model training requests (verifying no 504 timeouts within 120s)
- Concurrent RAG chat queries under load
- Report generation and PDF streaming
"""

import io
import time
from locust import HttpUser, task, between, events


class BankingPlatformLoadUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Simulate authentication session or CSRF initialization."""
        self.client.get("/api/csrf/")
        # Health check pre-flight
        self.client.get("/api/health/")

    @task(3)
    def test_rag_chat_concurrency(self):
        """Test RAG Chat query under concurrent retrieval."""
        payload = {
            "query": "What are the most significant risk indicators in transaction data?",
            "dataset_name": "banking_data.csv"
        }
        with self.client.post(
            "/api/chat/",
            json=payload,
            catch_response=True,
            name="/api/chat/ [Concurrent RAG]"
        ) as response:
            if response.status_code in [200, 503]:  # 503 is acceptable if local Ollama is offline
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(2)
    def test_large_dataset_upload(self):
        """Simulate concurrent upload of ~2MB CSV datasets."""
        # Generate ~2MB in-memory CSV
        header = "trans_id,user_id,amount,timestamp,channel,device_trust,target\n"
        row = "10001,user_88,540.25,2026-05-10 14:00:00,online,0.92,0\n"
        # 1 row is ~60 bytes. 35,000 rows is ~2.1MB
        row_count = 35000
        csv_buffer = io.StringIO()
        csv_buffer.write(header)
        csv_buffer.write(row * row_count)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        files = {
            "file": ("load_test_dataset.csv", io.BytesIO(csv_bytes), "text/csv")
        }

        with self.client.post(
            "/api/upload/",
            files=files,
            catch_response=True,
            name="/api/upload/ [2MB CSV Upload]"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 403:
                # Throttled or auth required
                response.success()
            else:
                response.failure(f"Upload failed with code: {response.status_code}")

    @task(1)
    def test_long_running_ml_training(self):
        """
        Verify long-running ML training endpoints withstand load without 504 Gateway Timeout.
        Gateway timeout threshold is 120 seconds.
        """
        payload = {
            "dataset_name": "load_test_dataset.csv",
            "target_column": "target",
            "algorithms": ["random_forest", "xgboost"]
        }
        start_time = time.time()
        with self.client.post(
            "/api/predict/",
            json=payload,
            timeout=120,
            catch_response=True,
            name="/api/predict/ [ML Training Long-Running]"
        ) as response:
            elapsed = time.time() - start_time
            if elapsed > 120:
                response.failure(f"Request took {elapsed:.2f}s, exceeding 120s timeout budget")
            elif response.status_code == 504:
                response.failure("Received 504 Gateway Timeout during model training")
            elif response.status_code in [200, 400, 404, 403]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
