"""
tests/conftest.py
=================
Global fixtures for Django REST Framework and ML tests.
"""
import io
import pytest
import pandas as pd
from django.contrib.auth.models import User
from rest_framework.test import APIClient

@pytest.fixture
def api_client():
    """Unauthenticated DRF test client."""
    return APIClient()

@pytest.fixture
def auth_user(db):
    """Create a standard authenticated test user."""
    return User.objects.create_user(
        username="testanalyst",
        email="analyst@bank.internal",
        password="TestPassword123!",
    )

@pytest.fixture
def auth_client(api_client, auth_user):
    """DRF test client pre-authenticated via session."""
    api_client.force_authenticate(user=auth_user)
    return api_client

@pytest.fixture
def valid_banking_dataframe():
    """
    Generate a clean synthetic DataFrame with realistic banking features.
    Provides at least 8 samples per class to satisfy SMOTE k_neighbors requirements.
    """
    data = {
        "Age": [25, 45, 38, 52, 60, 29, 41, 33, 50, 23, 27, 48, 36, 55, 62, 31, 44, 35, 53, 26],
        "Balance": [
            5000.0, 75000.5, 12000.0, 45000.2, 90000.0, 3200.0, 54000.0, 18000.0, 62000.0, 1500.0,
            6200.0, 81000.0, 14500.0, 49000.0, 95000.0, 4100.0, 58000.0, 21000.0, 66000.0, 2200.0
        ],
        "Transaction_Count": [
            120, 310, 140, 280, 450, 95, 260, 190, 340, 80,
            130, 325, 155, 295, 470, 105, 275, 205, 355, 90
        ],
        "Credit_Score": [
            680, 740, 620, 790, 810, 590, 710, 650, 760, 570,
            690, 750, 630, 800, 820, 600, 720, 660, 770, 580
        ],
        "Account_Type": [
            "Savings", "Current", "Savings", "Current", "Savings", "Savings", "Current", "Savings", "Current", "Savings",
            "Savings", "Current", "Savings", "Current", "Savings", "Savings", "Current", "Savings", "Current", "Savings"
        ],
        "target": [
            0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 1, 0, 0, 1, 0, 1, 0, 1
        ],
    }
    return pd.DataFrame(data)

@pytest.fixture
def valid_csv_bytes(valid_banking_dataframe):
    """Return raw CSV bytes in memory."""
    buf = io.StringIO()
    valid_banking_dataframe.to_csv(buf, index=False)
    return io.BytesIO(buf.getvalue().encode("utf-8"))

@pytest.fixture
def temp_csv_file(tmp_path, valid_banking_dataframe):
    """Write synthetic banking dataset to temporary file on disk."""
    file_path = tmp_path / "mock_customers.csv"
    valid_banking_dataframe.to_csv(file_path, index=False)
    return str(file_path)
