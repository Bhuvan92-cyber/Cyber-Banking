"""
tests/test_ml_pipeline.py
=========================
Validates the data encoding, SMOTE oversampling, and model training
pipeline across Random Forest, Gradient Boosting, SVM, and Logistic Regression.
"""
import numpy as np
import pandas as pd
import pytest
from ml_models.algorithm_runner import encode_and_split, run_all_algorithms
from api import services

def test_encode_and_split_resamples_and_encodes(valid_banking_dataframe):
    assert "Account_Type" in valid_banking_dataframe.columns
    
    X_train, X_test, y_train, y_test = encode_and_split(valid_banking_dataframe.copy())

    assert not X_train.select_dtypes(include=["object"]).shape[1]
    assert len(y_train) > 0
    assert len(X_test) > 0
    assert set(np.unique(y_train)) == {0, 1}

def test_run_all_algorithms_returns_valid_metrics(valid_banking_dataframe):
    results = run_all_algorithms(valid_banking_dataframe.copy())

    expected_models = ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"]
    for model_name in expected_models:
        assert model_name in results, f"{model_name} missing from results"
        res = results[model_name]
        
        assert 0.0 <= res["accuracy"] <= 1.0
        assert isinstance(res["matrix"], np.ndarray)
        assert res["matrix"].shape == (2, 2)
        assert "weighted avg" in res["report"]
        assert "f1-score" in res["report"]["weighted avg"]

def test_service_run_all_models_persists_best_model(temp_csv_file, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    summary = services.run_all_models(temp_csv_file)

    assert "best_label" in summary
    assert "best_accuracy" in summary
    assert len(summary["models"]) == 4

    saved_models = list((tmp_path / "models").glob("*.pkl"))
    assert len(saved_models) > 0
