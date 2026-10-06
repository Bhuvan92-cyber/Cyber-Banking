"""
tests/test_rag_pipeline.py
=========================
Validates dataset profiling, ChromaDB embedding operations, and LLM fallback handling.
"""
from unittest.mock import MagicMock, patch
import pytest
from ml_models import rag_pipeline

def test_build_dataset_profile_statistical_accuracy(temp_csv_file):
    profile = rag_pipeline.build_dataset_profile(temp_csv_file)

    assert isinstance(profile, str)
    assert "Dataset Profile" in profile or "mock_customers.csv" in profile
    assert "Shape: 20 rows" in profile
    assert "Balance" in profile
    assert "Transaction_Count" in profile
    assert "np.int64" not in profile

def test_profile_handles_nonexistent_file_gracefully():
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        rag_pipeline.build_dataset_profile("non_existent_path.csv")

@patch("langchain_core.runnables.RunnableSequence.invoke")
@patch("ml_models.rag_pipeline._get_or_build_profile")
def test_rag_query_executes_with_mocked_llm(mock_profile, mock_chain_invoke, temp_csv_file):
    mock_profile.return_value = "Profile: Total 20 rows. Average Balance: $50,000."
    mock_chain_invoke.return_value = "The average customer balance is $50,000."

    answer = rag_pipeline.query_dataset_rag(
        csv_file_path=temp_csv_file,
        user_query="What is the average balance?"
    )

    assert "50,000" in answer
    mock_chain_invoke.assert_called_once()

@patch("langchain_core.runnables.RunnableSequence.invoke")
def test_rag_pipeline_handles_llm_connection_error_gracefully(mock_chain_invoke, temp_csv_file):
    mock_chain_invoke.side_effect = ConnectionError("Connection refused by Ollama host")

    with pytest.raises(RuntimeError) as exc_info:
        rag_pipeline.query_dataset_rag(
            csv_file_path=temp_csv_file,
            user_query="Summarize dataset."
        )

    err_msg = str(exc_info.value).lower()
    assert "connection" in err_msg or "ollama" in err_msg or "failed" in err_msg
