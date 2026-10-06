"""
api/serializers.py
==================
DRF serializers act as the contract layer between HTTP and Python:
  - They validate incoming request data (type-safe, explicit error messages).
  - They serialize outgoing data to JSON.

All serializers here are Serializer (not ModelSerializer) because
the API works with files and ML results, not DB rows directly.
"""
from rest_framework import serializers


# ── Inbound: Upload ───────────────────────────────────────────────────────────

class DatasetUploadSerializer(serializers.Serializer):
    """Validates a multipart/form-data file upload."""
    file = serializers.FileField(
        help_text="CSV file to upload. Must have a 'target' column for ML training."
    )

    def validate_file(self, value):
        """Enforce CSV-only uploads and a sane file size limit (50 MB)."""
        MAX_BYTES = 50 * 1024 * 1024  # 50 MB

        if not value.name.lower().endswith(".csv"):
            raise serializers.ValidationError(
                "Only CSV files are accepted. Received: "
                f"'{value.name}'"
            )
        if value.size > MAX_BYTES:
            raise serializers.ValidationError(
                f"File too large ({value.size / 1_048_576:.1f} MB). "
                f"Maximum allowed: {MAX_BYTES // 1_048_576} MB."
            )
        return value


# ── Inbound: Predict ──────────────────────────────────────────────────────────

class PredictionRequestSerializer(serializers.Serializer):
    """
    Accepts either a previously-uploaded dataset name OR a one-off JSON
    payload of feature values for a single-record prediction.

    Rules:
      • dataset_name  → run all models on that file, return ranked results
      • input_data    → use the persisted best model for single-row inference
      • At least one of the two must be provided.
    """
    dataset_name = serializers.CharField(
        required=False,
        allow_blank=False,
        max_length=255,
        help_text="Filename of a previously uploaded CSV (e.g. 'customers.csv').",
    )
    input_data = serializers.DictField(
        child=serializers.CharField(allow_blank=True),
        required=False,
        help_text=(
            "Key-value dict for single-record inference using the saved best model. "
            "Example: {\"age\": \"35\", \"balance\": \"12000.50\", ...}"
        ),
    )

    def validate(self, data: dict) -> dict:
        if not data.get("dataset_name") and not data.get("input_data"):
            raise serializers.ValidationError(
                "Provide either 'dataset_name' (batch prediction on a CSV) "
                "or 'input_data' (single-record inference)."
            )
        return data


# ── Inbound: Chat / RAG ───────────────────────────────────────────────────────

class ChatQuerySerializer(serializers.Serializer):
    """Validates the natural-language dataset query request (Phase 4)."""
    dataset_id = serializers.CharField(
        help_text="Filename of the uploaded CSV to query (e.g. 'customers.csv')."
    )
    query = serializers.CharField(
        min_length=5,
        max_length=2000,
        help_text=(
            "Natural language question about the dataset. "
            "Example: 'What are the top 3 features driving loan defaults?'"
        ),
    )


# ── Outbound: Dataset Summary ─────────────────────────────────────────────────

class DatasetSummarySerializer(serializers.Serializer):
    """Read-only — used to document the shape of a summary response."""
    filename      = serializers.CharField(read_only=True)
    rows          = serializers.IntegerField(read_only=True)
    columns       = serializers.IntegerField(read_only=True)
    column_names  = serializers.ListField(child=serializers.CharField(), read_only=True)
    missing_values = serializers.DictField(child=serializers.IntegerField(), read_only=True)
    dtypes        = serializers.DictField(child=serializers.CharField(), read_only=True)
    preview       = serializers.ListField(read_only=True)
