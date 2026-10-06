"""
api/views.py
============
DRF API views — deliberately thin.

Each view does exactly three things:
  1. Validate the incoming request (via serializer).
  2. Call the appropriate service function.
  3. Translate the result (or exception) into an HTTP response.

Zero business logic here — that all lives in api/services.py.
"""
from __future__ import annotations

import os
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, HttpResponse
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import (
    ChatQuerySerializer,
    DatasetUploadSerializer,
    PredictionRequestSerializer,
)
from . import services


# ── helpers ────────────────────────────────────────────────────────────────────

def _dataset_path(filename: str) -> Path:
    """
    Securely resolve a filename to its full path inside MEDIA_ROOT/datasets/.
    Uses Path(filename).name and resolve() to strictly prevent directory traversal.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Directory traversal detected in filename: {filename}")
    safe_name = Path(filename).name
    base_dir = (Path(settings.MEDIA_ROOT) / "datasets").resolve()
    resolved = (base_dir / safe_name).resolve()
    if not str(resolved).startswith(str(base_dir)):
        raise PermissionError(f"Directory traversal path escaping sandbox: {filename}")
    return resolved


def _ensure_media_dirs() -> None:
    """Create required sub-directories inside MEDIA_ROOT if absent."""
    for sub in ("datasets", "reports"):
        (Path(settings.MEDIA_ROOT) / sub).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload/
# ─────────────────────────────────────────────────────────────────────────────

class DatasetUploadView(APIView):
    """
    Upload a CSV dataset.

    Request  : multipart/form-data  →  { "file": <csv> }
    Response : 201  →  dataset summary (rows, columns, dtypes, missing values)
             : 400  →  validation error or unreadable CSV

    Permissions: IsAuthenticated (session or basic auth)
    """
    parser_classes   = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = DatasetUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = serializer.validated_data["file"]

        # Security: strictly validate filename extension
        if not uploaded_file.name.lower().endswith(".csv"):
            return Response(
                {"error": "Invalid file type. Only .csv files are supported."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Security: Sniff first bytes for executable or script headers
        first_bytes = uploaded_file.read(256)
        uploaded_file.seek(0)
        if (
            first_bytes.startswith(b"MZ")
            or b"<script" in first_bytes.lower()
            or b"<svg" in first_bytes.lower()
            or first_bytes.startswith(b"#!")
        ):
            return Response(
                {"error": "Malicious or disguised file format rejected."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        _ensure_media_dirs()
        try:
            file_path = _dataset_path(uploaded_file.name)
        except (ValueError, PermissionError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        # Stream file to disk in chunks — safe for large CSVs
        with open(file_path, "wb+") as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

        try:
            summary = services.build_dataset_summary(str(file_path))
        except Exception as exc:
            # Remove corrupted file so storage stays clean
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            return Response(
                {"error": f"Failed to parse CSV: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {
                "message": "Dataset uploaded successfully.",
                "dataset_id": uploaded_file.name,   # use this in /predict/ and /report/
                "summary": summary,
            },
            status=status.HTTP_201_CREATED,
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/
# ─────────────────────────────────────────────────────────────────────────────

class PredictView(APIView):
    """
    Run ML predictions. Two modes:

    Mode A — Batch (train on uploaded CSV, rank all models):
        { "dataset_name": "customers.csv" }
        → Trains RF, GB, SVM, LR. Returns all model metrics, ranked by accuracy.
          Persists the best model to disk for Mode B.

    Mode B — Single-record inference (uses the persisted best model):
        { "input_data": { "age": "35", "balance": "12000", ... } }
        → Returns prediction label + feature importance.

    Response : 200
             : 400  validation / missing file
             : 404  dataset_name not found
             : 500  unexpected ML error
    """
    permission_classes = [IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        dataset_name = serializer.validated_data.get("dataset_name")
        input_data   = serializer.validated_data.get("input_data")

        # ── Mode A: Batch training ────────────────────────────────────────────
        if dataset_name:
            try:
                file_path = _dataset_path(dataset_name)
            except (ValueError, PermissionError) as exc:
                return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

            if not file_path.exists():
                return Response(
                    {
                        "error": f"Dataset '{dataset_name}' not found. "
                                 "Upload it first via POST /api/upload/."
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )
            try:
                results = services.run_all_models(str(file_path))
                return Response(
                    {
                        "mode":          "batch_training",
                        "dataset":       dataset_name,
                        "best_model":    results["best_label"],
                        "best_accuracy": results["best_accuracy"],
                        "all_models":    results["models"],
                        "note": (
                            "Best model persisted to disk. "
                            "You can now call POST /api/predict/ with "
                            "'input_data' for single-record inference."
                        ),
                    },
                    status=status.HTTP_200_OK,
                )
            except ValueError as exc:
                return Response(
                    {"error": f"Data error: {exc}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            except Exception as exc:
                error_msg = f"Training failed: {exc}" if settings.DEBUG else "Training failed due to an internal server error."
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        # ── Mode B: Single-record inference ───────────────────────────────────
        try:
            result = services.predict_single_record(input_data)
            return Response(
                {"mode": "single_inference", **result},
                status=status.HTTP_200_OK,
            )
        except FileNotFoundError as exc:
            return Response(
                {"error": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            error_msg = f"Inference failed: {exc}" if settings.DEBUG else "Inference failed due to an internal server error."
            return Response(
                {"error": error_msg},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/report/?dataset_name=<filename>
# ─────────────────────────────────────────────────────────────────────────────

class ReportView(APIView):
    """
    Generate and stream a PDF analytics report for an uploaded dataset.

    Query params : dataset_name=<filename>   e.g. ?dataset_name=customers.csv
    Response     : 200  application/pdf  (streamed as attachment)
                 : 400  missing param
                 : 404  dataset not found
                 : 500  generation error

    The PDF includes:
      • (Phase 5) AI Executive Summary at the top (empty until Phase 5)
      • Model accuracy comparison table
      • Pie + line accuracy charts
      • Confusion matrix heatmap
      • Classification report heatmap
    """
    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> HttpResponse:
        dataset_name = request.query_params.get("dataset_name")
        if not dataset_name:
            return Response(
                {"error": "'dataset_name' query parameter is required. "
                          "e.g. GET /api/report/?dataset_name=customers.csv"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            file_path = _dataset_path(dataset_name)
        except (ValueError, PermissionError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        if not file_path.exists():
            return Response(
                {"error": f"Dataset '{dataset_name}' not found. "
                          "Upload it first via POST /api/upload/."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            pdf_bytes = services.generate_pdf_report(
                csv_path=str(file_path),
            )
        except Exception as exc:
            error_msg = f"PDF generation failed: {exc}" if settings.DEBUG else "PDF generation failed due to an internal server error."
            return Response(
                {"error": error_msg},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Stream the PDF directly as attachment
        import io
        response = FileResponse(
            io.BytesIO(pdf_bytes),
            as_attachment=True,
            filename=f"banking_report_{dataset_name.removesuffix('.csv')}.pdf",
            content_type="application/pdf",
        )
        return response



# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/
# ─────────────────────────────────────────────────────────────────────────────

class ChatView(APIView):
    """
    Natural-language dataset querying via RAG + Ollama (Phase 4).

    Request
    -------
    POST /api/chat/
    Content-Type: application/json
    {
        "dataset_id": "customers.csv",       ← filename from /api/upload/ response
        "query":      "What drives defaults?" ← free-text question
    }

    Response (200)
    --------------
    {
        "success":      true,
        "dataset_name": "customers.csv",
        "query":        "What drives defaults?",
        "answer":       "Based on the dataset profile, ..."
    }

    Error responses
    ---------------
    400 — validation error or dataset not found
    503 — Ollama unavailable / model not pulled
    500 — unexpected pipeline error

    Requires Ollama to be running locally (or at OLLAMA_BASE_URL).
    Pull the model first: `ollama pull llama3`
    """
    permission_classes = [IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = ChatQuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # ChatQuerySerializer uses 'dataset_id' field name
        dataset_id = serializer.validated_data["dataset_id"]
        query      = serializer.validated_data["query"]

        # Delegate entirely to the service — no business logic in the view
        result = services.chat_with_dataset_service(
            dataset_name=dataset_id,
            user_query=query,
        )

        if not result["success"]:
            error_msg = result.get("error", "Unknown error")

            # Surface Ollama connectivity problems as 503 (service unavailable)
            # so the client knows to check their LLM server, not their request
            if "Ollama" in error_msg or "ollama pull" in error_msg:
                return Response(
                    {
                        "success":    False,
                        "error":      error_msg,
                        "hint": (
                            f"Ensure Ollama is running at "
                            f"{request.build_absolute_uri('/')[:-1]} "
                            "and that the model is pulled."
                        ),
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        return Response(
            {
                "success":      True,
                "dataset_name": dataset_id,
                "query":        query,
                "answer":       result["answer"],
            },
            status=status.HTTP_200_OK,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Authentication API (Session-based for Next.js)
# ─────────────────────────────────────────────────────────────────────────────
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie
from django.utils.decorators import method_decorator


class CSRFTokenView(APIView):
    """
    Sets CSRF cookie and returns token to the Next.js client.
    """
    permission_classes = [AllowAny]

    @method_decorator(ensure_csrf_cookie)
    def get(self, request: Request) -> Response:
        csrf_token = get_token(request)
        return Response({"csrfToken": csrf_token}, status=status.HTTP_200_OK)


class CurrentUserView(APIView):
    """
    Returns currently logged-in user profile or 401 if unauthenticated.
    """
    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        if request.user.is_authenticated:
            return Response({
                "isAuthenticated": True,
                "user": {
                    "id": request.user.id,
                    "username": request.user.username,
                    "email": request.user.email,
                    "is_staff": request.user.is_staff or request.user.is_superuser,
                    "is_superuser": request.user.is_superuser,
                }
            }, status=status.HTTP_200_OK)
        return Response({"isAuthenticated": False, "user": None}, status=status.HTTP_200_OK)


class LoginView(APIView):
    """
    Authenticates user and initiates Django session.
    """
    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        username = request.data.get("username")
        password = request.data.get("password")

        if not username or not password:
            return Response(
                {"error": "Username and password are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return Response({
                "message": "Login successful.",
                "csrfToken": get_token(request),
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_staff": user.is_staff or user.is_superuser,
                    "is_superuser": user.is_superuser,
                }
            }, status=status.HTTP_200_OK)
        return Response(
            {"error": "Invalid username or password."},
            status=status.HTTP_401_UNAUTHORIZED
        )


class LogoutView(APIView):
    """
    Destroys active Django session.
    """
    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        logout(request)
        return Response({"message": "Logout successful."}, status=status.HTTP_200_OK)


class RegisterView(APIView):
    """
    Registers a new user account.
    """
    permission_classes = [AllowAny]

    def post(self, request: Request) -> Response:
        username = request.data.get("username", "").strip()
        email = request.data.get("email", "").strip()
        password = request.data.get("password", "")
        confirm_password = request.data.get("confirm_password", "")

        if not username or not password:
            return Response({"error": "Username and password are required."}, status=status.HTTP_400_BAD_REQUEST)

        if password != confirm_password:
            return Response({"error": "Passwords do not match."}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({"error": "Username already taken."}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        return Response({
            "message": "User registered successfully.",
            "csrfToken": get_token(request),
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_staff": user.is_staff,
                "is_superuser": user.is_superuser,
            }
        }, status=status.HTTP_201_CREATED)


class HealthCheckView(APIView):
    """
    Production health check endpoint verifying database connectivity.
    """
    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        from django.db import connection
        status_info = {"status": "healthy", "database": "unknown"}
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
            status_info["database"] = "connected"
            return Response(status_info, status=status.HTTP_200_OK)
        except Exception as exc:
            status_info["status"] = "unhealthy"
            status_info["database"] = str(exc)
            return Response(status_info, status=status.HTTP_503_SERVICE_UNAVAILABLE)


