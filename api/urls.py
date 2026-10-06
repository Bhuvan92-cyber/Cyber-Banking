"""
api/urls.py
===========
All /api/* routes are registered here.
This module is included into the root urls.py under the 'api/' prefix.
"""
from django.urls import path
from .views import (
    CSRFTokenView,
    ChatView,
    CurrentUserView,
    DatasetUploadView,
    HealthCheckView,
    LoginView,
    LogoutView,
    PredictView,
    RegisterView,
    ReportView,
)

app_name = "api"

urlpatterns = [
    # ── Health Check (Production Pre-flight) ──────────────────────────────────
    path("health/", HealthCheckView.as_view(), name="health"),

    # ── Authentication & CSRF (Next.js Client) ────────────────────────────────
    path("csrf/", CSRFTokenView.as_view(), name="csrf"),
    path("auth/me/", CurrentUserView.as_view(), name="auth-me"),
    path("auth/login/", LoginView.as_view(), name="auth-login"),
    path("auth/logout/", LogoutView.as_view(), name="auth-logout"),
    path("auth/register/", RegisterView.as_view(), name="auth-register"),

    # ── Core ML API ──────────────────────────────────────────────────────────
    # POST  /api/upload/   → upload CSV, get dataset summary
    path("upload/",  DatasetUploadView.as_view(), name="upload"),

    # POST  /api/predict/  → batch training OR single-record inference
    path("predict/", PredictView.as_view(),       name="predict"),

    # GET   /api/report/?dataset_name=<filename>  → download PDF report
    path("report/",  ReportView.as_view(),         name="report"),

    # ── RAG / LLM Chat (Phase 4) ──────────────────────────────────────────────
    # POST  /api/chat/     → natural-language dataset query
    path("chat/",    ChatView.as_view(),           name="chat"),
]
