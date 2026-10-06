"""
cyber_physical_banking/urls.py
==============================
Root URL configuration.

Route map:
  /           → app (existing Django-template frontend)
  /api/*      → api (DRF REST API — v2)
  /admin/     → Django admin
  /media/*    → served by Django in DEBUG mode only (Nginx handles it in prod)
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/",  admin.site.urls),

    # ── REST API (v2) ─────────────────────────────────────────────────────────
    path("api/",    include("api.urls", namespace="api")),

    # ── Existing Django-template frontend ─────────────────────────────────────
    path("",        include("app.urls")),
]

# Media files — DEBUG only.
# In production Nginx serves /media/ directly (see nginx.conf).
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
