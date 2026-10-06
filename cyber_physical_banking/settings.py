"""
CyberPhysicalBanking v2 — Django Settings
==========================================
Environment variables are loaded from a .env file at project root
(via python-dotenv). In production/Docker, set them directly in the
container environment — no .env file needed.

Security contract: ZERO hardcoded secrets. Every sensitive value
comes from os.environ.get() with a safe, non-functional default.
"""
import os
from pathlib import Path

# ── Load .env file (development convenience) ──────────────────
# In Docker/production, env vars are injected directly.
# dotenv silently skips if .env is absent — safe to keep always.
from dotenv import load_dotenv
load_dotenv(override=True)  # .env takes precedence over ambient shell variables

# ── Base Directory ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security Settings ─────────────────────────────────────────
# SECURITY: Django will refuse to start in production if SECRET_KEY
# matches the placeholder. Set a real 50+ char key in your .env.
SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'INSECURE-placeholder-replace-before-any-deployment-!!!!'
)

DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 'yes')

# In local development with DEBUG=True, allow all hosts or explicitly listed hosts
_allowed_env = os.environ.get('ALLOWED_HOSTS')
if _allowed_env:
    ALLOWED_HOSTS = [h.strip() for h in _allowed_env.split(',') if h.strip()]
    if '127.0.0.1' not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append('127.0.0.1')
    if 'localhost' not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append('localhost')
    if 'testserver' not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append('testserver')
else:
    ALLOWED_HOSTS = ['*'] if DEBUG else ['localhost', '127.0.0.1', 'testserver']


# ── CORS & Security Headers ───────────────────────────────────
# In Django 4.0+, POST requests validate the Origin header against CSRF_TRUSTED_ORIGINS.
# Ports must be explicitly included when running on non-standard ports (e.g. :8000, :3000).
_raw_csrf_origins = os.environ.get(
    'CSRF_TRUSTED_ORIGINS',
    'http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1,http://localhost'
)
CSRF_TRUSTED_ORIGINS = [o.strip() for o in _raw_csrf_origins.split(',') if o.strip()]

# Guaranteed local frontend and backend origins for development
for _origin in [
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
]:
    if _origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(_origin)

SECURE_SSL_REDIRECT = os.environ.get('SECURE_SSL_REDIRECT', 'False') == 'True'
SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False') == 'True'
CSRF_COOKIE_SECURE = os.environ.get('CSRF_COOKIE_SECURE', 'False') == 'True'
# Recommended extra headers for production
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'


# ── Application Definition ────────────────────────────────────
INSTALLED_APPS = [
    'corsheaders',             # CORS headers for Next.js / frontend clients
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third-party
    'rest_framework',          # Django REST Framework (Phase 3)
    # Local apps
    'app',
    'api',                     # REST API layer (Phase 3)
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',        # Must be as high as possible
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # static files in production
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


ROOT_URLCONF = 'cyber_physical_banking.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'app', 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'cyber_physical_banking.wsgi.application'

# ── Database ──────────────────────────────────────────────────
# Strategy:
#   • If DATABASE_URL env var is set  → use PostgreSQL (production/staging)
#   • Otherwise                       → fall back to SQLite (local dev only)
#
# dj_database_url.config() parses the URL and returns a Django DATABASES dict.
# conn_max_age=600 enables persistent connections (recommended for Gunicorn).
import dj_database_url

_database_url = os.environ.get('DATABASE_URL')

if _database_url:
    DATABASES = {
        'default': dj_database_url.config(
            default=_database_url,
            conn_max_age=600,         # keep connections alive for 10 min
            conn_health_checks=True,  # discard stale connections automatically
        )
    }
else:
    # Local development fallback — SQLite
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# ── Password Validation ───────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ── Internationalization ──────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ── Static & Media Files ──────────────────────────────────────
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'app', 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')  # for collectstatic
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_URL = '/media/'
# If MEDIA_ROOT in env is an absolute Linux path like /app/media, fallback to local BASE_DIR / media on Windows/local dev
_env_media = os.environ.get('MEDIA_ROOT')
if _env_media and os.path.isabs(_env_media) and not os.path.exists(os.path.splitdrive(_env_media)[0] or '/'):
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
elif _env_media and not _env_media.startswith('/app'):
    MEDIA_ROOT = _env_media
else:
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Ensure media and dataset upload directories exist
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'datasets'), exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'reports'), exist_ok=True)

# ── Default Primary Key ───────────────────────────────────────

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── Authentication ────────────────────────────────────────────
LOGIN_URL = 'login'

# ── Django REST Framework ─────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        # Require login for all API endpoints by default.
        # Override per-view with AllowAny where needed (e.g. demo endpoints).
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',  # for file uploads
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '30/minute',
        'user': '120/minute',
    },
}

# ── LLM / RAG Configuration (Phase 4 & 5) ────────────────────
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
OLLAMA_LLM_MODEL = os.environ.get('OLLAMA_LLM_MODEL', os.environ.get('OLLAMA_MODEL', 'llama3'))
OLLAMA_MODEL = OLLAMA_LLM_MODEL  # backward compatibility
OLLAMA_EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
CHROMA_PERSIST_DIRECTORY = os.environ.get('CHROMA_PERSIST_DIRECTORY', 'chroma_db')
CHROMA_HOST = os.environ.get('CHROMA_HOST', '').strip() or None
CHROMA_PORT = int(os.environ.get('CHROMA_PORT', '8000')) if os.environ.get('CHROMA_PORT') else 8000

# ── CORS Settings (Next.js & Frontend Clients) ───────────────
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
CORS_ALLOW_CREDENTIALS = True

# ── CSRF & Session Cookie Settings for Production & Next.js ───
CSRF_COOKIE_HTTPONLY = False 
SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'None' if not DEBUG else 'Lax')
SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'True' if not DEBUG else 'False') == 'True'
CSRF_COOKIE_SAMESITE = os.environ.get('CSRF_COOKIE_SAMESITE', 'None' if not DEBUG else 'Lax')
CSRF_COOKIE_SECURE = os.environ.get('CSRF_COOKIE_SECURE', 'True' if not DEBUG else 'False') == 'True'

