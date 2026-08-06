"""Django settings for the price-model explorer (webui/).

Local, single-user data-viz tool: no auth, no database, no admin. Reads trained model
bundles from ``outputs/`` and sample parquets from ``inputs/`` at the repo root (one
level above this ``webui/`` folder), via the ``price_model`` package.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # webui/
REPO_ROOT = BASE_DIR.parent                                 # repo root (holds price_model/)
sys.path.insert(0, str(REPO_ROOT))                          # make `price_model` importable

# No auth/DB in this app, so the key only guards Django's CSRF/session machinery, which
# isn't in use either -- fine to keep a fixed placeholder rather than a managed secret.
SECRET_KEY = "django-insecure-price-model-webui-local-only"

DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "pricing",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "priceui.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "priceui.wsgi.application"

# No models/ORM usage anywhere in this app.
DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
