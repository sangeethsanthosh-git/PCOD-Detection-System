"""Django settings for the PCOS AI Clinical Support Platform."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = Path(os.getenv("PCOS_RESOURCE_DIR", BASE_DIR))
DATABASE_PATH = Path(os.getenv("PCOS_DB_PATH", BASE_DIR / "db.sqlite3"))
STATIC_ROOT_PATH = Path(os.getenv("PCOS_STATIC_ROOT", BASE_DIR / "staticfiles"))

SECRET_KEY = "pcos-ai-clinical-support-platform-dev-key"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "*"]


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "pcos_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [RESOURCE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "pcos_project.wsgi.application"


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATABASE_PATH,
    }
}


AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [RESOURCE_DIR / "static"]
STATIC_ROOT = STATIC_ROOT_PATH

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "pcos-ai-platform-cache",
        "TIMEOUT": 60 * 60,
    }
}

SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
