"""Utilities for locating packaged resources in development and PyInstaller builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resource_base_path() -> Path:
    """Return the directory that holds packaged or source resources."""
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))

    configured_dir = os.getenv("PCOS_RESOURCE_DIR", "").strip()
    if configured_dir:
        return Path(configured_dir)

    return PROJECT_ROOT


def resource_path(relative_path: str) -> str:
    """Return an absolute path for bundled or local project resources."""
    return os.path.join(str(_resource_base_path()), relative_path)


def resource_path_obj(relative_path: str) -> Path:
    """Return a Path object for bundled or local project resources."""
    return Path(resource_path(relative_path))


def log_path_obj(relative_path: str = "prediction_errors.log") -> Path:
    """Return a writable log path for packaged and development runs."""
    configured_dir = os.getenv("PCOS_LOG_DIR", "").strip()
    base_dir = Path(configured_dir) if configured_dir else PROJECT_ROOT / "logs"
    path = base_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
