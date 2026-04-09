"""Desktop launcher for the PCOS AI Clinical Support Platform."""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import time
from pathlib import Path
from shutil import copy2

import webview


HOST = "127.0.0.1"
DEFAULT_PORT = 8000
PORT_FALLBACK = 8001
MAX_PORT_ATTEMPTS = 25
STARTUP_TIMEOUT_SECONDS = 30
WINDOW_TITLE = "PCOS AI Clinical Support Platform"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
APP_FOLDER_NAME = "PCOSAIClinicalSupportPlatform"
WINDOW_APP_ID = "PCOSAIClinicalSupportPlatform.Desktop"


def main() -> int:
    """Start Django in the background and host it in a native window."""
    _set_windows_app_id()
    bundle_dir = _resolve_bundle_dir()
    runtime_dir = _resolve_runtime_dir()
    db_path = _prepare_database(bundle_dir, runtime_dir)
    _configure_environment(bundle_dir=bundle_dir, runtime_dir=runtime_dir, db_path=db_path)

    try:
        _initialize_django(bundle_dir)
    except Exception as exc:
        _show_error_dialog("PCOS AI Platform", f"Django initialization failed.\n\n{exc}")
        return 1

    try:
        port = _select_port(HOST, DEFAULT_PORT, PORT_FALLBACK)
    except RuntimeError as exc:
        _show_error_dialog("PCOS AI Platform", str(exc))
        return 1

    server_errors: list[BaseException] = []
    server_thread = threading.Thread(
        target=_run_server,
        args=(HOST, port, server_errors),
        name="pcos-django-server",
        daemon=True,
    )
    server_thread.start()

    if not _wait_for_server(HOST, port, STARTUP_TIMEOUT_SECONDS):
        error_message = (
            f"The local server did not start in time.\n\n{server_errors[0]}"
            if server_errors
            else "The local server did not start in time."
        )
        _show_error_dialog("PCOS AI Platform", error_message)
        return 1

    url = f"http://{HOST}:{port}"
    if os.getenv("PCOS_DISABLE_UI", "").strip() == "1":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    try:
        _open_desktop_window(url)
    except Exception as exc:
        _show_error_dialog(
            "PCOS AI Platform",
            f"Desktop window startup failed.\n\nOpen this URL manually if needed:\n{url}\n\n{exc}",
        )
        return 1

    return 0


def _resolve_bundle_dir() -> Path:
    """Return the directory containing packaged resources."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _set_windows_app_id() -> None:
    """Give Windows a stable application ID so the packaged app/icon behave consistently."""
    if sys.platform != "win32":
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOW_APP_ID)  # type: ignore[attr-defined]
    except Exception:
        pass


def _resolve_runtime_dir() -> Path:
    """Return a persistent writable directory for runtime data."""
    local_app_data = os.getenv("LOCALAPPDATA")
    base_dir = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    runtime_dir = base_dir / APP_FOLDER_NAME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _prepare_database(bundle_dir: Path, runtime_dir: Path) -> Path:
    """Copy the bundled SQLite database into a persistent location."""
    bundled_db = bundle_dir / "db.sqlite3"
    runtime_db = runtime_dir / "db.sqlite3"

    if bundled_db.exists() and not runtime_db.exists():
        copy2(bundled_db, runtime_db)
    elif not runtime_db.exists():
        runtime_db.touch()

    return runtime_db


def _configure_environment(bundle_dir: Path, runtime_dir: Path, db_path: Path) -> None:
    """Prepare environment variables before Django setup."""
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))

    os.chdir(bundle_dir)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pcos_project.settings")
    os.environ.setdefault("PCOS_RESOURCE_DIR", str(bundle_dir))
    os.environ.setdefault("PCOS_RUNTIME_DIR", str(runtime_dir))
    os.environ.setdefault("PCOS_LOG_DIR", str(runtime_dir / "logs"))
    os.environ.setdefault("PCOS_MODELS_DIR", str(bundle_dir / "models"))
    os.environ.setdefault("PCOS_DB_PATH", str(db_path))
    os.environ.setdefault("PCOS_STATIC_ROOT", str(runtime_dir / "staticfiles"))

    dataset_path = bundle_dir / "dataset" / "data1.csv"
    if dataset_path.exists():
        os.environ.setdefault("PCOS_DATASET_PATH", str(dataset_path))


def _initialize_django(bundle_dir: Path) -> None:
    """Initialize Django only after environment variables are ready."""
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))

    import django

    django.setup()


def _select_port(host: str, primary_port: int, fallback_port: int) -> int:
    """Use 8000 by default, 8001 first fallback, then continue scanning locally."""
    candidate_ports = [primary_port, fallback_port]
    candidate_ports.extend(
        port for port in range(fallback_port + 1, fallback_port + MAX_PORT_ATTEMPTS) if port not in candidate_ports
    )

    for port in candidate_ports:
        if _is_port_available(host, port):
            return port

    raise RuntimeError(
        f"No free local port was found starting at {primary_port}. "
        f"Tried: {', '.join(str(port) for port in candidate_ports)}."
    )


def _is_port_available(host: str, port: int) -> bool:
    """Return True when the host:port combination is currently unused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) != 0


def _run_server(host: str, port: int, error_bucket: list[BaseException]) -> None:
    """Run the Django development server in a background thread."""
    try:
        from django.core.management import execute_from_command_line

        execute_from_command_line(
            [
                "manage.py",
                "runserver",
                f"{host}:{port}",
                "--noreload",
            ]
        )
    except BaseException as exc:  # pragma: no cover - startup failures are user-facing
        error_bucket.append(exc)


def _wait_for_server(host: str, port: int, timeout_seconds: int) -> bool:
    """Wait until the server socket accepts local connections."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


def _open_desktop_window(url: str) -> None:
    """Open the Django UI inside a native pywebview window."""
    window = webview.create_window(
        WINDOW_TITLE,
        url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        resizable=True,
    )
    webview.start(debug=False)
    if window is None:
        raise RuntimeError("pywebview did not create the application window.")


def _show_error_dialog(title: str, message: str) -> None:
    """Display a Windows-native error dialog when no console is available."""
    try:
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)  # type: ignore[attr-defined]
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
