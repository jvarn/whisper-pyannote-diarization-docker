"""
Configure logging to file and console for debugging pipeline issues.
Logs are written to storage/logs/backend.log so they persist across process restarts.
"""
import logging
import os
import sys
from datetime import datetime

from .config import settings

LOG_DIR = os.path.join(settings.STORAGE_DIR, "logs")
LOG_FILE = (
    os.path.abspath(settings.LOG_FILE)
    if settings.LOG_FILE
    else os.path.abspath(os.path.join(LOG_DIR, "backend.log"))
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application. Safe to call multiple times."""
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    # Uvicorn adds handlers to root before we run, so we must add our file handler
    # even when root.handlers is non-empty. Avoid duplicates by checking for our file.
    has_file_handler = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "").replace("\\", "/").endswith("backend.log")
        for h in root.handlers
    )
    if not has_file_handler:
        formatter = logging.Formatter(
            "%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        # Bootstrap: write directly to verify path; print so user sees it
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} [logging configured] Log file: {LOG_FILE}\n")
        except OSError as e:
            print(f"WARNING: Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)
        else:
            print(f"Logging to: {LOG_FILE}", file=sys.stderr)
