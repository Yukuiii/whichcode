"""Shared filesystem paths for user-level whichcode data."""

from __future__ import annotations

import hashlib
from pathlib import Path

WHICHCODE_HOME_DIR_NAME = ".whichcode"
INDEX_CHUNK_DIR_NAME = "chunk"
MODEL_DIR_NAME = "models"


def whichcode_home() -> Path:
    """Return the user-level whichcode data directory."""
    return Path.home() / WHICHCODE_HOME_DIR_NAME


def project_index_key(root: str | Path) -> str:
    """Return the sha256 key for a resolved project root path."""
    root_path = Path(root).expanduser().resolve()
    return hashlib.sha256(str(root_path).encode("utf-8")).hexdigest()


def project_index_dir(root: str | Path) -> Path:
    """Return the global chunk index directory for a project root."""
    return whichcode_home() / INDEX_CHUNK_DIR_NAME / project_index_key(root)


def model_root_dir() -> Path:
    """Return the global directory for local model files."""
    return whichcode_home() / MODEL_DIR_NAME
