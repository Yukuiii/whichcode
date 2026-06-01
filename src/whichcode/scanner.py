"""Filesystem scanning helpers for chunk extraction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from whichcode.chunking import SUPPORTED_SUFFIXES, Chunk, chunk_source

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".whichcode",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
    "build",
    "dist",
    "case",
}
DEFAULT_SUFFIXES = SUPPORTED_SUFFIXES
DEFAULT_MAX_FILE_BYTES = 1_000_000


def scan_chunks(root: str | Path, suffixes: Sequence[str] | None = None) -> list[Chunk]:
    """Scan a project tree and return chunks for supported source files."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    resolved_suffixes = {suffix.lower() for suffix in (suffixes or DEFAULT_SUFFIXES)}
    chunks: list[Chunk] = []
    for file_path in _walk_files(root_path, resolved_suffixes):
        try:
            if file_path.stat().st_size > DEFAULT_MAX_FILE_BYTES:
                continue
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        if not source.strip():
            continue

        chunks.extend(chunk_source(source, file_path.relative_to(root_path).as_posix()))

    return chunks


def _walk_files(root: Path, suffixes: set[str]) -> Iterator[Path]:
    """Yield supported files under a root directory in deterministic order."""
    for item in sorted(root.iterdir(), key=lambda path: path.name):
        if item.is_symlink():
            continue
        if item.is_dir():
            if item.name in DEFAULT_IGNORED_DIRS:
                continue
            yield from _walk_files(item, suffixes)
        elif item.is_file() and item.suffix.lower() in suffixes:
            yield item
