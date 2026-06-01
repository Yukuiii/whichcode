"""Tests for project tree scanning."""

from pathlib import Path

import pytest

from whichcode.scanner import scan_chunks


def test_scan_chunks_reads_supported_files_and_skips_blank_or_ignored_files(tmp_path: Path) -> None:
    """scan_chunks should scan supported files and skip blank or ignored files."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (src / "app.js").write_text("function start() { return 1; }\n", encoding="utf-8")
    (src / "README.md").write_text("# Notes\n\nUse this project.\n", encoding="utf-8")
    (src / "blank.py").write_text(" \n\t\n", encoding="utf-8")
    (src / "notes.txt").write_text("def ignored():\n    return 0\n", encoding="utf-8")

    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "hidden.py").write_text("def hidden():\n    return 2\n", encoding="utf-8")

    chunks = scan_chunks(tmp_path)

    assert [chunk.file_path for chunk in chunks] == ["src/app.js", "src/app.py"]
    assert chunks[0].kind == "function"
    assert chunks[0].name == "start"
    assert chunks[1].kind == "function"
    assert chunks[1].name == "run"


def test_scan_chunks_raises_for_missing_path(tmp_path: Path) -> None:
    """scan_chunks should fail clearly when the root path does not exist."""
    with pytest.raises(FileNotFoundError):
        scan_chunks(tmp_path / "missing")


def test_scan_chunks_raises_for_file_path(tmp_path: Path) -> None:
    """scan_chunks should require a directory root."""
    file_path = tmp_path / "app.py"
    file_path.write_text("def run():\n    return 1\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_chunks(file_path)
