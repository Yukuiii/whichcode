"""Tests for the minimal tree-sitter chunker."""

from whichcode.chunking import Chunk, chunk_source


def test_chunk_source_skips_blank_files() -> None:
    """chunk_source should skip files that contain only whitespace."""
    assert chunk_source(" \n\t\n", "blank.py") == []


def test_chunk_source_splits_python_functions_and_preserves_context() -> None:
    """chunk_source should emit method chunks and keep surrounding module context."""
    source = (
        "import os\n"
        "\n"
        "VALUE = 1\n"
        "\n"
        "@decorator\n"
        "def top(x):\n"
        "    return x\n"
        "\n"
        "class Box:\n"
        "    \"\"\"Docstring.\"\"\"\n"
        "\n"
        "    def method(self, y):\n"
        "        return y + VALUE\n"
    )

    chunks = chunk_source(source, "sample.py")

    assert [chunk.kind for chunk in chunks] == ["module", "function", "module", "method"]
    assert chunks[1].name == "top"
    assert chunks[3].name == "Box.method"
    assert "import os" in chunks[0].content
    assert "class Box" in chunks[2].content
    assert chunks[1].start_line == 5
    assert chunks[1].end_line == 7
    assert chunks[3].start_line == 12
    assert chunks[3].end_line == 13


def test_chunk_source_falls_back_to_one_file_chunk_without_functions() -> None:
    """chunk_source should return a single file chunk when no function is present."""
    chunks = chunk_source("VALUE = 1\n", "config.py")

    assert chunks == [Chunk(content="VALUE = 1\n", file_path="config.py", start_line=1, end_line=1)]


def test_chunk_source_falls_back_for_unsupported_language() -> None:
    """chunk_source should keep the whole file when tree-sitter is not enabled."""
    chunks = chunk_source("fn main() {}\n", "main.rs")

    assert chunks == [Chunk(content="fn main() {}\n", file_path="main.rs", start_line=1, end_line=1)]
