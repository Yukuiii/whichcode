"""Tests for search result formatting."""

from whichcode.chunking import Chunk
from whichcode.formatting import format_results
from whichcode.types import SearchResult


def test_format_results_includes_chunk_content_and_metadata() -> None:
    """format_results should include the full chunk payload for display."""
    chunk = Chunk(
        content="def run():\n    return 1\n",
        file_path="src/app.py",
        start_line=10,
        end_line=11,
        kind="function",
        name="run",
        language="python",
    )

    formatted = format_results("run function", [SearchResult(chunk=chunk, score=0.5)])

    assert formatted["query"] == "run function"
    assert formatted["results"][0]["score"] == 0.5
    assert formatted["results"][0]["chunk"] == {
        "content": "def run():\n    return 1\n",
        "file_path": "src/app.py",
        "start_line": 10,
        "end_line": 11,
        "kind": "function",
        "name": "run",
        "language": "python",
        "location": "src/app.py:10-11",
    }


def test_format_results_handles_empty_results() -> None:
    """format_results should preserve the query when no results are present."""
    assert format_results("missing", []) == {"query": "missing", "results": []}
