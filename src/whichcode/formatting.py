"""Formatting helpers for retrieval results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from whichcode.chunking import Chunk
from whichcode.types import SearchResult


def format_results(query: str, results: Sequence[SearchResult]) -> dict[str, Any]:
    """Render search results as a JSON-serializable object."""
    return {"query": query, "results": [search_result_to_dict(result) for result in results]}


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    """Convert one search result to a JSON-serializable dictionary."""
    return {
        "score": result.score,
        "chunk": chunk_to_dict(result.chunk),
    }


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Convert one chunk to a JSON-serializable dictionary."""
    return {
        "content": chunk.content,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "kind": chunk.kind,
        "name": chunk.name,
        "language": chunk.language,
        "summary": chunk.summary,
        "location": chunk.location,
    }
