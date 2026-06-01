"""Shared result types for chunk retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from whichcode.chunking import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Represents one ranked chunk returned by a retrieval query."""

    chunk: Chunk
    score: float
