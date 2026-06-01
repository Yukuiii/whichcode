"""BM25 indexing helpers for ranked chunk search."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s

from whichcode.chunking import Chunk
from whichcode.types import SearchResult

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


@dataclass(frozen=True, slots=True)
class BM25Index:
    """Wraps a BM25 index and the chunks it ranks."""

    chunks: tuple[Chunk, ...]
    _index: bm25s.BM25

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> BM25Index:
        """Build a BM25 index from indexable chunks."""
        resolved_chunks = tuple(chunks)
        index = bm25s.BM25()
        if resolved_chunks:
            index.index([tokenize(enrich_for_bm25(chunk)) for chunk in resolved_chunks], show_progress=False)
        return cls(chunks=resolved_chunks, _index=index)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked by BM25 score for a query."""
        if top_k < 1 or not self.chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)[:top_k]
        return [SearchResult(chunk=self.chunks[index], score=float(scores[index])) for index in indices if scores[index] > 0]


def build_bm25_index(chunks: Sequence[Chunk]) -> BM25Index:
    """Build a searchable BM25 index from chunks."""
    return BM25Index.from_chunks(chunks)


def enrich_for_bm25(chunk: Chunk) -> str:
    """Add chunk metadata to content so path and symbol queries can match."""
    path = Path(chunk.file_path)
    stem = path.stem
    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    dir_text = " ".join(dir_parts[-3:])
    metadata = " ".join(part for part in (chunk.kind, chunk.name, chunk.language) if part)
    return f"{chunk.content} {stem} {stem} {dir_text} {metadata}"


def tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier-like tokens for BM25 indexing."""
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text):
        tokens.extend(split_identifier(token))
    return tokens


def split_identifier(token: str) -> list[str]:
    """Split a compound identifier while preserving the lowered original."""
    lower = token.lower()
    if "_" in token:
        parts = [part for part in lower.split("_") if part]
    else:
        parts = [match.lower() for match in _CAMEL_RE.findall(token)]
    if len(parts) >= 2:
        return [lower, *parts]
    return [lower]
