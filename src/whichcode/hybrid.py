"""Hybrid retrieval that combines BM25 and vector search results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from whichcode.bm25 import BM25Index, build_bm25_index
from whichcode.chunking import Chunk
from whichcode.types import SearchResult
from whichcode.vector import EmbeddingModel, VectorIndex, build_vector_index

_RRF_K = 60
_STRONG_PATH_PENALTY = 0.3
_MODERATE_PATH_PENALTY = 0.5
_MILD_PATH_PENALTY = 0.7
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:test_[^/]*\.\w+|[^/]*_test\.\w+|[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?)$"
)
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")
_LOW_SIGNAL_DIR_RE = re.compile(r"(?:^|/)(?:_?examples?|benchmarks?|docs?_src|compat|_compat|legacy)(?:/|$)")
_REEXPORT_FILENAMES = frozenset({"__init__.py", "package-info.java"})
_TYPE_DEFS_RE = re.compile(r"\.d\.ts$")


@dataclass(frozen=True, slots=True)
class HybridIndex:
    """Combines lexical and semantic indexes over the same chunk collection."""

    bm25: BM25Index
    vector: VectorIndex

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], model: EmbeddingModel | None = None) -> HybridIndex:
        """Build BM25 and vector indexes from chunks."""
        resolved_chunks = tuple(chunks)
        return cls(
            bm25=build_bm25_index(resolved_chunks),
            vector=build_vector_index(resolved_chunks, model=model),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        penalize_paths: bool = True,
    ) -> list[SearchResult]:
        """Return hybrid-ranked chunks for a query."""
        if top_k < 1 or not query.strip():
            return []
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        candidate_count = max(top_k * 5, 25)
        bm25_scores = _rrf_scores(self.bm25.search(query, top_k=candidate_count))
        vector_scores = _rrf_scores(self.vector.search(query, top_k=candidate_count))
        candidates = sorted({*bm25_scores, *vector_scores}, key=lambda chunk: (chunk.file_path, chunk.start_line))
        combined = [
            SearchResult(
                chunk=chunk,
                score=_apply_path_penalty(
                    chunk,
                    alpha * vector_scores.get(chunk, 0.0) + (1.0 - alpha) * bm25_scores.get(chunk, 0.0),
                    penalize_paths,
                ),
            )
            for chunk in candidates
        ]
        combined.sort(key=lambda result: result.score, reverse=True)
        return [result for result in combined[:top_k] if result.score > 0.0]


def build_hybrid_index(chunks: Sequence[Chunk], model: EmbeddingModel | None = None) -> HybridIndex:
    """Build a searchable hybrid index from chunks."""
    return HybridIndex.from_chunks(chunks, model=model)


def _rrf_scores(results: Sequence[SearchResult]) -> dict[Chunk, float]:
    """Convert ranked search results to reciprocal-rank-fusion scores."""
    return {result.chunk: 1.0 / (_RRF_K + rank) for rank, result in enumerate(results, 1)}


def _apply_path_penalty(chunk: Chunk, score: float, penalize_paths: bool) -> float:
    """Apply path-based score penalties when enabled."""
    if not penalize_paths:
        return score
    return score * _path_penalty(chunk.file_path)


def _path_penalty(file_path: str) -> float:
    """Return a multiplicative penalty for lower-signal file paths."""
    normalized = file_path.replace("\\", "/")
    penalty = 1.0
    if _TEST_FILE_RE.search(normalized) is not None or _TEST_DIR_RE.search(normalized) is not None:
        penalty *= _STRONG_PATH_PENALTY
    if _LOW_SIGNAL_DIR_RE.search(normalized) is not None:
        penalty *= _STRONG_PATH_PENALTY
    if Path(file_path).name in _REEXPORT_FILENAMES:
        penalty *= _MODERATE_PATH_PENALTY
    if _TYPE_DEFS_RE.search(normalized) is not None:
        penalty *= _MILD_PATH_PENALTY
    return penalty
