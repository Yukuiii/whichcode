"""Hybrid retrieval that combines BM25 and vector search results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from whichcode.bm25 import BM25Index, build_bm25_index
from whichcode.chunking import Chunk
from whichcode.types import SearchResult
from whichcode.vector import EmbeddingModel, VectorIndex, build_vector_index

_RRF_K = 60


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

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[SearchResult]:
        """Return hybrid-ranked chunks for a query."""
        if top_k < 1 or not query.strip():
            return []
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        candidate_count = top_k * 5
        bm25_scores = _rrf_scores(self.bm25.search(query, top_k=candidate_count))
        vector_scores = _rrf_scores(self.vector.search(query, top_k=candidate_count))
        candidates = sorted({*bm25_scores, *vector_scores}, key=lambda chunk: (chunk.file_path, chunk.start_line))
        combined = [
            SearchResult(
                chunk=chunk,
                score=alpha * vector_scores.get(chunk, 0.0) + (1.0 - alpha) * bm25_scores.get(chunk, 0.0),
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
