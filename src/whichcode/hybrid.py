"""Hybrid retrieval that combines BM25 and vector search results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from whichcode.bm25 import BM25Index, build_bm25_index, extract_search_terms, tokenize
from whichcode.chunking import Chunk
from whichcode.types import SearchResult
from whichcode.vector import EmbeddingModel, VectorIndex, build_vector_index

_RRF_K = 60
_STRONG_PATH_PENALTY = 0.3
_MODERATE_PATH_PENALTY = 0.5
_MILD_PATH_PENALTY = 0.7
_TINY_QUERY_BOOST = 1e-12
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:test_[^/]*\.\w+|[^/]*_test\.\w+|[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?)$"
)
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")
_LOW_SIGNAL_DIR_RE = re.compile(
    r"(?:^|/)(?:_?examples?|benchmarks?|docs?|docs?_src|agents?|compat|_compat|legacy)(?:/|$)"
)
_MARKDOWN_RE = re.compile(r"\.(?:md|markdown)$", re.IGNORECASE)
_REEXPORT_FILENAMES = frozenset({"__init__.py", "package-info.java"})
_LOW_SIGNAL_FILENAMES = frozenset({"readme.md", "changelog.md", "contributing.md"})
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
        query_terms = extract_search_terms(query) if alpha < 1.0 else []
        combined = [
            SearchResult(
                chunk=chunk,
                score=_apply_path_penalty(
                    chunk,
                    _apply_query_relevance(
                        chunk,
                        alpha * vector_scores.get(chunk, 0.0) + (1.0 - alpha) * bm25_scores.get(chunk, 0.0),
                        query_terms,
                    ),
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


def _apply_query_relevance(chunk: Chunk, score: float, query_terms: Sequence[str]) -> float:
    """Boost chunks whose metadata or content matches multiple query concepts."""
    if score <= 0.0 or not query_terms:
        return score

    metadata_tokens = set(_chunk_metadata_tokens(chunk))
    content_tokens = set(tokenize(chunk.content))
    name_tokens = set(tokenize(chunk.name or ""))
    metadata_hits = _count_term_hits(query_terms, metadata_tokens)
    content_hits = _count_term_hits(query_terms, content_tokens)
    name_hits = _count_term_hits(query_terms, name_tokens)
    total_hits = len(
        {
            term
            for term in query_terms
            if term in metadata_tokens or term in content_tokens
        }
    )

    factor = 1.0
    if name_hits:
        factor += min(name_hits, 3) * 0.25
    if metadata_hits:
        factor += min(metadata_hits, 4) * 0.15
    if total_hits >= 2:
        factor += min(total_hits, 5) * 0.08
    elif len(query_terms) >= 2 and content_hits == 0:
        factor *= 0.85

    return max(score * factor, _TINY_QUERY_BOOST)


def _chunk_metadata_tokens(chunk: Chunk) -> list[str]:
    """Return searchable tokens from stable chunk metadata."""
    path = Path(chunk.file_path)
    parts = [
        chunk.file_path,
        path.stem,
        path.parent.as_posix(),
        chunk.kind,
        chunk.name or "",
        chunk.language or "",
    ]
    return tokenize(" ".join(parts))


def _count_term_hits(query_terms: Sequence[str], candidate_terms: set[str]) -> int:
    """Count query terms that appear in a candidate term set."""
    return sum(1 for term in query_terms if term in candidate_terms)


def _path_penalty(file_path: str) -> float:
    """Return a multiplicative penalty for lower-signal file paths."""
    normalized = file_path.replace("\\", "/")
    filename = Path(file_path).name.lower()
    penalty = 1.0
    if _TEST_FILE_RE.search(normalized) is not None or _TEST_DIR_RE.search(normalized) is not None:
        penalty *= _STRONG_PATH_PENALTY
    if _LOW_SIGNAL_DIR_RE.search(normalized) is not None:
        penalty *= _STRONG_PATH_PENALTY
    if filename in _LOW_SIGNAL_FILENAMES:
        penalty *= _MODERATE_PATH_PENALTY
    if _MARKDOWN_RE.search(normalized) is not None:
        penalty *= _MILD_PATH_PENALTY
    if Path(file_path).name in _REEXPORT_FILENAMES:
        penalty *= _MODERATE_PATH_PENALTY
    if _TYPE_DEFS_RE.search(normalized) is not None:
        penalty *= _MILD_PATH_PENALTY
    return penalty
