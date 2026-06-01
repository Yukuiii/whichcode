"""Hybrid retrieval that combines BM25 and vector search results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from whichcode.bm25 import BM25Index, build_bm25_index
from whichcode.chunking import Chunk
from whichcode.ranking_rules import (
    ParsedQuery,
    apply_path_prior,
    apply_relevance_rules,
    chunk_matches_query_filters,
    parse_query,
)
from whichcode.types import SearchResult
from whichcode.vector import EmbeddingModel, VectorIndex, build_vector_index

_RRF_K = 60
_HYBRID_RECALL_CANDIDATES = 100
_FILE_COHERENCE_BOOST_FRACTION = 0.15
_FILE_SATURATION_THRESHOLD = 1
_FILE_SATURATION_DECAY = 0.65
_TINY_QUERY_BOOST = 1e-12
_SYMBOL_QUERY_RE = re.compile(
    r"^(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\|->|\.)[A-Za-z_][A-Za-z0-9_]*)+"
    r"|_[A-Za-z0-9_]*"
    r"|[A-Za-z][A-Za-z0-9]*[A-Z_][A-Za-z0-9_]*"
    r"|[A-Z][A-Za-z0-9]*"
    r")$"
)
_ALPHA_SYMBOL = 0.3
_ALPHA_NATURAL_LANGUAGE = 0.5


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
        alpha: float | None = None,
        penalize_paths: bool = True,
    ) -> list[SearchResult]:
        """Return hybrid-ranked chunks for a query."""
        if top_k < 1 or not query.strip():
            return []

        parsed_query = parse_query(query)
        search_query = parsed_query.search_text
        alpha_weight = _resolve_alpha(search_query, alpha)
        if not 0.0 <= alpha_weight <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        candidate_count = max(top_k * 10, _HYBRID_RECALL_CANDIDATES)
        bm25_scores = _rrf_scores(self.bm25.search(search_query, top_k=candidate_count))
        vector_scores = _rrf_scores(self.vector.search(search_query, top_k=candidate_count))
        candidates = [
            chunk
            for chunk in sorted({*bm25_scores, *vector_scores}, key=lambda chunk: (chunk.file_path, chunk.start_line))
            if chunk_matches_query_filters(chunk, parsed_query)
        ]
        combined_scores = {
            chunk: _apply_query_relevance(
                chunk,
                alpha_weight * vector_scores.get(chunk, 0.0) + (1.0 - alpha_weight) * bm25_scores.get(chunk, 0.0),
                parsed_query,
                alpha_weight,
            )
            for chunk in candidates
        }
        _boost_file_coherence(combined_scores)
        combined = [
            SearchResult(
                chunk=chunk,
                score=_apply_path_penalty(chunk, combined_scores[chunk], penalize_paths, search_query),
            )
            for chunk in candidates
        ]
        combined.sort(key=lambda result: result.score, reverse=True)
        ranked = [result for result in combined if result.score > 0.0]
        return _select_with_file_saturation(ranked, top_k)


def build_hybrid_index(chunks: Sequence[Chunk], model: EmbeddingModel | None = None) -> HybridIndex:
    """Build a searchable hybrid index from chunks."""
    return HybridIndex.from_chunks(chunks, model=model)


def _rrf_scores(results: Sequence[SearchResult]) -> dict[Chunk, float]:
    """Convert ranked search results to reciprocal-rank-fusion scores."""
    return {result.chunk: 1.0 / (_RRF_K + rank) for rank, result in enumerate(results, 1)}


def _resolve_alpha(query: str, alpha: float | None) -> float:
    """Return semantic-vs-lexical blend weight for the query."""
    if alpha is not None:
        return alpha
    return _ALPHA_SYMBOL if _is_symbol_query(query) else _ALPHA_NATURAL_LANGUAGE


def _is_symbol_query(query: str) -> bool:
    """Return whether the query looks like a bare code symbol."""
    return _SYMBOL_QUERY_RE.match(query.strip()) is not None


def _apply_path_penalty(chunk: Chunk, score: float, penalize_paths: bool, query: str) -> float:
    """Apply path-based score penalties when enabled."""
    if not penalize_paths:
        return score
    return apply_path_prior(chunk.file_path, score, query)


def _boost_file_coherence(scores: dict[Chunk, float]) -> None:
    """Lightly promote the best chunk from files with multiple relevant candidates."""
    if not scores:
        return
    max_score = max(scores.values())
    if max_score <= 0.0:
        return

    file_totals: dict[str, float] = {}
    best_by_file: dict[str, Chunk] = {}
    for chunk, score in scores.items():
        file_totals[chunk.file_path] = file_totals.get(chunk.file_path, 0.0) + score
        if chunk.file_path not in best_by_file or score > scores[best_by_file[chunk.file_path]]:
            best_by_file[chunk.file_path] = chunk

    max_file_total = max(file_totals.values())
    if max_file_total <= 0.0:
        return
    boost_unit = max_score * _FILE_COHERENCE_BOOST_FRACTION
    for file_path, chunk in best_by_file.items():
        scores[chunk] += boost_unit * file_totals[file_path] / max_file_total


def _select_with_file_saturation(results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
    """Select top results while decaying repeated chunks from the same file."""
    file_counts: dict[str, int] = {}
    selected: list[SearchResult] = []
    for result in results:
        count = file_counts.get(result.chunk.file_path, 0)
        effective_score = result.score
        if count >= _FILE_SATURATION_THRESHOLD:
            excess = count - _FILE_SATURATION_THRESHOLD + 1
            effective_score *= _FILE_SATURATION_DECAY**excess
        selected.append(SearchResult(chunk=result.chunk, score=effective_score))
        file_counts[result.chunk.file_path] = count + 1

    selected.sort(key=lambda result: result.score, reverse=True)
    return selected[:top_k]


def _apply_query_relevance(chunk: Chunk, score: float, query: ParsedQuery, alpha_weight: float) -> float:
    """Boost chunks whose metadata or content matches multiple query concepts."""
    if score <= 0.0 or alpha_weight >= 1.0:
        return score

    adjusted = apply_relevance_rules(chunk, score, query)
    return max(adjusted, _TINY_QUERY_BOOST)
