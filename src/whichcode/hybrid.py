"""Hybrid retrieval that combines BM25 and vector search results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from whichcode.bm25 import BM25Index, build_bm25_index, extract_search_terms, tokenize
from whichcode.chunking import Chunk
from whichcode.ranking_rules import apply_path_prior
from whichcode.types import SearchResult
from whichcode.vector import EmbeddingModel, VectorIndex, build_vector_index

_RRF_K = 60
_HYBRID_RECALL_CANDIDATES = 50
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
        alpha_weight = _resolve_alpha(query, alpha)
        if not 0.0 <= alpha_weight <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        candidate_count = max(top_k * 5, _HYBRID_RECALL_CANDIDATES)
        bm25_scores = _rrf_scores(self.bm25.search(query, top_k=candidate_count))
        vector_scores = _rrf_scores(self.vector.search(query, top_k=candidate_count))
        candidates = sorted({*bm25_scores, *vector_scores}, key=lambda chunk: (chunk.file_path, chunk.start_line))
        query_terms = extract_search_terms(query) if alpha_weight < 1.0 else []
        combined_scores = {
            chunk: _apply_query_relevance(
                chunk,
                alpha_weight * vector_scores.get(chunk, 0.0) + (1.0 - alpha_weight) * bm25_scores.get(chunk, 0.0),
                query_terms,
            )
            for chunk in candidates
        }
        _boost_file_coherence(combined_scores)
        combined = [
            SearchResult(
                chunk=chunk,
                score=_apply_path_penalty(chunk, combined_scores[chunk], penalize_paths),
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


def _apply_path_penalty(chunk: Chunk, score: float, penalize_paths: bool) -> float:
    """Apply path-based score penalties when enabled."""
    if not penalize_paths:
        return score
    return apply_path_prior(chunk.file_path, score)


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


def _apply_query_relevance(chunk: Chunk, score: float, query_terms: Sequence[str]) -> float:
    """Boost chunks whose metadata or content matches multiple query concepts."""
    if score <= 0.0 or not query_terms:
        return score

    metadata_tokens = set(_chunk_metadata_tokens(chunk))
    path_tokens = set(_chunk_path_tokens(chunk))
    content_tokens = set(tokenize(chunk.content))
    name_tokens = set(tokenize(chunk.name or ""))
    path_hits = _count_fuzzy_term_hits(query_terms, path_tokens)
    metadata_hits = _count_fuzzy_term_hits(query_terms, metadata_tokens)
    content_hits = _count_fuzzy_term_hits(query_terms, content_tokens)
    name_hits = _count_fuzzy_term_hits(query_terms, name_tokens)
    total_hits = len(
        {
            term
            for term in query_terms
            if _term_matches_any(term, metadata_tokens) or _term_matches_any(term, content_tokens)
        }
    )

    factor = 1.0
    if name_hits:
        factor += min(name_hits, 3) * 0.35
    if path_hits:
        factor += min(path_hits, 3) * 0.3
    if metadata_hits:
        factor += min(metadata_hits, 4) * 0.2
    if total_hits >= 2:
        factor += min(total_hits, 5) * 0.1
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


def _chunk_path_tokens(chunk: Chunk) -> list[str]:
    """Return searchable tokens from file and parent directory names."""
    path = Path(chunk.file_path)
    return tokenize(" ".join((path.stem, path.parent.as_posix())))


def _count_fuzzy_term_hits(query_terms: Sequence[str], candidate_terms: set[str]) -> int:
    """Count query terms with exact or prefix-compatible candidate matches."""
    return sum(1 for term in query_terms if _term_matches_any(term, candidate_terms))


def _term_matches_any(term: str, candidate_terms: set[str]) -> bool:
    """Return whether a query term matches any candidate token."""
    if term in candidate_terms:
        return True
    for candidate in candidate_terms:
        shorter, longer = (term, candidate) if len(term) <= len(candidate) else (candidate, term)
        if len(shorter) >= 3 and longer.startswith(shorter):
            return True
    return False
