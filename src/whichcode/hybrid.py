"""Hybrid retrieval that combines BM25 and vector search results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from whichcode.bm25 import BM25Index, build_bm25_index, extract_search_terms
from whichcode.chunking import Chunk
from whichcode.code_graph import CodeGraph, build_code_graph, chunk_node_id, file_node_id
from whichcode.ranking_rules import (
    ParsedQuery,
    apply_path_prior,
    apply_relevance_rules,
    chunk_matches_query_filters,
    name_match_bonus,
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
_GRAPH_SEED_COUNT = 20
_GRAPH_IMPORT_BOOST = 0.22
_GRAPH_DEFINITION_BOOST = 0.35
_GRAPH_REFERENCE_BOOST = 0.16
_GRAPH_QUERY_RE = re.compile(
    r"\b(imports?|dependencies|dependency|call graph|called by|callers?|callees?|references?|related files?|related code)\b",
    re.IGNORECASE,
)
_SYMBOL_QUERY_RE = re.compile(
    r"^(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\|->|\.)[A-Za-z_][A-Za-z0-9_]*)+"
    r"|_[A-Za-z0-9_]*"
    r"|[A-Za-z][A-Za-z0-9]*[A-Z_][A-Za-z0-9_]*"
    r"|[A-Z][A-Za-z0-9]*"
    r")$"
)
_IDENTIFIER_HINT_RE = re.compile(
    r"(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\|->|\.)[A-Za-z_][A-Za-z0-9_]*)+"
    r"|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+"
    r"|[A-Za-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*"
    r"|[A-Z]{2,}[A-Za-z0-9]*"
    r")"
)
_QUESTION_QUERY_RE = re.compile(r"^\s*(?:how|what|why|where|when|which|who|explain|describe)\b", re.IGNORECASE)
_ALPHA_SYMBOL = 0.3
_ALPHA_NATURAL_LANGUAGE = 0.5
_SYMBOL_DEFINITION_KINDS = frozenset(
    {"function", "method", "constructor", "class", "interface", "struct", "trait", "enum"}
)


@dataclass(frozen=True, slots=True)
class _ChannelWeights:
    """Stores per-channel weights used during weighted reciprocal-rank fusion."""

    content: float
    vector: float
    name: float
    path: float
    symbol: float


@dataclass(frozen=True, slots=True)
class HybridIndex:
    """Combines lexical and semantic indexes over the same chunk collection."""

    bm25: BM25Index
    vector: VectorIndex
    graph: CodeGraph | None = None
    graph_lookup: _GraphLookup | None = None

    def __post_init__(self) -> None:
        """Build graph lookup data once for repeated searches."""
        if self.graph is not None and self.graph_lookup is None:
            object.__setattr__(self, "graph_lookup", _build_graph_lookup(self.graph, _index_chunks(self.bm25, self.vector)))

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], model: EmbeddingModel | None = None) -> HybridIndex:
        """Build BM25 and vector indexes from chunks."""
        resolved_chunks = tuple(chunks)
        return cls(
            bm25=build_bm25_index(resolved_chunks),
            vector=build_vector_index(resolved_chunks, model=model),
            graph=build_code_graph(resolved_chunks),
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
        channel_weights = _resolve_channel_weights(search_query, alpha)
        all_chunks = _index_chunks(self.bm25, self.vector)
        channel_scores = _weighted_rrf_scores(
            (
                (
                    channel_weights.content,
                    _search_field(self.bm25, "search_content", search_query, candidate_count, "search"),
                ),
                (channel_weights.vector, self.vector.search(search_query, top_k=candidate_count)),
                (channel_weights.name, _search_field(self.bm25, "search_name", search_query, candidate_count)),
                (channel_weights.path, _search_field(self.bm25, "search_path", search_query, candidate_count)),
                (channel_weights.symbol, _rank_symbol_definitions(all_chunks, parsed_query, candidate_count)),
            )
        )
        if _should_expand_graph(search_query):
            _apply_graph_expansion(channel_scores, self.graph_lookup)
        candidates = [
            chunk
            for chunk in sorted(channel_scores, key=lambda chunk: (chunk.file_path, chunk.start_line))
            if chunk_matches_query_filters(chunk, parsed_query)
        ]
        combined_scores = {
            chunk: _apply_query_relevance(
                chunk,
                channel_scores.get(chunk, 0.0),
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


def _weighted_rrf_scores(channels: Sequence[tuple[float, Sequence[SearchResult]]]) -> dict[Chunk, float]:
    """Combine ranked channel outputs with weighted reciprocal-rank fusion."""
    scores: dict[Chunk, float] = {}
    for weight, results in channels:
        if weight <= 0.0:
            continue
        seen: set[Chunk] = set()
        for rank, result in enumerate(results, 1):
            if result.chunk in seen:
                continue
            seen.add(result.chunk)
            scores[result.chunk] = scores.get(result.chunk, 0.0) + weight / (_RRF_K + rank)
    return scores


def _search_field(
    index: object,
    method_name: str,
    query: str,
    top_k: int,
    fallback_name: str | None = None,
) -> list[SearchResult]:
    """Call a field-specific search method when the index supports it."""
    method = getattr(index, method_name, None)
    if callable(method):
        return list(method(query, top_k=top_k))
    if fallback_name is None:
        return []
    fallback = getattr(index, fallback_name, None)
    if callable(fallback):
        return list(fallback(query, top_k=top_k))
    return []


def _index_chunks(*indexes: object) -> tuple[Chunk, ...]:
    """Return the first chunk collection exposed by one of the indexes."""
    for index in indexes:
        chunks = getattr(index, "chunks", None)
        if chunks:
            return tuple(chunks)
    return ()


def _apply_graph_expansion(scores: dict[Chunk, float], lookup: _GraphLookup | None) -> None:
    """Add low-weight one-hop graph neighbors to the candidate score map."""
    if lookup is None or not scores:
        return

    seed_chunks = sorted(scores, key=lambda chunk: scores[chunk], reverse=True)[:_GRAPH_SEED_COUNT]
    for seed in seed_chunks:
        seed_score = scores.get(seed, 0.0)
        if seed_score <= 0.0:
            continue
        seed_chunk_id = chunk_node_id(seed)
        seed_file_id = file_node_id(seed.file_path)
        for related_file_id in lookup.import_neighbors_by_file.get(seed_file_id, ()):
            for chunk in lookup.chunks_by_file_id.get(related_file_id, ()):
                _add_graph_score(scores, chunk, seed_score * _GRAPH_IMPORT_BOOST)
        for identifier in lookup.references_by_chunk.get(seed_chunk_id, ()):
            for chunk in lookup.definitions_by_identifier.get(identifier, ()):
                _add_graph_score(scores, chunk, seed_score * _GRAPH_DEFINITION_BOOST)
        for identifier in lookup.definitions_by_chunk.get(seed_chunk_id, ()):
            for chunk in lookup.references_by_identifier.get(identifier, ()):
                _add_graph_score(scores, chunk, seed_score * _GRAPH_REFERENCE_BOOST)


@dataclass(frozen=True, slots=True)
class _GraphLookup:
    """Stores graph lookup maps used for one-hop candidate expansion."""

    chunks_by_file_id: dict[str, tuple[Chunk, ...]]
    import_neighbors_by_file: dict[str, tuple[str, ...]]
    definitions_by_identifier: dict[str, tuple[Chunk, ...]]
    references_by_identifier: dict[str, tuple[Chunk, ...]]
    definitions_by_chunk: dict[str, tuple[str, ...]]
    references_by_chunk: dict[str, tuple[str, ...]]


def _build_graph_lookup(graph: CodeGraph, chunks: Sequence[Chunk]) -> _GraphLookup | None:
    """Build file, definition, and reference lookup maps from a code graph."""
    chunk_by_id = {chunk_node_id(chunk): chunk for chunk in chunks}
    node_by_id = {node.id: node for node in graph.nodes}
    chunks_by_file: dict[str, list[Chunk]] = {}
    import_neighbors: dict[str, set[str]] = {}
    definitions_by_identifier: dict[str, list[Chunk]] = {}
    references_by_identifier: dict[str, list[Chunk]] = {}
    definitions_by_chunk: dict[str, set[str]] = {}
    references_by_chunk: dict[str, set[str]] = {}

    for chunk in chunks:
        chunks_by_file.setdefault(file_node_id(chunk.file_path), []).append(chunk)

    for edge in graph.edges:
        source_node = node_by_id.get(edge.source_id)
        target_node = node_by_id.get(edge.target_id)
        if source_node is None or target_node is None:
            continue
        if edge.kind == "imports" and source_node.kind == "file" and target_node.kind == "file":
            import_neighbors.setdefault(edge.source_id, set()).add(edge.target_id)
            import_neighbors.setdefault(edge.target_id, set()).add(edge.source_id)
        elif edge.kind == "defines" and source_node.kind == "chunk" and target_node.kind == "symbol":
            chunk = chunk_by_id.get(edge.source_id)
            if chunk is None:
                continue
            for identifier in _graph_identifier_terms(target_node.label):
                definitions_by_identifier.setdefault(identifier, []).append(chunk)
                definitions_by_chunk.setdefault(edge.source_id, set()).add(identifier)
        elif edge.kind == "references" and source_node.kind == "chunk" and target_node.kind == "identifier":
            chunk = chunk_by_id.get(edge.source_id)
            if chunk is None:
                continue
            identifier = target_node.label.lower()
            references_by_identifier.setdefault(identifier, []).append(chunk)
            references_by_chunk.setdefault(edge.source_id, set()).add(identifier)

    if not import_neighbors and not definitions_by_identifier and not references_by_identifier:
        return None
    return _GraphLookup(
        chunks_by_file_id={key: tuple(value) for key, value in chunks_by_file.items()},
        import_neighbors_by_file={key: tuple(sorted(value)) for key, value in import_neighbors.items()},
        definitions_by_identifier={key: tuple(dict.fromkeys(value)) for key, value in definitions_by_identifier.items()},
        references_by_identifier={key: tuple(dict.fromkeys(value)) for key, value in references_by_identifier.items()},
        definitions_by_chunk={key: tuple(sorted(value)) for key, value in definitions_by_chunk.items()},
        references_by_chunk={key: tuple(sorted(value)) for key, value in references_by_chunk.items()},
    )


def _graph_identifier_terms(text: str) -> tuple[str, ...]:
    """Return normalized identifier terms for graph symbol matching."""
    return tuple(
        dict.fromkeys(extract_search_terms(text, stems=False, include_stop_words=True, aliases=False))
    )


def _add_graph_score(scores: dict[Chunk, float], chunk: Chunk, score: float) -> None:
    """Add a graph-derived score without lowering an existing candidate score."""
    if score <= 0.0:
        return
    scores[chunk] = max(scores.get(chunk, 0.0), score)


def _rank_symbol_definitions(chunks: Sequence[Chunk], query: ParsedQuery, top_k: int) -> list[SearchResult]:
    """Rank definition-like chunks by exact, prefix, and compound symbol matches."""
    if top_k < 1 or not chunks:
        return []

    search_text = query.search_text
    if not search_text.strip():
        return []

    query_terms = extract_search_terms(search_text, stems=False, include_stop_words=True)
    scored: dict[Chunk, float] = {}
    for chunk in chunks:
        name_score = name_match_bonus(chunk.name or "", search_text)
        if name_score <= 0.0:
            continue
        score = name_score + _definition_line_match_bonus(chunk, query_terms)
        if chunk.kind.lower() in _SYMBOL_DEFINITION_KINDS:
            score *= 1.2
        else:
            score *= 0.7
        scored[chunk] = score

    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked]


def _definition_line_match_bonus(chunk: Chunk, query_terms: Sequence[str]) -> float:
    """Return a small symbol-channel bonus for query terms in the definition line."""
    if not query_terms or not chunk.content.strip():
        return 0.0
    first_line = chunk.content.lstrip().splitlines()[0].lower()
    hits = sum(1 for term in query_terms if term in first_line)
    if hits == 0:
        return 0.0
    return min(0.8, 0.2 * hits)


def _resolve_alpha(query: str, alpha: float | None) -> float:
    """Return semantic-vs-lexical blend weight for the query."""
    if alpha is not None:
        return alpha
    return _ALPHA_SYMBOL if _is_symbol_query(query) else _ALPHA_NATURAL_LANGUAGE


def _resolve_channel_weights(query: str, alpha: float | None) -> _ChannelWeights:
    """Return weighted-RRF channel weights for symbol or natural-language queries."""
    if alpha is not None:
        lexical_weight = 1.0 - alpha
        return _ChannelWeights(
            content=lexical_weight,
            vector=alpha,
            name=lexical_weight * 2.0,
            path=lexical_weight * 2.5,
            symbol=lexical_weight * 2.2,
        )
    if _is_symbol_query(query):
        return _ChannelWeights(content=1.5, vector=0.4, name=2.5, path=1.2, symbol=2.5)
    if _has_identifier_hint(query):
        return _ChannelWeights(content=1.1, vector=1.0, name=1.2, path=1.2, symbol=1.0)
    if _is_question_query(query):
        return _ChannelWeights(content=1.3, vector=1.3, name=0.2, path=0.9, symbol=0.2)
    return _ChannelWeights(content=1.2, vector=1.15, name=0.25, path=1.4, symbol=0.2)


def _is_symbol_query(query: str) -> bool:
    """Return whether the query looks like a bare code symbol."""
    return _SYMBOL_QUERY_RE.match(query.strip()) is not None


def _has_identifier_hint(query: str) -> bool:
    """Return whether a natural-language query contains an explicit code identifier."""
    return _IDENTIFIER_HINT_RE.search(query) is not None


def _is_question_query(query: str) -> bool:
    """Return whether a query looks like an architecture or behavior question."""
    return _QUESTION_QUERY_RE.search(query) is not None


def _should_expand_graph(query: str) -> bool:
    """Return whether a query explicitly asks for code relationships."""
    return _GRAPH_QUERY_RE.search(query) is not None


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
