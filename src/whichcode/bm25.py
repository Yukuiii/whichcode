"""BM25 indexing helpers for ranked chunk search."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import bm25s

from whichcode.chunking import Chunk
from whichcode.types import SearchResult

_CONTENT_FIELD_WEIGHT = 1.0
_NAME_FIELD_WEIGHT = 6.0
_PATH_FIELD_WEIGHT = 3.0
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_STOP_WORDS = frozenset(
    """
    a about after all also an and any are as at be been before being both but by
    called can code could did do does done during each every file files for from
    function give had has have having here how if in into is it its just like
    look made may me method might more most my need needs no not of on only or
    our out over should show some such tell than that the them then there they
    this to used using want was we were what when where which who why will with
    work works would you your
    """.split()
)


@dataclass(frozen=True, slots=True)
class BM25Index:
    """Wraps field-specific BM25 indexes over chunk content, names, and paths."""

    chunks: tuple[Chunk, ...]
    _content_index: bm25s.BM25 | None
    _name_index: bm25s.BM25 | None
    _path_index: bm25s.BM25 | None

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> BM25Index:
        """Build field-specific BM25 indexes from indexable chunks."""
        resolved_chunks = tuple(chunks)
        return cls(
            chunks=resolved_chunks,
            _content_index=_build_token_index([tokenize(chunk.content) for chunk in resolved_chunks]),
            _name_index=_build_token_index([_metadata_tokens(chunk.name or "") for chunk in resolved_chunks]),
            _path_index=_build_token_index([_path_tokens(chunk.file_path) for chunk in resolved_chunks]),
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked by weighted content, name, and path BM25 scores."""
        if top_k < 1 or not self.chunks:
            return []

        content_scores = self._field_score_map(self._content_index, _content_query_terms(query))
        name_scores = self._field_score_map(self._name_index, _metadata_query_terms(query))
        path_scores = self._field_score_map(self._path_index, _metadata_query_terms(query))
        if not content_scores and not name_scores and not path_scores:
            return []

        combined: dict[Chunk, float] = {}
        _add_weighted_scores(combined, content_scores, _CONTENT_FIELD_WEIGHT)
        _add_weighted_scores(combined, name_scores, _NAME_FIELD_WEIGHT)
        _add_weighted_scores(combined, path_scores, _PATH_FIELD_WEIGHT)
        return _rank_score_map(combined, top_k)

    def search_content(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked only by content BM25 score."""
        return _rank_score_map(self._field_score_map(self._content_index, _content_query_terms(query)), top_k)

    def search_name(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked only by symbol-name BM25 score."""
        return _rank_score_map(self._field_score_map(self._name_index, _metadata_query_terms(query)), top_k)

    def search_path(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked only by file-path BM25 score."""
        return _rank_score_map(self._field_score_map(self._path_index, _metadata_query_terms(query)), top_k)

    def _field_score_map(self, index: bm25s.BM25 | None, tokens: Sequence[str]) -> dict[Chunk, float]:
        """Return positive finite BM25 scores for one indexed field."""
        if index is None or not tokens:
            return {}
        scores = index.get_scores(list(tokens))
        result: dict[Chunk, float] = {}
        for chunk, score in zip(self.chunks, scores, strict=True):
            value = float(score)
            if value > 0.0 and isfinite(value):
                result[chunk] = value
        return result


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


def extract_search_terms(
    query: str,
    *,
    stems: bool = True,
    include_stop_words: bool = False,
) -> list[str]:
    """Extract useful lexical terms from a natural-language or symbol query."""
    tokens: list[str] = []
    seen: set[str] = set()
    for token in tokenize(query):
        if len(token) < 3 or (not include_stop_words and token in _STOP_WORDS) or token in seen:
            continue
        seen.add(token)
        tokens.append(token)

    if stems:
        for token in list(tokens):
            for variant in stem_variants(token):
                if variant not in seen and (include_stop_words or variant not in _STOP_WORDS):
                    seen.add(variant)
                    tokens.append(variant)
    return tokens


def stem_variants(term: str) -> list[str]:
    """Return cheap suffix-stripped variants for code-oriented FTS matching."""
    variants: set[str] = set()
    token = term.lower()

    if token.endswith("ing") and len(token) > 5:
        base = token[:-3]
        variants.add(base)
        variants.add(base + "e")
        if len(base) >= 2 and base[-1] == base[-2]:
            variants.add(base[:-1])

    if (token.endswith("tion") or token.endswith("sion")) and len(token) > 5:
        variants.add(token[:-3])

    if token.endswith("ment") and len(token) > 6:
        variants.add(token[:-4])

    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    elif token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
        if token[-3] not in {"s", "x", "z"}:
            variants.add(token[:-1])
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        variants.add(token[:-1])

    if token.endswith("ed") and not token.endswith("eed") and len(token) > 4:
        variants.add(token[:-1])
        variants.add(token[:-2])
        if token.endswith("ied") and len(token) > 5:
            variants.add(token[:-3] + "y")

    if token.endswith("er") and len(token) > 4:
        base = token[:-2]
        variants.add(base)
        variants.add(base + "e")
        if len(base) >= 2 and base[-1] == base[-2]:
            variants.add(base[:-1])

    return sorted(variant for variant in variants if len(variant) >= 3 and variant != token)


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


def _build_token_index(documents: Sequence[Sequence[str]]) -> bm25s.BM25 | None:
    """Build a BM25 index for tokenized field documents when any token exists."""
    resolved_documents = [list(document) for document in documents]
    if not resolved_documents or not any(resolved_documents):
        return None
    index = bm25s.BM25()
    index.index(resolved_documents, show_progress=False)
    return index


def _content_query_terms(query: str) -> list[str]:
    """Return natural-language query terms for content search."""
    return extract_search_terms(query)


def _metadata_query_terms(query: str) -> list[str]:
    """Return code-oriented query terms for path and symbol metadata search."""
    return extract_search_terms(query, include_stop_words=True)


def _metadata_tokens(text: str) -> list[str]:
    """Return field tokens plus stem variants for code metadata."""
    return _expand_index_tokens(tokenize(text))


def _path_tokens(file_path: str) -> list[str]:
    """Return file-name, file-stem, and directory tokens for path search."""
    path = Path(file_path.replace("\\", "/"))
    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    text = " ".join([path.name, path.stem, *dir_parts])
    return _metadata_tokens(text)


def _expand_index_tokens(tokens: Sequence[str]) -> list[str]:
    """Add cheap stem variants to indexed metadata tokens."""
    expanded: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            expanded.append(token)
        for variant in stem_variants(token):
            if variant not in seen:
                seen.add(variant)
                expanded.append(variant)
    return expanded


def _add_weighted_scores(target: dict[Chunk, float], scores: dict[Chunk, float], weight: float) -> None:
    """Add normalized field scores into a combined lexical score map."""
    if not scores or weight <= 0.0:
        return
    max_score = max(scores.values())
    if max_score <= 0.0:
        return
    for chunk, score in scores.items():
        target[chunk] = target.get(chunk, 0.0) + weight * score / max_score


def _rank_score_map(scores: dict[Chunk, float], top_k: int) -> list[SearchResult]:
    """Convert a score map into sorted search results."""
    if top_k < 1:
        return []
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked if score > 0.0]
