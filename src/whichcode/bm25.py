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

        tokens = extract_search_terms(query)
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


def extract_search_terms(query: str, *, stems: bool = True) -> list[str]:
    """Extract useful lexical terms from a natural-language or symbol query."""
    tokens: list[str] = []
    seen: set[str] = set()
    for token in tokenize(query):
        if len(token) < 3 or token in _STOP_WORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)

    if stems:
        for token in list(tokens):
            for variant in stem_variants(token):
                if variant not in seen and variant not in _STOP_WORDS:
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
