"""Shared rule-based ranking rules for retrieval candidates."""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from whichcode.bm25 import extract_search_terms, tokenize
from whichcode.chunking import Chunk

_STRONG_PATH_PENALTY = 0.3
_MODERATE_PATH_PENALTY = 0.5
_MILD_PATH_PENALTY = 0.7
_PRIVATE_MODULE_PENALTY = 0.85
_GENERATED_PATH_PENALTY = 0.25
_LOW_SIGNAL_PATH_PENALTY = 0.4

_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:test_[^/]*\.\w+|[^/]*_test\.\w+|[^/]*_spec\.\w+|[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?)$"
)
_CAMEL_TEST_FILE_RE = re.compile(r"(?:Test|Tests|TestCase|Tester|Spec|Specs)\.[A-Za-z0-9]+$")
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|specs?|testlib|testing)(?:/|$)")
_CAMEL_TEST_DIR_RE = re.compile(r"(?:^|/)[A-Za-z0-9]*(?:Test|Tests|Spec)/")
_LOW_SIGNAL_DIR_RE = re.compile(
    r"(?:^|/)(?:_?examples?|samples?|fixtures?|benchmarks?|demos?|docs?|docs?_src|compat|_compat|legacy)(?:/|$)"
)
_PRIVATE_MODULE_RE = re.compile(r"(?:^|/)_[^/]+\.\w+$")
_REEXPORT_FILENAMES = frozenset({"__init__.py", "package-info.java"})
_TYPE_DEFS_RE = re.compile(r"\.d\.ts$")
_QUERY_FILTER_RE = re.compile(r"^(kind|lang|language|path|name):(.+)$", re.IGNORECASE)
_TEST_QUERY_RE = re.compile(r"\b(tests?|specs?|testing)\b", re.IGNORECASE)
_IDENTIFIER_CHARS_RE = re.compile(r"[^a-zA-Z0-9_]+")
_GENERATED_PATH_RES = tuple(
    re.compile(pattern)
    for pattern in (
        r"\.pb\.go$",
        r"\.pulsar\.go$",
        r"_grpc\.pb\.go$",
        r"_mocks?\.go$",
        r"(?:^|/)mock_[^/]+\.go$",
        r"\.(?:generated|gen)\.[jt]sx?$",
        r"\.pb\.[jt]s$",
        r"_pb\.[jt]s$",
        r"_grpc_pb\.[jt]s$",
        r"_pb2(?:_grpc)?\.py$",
        r"_pb2\.pyi$",
        r"\.pb\.(?:cc|h)$",
        r"\.g\.cs$",
        r"grpc\.cs$",
        r"outerclass\.java$",
        r"grpc\.java$",
        r"\.pb\.swift$",
        r"\.g\.dart$",
        r"\.freezed\.dart$",
        r"\.pb\.dart$",
        r"\.pbgrpc\.dart$",
        r"\.chopper\.dart$",
        r"\.generated\.rs$",
    )
)
_KIND_BONUSES = {
    "function": 0.18,
    "method": 0.18,
    "constructor": 0.18,
    "class": 0.16,
    "interface": 0.16,
    "struct": 0.14,
    "trait": 0.14,
    "enum": 0.1,
    "module": 0.04,
    "file": 0.0,
}


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Stores free text plus optional structured filters from a search query."""

    raw: str
    text: str
    kinds: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    path_filters: tuple[str, ...] = ()
    name_filters: tuple[str, ...] = ()

    @property
    def search_text(self) -> str:
        """Return the text portion used by BM25 and vector retrieval."""
        return self.text or self.raw


def parse_query(raw: str) -> ParsedQuery:
    """Parse field filters from a query while preserving free-text search terms."""
    kinds: list[str] = []
    languages: list[str] = []
    path_filters: list[str] = []
    name_filters: list[str] = []
    text_parts: list[str] = []

    for token in _split_query_tokens(raw):
        match = _QUERY_FILTER_RE.match(token)
        if match is None:
            text_parts.append(token)
            continue
        key = match.group(1).lower()
        values = tuple(value.strip().lower() for value in match.group(2).split(",") if value.strip())
        if not values:
            text_parts.append(token)
        elif key == "kind":
            kinds.extend(values)
        elif key in {"lang", "language"}:
            languages.extend(values)
        elif key == "path":
            path_filters.extend(values)
        elif key == "name":
            name_filters.extend(values)

    return ParsedQuery(
        raw=raw,
        text=" ".join(text_parts).strip(),
        kinds=tuple(dict.fromkeys(kinds)),
        languages=tuple(dict.fromkeys(languages)),
        path_filters=tuple(dict.fromkeys(path_filters)),
        name_filters=tuple(dict.fromkeys(name_filters)),
    )


def chunk_matches_query_filters(chunk: Chunk, query: ParsedQuery) -> bool:
    """Return whether a chunk satisfies hard query filters."""
    if query.kinds and chunk.kind.lower() not in query.kinds:
        return False
    if query.languages and (chunk.language or "").lower() not in query.languages:
        return False
    if query.path_filters:
        file_path = _normalize_path(chunk.file_path).lower()
        if not any(path_filter in file_path for path_filter in query.path_filters):
            return False
    if query.name_filters:
        name = (chunk.name or "").lower()
        if not any(name_filter in name for name_filter in query.name_filters):
            return False
    return True


def apply_relevance_rules(chunk: Chunk, score: float, query: ParsedQuery) -> float:
    """Apply deterministic name, path, kind, and co-occurrence relevance rules."""
    if score <= 0.0:
        return score
    return score * chunk_relevance_factor(chunk, query)


def chunk_relevance_factor(chunk: Chunk, query: ParsedQuery) -> float:
    """Return a multiplicative relevance factor for one chunk and query."""
    search_text = query.search_text
    base_terms = extract_search_terms(search_text, stems=False)
    expanded_terms = extract_search_terms(search_text)
    if not base_terms and not expanded_terms:
        return 1.0

    groups = _concept_groups(expanded_terms or base_terms)
    name_bonus = name_match_bonus(chunk.name or "", search_text)
    path_bonus = path_relevance_bonus(chunk.file_path, base_terms)
    kind_bonus = kind_relevance_bonus(chunk.kind)
    concept_hits = _count_concept_group_hits(chunk, groups)
    factor = 1.0 + name_bonus + path_bonus + kind_bonus

    if concept_hits >= 2:
        factor += min(concept_hits, 5) * 0.12
    elif len(groups) >= 2 and name_bonus == 0.0 and path_bonus == 0.0:
        factor *= 0.75

    return max(factor, 0.1)


def name_match_bonus(chunk_name: str, query: str) -> float:
    """Return a boost for exact, prefix, compound, or substring symbol matches."""
    if not chunk_name.strip() or not query.strip():
        return 0.0

    name_compact = _compact_identifier(chunk_name)
    query_compact = _compact_identifier(query)
    if not name_compact or not query_compact:
        return 0.0
    if name_compact == query_compact:
        return 1.6

    query_tokens = [token.lower() for token in re.split(r"\s+", query) if len(token) >= 2]
    if len(query_tokens) > 1 and name_compact in {_compact_identifier(token) for token in query_tokens}:
        return 1.2

    if name_compact.startswith(query_compact):
        ratio = len(query_compact) / max(len(name_compact), 1)
        return 0.55 + 0.55 * ratio

    raw_terms = extract_search_terms(query, stems=False)
    name_terms = set(tokenize(chunk_name))
    if len(raw_terms) > 1 and all(_term_matches_any(term, name_terms) for term in raw_terms):
        return 0.7
    if query_compact in name_compact:
        return 0.45
    return 0.0


def path_relevance_bonus(file_path: str, query_terms: Sequence[str]) -> float:
    """Return a boost when query terms match the file name or directory path."""
    if not query_terms:
        return 0.0

    normalized = _normalize_path(file_path).lower()
    path = Path(normalized)
    file_name = path.name
    file_stem = path.stem
    dir_segments = [segment for segment in path.parent.as_posix().split("/") if segment and segment != "."]
    path_tokens = set(tokenize(normalized))
    bonus = 0.0
    for term in query_terms:
        if term == file_stem or _term_matches_any(term, set(tokenize(file_stem))):
            bonus += 0.28
        elif term in file_name:
            bonus += 0.2
        elif term in dir_segments:
            bonus += 0.16
        elif _term_matches_any(term, path_tokens):
            bonus += 0.08
    return min(bonus, 0.8)


def kind_relevance_bonus(kind: str) -> float:
    """Return a small boost for high-information chunk kinds."""
    return _KIND_BONUSES.get(kind.lower(), 0.0)


def path_prior(file_path: str, query: str = "") -> float:
    """Return a multiplicative ranking prior for lower-signal file paths."""
    normalized = _normalize_path(file_path).lower()
    prior = 1.0
    if is_test_file(file_path) and not is_test_query(query):
        prior *= _STRONG_PATH_PENALTY
    if _LOW_SIGNAL_DIR_RE.search(normalized) is not None:
        prior *= _LOW_SIGNAL_PATH_PENALTY
    if is_generated_file(file_path):
        prior *= _GENERATED_PATH_PENALTY
    if Path(file_path).name in _REEXPORT_FILENAMES:
        prior *= _MODERATE_PATH_PENALTY
    if _TYPE_DEFS_RE.search(normalized) is not None:
        prior *= _MILD_PATH_PENALTY
    if _PRIVATE_MODULE_RE.search(normalized) is not None:
        prior *= _PRIVATE_MODULE_PENALTY
    return prior


def apply_path_prior(file_path: str, score: float, query: str = "") -> float:
    """Apply the path prior to a candidate score."""
    return score * path_prior(file_path, query)


def is_test_file(file_path: str) -> bool:
    """Return whether a path looks like test or specification code."""
    normalized = _normalize_path(file_path)
    lower = normalized.lower()
    file_name = Path(normalized).name
    return (
        _TEST_FILE_RE.search(lower) is not None
        or _CAMEL_TEST_FILE_RE.search(file_name) is not None
        or _TEST_DIR_RE.search(lower) is not None
        or _CAMEL_TEST_DIR_RE.search(normalized) is not None
    )


def is_generated_file(file_path: str) -> bool:
    """Return whether a path looks like generated source output."""
    normalized = _normalize_path(file_path).lower()
    return any(pattern.search(normalized) is not None for pattern in _GENERATED_PATH_RES)


def is_test_query(query: str) -> bool:
    """Return whether a user query is explicitly asking for tests."""
    return _TEST_QUERY_RE.search(query) is not None


def _split_query_tokens(raw: str) -> list[str]:
    """Split a query into shell-like tokens while tolerating malformed quotes."""
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def _normalize_path(file_path: str) -> str:
    """Normalize path separators for platform-independent matching."""
    return file_path.replace("\\", "/")


def _compact_identifier(text: str) -> str:
    """Collapse text into a lowercase identifier-like string."""
    return _IDENTIFIER_CHARS_RE.sub("", text).lower()


def _concept_groups(terms: Sequence[str]) -> list[tuple[str, ...]]:
    """Group stem variants so one root concept is counted once."""
    groups: list[tuple[str, ...]] = []
    assigned: set[str] = set()
    for term in sorted(dict.fromkeys(terms), key=len, reverse=True):
        if term in assigned:
            continue
        group = [term]
        assigned.add(term)
        for other in terms:
            if other in assigned:
                continue
            if term in other or other in term:
                group.append(other)
                assigned.add(other)
        groups.append(tuple(group))
    return groups


def _count_concept_group_hits(chunk: Chunk, groups: Sequence[Sequence[str]]) -> int:
    """Count distinct query concept groups that match chunk metadata or content."""
    metadata_tokens = set(_chunk_metadata_tokens(chunk))
    content_tokens = set(tokenize(chunk.content))
    hits = 0
    for group in groups:
        if any(_term_matches_any(term, metadata_tokens) or _term_matches_any(term, content_tokens) for term in group):
            hits += 1
    return hits


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


def _term_matches_any(term: str, candidate_terms: set[str]) -> bool:
    """Return whether a query term matches any candidate token."""
    if term in candidate_terms:
        return True
    for candidate in candidate_terms:
        shorter, longer = (term, candidate) if len(term) <= len(candidate) else (candidate, term)
        if len(shorter) >= 3 and longer.startswith(shorter):
            return True
    return False
