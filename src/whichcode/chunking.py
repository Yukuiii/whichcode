"""Tree-sitter based source chunking for indexing code."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from tree_sitter_language_pack import (
    DownloadError,
    LanguageNotFoundError,
    ProcessConfig,
    detect_language_from_path,
    get_parser,
    process,
)

FUNCTION_KINDS = {"function", "method", "constructor"}
CLASS_KIND = "class"
SUPPORTED_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cxx",
    ".dart",
    ".ex",
    ".exs",
    ".go",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".java",
    ".js",
    ".cjs",
    ".jsx",
    ".kt",
    ".kts",
    ".lua",
    ".m",
    ".mjs",
    ".mm",
    ".php",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".scala",
    ".sc",
    ".swift",
    ".sh",
    ".bash",
    ".zsh",
    ".ts",
    ".tsx",
    ".zig",
}


@dataclass(frozen=True, slots=True)
class Chunk:
    """Represents one source fragment selected for indexing."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    kind: str = "file"
    name: str | None = None
    language: str | None = None

    @property
    def location(self) -> str:
        """Return the file path and line range for this chunk."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True, slots=True)
class _Span:
    """Stores a byte range and metadata found from the syntax tree."""

    start_byte: int
    end_byte: int
    kind: str
    name: str | None
    language: str


def chunk_source(source: str, file_path: str, language: str | None = None) -> list[Chunk]:
    """Split source into method-focused chunks using tree-sitter when possible."""
    if not source.strip():
        return []
    if Path(file_path).suffix.lower() not in SUPPORTED_SUFFIXES:
        return []

    source_bytes = source.encode("utf-8")
    resolved_language = _resolve_language(file_path, language)
    if resolved_language is None or not _can_parse_language(resolved_language):
        return [_make_chunk(source_bytes, 0, len(source_bytes), file_path, "file", None, resolved_language)]

    spans = _extract_structure_spans(source, source_bytes, resolved_language)

    if not spans:
        return [_make_chunk(source_bytes, 0, len(source_bytes), file_path, "file", None, resolved_language)]

    return _build_chunks(source_bytes, file_path, sorted(spans, key=lambda span: span.start_byte))


@cache
def _can_parse_language(language: str) -> bool:
    """Return whether a tree-sitter parser can be loaded for a language."""
    try:
        get_parser(language)
        return True
    except (DownloadError, LanguageNotFoundError):
        return False


def _resolve_language(file_path: str, language: str | None) -> str | None:
    """Resolve an explicit or path-inferred tree-sitter language name."""
    if language:
        return language.lower()
    detected = detect_language_from_path(file_path)
    return detected.lower() if detected else None


def _extract_structure_spans(source: str, source_bytes: bytes, language: str) -> list[_Span]:
    """Extract function and method spans from tree-sitter language-pack metadata."""
    try:
        result = process(source, ProcessConfig.all(language))
    except (DownloadError, LanguageNotFoundError):
        return []

    structure = result.get("structure")
    if not isinstance(structure, list):
        return []
    return _collect_structure_spans(structure, source_bytes, language, ())


def _collect_structure_spans(
    items: list[dict[str, Any]],
    source_bytes: bytes,
    language: str,
    class_stack: tuple[str, ...],
) -> list[_Span]:
    """Collect function-like structure spans while qualifying class methods."""
    spans: list[_Span] = []
    for item in items:
        kind = str(item.get("kind", "")).lower()
        name = item.get("name")
        name_text = name if isinstance(name, str) and name else None

        if kind in FUNCTION_KINDS:
            span = item.get("span")
            if isinstance(span, dict):
                chunk = _span_to_chunk_span(span, source_bytes, kind, name_text, language, class_stack)
                if chunk is not None:
                    spans.append(chunk)
            continue

        children = item.get("children")
        next_class_stack = class_stack
        if kind == CLASS_KIND and name_text:
            class_span = _class_header_span(item, source_bytes, language)
            if class_span is not None:
                spans.append(class_span)
            next_class_stack = (*class_stack, name_text)
        if isinstance(children, list):
            spans.extend(_collect_structure_spans(children, source_bytes, language, next_class_stack))

    return spans


def _class_header_span(item: dict[str, Any], source_bytes: bytes, language: str) -> _Span | None:
    """Convert a class structure item into a header/docstring span."""
    name = item.get("name")
    name_text = name if isinstance(name, str) and name else None
    span = item.get("span")
    if not isinstance(span, dict):
        return None

    start_byte = span.get("start_byte")
    end_byte = span.get("end_byte")
    if not isinstance(start_byte, int) or not isinstance(end_byte, int) or end_byte <= start_byte:
        return None

    children = item.get("children")
    child_start = _first_child_start_byte(children, source_bytes) if isinstance(children, list) else None
    header_end = child_start if child_start is not None and child_start > start_byte else end_byte
    if not source_bytes[start_byte:header_end].strip():
        return None
    return _Span(start_byte, header_end, "class", name_text, language)


def _first_child_start_byte(children: list[dict[str, Any]], source_bytes: bytes) -> int | None:
    """Return the earliest child span start, including leading metadata lines."""
    starts = []
    for child in children:
        span = child.get("span")
        if not isinstance(span, dict):
            continue
        start_byte = span.get("start_byte")
        if isinstance(start_byte, int):
            starts.append(_expand_start_to_leading_metadata(source_bytes, start_byte))
    return min(starts) if starts else None


def _span_to_chunk_span(
    span: dict[str, Any],
    source_bytes: bytes,
    kind: str,
    name: str | None,
    language: str,
    class_stack: tuple[str, ...],
) -> _Span | None:
    """Convert a language-pack span into a chunk span."""
    start_byte = span.get("start_byte")
    end_byte = span.get("end_byte")
    if not isinstance(start_byte, int) or not isinstance(end_byte, int) or end_byte <= start_byte:
        return None

    expanded_start = _expand_start_to_leading_metadata(source_bytes, start_byte)
    qualified_name = ".".join([*class_stack, name]) if name and class_stack else name
    chunk_kind = "method" if class_stack or kind in {"method", "constructor"} else "function"
    return _Span(expanded_start, end_byte, chunk_kind, qualified_name, language)


def _build_chunks(source_bytes: bytes, file_path: str, spans: list[_Span]) -> list[Chunk]:
    """Build ordered chunks while preserving code outside selected method spans."""
    chunks: list[Chunk] = []
    cursor = 0
    for span in spans:
        if span.start_byte > cursor:
            _append_gap_chunk(chunks, source_bytes, file_path, cursor, span.start_byte, span.language)
        chunks.append(
            _make_chunk(source_bytes, span.start_byte, span.end_byte, file_path, span.kind, span.name, span.language)
        )
        cursor = max(cursor, span.end_byte)

    if cursor < len(source_bytes):
        _append_gap_chunk(chunks, source_bytes, file_path, cursor, len(source_bytes), spans[-1].language)

    return chunks


def _append_gap_chunk(
    chunks: list[Chunk],
    source_bytes: bytes,
    file_path: str,
    start_byte: int,
    end_byte: int,
    language: str,
) -> None:
    """Append a module chunk for non-empty source outside function spans."""
    if source_bytes[start_byte:end_byte].strip():
        chunks.append(_make_chunk(source_bytes, start_byte, end_byte, file_path, "module", None, language))


def _make_chunk(
    source_bytes: bytes,
    start_byte: int,
    end_byte: int,
    file_path: str,
    kind: str,
    name: str | None,
    language: str | None,
) -> Chunk:
    """Create a chunk from byte offsets in the original source."""
    return Chunk(
        content=_decode_slice(source_bytes, start_byte, end_byte),
        file_path=file_path,
        start_line=_line_number(source_bytes, start_byte),
        end_line=_line_number(source_bytes, max(start_byte, end_byte - 1)),
        kind=kind,
        name=name,
        language=language,
    )


def _expand_start_to_leading_metadata(source_bytes: bytes, start_byte: int) -> int:
    """Include decorators or annotations immediately above a function span."""
    expanded = _line_start(source_bytes, start_byte)
    while expanded > 0:
        previous_start = _previous_line_start(source_bytes, expanded)
        previous_line = source_bytes[previous_start:expanded].strip()
        if not previous_line.startswith(b"@"):
            break
        expanded = previous_start
    return expanded


def _line_start(source_bytes: bytes, byte_index: int) -> int:
    """Return the byte offset for the beginning of the current line."""
    return source_bytes.rfind(b"\n", 0, byte_index) + 1


def _previous_line_start(source_bytes: bytes, line_start: int) -> int:
    """Return the byte offset for the beginning of the previous line."""
    if line_start <= 0:
        return 0
    return source_bytes.rfind(b"\n", 0, max(0, line_start - 1)) + 1


def _line_number(source_bytes: bytes, byte_index: int) -> int:
    """Convert a byte offset into a one-based source line number."""
    return _decode_slice(source_bytes, 0, byte_index).count("\n") + 1


def _decode_slice(source_bytes: bytes, start_byte: int, end_byte: int) -> str:
    """Decode a byte slice from the original UTF-8 source."""
    return source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
