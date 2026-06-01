"""Tree-sitter based source chunking for indexing code."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tree_sitter import Node, Parser
from tree_sitter_language_pack import DownloadError, LanguageNotFoundError, get_parser

DEFAULT_LANGUAGE = "python"
PYTHON_SUFFIXES = {".py", ".pyi"}
FUNCTION_NODE_TYPE = "function_definition"
CLASS_NODE_TYPE = "class_definition"
DECORATED_NODE_TYPE = "decorated_definition"


@dataclass(frozen=True, slots=True)
class Chunk:
    """Represents one source fragment selected for indexing."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    kind: str = "file"
    name: str | None = None

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


def chunk_source(source: str, file_path: str, language: str | None = None) -> list[Chunk]:
    """Split source into method-focused chunks using tree-sitter when possible."""
    if not source.strip():
        return []

    resolved_language = (language or _infer_language(file_path))
    if resolved_language is not None:
        resolved_language = resolved_language.lower()
    source_bytes = source.encode("utf-8")
    if resolved_language != DEFAULT_LANGUAGE:
        return [_make_chunk(source_bytes, 0, len(source_bytes), file_path, "file", None)]

    parser = _load_parser(resolved_language)
    if parser is None:
        return [_make_chunk(source_bytes, 0, len(source_bytes), file_path, "file", None)]

    root = parser.parse(source_bytes).root_node
    spans: list[_Span] = []
    _collect_spans(root, source_bytes, [], spans)

    if not spans:
        return [_make_chunk(source_bytes, 0, len(source_bytes), file_path, "file", None)]

    return _build_chunks(source_bytes, file_path, sorted(spans, key=lambda span: span.start_byte))


@cache
def _load_parser(language: str) -> Parser | None:
    """Load and cache a tree-sitter parser for the requested language."""
    try:
        return get_parser(language)
    except (DownloadError, LanguageNotFoundError):
        return None


def _infer_language(file_path: str) -> str | None:
    """Infer the supported language name from a file path."""
    suffix = Path(file_path).suffix.lower()
    if suffix in PYTHON_SUFFIXES:
        return DEFAULT_LANGUAGE
    return None


def _collect_spans(node: Node, source_bytes: bytes, class_stack: list[str], spans: list[_Span]) -> None:
    """Collect function and method spans from a Python syntax tree."""
    if node.type == DECORATED_NODE_TYPE:
        definition = _first_definition_child(node)
        if definition is not None and definition.type == FUNCTION_NODE_TYPE:
            _append_function_span(node, definition, source_bytes, class_stack, spans)
        return

    if node.type == CLASS_NODE_TYPE:
        class_name = _node_name(node, source_bytes)
        next_class_stack = [*class_stack, class_name] if class_name else class_stack
        for child in node.children:
            _collect_spans(child, source_bytes, next_class_stack, spans)
        return

    if node.type == FUNCTION_NODE_TYPE:
        _append_function_span(node, node, source_bytes, class_stack, spans)
        return

    for child in node.children:
        _collect_spans(child, source_bytes, class_stack, spans)


def _first_definition_child(node: Node) -> Node | None:
    """Return the wrapped definition node from a decorated definition."""
    for child in node.children:
        if child.type in {FUNCTION_NODE_TYPE, CLASS_NODE_TYPE}:
            return child
    return None


def _append_function_span(
    chunk_node: Node,
    name_node: Node,
    source_bytes: bytes,
    class_stack: list[str],
    spans: list[_Span],
) -> None:
    """Append a function or method span to the collected span list."""
    name = _node_name(name_node, source_bytes)
    qualified_name = ".".join([*class_stack, name]) if name and class_stack else name
    kind = "method" if class_stack else "function"
    spans.append(_Span(chunk_node.start_byte, chunk_node.end_byte, kind, qualified_name))


def _node_name(node: Node, source_bytes: bytes) -> str | None:
    """Read the tree-sitter name field for a definition node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return _decode_slice(source_bytes, name_node.start_byte, name_node.end_byte)


def _build_chunks(source_bytes: bytes, file_path: str, spans: list[_Span]) -> list[Chunk]:
    """Build ordered chunks while preserving code outside selected method spans."""
    chunks: list[Chunk] = []
    cursor = 0
    for span in spans:
        if span.start_byte > cursor:
            _append_gap_chunk(chunks, source_bytes, file_path, cursor, span.start_byte)
        chunks.append(_make_chunk(source_bytes, span.start_byte, span.end_byte, file_path, span.kind, span.name))
        cursor = max(cursor, span.end_byte)

    if cursor < len(source_bytes):
        _append_gap_chunk(chunks, source_bytes, file_path, cursor, len(source_bytes))

    return chunks


def _append_gap_chunk(chunks: list[Chunk], source_bytes: bytes, file_path: str, start_byte: int, end_byte: int) -> None:
    """Append a module chunk for non-empty source outside function spans."""
    if source_bytes[start_byte:end_byte].strip():
        chunks.append(_make_chunk(source_bytes, start_byte, end_byte, file_path, "module", None))


def _make_chunk(
    source_bytes: bytes,
    start_byte: int,
    end_byte: int,
    file_path: str,
    kind: str,
    name: str | None,
) -> Chunk:
    """Create a chunk from byte offsets in the original source."""
    return Chunk(
        content=_decode_slice(source_bytes, start_byte, end_byte),
        file_path=file_path,
        start_line=_line_number(source_bytes, start_byte),
        end_line=_line_number(source_bytes, max(start_byte, end_byte - 1)),
        kind=kind,
        name=name,
    )


def _line_number(source_bytes: bytes, byte_index: int) -> int:
    """Convert a byte offset into a one-based source line number."""
    return _decode_slice(source_bytes, 0, byte_index).count("\n") + 1


def _decode_slice(source_bytes: bytes, start_byte: int, end_byte: int) -> str:
    """Decode a byte slice from the original UTF-8 source."""
    return source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
