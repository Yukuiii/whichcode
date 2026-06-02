"""Lightweight code relationship graph extraction and persistence."""

from __future__ import annotations

import posixpath
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from whichcode.chunking import SUPPORTED_SUFFIXES, Chunk

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PY_IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")
_PY_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+(.+)$")
_JS_IMPORT_RE = re.compile(r"""(?:import|export)\s+(?:[^'"]*?\s+from\s+)?["']([^"']+)["']""")
_JS_REQUIRE_RE = re.compile(r"""require\s*\(\s*["']([^"']+)["']\s*\)""")
_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([A-Za-z_][\w.]*)(?:\.\*)?\s*;")
_GO_IMPORT_RE = re.compile(r'^\s*(?:import\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\s+)?["]([^"]+)["]\s*;?$')
_INCLUDE_RE = re.compile(r"""^\s*#\s*include\s+[<"]([^>"]+)[>"]""")
_RUST_USE_RE = re.compile(r"^\s*use\s+([A-Za-z_][\w:]*)(?:\s+as\s+\w+)?\s*;")
_MAX_IDENTIFIERS_PER_CHUNK = 200
_REFERENCE_CONFIDENCE = 0.55
_IMPORT_MODULE_CONFIDENCE = 0.95
_IMPORT_FILE_CONFIDENCE = 1.0
_DEFINE_CONFIDENCE = 1.0
_CONTAINS_CONFIDENCE = 1.0
_IDENTIFIER_STOP_WORDS = frozenset(
    """
    and as async await break case catch class const continue def default defer do
    else elif enum except export extends false final finally for from func function
    if impl import in include interface is lambda let match mod namespace new nil
    none not null package pass private protected public raise require return self
    static struct super switch this throw throws true try type undefined use using
    var void while with yield
    """.split()
)


@dataclass(frozen=True, slots=True)
class GraphNode:
    """Represents one file, chunk, symbol, module, or identifier graph node."""

    id: str
    kind: str
    label: str
    file_path: str | None = None
    chunk_id: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """Represents one directed relationship between graph nodes."""

    source_id: str
    target_id: str
    kind: str
    confidence: float


@dataclass(frozen=True, slots=True)
class CodeGraph:
    """Stores lightweight code relationship nodes and edges."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


def build_code_graph(chunks: Sequence[Chunk]) -> CodeGraph:
    """Build import, definition, and identifier-reference graph edges from chunks."""
    resolved_chunks = tuple(chunks)
    file_paths = tuple(sorted({chunk.file_path for chunk in resolved_chunks}))
    file_path_set = set(file_paths)
    module_file_map = _build_module_file_map(file_paths)
    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str, str], GraphEdge] = {}

    for file_path in file_paths:
        _add_node(nodes, GraphNode(file_node_id(file_path), "file", file_path, file_path=file_path))

    for chunk in resolved_chunks:
        file_id = file_node_id(chunk.file_path)
        chunk_id = chunk_node_id(chunk)
        _add_node(
            nodes,
            GraphNode(
                chunk_id,
                "chunk",
                chunk.location,
                file_path=chunk.file_path,
                chunk_id=chunk_id,
                language=chunk.language,
            ),
        )
        _add_edge(edges, file_id, chunk_id, "contains", _CONTAINS_CONFIDENCE)
        _add_definition_edges(nodes, edges, chunk, file_id, chunk_id)
        _add_import_edges(nodes, edges, chunk, file_id, file_path_set, module_file_map)
        _add_identifier_edges(nodes, edges, chunk, chunk_id)

    return CodeGraph(
        nodes=tuple(sorted(nodes.values(), key=lambda node: node.id)),
        edges=tuple(sorted(edges.values(), key=lambda edge: (edge.source_id, edge.kind, edge.target_id))),
    )


def save_code_graph(path: Path, graph: CodeGraph) -> None:
    """Persist a code graph to a SQLite database file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        _initialize_schema(connection)
        connection.executemany(
            """
            INSERT INTO nodes (id, kind, label, file_path, chunk_id, language)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (node.id, node.kind, node.label, node.file_path, node.chunk_id, node.language)
                for node in graph.nodes
            ),
        )
        connection.executemany(
            """
            INSERT INTO edges (source_id, target_id, kind, confidence)
            VALUES (?, ?, ?, ?)
            """,
            ((edge.source_id, edge.target_id, edge.kind, edge.confidence) for edge in graph.edges),
        )


def load_code_graph(path: Path) -> CodeGraph:
    """Load a persisted code graph from a SQLite database file."""
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        node_rows = connection.execute(
            "SELECT id, kind, label, file_path, chunk_id, language FROM nodes ORDER BY id"
        ).fetchall()
        edge_rows = connection.execute(
            "SELECT source_id, target_id, kind, confidence FROM edges ORDER BY source_id, kind, target_id"
        ).fetchall()
    return CodeGraph(
        nodes=tuple(
            GraphNode(
                id=str(row["id"]),
                kind=str(row["kind"]),
                label=str(row["label"]),
                file_path=row["file_path"] if isinstance(row["file_path"], str) else None,
                chunk_id=row["chunk_id"] if isinstance(row["chunk_id"], str) else None,
                language=row["language"] if isinstance(row["language"], str) else None,
            )
            for row in node_rows
        ),
        edges=tuple(
            GraphEdge(
                source_id=str(row["source_id"]),
                target_id=str(row["target_id"]),
                kind=str(row["kind"]),
                confidence=float(row["confidence"]),
            )
            for row in edge_rows
        ),
    )


def file_node_id(file_path: str) -> str:
    """Return a stable graph node id for a file."""
    return f"file:{file_path}"


def chunk_node_id(chunk: Chunk) -> str:
    """Return a stable graph node id for a chunk."""
    return f"chunk:{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.kind}:{chunk.name or ''}"


def identifier_node_id(identifier: str) -> str:
    """Return a stable graph node id for an identifier reference."""
    return f"identifier:{identifier.lower()}"


def _initialize_schema(connection: sqlite3.Connection) -> None:
    """Create graph tables and clear previous generated graph rows."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            label TEXT NOT NULL,
            file_path TEXT,
            chunk_id TEXT,
            language TEXT
        );
        CREATE TABLE IF NOT EXISTS edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            confidence REAL NOT NULL,
            PRIMARY KEY (source_id, target_id, kind)
        );
        DELETE FROM edges;
        DELETE FROM nodes;
        """
    )


def _add_definition_edges(
    nodes: dict[str, GraphNode],
    edges: dict[tuple[str, str, str], GraphEdge],
    chunk: Chunk,
    file_id: str,
    chunk_id: str,
) -> None:
    """Add symbol definition nodes and edges for named chunks."""
    if not chunk.name:
        return
    symbol_id = _symbol_id(chunk.file_path, chunk.name)
    _add_node(
        nodes,
        GraphNode(
            symbol_id,
            "symbol",
            chunk.name,
            file_path=chunk.file_path,
            chunk_id=chunk_id,
            language=chunk.language,
        ),
    )
    _add_edge(edges, file_id, symbol_id, "defines", _DEFINE_CONFIDENCE)
    _add_edge(edges, chunk_id, symbol_id, "defines", _DEFINE_CONFIDENCE)


def _add_import_edges(
    nodes: dict[str, GraphNode],
    edges: dict[tuple[str, str, str], GraphEdge],
    chunk: Chunk,
    file_id: str,
    file_path_set: set[str],
    module_file_map: dict[str, str],
) -> None:
    """Add module and resolved file import edges found in a chunk."""
    for module in _extract_imports(chunk.content, chunk.language):
        module_id = _module_id(module)
        _add_node(nodes, GraphNode(module_id, "module", module))
        _add_edge(edges, file_id, module_id, "imports", _IMPORT_MODULE_CONFIDENCE)
        target_file = _resolve_import_target(chunk.file_path, module, file_path_set, module_file_map)
        if target_file is not None:
            _add_edge(edges, file_id, file_node_id(target_file), "imports", _IMPORT_FILE_CONFIDENCE)


def _add_identifier_edges(
    nodes: dict[str, GraphNode],
    edges: dict[tuple[str, str, str], GraphEdge],
    chunk: Chunk,
    chunk_id: str,
) -> None:
    """Add low-confidence identifier reference edges for one chunk."""
    for identifier in _extract_identifiers(chunk.content):
        identifier_id = _identifier_id(identifier)
        _add_node(nodes, GraphNode(identifier_id, "identifier", identifier))
        _add_edge(edges, chunk_id, identifier_id, "references", _REFERENCE_CONFIDENCE)


def _add_node(nodes: dict[str, GraphNode], node: GraphNode) -> None:
    """Insert a graph node if it has not already been added."""
    nodes.setdefault(node.id, node)


def _add_edge(
    edges: dict[tuple[str, str, str], GraphEdge],
    source_id: str,
    target_id: str,
    kind: str,
    confidence: float,
) -> None:
    """Insert or strengthen one graph edge."""
    key = (source_id, target_id, kind)
    existing = edges.get(key)
    if existing is None or confidence > existing.confidence:
        edges[key] = GraphEdge(source_id, target_id, kind, confidence)


def _extract_imports(source: str, language: str | None = None) -> tuple[str, ...]:
    """Extract import-like module names from source text with language-aware regexes."""
    modules: set[str] = set()
    language_name = (language or "").lower()
    for line in source.splitlines():
        if language_name == "python":
            modules.update(_extract_python_imports(line))
        elif language_name in {"javascript", "typescript", "tsx", "jsx"}:
            modules.update(_extract_pattern_values(line, (_JS_IMPORT_RE, _JS_REQUIRE_RE)))
        elif language_name == "go":
            modules.update(_extract_pattern_values(line, (_GO_IMPORT_RE,)))
        elif language_name == "rust":
            modules.update(_extract_pattern_values(line, (_RUST_USE_RE,)))
        elif language_name in {"java", "kotlin", "scala"}:
            modules.update(_extract_pattern_values(line, (_JAVA_IMPORT_RE,)))
        elif language_name in {"c", "cpp", "c_sharp", "objc", "objective-c"}:
            modules.update(_extract_pattern_values(line, (_INCLUDE_RE,)))
        else:
            modules.update(_extract_python_imports(line))
            modules.update(_extract_pattern_values(line, (_JS_IMPORT_RE, _JS_REQUIRE_RE, _JAVA_IMPORT_RE)))
            modules.update(_extract_pattern_values(line, (_GO_IMPORT_RE, _INCLUDE_RE, _RUST_USE_RE)))
    return tuple(sorted(module for module in modules if module))


def _extract_python_imports(line: str) -> tuple[str, ...]:
    """Extract Python import targets from one line."""
    import_match = _PY_IMPORT_RE.match(line)
    if import_match is not None:
        return tuple(_first_import_name(part) for part in import_match.group(1).split(",") if _first_import_name(part))

    from_match = _PY_FROM_IMPORT_RE.match(line)
    if from_match is None:
        return ()

    module = from_match.group(1).strip()
    if module.strip("."):
        return (module,)

    return tuple(
        f"{module}{_first_import_name(part)}"
        for part in from_match.group(2).split(",")
        if _first_import_name(part) and _first_import_name(part) != "*"
    )


def _extract_pattern_values(line: str, patterns: Sequence[re.Pattern[str]]) -> Iterable[str]:
    """Extract the first capture group from all regex matches on a line."""
    for pattern in patterns:
        for match in pattern.finditer(line):
            value = match.group(1).strip()
            if value:
                yield value


def _first_import_name(text: str) -> str:
    """Return the imported name before alias syntax and whitespace."""
    return text.strip().rstrip(";").split(" as ", 1)[0].strip().split()[0] if text.strip() else ""


def _extract_identifiers(source: str) -> tuple[str, ...]:
    """Extract searchable identifier names from source text."""
    identifiers: set[str] = set()
    for match in _IDENTIFIER_RE.finditer(source):
        identifier = match.group(0)
        normalized = identifier.lower()
        if len(identifier) < 2 or normalized in _IDENTIFIER_STOP_WORDS:
            continue
        identifiers.add(identifier)
        if len(identifiers) >= _MAX_IDENTIFIERS_PER_CHUNK:
            break
    return tuple(sorted(identifiers, key=str.lower))


def _build_module_file_map(file_paths: Sequence[str]) -> dict[str, str]:
    """Build unique dotted and slash module names for known project files."""
    candidates: dict[str, set[str]] = {}
    for file_path in file_paths:
        for module_name in _module_name_candidates(file_path):
            candidates.setdefault(module_name, set()).add(file_path)
    return {module_name: next(iter(paths)) for module_name, paths in candidates.items() if len(paths) == 1}


def _module_name_candidates(file_path: str) -> tuple[str, ...]:
    """Return module-name candidates that could resolve to a file path."""
    stem_path = _strip_supported_suffix(file_path)
    if stem_path.endswith("/__init__"):
        stem_path = stem_path[: -len("/__init__")]
    parts = [part for part in stem_path.split("/") if part and part != "."]
    candidates: set[str] = set()
    for index in range(len(parts)):
        suffix_parts = parts[index:]
        candidates.add(".".join(suffix_parts))
        candidates.add("/".join(suffix_parts))
    return tuple(sorted(candidate for candidate in candidates if candidate))


def _strip_supported_suffix(file_path: str) -> str:
    """Remove a supported source suffix from a file path when present."""
    suffix = Path(file_path).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return file_path[: -len(suffix)]
    return file_path


def _resolve_import_target(
    importer_path: str,
    module: str,
    file_paths: set[str],
    module_file_map: dict[str, str],
) -> str | None:
    """Resolve an import target to a known project file when possible."""
    if module.startswith("."):
        return _resolve_relative_import(importer_path, module, file_paths)

    for candidate in _module_lookup_candidates(module):
        target = module_file_map.get(candidate)
        if target is not None:
            return target
    return None


def _module_lookup_candidates(module: str) -> tuple[str, ...]:
    """Return exact and suffix module names to try against known files."""
    normalized = module.strip().strip(";")
    if not normalized:
        return ()
    parts = re.split(r"[/.]", normalized)
    candidates = {normalized}
    for index in range(len(parts)):
        suffix_parts = [part for part in parts[index:] if part]
        if suffix_parts:
            candidates.add(".".join(suffix_parts))
            candidates.add("/".join(suffix_parts))
    return tuple(sorted(candidates, key=lambda value: (len(value), value), reverse=True))


def _resolve_relative_import(importer_path: str, module: str, file_paths: set[str]) -> str | None:
    """Resolve a relative import path against the importing file."""
    base_dir = posixpath.dirname(importer_path)
    if module.startswith("./") or module.startswith("../"):
        base = posixpath.normpath(posixpath.join(base_dir, module))
    else:
        dot_count = len(module) - len(module.lstrip("."))
        remainder = module[dot_count:].replace(".", "/")
        base = base_dir
        for _ in range(max(dot_count - 1, 0)):
            base = posixpath.dirname(base)
        base = posixpath.normpath(posixpath.join(base, remainder))

    for candidate in _file_candidates(base):
        if candidate in file_paths:
            return candidate
    return None


def _file_candidates(base: str) -> tuple[str, ...]:
    """Return possible source file paths for an import base path."""
    normalized = base.removeprefix("./")
    candidates = [normalized]
    candidates.extend(f"{normalized}{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
    candidates.extend(f"{normalized}/index{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
    candidates.extend(f"{normalized}/__init__{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate and candidate != "."))


def _symbol_id(file_path: str, name: str) -> str:
    """Return a stable graph node id for a symbol definition."""
    return f"symbol:{file_path}:{name}"


def _module_id(module: str) -> str:
    """Return a stable graph node id for an imported module."""
    return f"module:{module}"


def _identifier_id(identifier: str) -> str:
    """Return a stable graph node id for an identifier reference."""
    return identifier_node_id(identifier)
