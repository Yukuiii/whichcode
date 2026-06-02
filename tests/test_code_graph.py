"""Tests for lightweight code graph extraction."""

from pathlib import Path

from whichcode.chunking import Chunk
from whichcode.code_graph import CodeGraph, build_code_graph, load_code_graph, save_code_graph


def test_build_code_graph_extracts_import_definitions_and_references() -> None:
    """build_code_graph should extract import, definition, and identifier reference edges."""
    chunks = (
        Chunk(
            "from auth.token import verify_token\n",
            "src/app.py",
            1,
            1,
            "module",
            language="python",
        ),
        Chunk(
            "def run(request):\n    return verify_token(request.token)\n",
            "src/app.py",
            3,
            4,
            "function",
            name="run",
            language="python",
        ),
        Chunk(
            "def verify_token(token):\n    return token\n",
            "src/auth/token.py",
            1,
            2,
            "function",
            name="verify_token",
            language="python",
        ),
    )

    graph = build_code_graph(chunks)
    nodes = {node.id: node for node in graph.nodes}

    assert "file:src/app.py" in nodes
    assert _has_edge(graph, "file:src/app.py", "module:auth.token", "imports")
    assert _has_edge(graph, "file:src/app.py", "file:src/auth/token.py", "imports")
    assert _has_edge(
        graph,
        "chunk:src/auth/token.py:1:2:function:verify_token",
        "symbol:src/auth/token.py:verify_token",
        "defines",
    )
    assert _has_edge(graph, "chunk:src/app.py:3:4:function:run", "identifier:verify_token", "references")


def test_build_code_graph_resolves_relative_javascript_imports() -> None:
    """build_code_graph should resolve relative JavaScript imports to known source files."""
    chunks = (
        Chunk(
            'import { helper } from "./utils";\n',
            "src/app.ts",
            1,
            1,
            "module",
            language="typescript",
        ),
        Chunk(
            "export function helper() { return 1; }\n",
            "src/utils.ts",
            1,
            1,
            "function",
            name="helper",
            language="typescript",
        ),
    )

    graph = build_code_graph(chunks)

    assert _has_edge(graph, "file:src/app.ts", "module:./utils", "imports")
    assert _has_edge(graph, "file:src/app.ts", "file:src/utils.ts", "imports")


def test_code_graph_persistence_round_trips_sqlite(tmp_path: Path) -> None:
    """save_code_graph and load_code_graph should round-trip graph data through SQLite."""
    graph = build_code_graph(
        (
            Chunk("def run():\n    return value\n", "app.py", 1, 2, "function", name="run", language="python"),
        )
    )
    path = tmp_path / "graph.sqlite"

    save_code_graph(path, graph)
    loaded = load_code_graph(path)

    assert loaded == graph


def _has_edge(graph: CodeGraph, source_id: str, target_id: str, kind: str) -> bool:
    """Return whether the graph contains one expected edge."""
    return any(
        edge.source_id == source_id and edge.target_id == target_id and edge.kind == kind for edge in graph.edges
    )
