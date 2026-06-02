"""Tests for persisted whichcode indexes."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from whichcode.chunking import Chunk
from whichcode.paths import project_index_key
from whichcode.storage import (
    GRAPH_FILE_NAME,
    index_dir,
    index_exists,
    load_chunks_and_vectors,
    load_code_graph,
    load_or_build_hybrid_index,
)


class ConstantEmbeddingModel:
    """Embedding model stub that returns the same vector for every text."""

    def encode(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        """Encode each text as a single-dimensional vector."""
        return [[1.0] for _ in texts]


class RecordingEmbeddingModel:
    """Embedding model stub that records embedded texts."""

    def __init__(self) -> None:
        """Initialize the recorded text list."""
        self.texts: list[str] = []

    def encode(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        """Record each text and return deterministic vectors."""
        self.texts.extend(texts)
        return [[1.0] for _ in texts]


@pytest.fixture(autouse=True)
def isolated_whichcode_home(monkeypatch, tmp_path) -> None:
    """Redirect user-level whichcode files to each test's temp home."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_load_or_build_hybrid_index_persists_chunks_and_vectors(monkeypatch, tmp_path) -> None:
    """load_or_build_hybrid_index should save chunks and vectors under ~/.whichcode."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    _patch_scan_chunks(monkeypatch, [_chunk("def run():\n    return 1\n", "app.py", "run")])

    index = load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    chunks, vectors = load_chunks_and_vectors(tmp_path)
    expected_index_dir = Path.home() / ".whichcode" / "chunk" / project_index_key(tmp_path)

    assert index_exists(tmp_path)
    assert index.search("run", top_k=1)
    assert index_dir(tmp_path) == expected_index_dir
    assert (expected_index_dir / "chunks.jsonl").exists()
    assert (expected_index_dir / GRAPH_FILE_NAME).exists()
    assert not (tmp_path / ".whichcode").exists()
    assert len(chunks) == 1
    assert chunks[0].name == "run"
    assert vectors.shape == (1, 1)
    assert vectors.dtype == np.float32


def test_load_or_build_hybrid_index_persists_code_graph(monkeypatch, tmp_path) -> None:
    """load_or_build_hybrid_index should persist import, definition, and reference graph data."""
    (tmp_path / "auth.py").write_text("def verify(token):\n    return token\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "import auth\n\n"
        "def run():\n"
        "    return auth\n",
        encoding="utf-8",
    )
    _patch_scan_chunks(
        monkeypatch,
        [
            Chunk("import auth\n", "app.py", 1, 1, "module", language="python"),
            _chunk("def run():\n    return auth\n", "app.py", "run", start_line=3, end_line=4),
            _chunk("def verify(token):\n    return token\n", "auth.py", "verify"),
        ],
    )

    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    graph = load_code_graph(tmp_path)

    assert _has_edge(graph, "file:app.py", "module:auth", "imports")
    assert _has_edge(graph, "file:app.py", "file:auth.py", "imports")
    assert _has_edge(graph, "chunk:auth.py:1:2:function:verify", "symbol:auth.py:verify", "defines")
    assert _has_edge(graph, "chunk:app.py:3:4:function:run", "identifier:auth", "references")


def test_index_exists_does_not_require_code_graph(monkeypatch, tmp_path) -> None:
    """index_exists should allow old indexes without graph files until graph data is requested."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    _patch_scan_chunks(monkeypatch, [_chunk("def run():\n    return 1\n", "app.py", "run")])
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    graph_path = index_dir(tmp_path) / GRAPH_FILE_NAME
    graph_path.unlink()

    assert index_exists(tmp_path)
    assert not graph_path.exists()

    graph = load_code_graph(tmp_path)

    assert graph_path.exists()
    assert graph.nodes


def test_index_exists_ignores_metadata_version_field(monkeypatch, tmp_path) -> None:
    """index_exists should rely on explicit rebuilds instead of metadata version fields."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    _patch_scan_chunks(monkeypatch, [_chunk("def run():\n    return 1\n", "app.py", "run")])
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    metadata_path = index_dir(tmp_path) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert index_exists(tmp_path)


def test_index_exists_rejects_metadata_for_a_different_project_path(monkeypatch, tmp_path) -> None:
    """index_exists should reject global index metadata with a mismatched project hash."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    _patch_scan_chunks(monkeypatch, [_chunk("def run():\n    return 1\n", "app.py", "run")])
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    metadata_path = index_dir(tmp_path) / "metadata.json"
    metadata = metadata_path.read_text(encoding="utf-8").replace(project_index_key(tmp_path), "0" * 64)
    metadata_path.write_text(metadata, encoding="utf-8")

    assert not index_exists(tmp_path)


def test_load_or_build_hybrid_index_reuses_existing_files(monkeypatch, tmp_path) -> None:
    """load_or_build_hybrid_index should reuse existing files unless rebuild is requested."""
    source = tmp_path / "app.py"
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    _patch_scan_chunks(monkeypatch, _scan_app_chunk_from_file)
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    source.write_text("def changed():\n    return 2\n", encoding="utf-8")

    cached_chunks, _ = load_chunks_and_vectors(tmp_path)
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    rebuilt = load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel(), rebuild=True)

    rebuilt_chunks, _ = load_chunks_and_vectors(tmp_path)
    assert cached_chunks[0].name == "run"
    assert rebuilt_chunks[0].name == "changed"
    assert rebuilt.search("changed", top_k=1)


def test_load_or_build_hybrid_index_uses_indexed_vector_text(monkeypatch, tmp_path) -> None:
    """load_or_build_hybrid_index should build embedding text from code metadata only."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    _patch_scan_chunks(monkeypatch, [_chunk("def run():\n    return 1\n", "app.py", "run")])
    embedding_model = RecordingEmbeddingModel()

    load_or_build_hybrid_index(tmp_path, model=embedding_model, rebuild=True)

    assert embedding_model.texts
    assert "name: run" in embedding_model.texts[0]
    assert "def run()" in embedding_model.texts[0]


def _has_edge(graph, source_id: str, target_id: str, kind: str) -> bool:
    """Return whether the graph contains one expected edge."""
    return any(
        edge.source_id == source_id and edge.target_id == target_id and edge.kind == kind for edge in graph.edges
    )


def _patch_scan_chunks(monkeypatch, chunks_or_factory) -> None:
    """Patch storage scanning so persistence tests do not depend on parser startup."""
    if callable(chunks_or_factory):
        monkeypatch.setattr("whichcode.storage.scan_chunks", chunks_or_factory)
        return
    monkeypatch.setattr("whichcode.storage.scan_chunks", lambda root: list(chunks_or_factory))


def _scan_app_chunk_from_file(root: Path) -> list[Chunk]:
    """Return a synthetic app chunk matching the current test file content."""
    source = (root / "app.py").read_text(encoding="utf-8")
    name = "changed" if "def changed" in source else "run"
    return [_chunk(source, "app.py", name)]


def _chunk(
    content: str,
    file_path: str,
    name: str,
    *,
    start_line: int = 1,
    end_line: int = 2,
) -> Chunk:
    """Create a Python function chunk for storage tests."""
    return Chunk(content, file_path, start_line, end_line, "function", name=name, language="python")
