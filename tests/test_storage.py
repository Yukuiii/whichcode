"""Tests for persisted whichcode indexes."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from whichcode.chunking import Chunk
from whichcode.paths import project_index_key
from whichcode.storage import INDEX_VERSION, index_dir, index_exists, load_chunks_and_vectors, load_or_build_hybrid_index


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


def test_load_or_build_hybrid_index_persists_chunks_and_vectors(tmp_path) -> None:
    """load_or_build_hybrid_index should save chunks and vectors under ~/.whichcode."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    index = load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    chunks, vectors = load_chunks_and_vectors(tmp_path)
    expected_index_dir = Path.home() / ".whichcode" / "chunk" / project_index_key(tmp_path)

    assert index_exists(tmp_path)
    assert index.search("run", top_k=1)
    assert index_dir(tmp_path) == expected_index_dir
    assert (expected_index_dir / "chunks.jsonl").exists()
    assert not (tmp_path / ".whichcode").exists()
    assert len(chunks) == 1
    assert chunks[0].name == "run"
    assert vectors.shape == (1, 1)
    assert vectors.dtype == np.float32


def test_index_exists_rejects_stale_metadata_version(tmp_path) -> None:
    """index_exists should reject stale persisted indexes after format changes."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    metadata_path = index_dir(tmp_path) / "metadata.json"
    metadata = metadata_path.read_text(encoding="utf-8").replace(f'"version": {INDEX_VERSION}', '"version": 1')
    metadata_path.write_text(metadata, encoding="utf-8")

    assert not index_exists(tmp_path)


def test_index_exists_rejects_metadata_for_a_different_project_path(tmp_path) -> None:
    """index_exists should reject global index metadata with a mismatched project hash."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    metadata_path = index_dir(tmp_path) / "metadata.json"
    metadata = metadata_path.read_text(encoding="utf-8").replace(project_index_key(tmp_path), "0" * 64)
    metadata_path.write_text(metadata, encoding="utf-8")

    assert not index_exists(tmp_path)


def test_load_or_build_hybrid_index_reuses_existing_files(tmp_path) -> None:
    """load_or_build_hybrid_index should reuse existing files unless rebuild is requested."""
    source = tmp_path / "app.py"
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    source.write_text("def changed():\n    return 2\n", encoding="utf-8")

    cached_chunks, _ = load_chunks_and_vectors(tmp_path)
    load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel())
    rebuilt = load_or_build_hybrid_index(tmp_path, model=ConstantEmbeddingModel(), rebuild=True)

    rebuilt_chunks, _ = load_chunks_and_vectors(tmp_path)
    assert cached_chunks[0].name == "run"
    assert rebuilt_chunks[0].name == "changed"
    assert rebuilt.search("changed", top_k=1)


def test_load_or_build_hybrid_index_uses_indexed_vector_text(tmp_path) -> None:
    """load_or_build_hybrid_index should build embedding text from code metadata only."""
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    embedding_model = RecordingEmbeddingModel()

    load_or_build_hybrid_index(tmp_path, model=embedding_model, rebuild=True)

    assert embedding_model.texts
    assert "name: run" in embedding_model.texts[0]
    assert "def run()" in embedding_model.texts[0]
