"""Tests for vector chunk search."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from whichcode.chunking import Chunk
from whichcode.vector import VectorIndex, chunk_to_embedding_text, embed_texts


class FakeEmbeddingModel:
    """Embedding model stub that maps text to configured vectors."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        """Store deterministic vectors keyed by text."""
        self.mapping = mapping

    def encode(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        """Encode texts by exact lookup."""
        return [self.mapping[text] for text in texts]


def test_chunk_to_embedding_text_includes_metadata() -> None:
    """chunk_to_embedding_text should include location metadata and content."""
    chunk = Chunk(
        content="def authenticate(): pass",
        file_path="src/auth.py",
        start_line=1,
        end_line=1,
        kind="function",
        name="authenticate",
        language="python",
    )

    text = chunk_to_embedding_text(chunk)

    assert "path: src/auth.py" in text
    assert "kind: function" in text
    assert "name: authenticate" in text
    assert "language: python" in text
    assert "def authenticate(): pass" in text


def test_chunk_to_embedding_text_ignores_summary_when_present() -> None:
    """chunk_to_embedding_text should not use deprecated summary metadata."""
    chunk = Chunk(
        content="def build(): pass",
        file_path="src/build.py",
        start_line=1,
        end_line=1,
        kind="function",
        name="build",
        language="python",
        summary='{"purpose":"Builds the index.","key_terms":["indexing"]}',
    )

    text = chunk_to_embedding_text(chunk)

    assert "summary:" not in text
    assert "def build(): pass" in text


def test_embed_texts_returns_float32_matrix() -> None:
    """embed_texts should normalize model output shape and dtype."""
    model = FakeEmbeddingModel({"hello": [1.0, 2.0]})

    vectors = embed_texts(model, ["hello"])

    assert vectors.dtype == np.float32
    assert vectors.shape == (1, 2)


def test_vector_index_search_ranks_by_cosine_similarity() -> None:
    """VectorIndex.search should rank chunks by query cosine similarity."""
    chunks = [
        Chunk("def authenticate(): pass", "src/auth.py", 1, 1, "function"),
        Chunk("def render(): pass", "src/view.py", 1, 1, "function"),
    ]
    model = FakeEmbeddingModel({"login flow": [1.0, 0.0]})
    index = VectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]], model)

    results = index.search("login flow", top_k=2)

    assert [result.chunk.file_path for result in results] == ["src/auth.py", "src/view.py"]
    assert results[0].score > results[1].score


def test_vector_index_validates_embedding_shape() -> None:
    """VectorIndex.from_embeddings should reject mismatched chunk and vector counts."""
    chunks = [Chunk("def run(): pass", "src/app.py", 1, 1, "function")]
    model = FakeEmbeddingModel({})

    try:
        VectorIndex.from_embeddings(chunks, [[1.0], [0.0]], model)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_vector_index_returns_empty_for_blank_query_or_empty_index() -> None:
    """VectorIndex.search should return no results for blank queries or empty indexes."""
    model = FakeEmbeddingModel({"query": [1.0]})
    index = VectorIndex.from_embeddings([], np.empty((0, 1), dtype=np.float32), model)

    assert index.search("query") == []

    non_empty = VectorIndex.from_embeddings([Chunk("x", "x.py", 1, 1)], [[1.0]], model)
    assert non_empty.search("  ") == []
