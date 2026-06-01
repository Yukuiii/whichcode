"""Tests for hybrid chunk search."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from whichcode.bm25 import build_bm25_index
from whichcode.chunking import Chunk
from whichcode.hybrid import HybridIndex, build_hybrid_index
from whichcode.vector import VectorIndex


class FakeEmbeddingModel:
    """Embedding model stub that maps text to configured vectors."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        """Store deterministic vectors keyed by text."""
        self.mapping = mapping

    def encode(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        """Encode texts by exact lookup."""
        return [self.mapping[text] for text in texts]


class RecordingSearchIndex:
    """Search index stub that records requested candidate counts."""

    def __init__(self) -> None:
        """Initialize the recorded top_k value."""
        self.requested_top_k: int | None = None

    def search(self, query: str, top_k: int = 5) -> list:
        """Record top_k and return no results."""
        self.requested_top_k = top_k
        return []


def test_hybrid_search_merges_bm25_and_vector_results() -> None:
    """HybridIndex.search should return candidates from both retrieval paths."""
    chunks = [
        Chunk("def authenticate_token(token): return verify(token)", "src/auth.py", 1, 1, "function"),
        Chunk("def render_template(context): return html", "src/view.py", 1, 1, "function"),
    ]
    model = FakeEmbeddingModel({"authenticate_token": [0.0, 1.0]})
    vector = VectorIndex.from_embeddings(chunks, [[0.0, 1.0], [1.0, 0.0]], model)
    index = HybridIndex(bm25=build_bm25_index(chunks), vector=vector)

    results = index.search("authenticate_token", top_k=2, alpha=0.5)

    assert {result.chunk.file_path for result in results} == {"src/auth.py", "src/view.py"}


def test_hybrid_search_alpha_controls_weighting() -> None:
    """HybridIndex.search should let semantic or lexical ranking dominate."""
    chunks = [
        Chunk("def authenticate_token(token): return verify(token)", "src/auth.py", 1, 1, "function"),
        Chunk("def render_template(context): return html", "src/view.py", 1, 1, "function"),
    ]
    model = FakeEmbeddingModel({"authenticate_token": [0.0, 1.0]})
    vector = VectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]], model)
    bm25 = build_bm25_index(chunks)
    index = HybridIndex(bm25=bm25, vector=vector)

    lexical_first = index.search("authenticate_token", top_k=2, alpha=0.0)
    semantic_first = index.search("authenticate_token", top_k=2, alpha=1.0)

    assert lexical_first[0].chunk.file_path == "src/auth.py"
    assert semantic_first[0].chunk.file_path == "src/view.py"


def test_hybrid_search_penalizes_lower_signal_paths() -> None:
    """HybridIndex.search should demote tests when path penalties are enabled."""
    chunks = [
        Chunk("def impl(): pass", "src/impl.py", 1, 1, "function"),
        Chunk("def impl(): pass", "tests/test_impl.py", 1, 1, "function"),
    ]
    model = FakeEmbeddingModel({"implementation": [0.0, 1.0]})
    vector = VectorIndex.from_embeddings(chunks, [[1.0, 0.0], [0.0, 1.0]], model)
    index = HybridIndex(bm25=build_bm25_index(chunks), vector=vector)

    penalized = index.search("implementation", top_k=2, alpha=1.0)
    raw = index.search("implementation", top_k=2, alpha=1.0, penalize_paths=False)

    assert penalized[0].chunk.file_path == "src/impl.py"
    assert raw[0].chunk.file_path == "tests/test_impl.py"


def test_build_hybrid_index_builds_both_indexes() -> None:
    """build_hybrid_index should build BM25 and vector indexes from chunks."""
    chunk = Chunk("def run(): pass", "src/app.py", 1, 1, "function")
    embedding_text = "path: src/app.py\nkind: function\n\ndef run(): pass"
    model = FakeEmbeddingModel({embedding_text: [1.0], "run": [1.0]})

    index = build_hybrid_index([chunk], model=model)

    assert index.search("run", top_k=1)[0].chunk == chunk


def test_hybrid_search_rejects_invalid_alpha() -> None:
    """HybridIndex.search should validate alpha to keep scores interpretable."""
    chunk = Chunk("def run(): pass", "src/app.py", 1, 1, "function")
    model = FakeEmbeddingModel({"run": [1.0]})
    vector = VectorIndex.from_embeddings([chunk], np.array([[1.0]], dtype=np.float32), model)
    index = HybridIndex(bm25=build_bm25_index([chunk]), vector=vector)

    try:
        index.search("run", alpha=1.5)
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hybrid_search_overfetches_at_least_twenty_five_candidates() -> None:
    """HybridIndex.search should keep enough candidates even for top_k=1."""
    bm25 = RecordingSearchIndex()
    vector = RecordingSearchIndex()
    index = HybridIndex(bm25=bm25, vector=vector)

    assert index.search("query", top_k=1) == []
    assert bm25.requested_top_k == 25
    assert vector.requested_top_k == 25
