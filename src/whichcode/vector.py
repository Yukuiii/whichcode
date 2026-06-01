"""Vector indexing helpers for semantic chunk search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from whichcode.chunking import Chunk
from whichcode.types import SearchResult

DEFAULT_EMBEDDING_MODEL = "minishlab/potion-code-16M"
_EPSILON = 1e-12


class EmbeddingModel(Protocol):
    """Defines the minimal model interface needed for text embeddings."""

    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any:
        """Encode text strings into a two-dimensional vector array."""
        ...


@dataclass(frozen=True, slots=True)
class VectorIndex:
    """Stores normalized chunk vectors and ranks them by cosine similarity."""

    chunks: tuple[Chunk, ...]
    vectors: npt.NDArray[np.float32]
    model: EmbeddingModel

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], model: EmbeddingModel | None = None) -> VectorIndex:
        """Build a vector index by embedding chunks with a model."""
        resolved_model = model or load_embedding_model()
        resolved_chunks = tuple(chunks)
        embeddings = embed_chunks(resolved_model, resolved_chunks)
        return cls.from_embeddings(resolved_chunks, embeddings, resolved_model)

    @classmethod
    def from_embeddings(
        cls,
        chunks: Sequence[Chunk],
        embeddings: npt.ArrayLike,
        model: EmbeddingModel,
    ) -> VectorIndex:
        """Build a vector index from precomputed embeddings."""
        resolved_chunks = tuple(chunks)
        vectors = _normalize_vectors(np.asarray(embeddings, dtype=np.float32))
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if len(resolved_chunks) != vectors.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")
        return cls(chunks=resolved_chunks, vectors=vectors, model=model)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked by semantic similarity to a text query."""
        if top_k < 1 or not self.chunks or not query.strip():
            return []
        query_vector = embed_texts(self.model, [query])
        return self.search_vector(query_vector[0], top_k=top_k)

    def search_vector(self, query_vector: npt.ArrayLike, top_k: int = 5) -> list[SearchResult]:
        """Return chunks ranked by cosine similarity to a query vector."""
        if top_k < 1 or not self.chunks:
            return []
        vector = _normalize_vectors(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        if vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("query vector dimension does not match index dimension")
        scores = self.vectors @ vector
        indices = np.argsort(-scores)[:top_k]
        return [SearchResult(chunk=self.chunks[index], score=float(scores[index])) for index in indices]


@cache
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingModel:
    """Load and cache the default static embedding model."""
    from model2vec import StaticModel

    return StaticModel.from_pretrained(model_name, force_download=False)


def build_vector_index(chunks: Sequence[Chunk], model: EmbeddingModel | None = None) -> VectorIndex:
    """Build a searchable vector index from chunks."""
    return VectorIndex.from_chunks(chunks, model=model)


def embed_chunks(model: EmbeddingModel, chunks: Sequence[Chunk]) -> npt.NDArray[np.float32]:
    """Embed chunks using the text representation selected for vector search."""
    return embed_texts(model, [chunk_to_embedding_text(chunk) for chunk in chunks])


def embed_texts(model: EmbeddingModel, texts: Sequence[str]) -> npt.NDArray[np.float32]:
    """Embed raw texts with a model and return a float32 matrix."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    try:
        encoded = model.encode(texts, use_multiprocessing=False)
    except TypeError:
        encoded = model.encode(texts)
    vectors = np.asarray(encoded, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("model.encode must return a two-dimensional array")
    return vectors


def chunk_to_embedding_text(chunk: Chunk) -> str:
    """Create the text sent to the embedding model for a chunk."""
    metadata = [
        f"path: {chunk.file_path}",
        f"kind: {chunk.kind}",
    ]
    if chunk.name:
        metadata.append(f"name: {chunk.name}")
    if chunk.language:
        metadata.append(f"language: {chunk.language}")
    return "\n".join([*metadata, "", chunk.content])


def _normalize_vectors(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return row-normalized vectors for cosine similarity."""
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, _EPSILON)
