"""Persistence helpers for local whichcode indexes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from whichcode.bm25 import build_bm25_index
from whichcode.chunking import Chunk
from whichcode.hybrid import HybridIndex
from whichcode.scanner import scan_chunks
from whichcode.vector import DEFAULT_EMBEDDING_MODEL, EmbeddingModel, VectorIndex, build_vector_index

INDEX_DIR_NAME = ".whichcode"
CHUNKS_FILE_NAME = "chunks.jsonl"
VECTORS_FILE_NAME = "vectors.npy"
METADATA_FILE_NAME = "metadata.json"
INDEX_VERSION = 1


def load_or_build_hybrid_index(
    root: str | Path,
    model: EmbeddingModel | None = None,
    *,
    rebuild: bool = False,
) -> HybridIndex:
    """Load a persisted hybrid index or build and save one when missing."""
    root_path = _resolve_root(root)
    if not rebuild and index_exists(root_path):
        chunks, vectors = load_chunks_and_vectors(root_path)
        resolved_model = model or _load_default_model()
        return HybridIndex(
            bm25=build_bm25_index(chunks),
            vector=VectorIndex.from_embeddings(chunks, vectors, resolved_model),
        )

    chunks = tuple(scan_chunks(root_path))
    vector = build_vector_index(chunks, model=model)
    save_chunks_and_vectors(root_path, chunks, vector.vectors)
    return HybridIndex(bm25=build_bm25_index(chunks), vector=vector)


def index_exists(root: str | Path) -> bool:
    """Return whether the required persisted index files exist."""
    index_path = index_dir(root)
    return all(
        (index_path / file_name).exists()
        for file_name in (CHUNKS_FILE_NAME, VECTORS_FILE_NAME, METADATA_FILE_NAME)
    )


def save_chunks_and_vectors(
    root: str | Path,
    chunks: tuple[Chunk, ...],
    vectors: npt.NDArray[np.float32],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Persist chunks and vectors under the root .whichcode directory."""
    root_path = _resolve_root(root)
    output_dir = index_dir(root_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_chunks(output_dir / CHUNKS_FILE_NAME, chunks)
    np.save(output_dir / VECTORS_FILE_NAME, np.asarray(vectors, dtype=np.float32))
    _write_metadata(output_dir / METADATA_FILE_NAME, root_path, chunks, vectors, model_name)


def load_chunks_and_vectors(root: str | Path) -> tuple[tuple[Chunk, ...], npt.NDArray[np.float32]]:
    """Load persisted chunks and vectors from the root .whichcode directory."""
    root_path = _resolve_root(root)
    input_dir = index_dir(root_path)
    chunks = _read_chunks(input_dir / CHUNKS_FILE_NAME)
    vectors = np.load(input_dir / VECTORS_FILE_NAME).astype(np.float32, copy=False)
    if vectors.ndim != 2:
        raise ValueError("persisted vectors must be a two-dimensional array")
    if len(chunks) != vectors.shape[0]:
        raise ValueError("persisted chunks and vectors have different lengths")
    return chunks, vectors


def index_dir(root: str | Path) -> Path:
    """Return the index directory under a project root."""
    return _resolve_root(root) / INDEX_DIR_NAME


def _resolve_root(root: str | Path) -> Path:
    """Resolve and validate a project root directory."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")
    return root_path


def _load_default_model() -> EmbeddingModel:
    """Load the default embedding model without exposing vector internals."""
    from whichcode.vector import load_embedding_model

    return load_embedding_model()


def _write_chunks(path: Path, chunks: tuple[Chunk, ...]) -> None:
    """Write chunks as JSON Lines."""
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(_chunk_to_dict(chunk), ensure_ascii=False) + "\n")


def _read_chunks(path: Path) -> tuple[Chunk, ...]:
    """Read chunks from JSON Lines."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(_chunk_from_dict(json.loads(line)))
    return tuple(chunks)


def _write_metadata(
    path: Path,
    root: Path,
    chunks: tuple[Chunk, ...],
    vectors: npt.NDArray[np.float32],
    model_name: str,
) -> None:
    """Write metadata for a persisted index."""
    metadata = {
        "version": INDEX_VERSION,
        "root_path": str(root),
        "created_at": time.time(),
        "model_name": model_name,
        "chunk_count": len(chunks),
        "vector_shape": list(vectors.shape),
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Convert a chunk to a JSON-compatible dictionary."""
    return {
        "content": chunk.content,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "kind": chunk.kind,
        "name": chunk.name,
        "language": chunk.language,
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    """Create a chunk from persisted dictionary data."""
    return Chunk(
        content=str(data["content"]),
        file_path=str(data["file_path"]),
        start_line=int(data["start_line"]),
        end_line=int(data["end_line"]),
        kind=str(data.get("kind", "file")),
        name=data.get("name") if isinstance(data.get("name"), str) else None,
        language=data.get("language") if isinstance(data.get("language"), str) else None,
    )
