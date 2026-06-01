"""Public entry points for the whichcode package."""

from whichcode.bm25 import BM25Index, build_bm25_index
from whichcode.chunking import Chunk, chunk_source
from whichcode.formatting import format_results
from whichcode.hybrid import HybridIndex, build_hybrid_index
from whichcode.scanner import scan_chunks
from whichcode.types import SearchResult
from whichcode.vector import VectorIndex, build_vector_index, load_embedding_model

__all__ = [
    "BM25Index",
    "Chunk",
    "HybridIndex",
    "SearchResult",
    "VectorIndex",
    "build_bm25_index",
    "build_hybrid_index",
    "build_vector_index",
    "chunk_source",
    "format_results",
    "load_embedding_model",
    "scan_chunks",
]
