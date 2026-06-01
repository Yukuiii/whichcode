# whichcode

`whichcode` is a minimal tree-sitter chunking prototype for code indexing.

It currently supports Python, JavaScript, TypeScript, Go, Java, Rust, Ruby, PHP, Kotlin, Swift, C, C++, C#, Lua, Dart, Elixir, Shell, and Zig files when the corresponding tree-sitter parser is available. Markdown files are indexed as whole-file chunks.

## Development

```bash
uv sync
uv run pytest
```

## Usage

```python
from whichcode import scan_chunks

chunks = scan_chunks(".")
```

Chunks are emitted as `module`, `class`, `function`, `method`, or `file` records.

```python
from whichcode import build_bm25_index, scan_chunks

chunks = scan_chunks(".")
index = build_bm25_index(chunks)
results = index.search("authentication token", top_k=5)
```

```python
from whichcode import build_vector_index, scan_chunks

chunks = scan_chunks(".")
index = build_vector_index(chunks)
results = index.search("how authentication is handled", top_k=5)
```

```python
from whichcode import build_hybrid_index, scan_chunks

chunks = scan_chunks(".")
index = build_hybrid_index(chunks)
results = index.search("authenticate_token behavior", top_k=5)
```
