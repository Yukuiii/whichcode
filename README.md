# whichcode

`whichcode` is a minimal tree-sitter chunking prototype for code indexing.

It currently scans Python, JavaScript, TypeScript, Go, Java, Rust, Ruby, PHP, Kotlin, Swift, C, C++, C#,
Lua, Dart, Elixir, Shell, and Zig files when the corresponding tree-sitter parser is available. Markdown
files are not included in the default project scan.

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
results = index.search("authenticate_token behavior")
```

```python
from whichcode import format_results

payload = format_results("authenticate_token behavior", results)
```

The CLI builds a user-level index under `~/.whichcode/chunk/<sha256(resolved-project-path)>` on first use
and reuses it on later runs.

```bash
uv run whichcode . "how authentication is handled"
```

The CLI retrieves 50 hybrid candidates from the user-level `.whichcode` index, applies lightweight code-search
ranking rules, then returns the best 10 chunks. Query-time ranking is deterministic and does not load any
separate ranking model.

```bash
uv run whichcode . "how authentication is handled" --rebuild
```
