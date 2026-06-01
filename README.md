# whichcode

`whichcode` is a minimal tree-sitter chunking prototype for code indexing.

It currently supports Python, JavaScript, TypeScript, Go, Java, Rust, Ruby, PHP, Kotlin, Swift, C, C++, C#, Lua, Dart, Elixir, Shell, and Zig files when the corresponding tree-sitter parser is available.

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
