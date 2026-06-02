"""Tests for BM25 chunk search."""

from whichcode.bm25 import build_bm25_index, enrich_for_bm25, extract_search_terms, split_identifier, tokenize
from whichcode.chunking import Chunk
from whichcode.query_aliases import query_aliases


def test_tokenize_splits_compound_identifiers() -> None:
    """tokenize should preserve originals and add useful identifier parts."""
    tokens = tokenize("getHTTPResponse my_func simple")

    assert "gethttpresponse" in tokens
    assert "get" in tokens
    assert "http" in tokens
    assert "response" in tokens
    assert "my_func" in tokens
    assert "my" in tokens
    assert "func" in tokens
    assert "simple" in tokens


def test_split_identifier_keeps_simple_identifier_once() -> None:
    """split_identifier should avoid duplicate tokens for simple words."""
    assert split_identifier("simple") == ["simple"]


def test_extract_search_terms_filters_noise_and_adds_stems() -> None:
    """extract_search_terms should keep code terms and expand common suffixes."""
    terms = extract_search_terms("How are indexes cached by CacheBuilder?")

    assert "how" not in terms
    assert "indexes" in terms
    assert "index" in terms
    assert "cached" in terms
    assert "cache" in terms
    assert "cachebuilder" in terms
    assert "builder" in terms


def test_extract_search_terms_expands_query_aliases() -> None:
    """extract_search_terms should expand common code abbreviations at query time."""
    terms = extract_search_terms("ctx db req cfg opts fn cmd tx hdr spec", include_stop_words=True)

    assert "ctx" in terms
    assert "context" in terms
    assert "db" in terms
    assert "database" in terms
    assert "req" in terms
    assert "request" in terms
    assert "cfg" in terms
    assert "config" in terms
    assert "configuration" in terms
    assert "opts" in terms
    assert "option" in terms
    assert "options" in terms
    assert "fn" in terms
    assert "function" in terms
    assert "cmd" in terms
    assert "command" in terms
    assert "tx" in terms
    assert "transaction" in terms
    assert "hdr" in terms
    assert "header" in terms
    assert "headers" in terms
    assert "spec" in terms
    assert "test" in terms


def test_query_aliases_returns_aliases_from_dedicated_module() -> None:
    """query_aliases should expose the shared alias table outside BM25."""
    assert query_aliases("ctx") == ("context",)
    assert "cfg" in query_aliases("configuration")
    assert "endpoint" in query_aliases("api")
    assert "repository" in query_aliases("dao")
    assert "filesystem" in query_aliases("fs")
    assert "serde" in query_aliases("deserialize")
    assert "validator" in query_aliases("validation")


def test_enrich_for_bm25_adds_path_and_chunk_metadata() -> None:
    """enrich_for_bm25 should include path and chunk metadata for exact matches."""
    chunk = Chunk(
        content="return token",
        file_path="src/auth/session_store.py",
        start_line=1,
        end_line=1,
        kind="function",
        name="load_session",
        language="python",
    )

    enriched = enrich_for_bm25(chunk)

    assert "session_store session_store" in enriched
    assert "src auth" in enriched
    assert "function load_session python" in enriched


def test_bm25_search_ranks_matching_chunk_first() -> None:
    """BM25Index.search should rank the most lexically relevant chunk first."""
    chunks = [
        Chunk("def authenticate_token(token):\n    return verify(token)\n", "src/auth.py", 1, 2, "function"),
        Chunk("def render_template(context):\n    return html\n", "src/view.py", 1, 2, "function"),
    ]
    index = build_bm25_index(chunks)

    results = index.search("authenticate token", top_k=2)

    assert results
    assert results[0].chunk.file_path == "src/auth.py"
    assert results[0].score > 0


def test_bm25_search_uses_file_path_terms() -> None:
    """BM25Index.search should find chunks by file path tokens."""
    chunks = [
        Chunk("value = read()", "src/cache/store.py", 1, 1, "module"),
        Chunk("value = read()", "src/http/client.py", 1, 1, "module"),
    ]
    index = build_bm25_index(chunks)

    results = index.search("cache", top_k=2)

    assert results
    assert results[0].chunk.file_path == "src/cache/store.py"


def test_bm25_search_uses_query_aliases() -> None:
    """BM25Index.search should expand short query aliases before scoring."""
    chunks = [
        Chunk("def current_context():\n    return context\n", "src/context.py", 1, 2, "function"),
        Chunk("def render_template():\n    return html\n", "src/view.py", 1, 2, "function"),
    ]
    index = build_bm25_index(chunks)

    results = index.search("ctx", top_k=1)

    assert results
    assert results[0].chunk.file_path == "src/context.py"


def test_bm25_search_uses_structural_query_aliases() -> None:
    """BM25Index.search should use structural aliases for source-code vocabulary."""
    chunks = [
        Chunk("function buildCommand() { return run(); }", "src/commands/build.ts", 1, 1, "function"),
        Chunk("class View { render() {} }", "src/view.ts", 1, 1, "class"),
    ]
    index = build_bm25_index(chunks)

    results = index.search("fn cmd", top_k=1)

    assert results
    assert results[0].chunk.file_path == "src/commands/build.ts"


def test_bm25_search_exposes_field_specific_scores() -> None:
    """BM25Index should score content, name, and path fields independently."""
    content_chunk = Chunk("def build_cache(): return value", "src/app.py", 1, 1, "function", name="build")
    metadata_chunk = Chunk("return value", "src/click/types.py", 1, 1, "function", name="ParamType")
    index = build_bm25_index([content_chunk, metadata_chunk])

    assert index.search_content("build cache", top_k=1)[0].chunk == content_chunk
    assert index.search_name("ParamType", top_k=1)[0].chunk == metadata_chunk
    assert index.search_path("type", top_k=1)[0].chunk == metadata_chunk
    assert index.search("ParamType", top_k=1)[0].chunk == metadata_chunk


def test_bm25_search_returns_empty_for_blank_or_missing_queries() -> None:
    """BM25Index.search should return no results for empty or unmatched queries."""
    index = build_bm25_index([Chunk("def run(): pass", "src/app.py", 1, 1, "function")])

    assert index.search("   ") == []
    assert index.search("zzzznonexistentterm") == []
    assert build_bm25_index([]).search("run") == []
