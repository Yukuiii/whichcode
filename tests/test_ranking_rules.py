"""Tests for deterministic ranking rules."""

from whichcode.chunking import Chunk
from whichcode.ranking_rules import (
    chunk_matches_query_filters,
    chunk_relevance_factor,
    is_generated_file,
    is_test_file,
    name_match_bonus,
    parse_query,
    path_prior,
)


def test_path_prior_demotes_lower_signal_paths() -> None:
    """path_prior should rank public source files above lower-signal paths."""
    assert path_prior("src/public.py") == 1.0
    assert path_prior("tests/test_public.py") < path_prior("src/public.py")
    assert path_prior("src/_private.py") < path_prior("src/public.py")
    assert path_prior("examples/demo.py") < path_prior("src/public.py")


def test_path_prior_keeps_tests_when_query_asks_for_tests() -> None:
    """path_prior should not demote test files for explicit test queries."""
    test_path = "tests/test_public.py"

    assert path_prior(test_path, "public tests") > path_prior(test_path, "public implementation")


def test_path_prior_demotes_generated_files() -> None:
    """path_prior should rank generated files below handwritten source."""
    assert is_generated_file("api/user.pb.go")
    assert path_prior("api/user.pb.go") < path_prior("api/user.go")


def test_is_test_file_handles_common_language_patterns() -> None:
    """is_test_file should cover common filename and source-set conventions."""
    assert is_test_file("pkg/cache/cache_test.go")
    assert is_test_file("src/jvmTest/kotlin/CacheTest.kt")
    assert not is_test_file("src/latest.kt")


def test_parse_query_extracts_structured_filters() -> None:
    """parse_query should separate filters from the free-text query."""
    parsed = parse_query('kind:function,method path:"src/auth" lang:python authenticate token')

    assert parsed.text == "authenticate token"
    assert parsed.kinds == ("function", "method")
    assert parsed.languages == ("python",)
    assert parsed.path_filters == ("src/auth",)


def test_chunk_matches_query_filters_uses_chunk_metadata() -> None:
    """chunk_matches_query_filters should enforce kind, language, path, and name filters."""
    chunk = Chunk(
        content="def authenticate_token(): pass",
        file_path="src/auth/session.py",
        start_line=1,
        end_line=1,
        kind="function",
        name="authenticate_token",
        language="python",
    )

    assert chunk_matches_query_filters(chunk, parse_query("kind:function path:auth name:token lang:python"))
    assert not chunk_matches_query_filters(chunk, parse_query("kind:class path:auth"))


def test_name_match_bonus_prefers_exact_names_over_prefixes() -> None:
    """name_match_bonus should rank exact symbol names above prefix matches."""
    exact = name_match_bonus("CacheBuilder", "CacheBuilder")
    prefix = name_match_bonus("CacheBuilderFactory", "CacheBuilder")
    substring = name_match_bonus("DefaultCacheBuilderFactory", "CacheBuilder")

    assert exact > prefix > substring > 0


def test_chunk_relevance_factor_rewards_multi_term_cooccurrence() -> None:
    """chunk_relevance_factor should boost chunks matching multiple query concepts."""
    direct_chunk = Chunk(
        content="def execute_search_request(shard): return shard",
        file_path="src/search/request.py",
        start_line=1,
        end_line=1,
        kind="function",
        name="execute_search_request",
        language="python",
    )
    generic_chunk = Chunk(
        content="def execute(value): return value",
        file_path="src/utils/execution.py",
        start_line=1,
        end_line=1,
        kind="function",
        name="execute",
        language="python",
    )
    query = parse_query("search execution request shard")

    assert chunk_relevance_factor(direct_chunk, query) > chunk_relevance_factor(generic_chunk, query)
