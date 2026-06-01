"""Tests for the whichcode command-line interface."""

from whichcode.chunking import Chunk
from whichcode.cli import main
from whichcode.types import SearchResult


class FakeIndex:
    """Search index stub for CLI output tests."""

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return one deterministic result."""
        return [
            SearchResult(
                chunk=Chunk(
                    content="def run():\n    return 1\n",
                    file_path="src/app.py",
                    start_line=1,
                    end_line=2,
                    kind="function",
                    name="run",
                    language="python",
                ),
                score=0.5,
            )
        ]


def test_main_prints_json_results(monkeypatch, capsys, tmp_path) -> None:
    """main should print formatted results that include chunk content."""
    monkeypatch.setattr("whichcode.cli.load_or_build_hybrid_index", lambda path, rebuild=False: FakeIndex())

    main([str(tmp_path), "run"])

    output = capsys.readouterr().out
    assert '"query": "run"' in output
    assert '"content": "def run():\\n    return 1\\n"' in output
    assert '"location": "src/app.py:1-2"' in output
