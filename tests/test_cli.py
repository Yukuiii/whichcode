"""Tests for the whichcode command-line interface."""

from whichcode.chunking import Chunk
from whichcode.cli import main
from whichcode.types import SearchResult


class FakeReranker:
    """Reranker stub passed through the CLI search call."""


class FakeIndex:
    """Search index stub for CLI output tests."""

    def __init__(self) -> None:
        """Initialize captured search arguments."""
        self.query: str | None = None
        self.top_k: int | None = None
        self.reranker = None

    def search(self, query: str, top_k: int = 5, reranker=None) -> list[SearchResult]:
        """Return one deterministic result."""
        self.query = query
        self.top_k = top_k
        self.reranker = reranker
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
    captured = {}
    fake_index = FakeIndex()
    fake_reranker = FakeReranker()

    def fake_load_or_build_hybrid_index(path, **kwargs):
        """Record CLI index construction arguments and return a fake index."""
        captured["path"] = path
        captured["kwargs"] = kwargs
        return fake_index

    def fake_create_default_reranker(path):
        """Record reranker construction and return a fake reranker."""
        captured["reranker_path"] = path
        return fake_reranker

    monkeypatch.setattr("whichcode.cli.load_or_build_hybrid_index", fake_load_or_build_hybrid_index)
    monkeypatch.setattr("whichcode.cli.create_default_reranker", fake_create_default_reranker)

    main([str(tmp_path), "run"])

    output = capsys.readouterr().out
    assert captured == {"path": str(tmp_path), "kwargs": {"rebuild": False}, "reranker_path": str(tmp_path)}
    assert fake_index.reranker is fake_reranker
    assert '"query": "run"' in output
    assert '"content": "def run():\\n    return 1\\n"' in output
    assert '"location": "src/app.py:1-2"' in output


def test_main_passes_rebuild_flag(monkeypatch, capsys, tmp_path) -> None:
    """main should pass only index-level options through the CLI."""
    captured = {}

    def fake_load_or_build_hybrid_index(path, **kwargs):
        """Record index construction arguments and return a fake index."""
        captured["path"] = path
        captured.update(kwargs)
        return FakeIndex()

    monkeypatch.setattr("whichcode.cli.load_or_build_hybrid_index", fake_load_or_build_hybrid_index)
    monkeypatch.setattr("whichcode.cli.create_default_reranker", lambda path: FakeReranker())

    main([str(tmp_path), "run", "--rebuild"])

    capsys.readouterr()
    assert captured == {"path": str(tmp_path), "rebuild": True}
