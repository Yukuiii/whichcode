"""Tests for the whichcode command-line interface."""

from whichcode.chunking import Chunk
from whichcode.cli import main
from whichcode.types import SearchResult


class FakeIndex:
    """Search index stub for CLI output tests."""

    def __init__(self) -> None:
        """Initialize captured search arguments."""
        self.query: str | None = None
        self.top_k: int | None = None

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return one deterministic result."""
        self.query = query
        self.top_k = top_k
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

    def fake_load_or_build_hybrid_index(path, **kwargs):
        """Record CLI index construction arguments and return a fake index."""
        captured["path"] = path
        captured["kwargs"] = kwargs
        return fake_index

    monkeypatch.setattr("whichcode.cli.load_or_build_hybrid_index", fake_load_or_build_hybrid_index)

    main([str(tmp_path), "run"])

    output = capsys.readouterr().out
    assert captured == {"path": str(tmp_path), "kwargs": {"rebuild": False}}
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

    main([str(tmp_path), "run", "--rebuild"])

    capsys.readouterr()
    assert captured == {"path": str(tmp_path), "rebuild": True}
