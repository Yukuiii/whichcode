"""Tests for local LLM model resolution and query-time reranking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from whichcode.chunking import Chunk
from whichcode.local_llm import LocalModelConfig, resolve_local_model_path
from whichcode.reranker import LlamaCppReranker
from whichcode.types import SearchResult


class FakeChatModel:
    """Chat model stub that records prompts and returns configured content."""

    def __init__(self, content: str) -> None:
        """Store the response content and initialize captured calls."""
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        """Record the call and return a llama.cpp-compatible response."""
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.content}}]}


def _results(contents: Sequence[str]) -> list[SearchResult]:
    """Create deterministic search results for reranker tests."""
    return [
        SearchResult(
            chunk=Chunk(content=content, file_path=f"src/file{index}.py", start_line=1, end_line=1, kind="function"),
            score=float(index),
        )
        for index, content in enumerate(contents)
    ]


@pytest.fixture(autouse=True)
def isolated_whichcode_home(monkeypatch, tmp_path) -> None:
    """Redirect user-level model files to each test's temp home."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_local_model_config_uses_full_qwen_context() -> None:
    """LocalModelConfig should use the Qwen GGUF context length by default."""
    assert LocalModelConfig().n_ctx == 262_144


def test_resolve_local_model_path_accepts_existing_local_file(tmp_path) -> None:
    """resolve_local_model_path should return an existing local GGUF path."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    resolved = resolve_local_model_path(tmp_path, LocalModelConfig(model_name=str(model_path)))

    assert resolved == model_path.resolve()


def test_resolve_local_model_path_downloads_missing_remote_file(monkeypatch, tmp_path) -> None:
    """resolve_local_model_path should download a missing remote GGUF into ~/.whichcode."""
    captured: dict[str, object] = {}

    def fake_download(repo_id: str, filename: str, local_dir: Path) -> str:
        """Create a fake downloaded model and record download arguments."""
        captured.update({"repo_id": repo_id, "filename": filename, "local_dir": local_dir})
        output_path = local_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"gguf")
        return str(output_path)

    monkeypatch.setattr("whichcode.local_llm._download_hf_file", fake_download)

    resolved = resolve_local_model_path(tmp_path, LocalModelConfig())

    assert resolved.exists()
    assert resolved.is_relative_to(Path.home() / ".whichcode" / "models")
    assert captured["repo_id"] == "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
    assert captured["filename"] == "Qwen_Qwen3.5-0.8B-Q5_K_S.gguf"
    assert captured["local_dir"] == Path.home() / ".whichcode" / "models" / "bartowski__Qwen_Qwen3.5-0.8B-GGUF"


@pytest.mark.parametrize("model_file", ["/model.gguf", "../model.gguf", "model.bin"])
def test_resolve_local_model_path_rejects_invalid_remote_file(tmp_path, model_file) -> None:
    """resolve_local_model_path should reject unsafe or non-GGUF remote filenames."""
    config = LocalModelConfig(model_name="owner/repo", model_file=model_file)

    with pytest.raises(ValueError):
        resolve_local_model_path(tmp_path, config)


def test_llama_cpp_reranker_reorders_results_and_keeps_full_content() -> None:
    """LlamaCppReranker should use full candidate content and apply returned ranking ids."""
    full_content = "def large():\n    return '" + ("x" * 7000) + "'\n"
    model = FakeChatModel('{"ranking":[2,0,1]}')
    reranker = LlamaCppReranker(model)

    ranked = reranker.rerank("find large implementation", _results(["alpha", "beta", full_content]), top_k=2)
    system_prompt = model.calls[0]["messages"][0]["content"]
    prompt = model.calls[0]["messages"][1]["content"]

    assert [result.chunk.file_path for result in ranked] == ["src/file2.py", "src/file0.py"]
    assert "canonical definitions, public APIs, and owner modules" in system_prompt
    assert "Treat hybrid_score as a strong prior" in system_prompt
    assert "most directly useful to least useful" in prompt
    assert full_content in prompt


def test_llama_cpp_reranker_falls_back_on_invalid_output() -> None:
    """LlamaCppReranker should preserve hybrid order when the model output is invalid."""
    results = _results(["alpha", "beta"])
    reranker = LlamaCppReranker(FakeChatModel("not json"))

    assert reranker.rerank("query", results, top_k=2) == results
