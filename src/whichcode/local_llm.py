"""Local llama.cpp model configuration and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

DEFAULT_LLM_BACKEND = "llama-cpp"
DEFAULT_LLM_MODEL_NAME = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
DEFAULT_LLM_MODEL_FILE = "Qwen_Qwen3.5-0.8B-Q5_K_S.gguf"
DEFAULT_LLM_CONTEXT = 262_144
DEFAULT_LLM_GPU_LAYERS = -1


class ChatModel(Protocol):
    """Defines the minimal chat-completion interface used by local LLM features."""

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        """Return a llama.cpp-compatible chat completion response."""
        ...


@dataclass(frozen=True, slots=True)
class LocalModelConfig:
    """Stores fixed local LLM settings for query-time reasoning."""

    backend: str = DEFAULT_LLM_BACKEND
    model_name: str = DEFAULT_LLM_MODEL_NAME
    model_file: str = DEFAULT_LLM_MODEL_FILE
    n_ctx: int = DEFAULT_LLM_CONTEXT
    n_gpu_layers: int = DEFAULT_LLM_GPU_LAYERS


def load_chat_model(root: str | Path, config: LocalModelConfig | None = None) -> ChatModel:
    """Load and cache the configured local chat model for a project root."""
    resolved_config = config or LocalModelConfig()
    if resolved_config.backend != DEFAULT_LLM_BACKEND:
        raise ValueError(f"Unsupported local LLM backend: {resolved_config.backend}")
    model_path = resolve_local_model_path(root, resolved_config)
    return _load_llama_cpp_model(str(model_path), resolved_config.n_ctx, resolved_config.n_gpu_layers)


def resolve_local_model_path(root: str | Path, config: LocalModelConfig) -> Path:
    """Resolve a local GGUF path or download the configured remote model under .whichcode."""
    if _looks_like_local_path(config.model_name):
        model_path = Path(config.model_name).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Local LLM model file does not exist: {model_path}")
        return model_path.resolve()

    relative_file = _validate_remote_model_file(config.model_file)
    root_path = Path(root).expanduser().resolve()
    model_dir = root_path / ".whichcode" / "models" / _safe_model_dir(config.model_name)
    model_path = model_dir / relative_file
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = _download_hf_file(config.model_name, config.model_file, model_dir)
    return Path(downloaded).expanduser().resolve()


@cache
def _load_llama_cpp_model(model_path: str, n_ctx: int, n_gpu_layers: int) -> ChatModel:
    """Load a llama.cpp model once for each model path and runtime setting."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is required for local LLM reranking. Install it with: uv sync") from exc
    return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)


def _looks_like_local_path(model_name: str) -> bool:
    """Return whether a model name should be treated as a filesystem path."""
    path = Path(model_name).expanduser()
    return path.exists() or path.suffix.lower() == ".gguf" or model_name.startswith((".", "~", "/"))


def _validate_remote_model_file(model_file: str) -> Path:
    """Validate and convert a Hugging Face repo filename to a relative path."""
    posix_path = PurePosixPath(model_file)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError("local LLM model file must be a relative path inside the remote repo")
    if posix_path.suffix.lower() != ".gguf":
        raise ValueError("local LLM model file must be a .gguf file")
    return Path(*posix_path.parts)


def _safe_model_dir(repo_id: str) -> str:
    """Convert a remote repo id into a safe local cache directory name."""
    safe = "".join(char if char.isalnum() or char in "._-" else "__" for char in repo_id)
    return safe or "model"


def _download_hf_file(repo_id: str, filename: str, local_dir: Path) -> str:
    """Download one file from Hugging Face into a local project cache directory."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download local LLM models. Install it with: uv sync"
        ) from exc
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
