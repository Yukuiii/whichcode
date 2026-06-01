"""Query-time reranking for hybrid search results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from whichcode.local_llm import ChatModel, LocalModelConfig, load_chat_model
from whichcode.types import SearchResult

LLM_RERANK_CANDIDATES = 20
LLM_RERANK_MAX_TOKENS = 512


class ResultReranker(Protocol):
    """Defines the interface for query-time result rerankers."""

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        """Return the given results reordered for the query."""
        ...


@dataclass(slots=True)
class LlamaCppReranker:
    """Uses a local llama.cpp chat model to rerank candidate chunks."""

    model: ChatModel
    max_tokens: int = LLM_RERANK_MAX_TOKENS

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        """Return LLM-ranked candidates, falling back to the original order on invalid output."""
        candidates = list(results)
        if top_k < 1 or not candidates:
            return []
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _user_prompt(query, candidates)},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
            )
            ranked_ids = _parse_ranking(_extract_chat_content(response), len(candidates))
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
            return candidates[:top_k]

        by_id = {index: result for index, result in enumerate(candidates)}
        ordered = [by_id[index] for index in ranked_ids if index in by_id]
        seen = set(ranked_ids)
        ordered.extend(result for index, result in enumerate(candidates) if index not in seen)
        return ordered[:top_k]


def create_default_reranker(root: str | Path, config: LocalModelConfig | None = None) -> ResultReranker:
    """Create the fixed local LLM reranker for a project root."""
    return LlamaCppReranker(load_chat_model(root, config))


def _system_prompt() -> str:
    """Return the fixed instruction for local code-search reranking."""
    return (
        "You rerank code search results for a developer. "
        "Use only the provided query and candidate chunks. "
        "Return strict JSON only, with the shape {\"ranking\":[id,...]}. "
        "The first id must be the most relevant candidate. "
        "Do not include ids that were not provided."
    )


def _user_prompt(query: str, results: Sequence[SearchResult]) -> str:
    """Build a full-content reranking prompt from hybrid candidates."""
    parts = ["Query:", query, "", "Candidates:"]
    for index, result in enumerate(results):
        chunk = result.chunk
        metadata = [
            f"id: {index}",
            f"path: {chunk.file_path}",
            f"lines: {chunk.start_line}-{chunk.end_line}",
            f"kind: {chunk.kind}",
            f"name: {chunk.name or ''}",
            f"language: {chunk.language or ''}",
            f"hybrid_score: {result.score:.8f}",
        ]
        parts.extend(["", "<candidate>", *metadata, "content:", chunk.content, "</candidate>"])
    parts.append("")
    parts.append("Return JSON now.")
    return "\n".join(parts)


def _extract_chat_content(response: dict[str, Any]) -> str:
    """Extract assistant text from a llama.cpp chat completion response."""
    return str(response["choices"][0]["message"]["content"])


def _parse_ranking(text: str, candidate_count: int) -> list[int]:
    """Parse a ranking list from strict or lightly wrapped JSON model output."""
    payload = _extract_json_object(text)
    raw_ranking = payload.get("ranking")
    if not isinstance(raw_ranking, list):
        raise ValueError("reranker output must contain a ranking list")

    ranking: list[int] = []
    for item in raw_ranking:
        candidate_id = _coerce_ranked_id(item)
        if candidate_id is None or candidate_id in ranking:
            continue
        if 0 <= candidate_id < candidate_count:
            ranking.append(candidate_id)
    if not ranking:
        raise ValueError("reranker output did not contain valid candidate ids")
    return ranking


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("reranker output must be a JSON object")
    return payload


def _coerce_ranked_id(item: object) -> int | None:
    """Return a candidate id from either a bare int or an object with an id field."""
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return item
    if isinstance(item, str) and item.isdecimal():
        return int(item)
    if isinstance(item, dict):
        raw_id = item.get("id")
        if isinstance(raw_id, bool):
            return None
        if isinstance(raw_id, int):
            return raw_id
        if isinstance(raw_id, str) and raw_id.isdecimal():
            return int(raw_id)
    return None
