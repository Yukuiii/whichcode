"""Shared rule-based ranking priors for retrieval candidates."""

from __future__ import annotations

import re
from pathlib import Path

_STRONG_PATH_PENALTY = 0.3
_MODERATE_PATH_PENALTY = 0.5
_MILD_PATH_PENALTY = 0.7
_PRIVATE_MODULE_PENALTY = 0.85

_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:test_[^/]*\.\w+|[^/]*_test\.\w+|[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?)$"
)
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")
_LOW_SIGNAL_DIR_RE = re.compile(r"(?:^|/)(?:_?examples?|benchmarks?|docs?|docs?_src|compat|_compat|legacy)(?:/|$)")
_PRIVATE_MODULE_RE = re.compile(r"(?:^|/)_[^/]+\.\w+$")
_REEXPORT_FILENAMES = frozenset({"__init__.py", "package-info.java"})
_TYPE_DEFS_RE = re.compile(r"\.d\.ts$")


def path_prior(file_path: str) -> float:
    """Return a multiplicative ranking prior for lower-signal file paths."""
    normalized = file_path.replace("\\", "/")
    prior = 1.0
    if _TEST_FILE_RE.search(normalized) is not None or _TEST_DIR_RE.search(normalized) is not None:
        prior *= _STRONG_PATH_PENALTY
    if _LOW_SIGNAL_DIR_RE.search(normalized) is not None:
        prior *= _STRONG_PATH_PENALTY
    if Path(file_path).name in _REEXPORT_FILENAMES:
        prior *= _MODERATE_PATH_PENALTY
    if _TYPE_DEFS_RE.search(normalized) is not None:
        prior *= _MILD_PATH_PENALTY
    if _PRIVATE_MODULE_RE.search(normalized) is not None:
        prior *= _PRIVATE_MODULE_PENALTY
    return prior


def apply_path_prior(file_path: str, score: float) -> float:
    """Apply the path prior to a candidate score."""
    return score * path_prior(file_path)
