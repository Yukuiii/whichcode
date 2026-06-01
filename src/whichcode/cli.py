"""Command-line interface for whichcode search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from whichcode.formatting import format_results
from whichcode.storage import load_or_build_hybrid_index

DEFAULT_RESULT_COUNT = 10


def main(argv: Sequence[str] | None = None) -> None:
    """Run a query against a project path and print JSON results."""
    parser = argparse.ArgumentParser(prog="whichcode", description="Search a codebase with a local chunk index.")
    parser.add_argument("path", help="Project directory to scan or load from .whichcode.")
    parser.add_argument("query", help="Search query.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild .whichcode before searching.")
    args = parser.parse_args(argv)

    index = load_or_build_hybrid_index(args.path, rebuild=args.rebuild)
    results = index.search(args.query, top_k=DEFAULT_RESULT_COUNT)
    print(json.dumps(format_results(args.query, results), ensure_ascii=False, indent=2))
