"""Tests for deterministic ranking rules."""

from whichcode.ranking_rules import apply_path_prior, path_prior


def test_path_prior_demotes_lower_signal_paths() -> None:
    """path_prior should rank public source files above lower-signal paths."""
    assert path_prior("src/public.py") == 1.0
    assert path_prior("tests/test_public.py") < path_prior("src/public.py")
    assert path_prior("src/_private.py") < path_prior("src/public.py")
    assert path_prior("examples/demo.py") < path_prior("src/public.py")


def test_apply_path_prior_multiplies_scores() -> None:
    """apply_path_prior should scale a score by the path prior."""
    public_score = apply_path_prior("src/public.py", 2.0)
    test_score = apply_path_prior("tests/test_public.py", 2.0)

    assert public_score == 2.0
    assert test_score < public_score
