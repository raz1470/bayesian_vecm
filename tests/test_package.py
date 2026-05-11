"""Sanity tests that the package is correctly installed and importable."""

from __future__ import annotations

import bayesian_vecm


def test_package_is_importable() -> None:
    """The package itself should import without errors."""
    assert bayesian_vecm is not None


def test_version_is_set() -> None:
    """__version__ should be a non-empty string."""
    assert isinstance(bayesian_vecm.__version__, str)
    assert bayesian_vecm.__version__ != ""


def test_version_follows_semver() -> None:
    """Version should follow MAJOR.MINOR.PATCH (semantic versioning)."""
    parts = bayesian_vecm.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
