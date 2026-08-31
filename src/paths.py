"""Locating the frozen catalog.

Both docs/submission_rules.md and docs/competition_specification.md publish the
required interface as a class with `reset` and `respond` and no constructor
argument. A private harness may therefore import this module and call `Agent()`
from its own working directory -- where a bare relative path resolves to nothing
and raises FileNotFoundError before `respond`'s exception guard can ever run.

So: try the package's own repository first, then the working directory, then an
explicit environment override, and if all of that fails say exactly what was
tried rather than naming one path that was never going to work.
"""
from __future__ import annotations

import os
from pathlib import Path

RELATIVE = "data/catalog.jsonl"
ENV_VAR = "TECHJAM_CATALOG"

# src/paths.py -> src/ -> repository root
_ROOT = Path(__file__).resolve().parents[1]
# Exported so other modules can locate committed build-time assets the same way
# the catalog is located: from the package, never from the current directory.
ROOT = _ROOT


class CatalogNotFound(FileNotFoundError):
    """Raised with every location that was searched, so the failure is actionable."""


def candidates(catalog_path: str | Path | None = None) -> list[Path]:
    """Every location worth trying, most specific first, in order.

    An explicitly supplied path is authoritative: if the caller names a file, we
    look for that file and nothing else. Silently substituting a different catalog
    for a path the caller asked for would turn a typo into a wrong-data run, which
    is far worse than an error.
    """
    seen: list[Path] = []

    def add(value) -> None:
        if not value:
            return
        resolved = Path(value).expanduser()
        if resolved not in seen:
            seen.append(resolved)

    if catalog_path is not None:
        add(catalog_path)
        # A relative path is still worth trying against the repository root, so
        # `Agent("data/catalog.jsonl")` works from any working directory.
        if not Path(catalog_path).is_absolute():
            add(_ROOT / catalog_path)
        return seen

    add(os.environ.get(ENV_VAR))
    add(_ROOT / RELATIVE)
    add(Path.cwd() / RELATIVE)
    return seen


def resolve(catalog_path: str | Path | None = None) -> Path:
    """The first location that exists, or CatalogNotFound listing all of them."""
    tried = candidates(catalog_path)
    for option in tried:
        if option.is_file():
            return option
    listing = "\n  ".join(str(option) for option in tried)
    raise CatalogNotFound(
        f"could not find the frozen catalog. Tried:\n  {listing}\n"
        f"Set {ENV_VAR} to its location, or pass the path to Agent()."
    )
