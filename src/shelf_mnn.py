"""Runtime adapter for the V6 catalog-only MNN shelf map (Candidate B).

Import-clean: no I/O at import time and no module-global mutable state.
The frozen asset lives at assets/v6_shelf_mnn.json and is loaded through
src.shelf_transform.load("mnn", ...); this module only wraps an already
loaded mapping in the shared transform machinery.
"""
from __future__ import annotations

from .shelf_transform import MappingTransform


def build_transform(mapping: dict, payload_sha256: str) -> MappingTransform:
    """The immutable mnn-mode transform over a frozen mapping."""
    return MappingTransform("mnn", mapping, payload_sha256)
