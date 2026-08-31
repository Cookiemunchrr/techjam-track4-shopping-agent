"""V6 candidate A (blind aliases): the serving-side adapter.

Thin by contract: the whole mechanism lives in src/shelf_transform.py; this
module only names the candidate and wraps its frozen mapping in the shared
MappingTransform. No I/O at import time.
"""
from __future__ import annotations

from .shelf_transform import MappingTransform

CANDIDATE_NAME = "aliases"


def build_transform(mapping: dict, payload_sha256: str) -> MappingTransform:
    """Wrap the frozen alias mapping in the shared span-matcher."""
    return MappingTransform(CANDIDATE_NAME, mapping, payload_sha256)
