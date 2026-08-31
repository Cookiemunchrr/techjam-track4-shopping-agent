"""The V6 shared shelf-transform contract, frozen at Phase 2.

Both candidates (blind aliases, catalog-only MNN) are the same mechanism: a
frozen mapping from normalized alias spans to canonical shelf heads, applied to
the category phrase only, returning replacement phrases built from the phrase's
own residual tokens plus a canonical head. This module owns everything the two
candidates share -- the interface, the span matching, the loader, the RRF
merge, and the head-eligibility filter -- so the only thing a candidate can
differ in is the mapping itself (G8: one mechanism per experiment).

Runtime rules (V6 §5): transform() performs no I/O and touches no module-global
mutable state; empty means no evidence; more than two heads emits nothing; the
exact route never consults a transform. Invalid modes, missing or corrupt
assets, and catalog mismatches all fail closed to `off`.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .text import normalise, tokens

MODES = ("off", "aliases", "mnn")
ASSETS = {
    "aliases": "assets/v6_shelf_aliases.json",
    "mnn": "assets/v6_shelf_mnn.json",
}
ASSET_SCHEMA = "techjam-v6-shelf-transform-v1"

# Serving constants, registered in analysis/v6_cycle_protocol.json before any
# development evaluation: standard unweighted RRF (k=60), at most five ranked
# shelves per transformed phrase, at most two added shelves.
RRF_K = 60
OVERLAP_LIMIT = 5
MAX_ADDED_SHELVES = 2


class ShelfTransform:
    """The immutable instance interface (V6 §5). The default is `off`."""

    mode = "off"
    payload_sha256 = ""

    def transform(self, category_phrase: str) -> list[str]:
        return []


class MappingTransform(ShelfTransform):
    """Longest-span matching over a frozen alias -> canonical-heads mapping."""

    def __init__(self, mode: str, mapping: dict, payload_sha256: str) -> None:
        self.mode = mode
        self.payload_sha256 = payload_sha256
        aliases: dict[tuple[str, ...], tuple[str, ...]] = {}
        for alias, heads in mapping.items():
            key = tuple(tokens(normalise(alias)))
            if key and heads:
                aliases[key] = tuple(heads)
        self._aliases = aliases
        self._max_len = max((len(key) for key in aliases), default=0)

    def transform(self, category_phrase: str) -> list[str]:
        words = tokens(normalise(category_phrase or ""))
        if not words or not self._aliases:
            return []
        # Longest normalized alias span wins; ties take the rightmost span,
        # then the lexical alias key. One scan, no backtracking.
        best: tuple[tuple[str, ...], int, int] | None = None
        for start in range(len(words)):
            for length in range(min(self._max_len, len(words) - start), 0, -1):
                key = tuple(words[start:start + length])
                if key in self._aliases:
                    if best is None or (length, start, key) > (best[1], best[2], best[0]):
                        best = (key, length, start)
                    break
        if best is None:
            return []
        key, length, start = best
        heads = self._aliases[key]
        if len(heads) > 2:
            return []
        residual = words[:start] + words[start + length:]
        return [" ".join([head, *residual]) for head in heads]


def rrf_merge(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Unweighted reciprocal-rank fusion; equal fused ranks break lexically."""
    fused: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, name in enumerate(ranking):
            fused[name] = fused.get(name, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda name: (-fused[name], name))


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def payload_hash(mapping: dict) -> str:
    return hashlib.sha256(_canonical_json(mapping).encode("utf-8")).hexdigest()


def load(mode: str | None, catalog_path: str | Path) -> ShelfTransform:
    """Load exactly one immutable transform; any problem fails closed to off."""
    if not mode or mode == "off":
        return ShelfTransform()
    if mode not in ASSETS:
        return ShelfTransform()
    try:
        path = Path(ASSETS[mode])
        asset = json.loads(path.read_text(encoding="utf-8"))
        if asset.get("schema_version") != ASSET_SCHEMA:
            return ShelfTransform()
        if asset.get("candidate_name") != mode:
            return ShelfTransform()
        mapping = asset.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            return ShelfTransform()
        if asset.get("payload_sha256") != payload_hash(mapping):
            return ShelfTransform()
        catalog_sha = hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest()
        if asset.get("catalog_sha256") != catalog_sha:
            return ShelfTransform()
        return MappingTransform(mode, mapping, asset["payload_sha256"])
    except Exception:
        return ShelfTransform()


# ---------------------------------------------------------------- build side
# Shared by both candidate builders so the eligibility rule is identical by
# construction (addendum B1). Build-time only; never imported by serving.

_JUNK_HEAD = re.compile(r"^(?:\d+(?:\.\d+)?|\$\d+|(?:19|20)\d\d|v\d+)$")
_BANNER = re.compile(
    r"%|\boff\b|\bsavings\b|\bdeals\b|\bprime\b|\bexclusives\b|\bsale\b"
    r"|new arrivals|\bessentials\b|top rated|best seller|under\s*\$\s*\d+",
    re.IGNORECASE)


def catalog_shelf_heads(catalog) -> dict[str, str]:
    """Every coarse shelf label to its canonical head token, catalog-derived."""
    heads = {}
    for label in catalog.buckets:
        words = tokens(normalise(label))
        heads[label] = words[-1] if words else ""
    return heads


def eligible_heads(catalog, min_products: int = 10):
    """The addendum-B1 head filter, identical for both candidates.

    A head is ineligible when its normalized form is numeric, a currency amount,
    a four-digit year, or ^vN; when its surface form is all-uppercase beyond one
    character; when it normalizes to nothing content-bearing (stopword or single
    character, which tokens() already removes); when every shelf label sharing
    the head matches the frozen promotional-banner patterns; or when the head's
    shelves hold fewer than `min_products` catalog products in total.

    Returns (eligible, counts): eligible maps head -> sorted labels; counts
    records per-rule exclusions plus the considered/retained totals for the
    asset manifest (T24).
    """
    heads = catalog_shelf_heads(catalog)
    by_head: dict[str, list[str]] = {}
    for label, head in heads.items():
        if head:
            by_head.setdefault(head, []).append(label)

    counts = {"considered": len(by_head), "junk_token": 0, "uppercase_fragment": 0,
              "empty_after_normalization": 0, "banner_only": 0,
              "product_floor": 0, "retained": 0}
    eligible: dict[str, list[str]] = {}
    for head, labels in sorted(by_head.items()):
        if _JUNK_HEAD.match(head):
            counts["junk_token"] += 1
            continue
        # The surface form behind the normalized head, from its shortest label.
        surface = min(labels, key=len).split()[-1]
        if len(surface) > 1 and surface.upper() == surface:
            counts["uppercase_fragment"] += 1
            continue
        if not tokens(normalise(head)):
            counts["empty_after_normalization"] += 1
            continue
        if all(_BANNER.search(label) for label in labels):
            counts["banner_only"] += 1
            continue
        if sum(len(catalog.buckets[label]) for label in labels) < min_products:
            counts["product_floor"] += 1
            continue
        eligible[head] = sorted(labels)
    counts["retained"] = len(eligible)
    return eligible, counts
