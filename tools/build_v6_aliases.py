"""Builds the blind shelf-alias asset (V6 candidate A).

Two modes:

    python -m tools.build_v6_aliases --heads
        Load the frozen catalog, apply the B1 eligibility filter, and write
        analysis/_v6_eligible_heads.json (eligible head -> labels, plus the
        per-rule exclusion counts). Deterministic, sorted.

    python -m tools.build_v6_aliases --build
        Read analysis/_v6_aliases_raw.json (the generator's raw aliases),
        normalize, filter, cap and resolve collisions, then emit
        assets/v6_shelf_aliases.json and assets/v6_shelf_aliases.manifest.json.

Build-time only; never imported by serving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.catalog import Catalog  # noqa: E402
from src.shelf_transform import ASSET_SCHEMA, eligible_heads, payload_hash  # noqa: E402
from src.text import normalise, tokens  # noqa: E402

CATALOG_PATH = ROOT / "data" / "catalog.jsonl"
HEADS_PATH = ROOT / "analysis" / "_v6_eligible_heads.json"
RAW_PATH = ROOT / "analysis" / "_v6_aliases_raw.json"
# The registered outcome-blind cross-review of this candidate. --build applies
# its dispositions: removals only, no additions, no other edits.
AUDIT_PATH = ROOT / "analysis" / "_v6_review_aliases_by_mnn.json"
ASSET_PATH = ROOT / "assets" / "v6_shelf_aliases.json"
MANIFEST_PATH = ROOT / "assets" / "v6_shelf_aliases.manifest.json"
BUILDER_PATH = Path(__file__).resolve()

# The frozen generation contract, embedded so its hash is reproducible. This is
# the exact instruction text the generator (Candidate A) was bound by.
GENERATION_CONTRACT = """\
For every retained eligible shelf head, author up to 4 aliases.
Allowed classes ONLY: synonyms, regional or common-usage variants, and
one-level hypernyms. NEVER: brands, materials, colors, use cases, sizes, or
product attributes. Each alias at most 3 normalized tokens. Ordinary shopper
English. If a head has no honest alias, give it an empty list.
"""

GENERATOR_IDENTITY = "opencode subagent, aiand glm model family"
GENERATOR_DATE = "2026-08-30"

MAX_ALIASES_PER_HEAD = 4
MAX_ALIAS_TOKENS = 3


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def cmd_heads() -> None:
    catalog = Catalog(str(CATALOG_PATH))
    eligible, counts = eligible_heads(catalog)
    _dump(HEADS_PATH, {
        "eligible": {head: eligible[head] for head in sorted(eligible)},
        "counts": counts,
    })
    print(f"considered={counts['considered']} retained={counts['retained']}")
    for rule in ("junk_token", "uppercase_fragment", "empty_after_normalization",
                 "banner_only", "product_floor"):
        print(f"  {rule}={counts[rule]}")


def cmd_build() -> None:
    heads_doc = json.loads(HEADS_PATH.read_text(encoding="utf-8"))
    eligible: dict[str, list[str]] = heads_doc["eligible"]
    exclusion_counts: dict[str, int] = heads_doc["counts"]
    raw: dict[str, list[str]] = json.loads(RAW_PATH.read_text(encoding="utf-8"))

    head_token_sets = {head: tuple(tokens(normalise(head))) for head in eligible}
    canonical_head_tokens = set(eligible)  # heads are already normalized tokens

    dropped = {"empty_after_normalization": 0, "canonical_identical": 0,
               "too_long": 0, "cap_per_head": 0, "unknown_head": 0,
               "collision_other_head": 0, "collision_multi_head": 0,
               "audit_removal": 0}
    aliases_in = 0

    # (a)-(d): normalize and filter per head, cap at 4 (lexically first).
    per_head: dict[str, list[str]] = {}
    for head, alias_list in raw.items():
        if head not in eligible:
            dropped["unknown_head"] += len(alias_list)
            continue
        kept: set[str] = set()
        for alias in alias_list or []:
            aliases_in += 1
            alias_tokens = tuple(tokens(normalise(str(alias))))
            if not alias_tokens:
                # tokens() removes stopwords, so stopword-only lands here too.
                dropped["empty_after_normalization"] += 1
                continue
            norm = " ".join(alias_tokens)
            if alias_tokens == head_token_sets[head]:
                dropped["canonical_identical"] += 1
                continue
            if len(alias_tokens) > MAX_ALIAS_TOKENS:
                dropped["too_long"] += 1
                continue
            kept.add(norm)
        capped = sorted(kept)
        if len(capped) > MAX_ALIASES_PER_HEAD:
            dropped["cap_per_head"] += len(capped) - MAX_ALIASES_PER_HEAD
            capped = capped[:MAX_ALIASES_PER_HEAD]
        per_head[head] = capped

    # (e): collisions. An alias equal to a DIFFERENT canonical head token drops.
    # One head keeps it; two heads keep both; more than two drops entirely.
    alias_to_heads: dict[str, list[str]] = {}
    for head in sorted(per_head):
        for alias in per_head[head]:
            alias_to_heads.setdefault(alias, []).append(head)

    mapping: dict[str, list[str]] = {}
    collisions = {"alias_equals_other_head": 0, "multi_head_dropped": 0,
                  "two_head_hedge_kept": 0}
    for alias in sorted(alias_to_heads):
        heads = sorted(alias_to_heads[alias])
        if alias in canonical_head_tokens:
            # Canonical-identical aliases never reach here (dropped in (b)),
            # so this alias equals a DIFFERENT head's canonical token.
            dropped["collision_other_head"] += len(heads)
            collisions["alias_equals_other_head"] += 1
            continue
        if len(heads) > 2:
            dropped["collision_multi_head"] += len(heads)
            collisions["multi_head_dropped"] += 1
            continue
        if len(heads) == 2:
            collisions["two_head_hedge_kept"] += 1
        mapping[alias] = heads

    pre_removal = sum(len(v) for v in mapping.values())

    # (g): registered cross-review dispositions -- removals ONLY, no additions.
    audit_record = None
    if AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        removals = audit.get("removals", [])
        removed = []
        for entry in removals:
            norm = " ".join(tokens(normalise(str(entry["alias"]))))
            if norm in mapping:
                heads = mapping.pop(norm)
                dropped["audit_removal"] += len(heads)
                removed.append({"alias": norm, "heads": heads,
                                "reason": entry["reason"]})
        removed.sort(key=lambda r: r["alias"])
        audit_record = {
            "file": "analysis/_v6_review_aliases_by_mnn.json",
            "reviewer": audit.get("reviewer"),
            "sampled": audit.get("sampled"),
            "supported": audit.get("supported"),
            "precision": audit.get("precision"),
            "wilson95": audit.get("wilson95"),
            "removed": removed,
            "counts": {"pre_removal": pre_removal,
                       "removed": sum(len(r["heads"]) for r in removed),
                       "post_removal": sum(len(v) for v in mapping.values())},
        }

    retained = sum(len(v) for v in mapping.values())
    support = {"aliases_in": aliases_in, "dropped": dropped, "retained": retained}

    catalog_sha = _sha256_file(CATALOG_PATH)
    builder_sha = _sha256_file(BUILDER_PATH)
    input_sha = _sha256_file(HEADS_PATH)
    raw_sha = _sha256_file(RAW_PATH)
    prompt_sha = _sha256_bytes(GENERATION_CONTRACT.encode("utf-8"))
    payload_sha = payload_hash(mapping)

    asset = {
        "schema_version": ASSET_SCHEMA,
        "candidate_name": "aliases",
        "catalog_sha256": catalog_sha,
        "builder_source_sha256": builder_sha,
        "build_config": {
            "max_aliases_per_head": MAX_ALIASES_PER_HEAD,
            "max_alias_tokens": MAX_ALIAS_TOKENS,
            "allowed_classes": ["synonym", "regional_variant",
                                "common_usage_variant", "one_level_hypernym"],
        },
        "input_sha256": input_sha,
        "generator": {
            "identity": GENERATOR_IDENTITY,
            "date": GENERATOR_DATE,
            "prompt_sha256": prompt_sha,
            "raw_output_sha256": raw_sha,
        },
        "mapping": {alias: mapping[alias] for alias in sorted(mapping)},
        "support": support,
        "payload_sha256": payload_sha,
    }
    _dump(ASSET_PATH, asset)

    manifest = {
        "schema_version": ASSET_SCHEMA,
        "candidate_name": "aliases",
        "b1_exclusion_counts": exclusion_counts,
        "heads": {"considered": exclusion_counts["considered"],
                  "retained": exclusion_counts["retained"]},
        "collisions": collisions,
        "support": support,
        "audit": audit_record,
        "hashes": {
            "catalog_sha256": catalog_sha,
            "builder_source_sha256": builder_sha,
            "input_sha256": input_sha,
            "prompt_sha256": prompt_sha,
            "raw_output_sha256": raw_sha,
            "payload_sha256": payload_sha,
        },
    }
    _dump(MANIFEST_PATH, manifest)

    print(f"aliases_in={aliases_in} retained={retained} heads={len(per_head)}")
    print(f"dropped={dropped}")
    print(f"collisions={collisions}")
    if audit_record:
        print(f"audit: removed {audit_record['counts']['removed']} aliases "
              f"({pre_removal} -> {audit_record['counts']['post_removal']})")
    print(f"payload_sha256={payload_sha}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--heads", action="store_true")
    group.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.heads:
        cmd_heads()
    else:
        cmd_build()


if __name__ == "__main__":
    main()
