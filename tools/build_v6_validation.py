"""Build the V6 sealed validation bodies and their public manifest (Phase 1.3).

S1 is the sealed lexical-shift confirmation instrument over sessions 101-200:
a deterministic, catalog-only rewrite of the customer messages -- no target ids,
no agent output, no candidate assets, and never the repository's adversarial
dictionary (tools/adversarial.py), which the builders are forbidden to read.

S2 is the target-free shelf-language audit: shelf/head families selected on
catalog-only strata (frequency and ambiguity), each carrying natural shopper
phrases with their acceptable canonical heads. No product targets or public
sessions are involved.

Both bodies are ignored sidecars; the committed artifact is the manifest of
hashes, counts, strata, provenance, and the consumption ledger's starting state.
Candidate builders never read the bodies.

    python3 -m tools.build_v6_validation
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from src.catalog import Catalog
from src.text import normalise, tokens

CATALOG = "data/catalog.jsonl"

S1_BODY = Path("analysis/_v6_s1_shifts.json")
S2_BODY = Path("analysis/_v6_s2_phrases.json")
MANIFEST = Path("analysis/v6_validation_manifest.json")

# The fixed seed schedule: every confirmation-side choice derives from this
# constant and the session ordinal, frozen here before any candidate exists.
S1_SEED_BASE = 913_000
S2_SEED = 41_203

# The opening template cycle. Generic customer phrasing; the category shift is
# what stresses shelf resolution.
S1_OPENING_TEMPLATES = (
    "I'm looking for {category}.",
    "I'm after {category}.",
    "I need {category}.",
    "Do you carry {category}?",
    "Got any {category}?",
)


def _sha256_json(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def shelf_heads(catalog: Catalog) -> dict[str, str]:
    """Every coarse shelf label to its canonical head token (catalog-derived)."""
    heads = {}
    for label in catalog.buckets:
        words = tokens(normalise(label))
        heads[label] = words[-1] if words else ""
    return heads


def build_s1_body(catalog: Catalog) -> dict:
    """Deterministic catalog-only lexical shifts for every shelf label.

    The shifts never leave the catalog's own vocabulary: they drop or reorder
    the label's own tokens. What they never do is synonymise the head from any
    external dictionary -- that vocabulary is the sealed part of the dev axis,
    and reusing it here would make the confirmation a rerun of development.
    """
    shifts: dict[str, list[str]] = {}
    for label, head in shelf_heads(catalog).items():
        words = tokens(normalise(label))
        variants = []
        if len(words) >= 2:
            variants.append(" ".join(words[-2:]))     # trailing modifier + head
            variants.append(" ".join(words[::-1]))    # reversed word order
        if head:
            variants.append(head)                     # bare head
        # De-duplicate while preserving order; the label itself is not a shift.
        seen = []
        for variant in variants:
            if variant and variant != normalise(label) and variant not in seen:
                seen.append(variant)
        shifts[label] = seen
    return {
        "seed_base": S1_SEED_BASE,
        "opening_templates": list(S1_OPENING_TEMPLATES),
        "shifts": shifts,
    }


def strata_families(catalog: Catalog) -> list[dict]:
    """Shelf/head families on catalog-only strata: frequency and ambiguity."""
    heads = shelf_heads(catalog)
    by_head: dict[str, list[str]] = defaultdict(list)
    for label, head in heads.items():
        if head:
            by_head[head].append(label)
    families = []
    for head, labels in sorted(by_head.items()):
        products = sum(len(catalog.buckets[label]) for label in labels)
        families.append({
            "head": head,
            "labels": sorted(labels),
            "product_count": products,
            "ambiguity": len(labels),
        })
    return families


def select_s2_families(families: list[dict], quota: int = 60) -> list[dict]:
    """A seeded stratified sample across frequency terciles x ambiguity tiers."""
    counts = sorted(f["product_count"] for f in families)
    t1 = counts[len(counts) // 3]
    t2 = counts[2 * len(counts) // 3]

    def tier(family) -> str:
        freq = ("low" if family["product_count"] <= t1
                else "mid" if family["product_count"] <= t2 else "high")
        amb = ("unique" if family["ambiguity"] == 1
               else "few" if family["ambiguity"] <= 3 else "many")
        return f"{freq}/{amb}"

    rng = random.Random(S2_SEED)
    strata: dict[str, list[dict]] = defaultdict(list)
    for family in families:
        strata[tier(family)].append(family)
    per_stratum = max(1, quota // max(len(strata), 1))
    chosen: list[dict] = []
    for name in sorted(strata):
        pool = sorted(strata[name], key=lambda f: f["head"])
        take = rng.sample(pool, min(per_stratum + 2, len(pool)))
        chosen.extend(take)
    rng.shuffle(chosen)
    return chosen[:max(quota, 60)]


def main() -> int:
    catalog = Catalog(CATALOG)

    s1_body = build_s1_body(catalog)
    S1_BODY.write_text(json.dumps(s1_body, indent=2) + "\n", encoding="utf-8")

    families = strata_families(catalog)
    chosen = select_s2_families(families)
    worksheet = {
        "seed": S2_SEED,
        "family_count": len(chosen),
        "families": chosen,
    }
    # The worksheet is the authoring input; the phrases are authored against it
    # and sealed separately. Both stay ignored sidecars.
    Path("analysis/_v6_s2_worksheet.json").write_text(
        json.dumps(worksheet, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "techjam-v6-validation-manifest-v1",
        "s1": {
            "kind": "sealed lexical-shift confirmation instrument",
            "body": str(S1_BODY),
            "body_sha256": _sha256_json(s1_body),
            "seed_base": S1_SEED_BASE,
            "label_count": len(s1_body["shifts"]),
            "empty_shift_labels": sum(1 for v in s1_body["shifts"].values() if not v),
        },
        "s2": {
            "kind": "target-free shelf-language audit",
            "worksheet": "analysis/_v6_s2_worksheet.json",
            "worksheet_sha256": _sha256_json(worksheet),
            "seed": S2_SEED,
            "family_count": len(chosen),
            "strata": "frequency tercile x ambiguity (unique/few/many)",
            "phrases": "authored against the worksheet, sealed as "
                       "analysis/_v6_s2_phrases.json; hash lands here at freeze",
            "phrases_sha256": None,
        },
        "generator": {
            "identity": "opencode agent (aiand/zai-org/glm-5.2)",
            "separation": "process-disjoint, not family-disjoint: the validation "
                          "author and the candidate-A alias generator are the same "
                          "model family (STOP-10 limitation; no family-disjoint "
                          "claim is made)",
            "never_read": ["tools/adversarial.py and HEAD_NOUNS", "evaluator transcripts",
                           "per-session misses", "candidate outcome tables",
                           "target labels"],
        },
        "consumption": "each sealed body may be consumed exactly once, by the "
                       "selected winner, through analysis/v6_validation_ledger.json",
        "ledger": "analysis/v6_validation_ledger.json",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"S1: {len(s1_body['shifts'])} labels, "
          f"{manifest['s1']['empty_shift_labels']} without a shift")
    print(f"S2: {len(chosen)} families across {len(strata_families(catalog))} heads")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
