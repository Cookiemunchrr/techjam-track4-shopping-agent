"""Build the V6 catalog-only sparse mutual-nearest-neighbour shelf map (Candidate B).

Distinct from src/semantic.py's mining: the semantic route keeps a broad
word -> shelf distribution; this builder retains only reciprocal
term <-> canonical-head relationships under the registered fixed formula.
The only thing shared with src/semantic.py is the stem() helper, used for
the stem-equality exclusion exactly as registered.

Formula (frozen; no sweeps, no tuning):

  * The catalog is re-read with stdlib json, line by line. Only title,
    features, details and the categories text are tokenized
    (src.text.normalise + src.text.tokens). Store/brand and free
    description prose never contribute. Document PRESENCE is counted:
    a term present in a product counts once.
  * Shelves are the catalog's category buckets. Membership follows each
    product's own categories text via src.catalog.coarse_category, which
    is exactly how Catalog.buckets is built; Catalog itself is used for
    the frozen B1 head filter (src.shelf_transform.eligible_heads).
  * c(w, b) = number of shelf-b products containing w; v_w[b] = c(w,b)/|b|
    over RETAINED shelves only. Head vectors v_h are built identically,
    with the head token treated as a term.
  * Vectors are L2-normalized; similarity(w, h) = cosine(v_w, v_h).

Fixed filters (all registered, none adjustable):

  B2  shelves with fewer than MIN_SHELF_SIZE products contribute no
      coordinate to any vector; a term must then still have document
      frequency >= MIN_DOCUMENT_FREQUENCY among retained shelves, and may
      occur in no more than MAX_SHELF_SPREAD of the retained shelves.
  *   w is not the same stemmed token as h (src.semantic.stem).
  *   h is among w's top TOP_HEADS_PER_TERM canonical-head neighbours
      (cosine desc, lexical tie-break).
  *   w is among h's top TOP_TERMS_PER_HEAD noncanonical-term neighbours
      (same ordering).
  *   cosine >= MIN_COSINE.
  *   at most MAX_ALIASES_PER_HEAD aliases per canonical head
      (cosine desc, lexical tie-break).
  *   heads come from eligible_heads(catalog); ineligible heads get no
      mappings. A term with two reciprocal heads emits both.

    python -m tools.build_v6_mnn
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from src.catalog import Catalog, coarse_category
from src.semantic import stem
from src.shelf_transform import ASSET_SCHEMA, eligible_heads, payload_hash
from src.text import flatten, normalise, tokens

CATALOG_PATH = "data/catalog.jsonl"
ASSET_PATH = Path("assets/v6_shelf_mnn.json")
MANIFEST_PATH = Path("assets/v6_shelf_mnn.manifest.json")

CANDIDATE = "mnn"
TOKEN_FIELDS = ("title", "features", "details", "categories")
DERIVATION = "catalog-only algorithmic derivation"

# Registered fixed filter values.
MIN_SHELF_SIZE = 10
MIN_DOCUMENT_FREQUENCY = 5
MAX_SHELF_SPREAD = 0.25
TOP_HEADS_PER_TERM = 2
TOP_TERMS_PER_HEAD = 8
MIN_COSINE = 0.80
MAX_ALIASES_PER_HEAD = 4

BUILD_CONFIG = {
    "min_shelf_size": MIN_SHELF_SIZE,
    "min_document_frequency": MIN_DOCUMENT_FREQUENCY,
    "max_shelf_spread": MAX_SHELF_SPREAD,
    "top_heads_per_term": TOP_HEADS_PER_TERM,
    "top_terms_per_head": TOP_TERMS_PER_HEAD,
    "min_cosine": MIN_COSINE,
    "max_aliases_per_head": MAX_ALIASES_PER_HEAD,
    "token_fields": list(TOKEN_FIELDS),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def product_terms(product: dict) -> set:
    """Document-presence token set from the registered fields only."""
    blob = " ".join(flatten(product.get(field)) for field in TOKEN_FIELDS)
    return set(tokens(normalise(blob)))


def shelf_term_counts(catalog_path):
    """(shelf sizes, term -> shelf -> document-presence count).

    Membership comes from each product's own categories text via
    coarse_category, identical to how Catalog.buckets assigns it; counts
    come from this module's own tokenization, never from Catalog.tf or
    Catalog.corpus (those include the excluded fields).
    """
    sizes: dict = defaultdict(int)
    counts: dict = defaultdict(lambda: defaultdict(int))
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            shelf = coarse_category(product.get("categories"))
            sizes[shelf] += 1
            for term in product_terms(product):
                counts[term][shelf] += 1
    return dict(sizes), counts


def _norm(vec: dict) -> float:
    return math.sqrt(sum(value * value for value in vec.values()))


def build(catalog_path: str = CATALOG_PATH) -> dict:
    """Run the frozen pipeline; return {"asset": ..., "manifest": ...}."""
    catalog_path = str(catalog_path)
    catalog = Catalog(catalog_path)
    eligible, b1_counts = eligible_heads(catalog, MIN_SHELF_SIZE)

    sizes, counts = shelf_term_counts(catalog_path)
    retained = sorted(shelf for shelf, n in sizes.items() if n >= MIN_SHELF_SIZE)
    retained_set = set(retained)
    shelf_excluded = len(sizes) - len(retained)

    # B2: restrict every vector to retained shelves, then the document
    # frequency and shelf-spread gates apply to terms.
    drop_df = drop_spread = 0
    considered = 0
    surviving: dict = {}
    for term, per_shelf in counts.items():
        vec = {}
        doc_freq = 0
        for shelf, n in per_shelf.items():
            if shelf in retained_set:
                vec[shelf] = n / sizes[shelf]
                doc_freq += n
        if not vec:
            continue
        considered += 1
        if doc_freq < MIN_DOCUMENT_FREQUENCY:
            drop_df += 1
            continue
        if len(vec) * 4 > len(retained):  # occurrence in > 25% of retained
            drop_spread += 1
            continue
        surviving[term] = vec

    # Canonical heads are never alias sources: only noncanonical terms map.
    alias_terms = {t: v for t, v in surviving.items() if t not in eligible}
    head_terms = len(surviving) - len(alias_terms)

    # Head vectors are built identically (head as a term) but are not
    # subject to the term gates.
    head_vecs: dict = {}
    for head in sorted(eligible):
        per_shelf = counts.get(head)
        if not per_shelf:
            continue
        vec = {shelf: n / sizes[shelf]
               for shelf, n in per_shelf.items() if shelf in retained_set}
        if vec:
            head_vecs[head] = vec

    term_norms = {t: _norm(v) for t, v in alias_terms.items()}
    head_norms = {h: _norm(v) for h, v in head_vecs.items()}
    term_stems = {t: stem(t) for t in alias_terms}
    head_stems = {h: stem(h) for h in head_vecs}

    # Inverted views, in canonical (sorted) order so float accumulation is
    # bit-identical regardless of input line order or id naming.
    heads_by_shelf: dict = {shelf: {} for shelf in retained}
    for head in sorted(head_vecs):
        for shelf in sorted(head_vecs[head]):
            heads_by_shelf[shelf][head] = head_vecs[head][shelf]
    terms_by_shelf: dict = defaultdict(list)
    for term in sorted(alias_terms):
        for shelf in sorted(alias_terms[term]):
            terms_by_shelf[shelf].append((term, alias_terms[term][shelf]))

    # Term side: exact cosine against every co-occurring head, then top-2.
    drop_no_head = 0
    term_top: dict = {}
    for term in sorted(alias_terms):
        dots: dict = {}
        for shelf in sorted(alias_terms[term]):
            value = alias_terms[term][shelf]
            for head, head_value in heads_by_shelf[shelf].items():
                if head_stems[head] == term_stems[term]:
                    continue
                dots[head] = dots.get(head, 0.0) + value * head_value
        if not dots:
            drop_no_head += 1
            continue
        norm_t = term_norms[term]
        scored = [(head, dot / (norm_t * head_norms[head]))
                  for head, dot in dots.items()]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        term_top[term] = scored[:TOP_HEADS_PER_TERM]

    # Head side: exact cosine against every co-occurring noncanonical term,
    # then top-8.
    head_top: dict = {}
    for head in sorted(head_vecs):
        acc: dict = {}
        for shelf in sorted(head_vecs[head]):
            head_value = head_vecs[head][shelf]
            for term, term_value in terms_by_shelf.get(shelf, ()):  # sorted
                if term_stems[term] == head_stems[head]:
                    continue
                acc[term] = acc.get(term, 0.0) + term_value * head_value
        norm_h = head_norms[head]
        scored = [(term, dot / (norm_h * term_norms[term]))
                  for term, dot in acc.items()]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        head_top[head] = scored[:TOP_TERMS_PER_HEAD]
    head_top_sets = {head: {term for term, _ in ranked}
                     for head, ranked in head_top.items()}

    # Reciprocity + cosine gate. A term can hold at most TOP_HEADS_PER_TERM
    # heads by construction; more than two would emit nothing.
    drop_cosine = drop_reciprocity = drop_multi = 0
    pairs: list = []
    for term in sorted(term_top):
        passing = [(h, c) for h, c in term_top[term] if c >= MIN_COSINE]
        if not passing:
            drop_cosine += 1
            continue
        mutual = [(h, c) for h, c in passing
                  if term in head_top_sets.get(h, ())]
        if not mutual:
            drop_reciprocity += 1
            continue
        if len(mutual) > 2:
            drop_multi += 1
            continue
        for head, cos in mutual:
            pairs.append((term, head, cos))

    # Per-head alias cap.
    by_head: dict = defaultdict(list)
    for term, head, cos in pairs:
        by_head[head].append((term, cos))
    drop_alias_cap = 0
    kept: dict = defaultdict(list)
    for head in sorted(by_head):
        ranked = sorted(by_head[head], key=lambda pair: (-pair[1], pair[0]))
        for term, cos in ranked[:MAX_ALIASES_PER_HEAD]:
            kept[term].append((head, cos))
        drop_alias_cap += max(0, len(ranked) - MAX_ALIASES_PER_HEAD)

    mapping = {}
    for term in sorted(kept):
        heads = sorted(kept[term], key=lambda pair: (-pair[1], pair[0]))
        mapping[term] = [head for head, _ in heads]

    support = {
        "shelves_total": len(sizes),
        "shelves_retained": len(retained),
        "shelves_excluded_min_size": shelf_excluded,
        "terms_considered": considered,
        "drop_document_frequency": drop_df,
        "drop_shelf_spread": drop_spread,
        "canonical_head_terms": head_terms,
        "candidate_terms": len(alias_terms),
        "drop_no_head_neighbour": drop_no_head,
        "drop_cosine": drop_cosine,
        "drop_reciprocity": drop_reciprocity,
        "drop_multi_head": drop_multi,
        "drop_alias_cap": drop_alias_cap,
        "terms_retained": len(mapping),
        "emitted_pairs": sum(len(v) for v in mapping.values()),
        "multi_head_terms": sum(1 for v in mapping.values() if len(v) == 2),
    }

    catalog_sha = _sha256_bytes(Path(catalog_path).read_bytes())
    builder_sha = _sha256_bytes(Path(__file__).read_bytes())
    input_sha = _sha256_bytes(_canonical_json(eligible).encode("utf-8"))
    payload_sha = payload_hash(mapping)

    asset = {
        "schema_version": ASSET_SCHEMA,
        "candidate_name": CANDIDATE,
        "catalog_sha256": catalog_sha,
        "builder_source_sha256": builder_sha,
        "build_config": BUILD_CONFIG,
        "input_sha256": input_sha,
        "generator": None,
        "mapping": mapping,
        "support": support,
        "payload_sha256": payload_sha,
    }
    manifest = {
        "schema": "techjam-v6-shelf-mnn-manifest-v1",
        "candidate_name": CANDIDATE,
        "derivation": DERIVATION,
        "generator": None,
        "b1_exclusion_counts": b1_counts,
        "shelf_size_exclusion_count": shelf_excluded,
        "shelves": {
            "total": len(sizes),
            "retained": retained and len(retained) or 0,
            "excluded_min_size": shelf_excluded,
        },
        "terms": {"considered": considered, "retained": len(mapping)},
        "mapping_size": len(mapping),
        "support": support,
        "hashes": {
            "catalog_sha256": catalog_sha,
            "builder_source_sha256": builder_sha,
            "input_sha256": input_sha,
            "payload_sha256": payload_sha,
            "asset_sha256": _sha256_bytes(_canonical_json(asset).encode("utf-8")),
        },
    }
    return {"asset": asset, "manifest": manifest}


def main() -> int:
    result = build(CATALOG_PATH)
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSET_PATH.write_text(
        json.dumps(result["asset"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    support = result["asset"]["support"]
    print(f"retained shelves: {support['shelves_retained']} "
          f"(excluded {support['shelves_excluded_min_size']} of "
          f"{support['shelves_total']} below size {MIN_SHELF_SIZE})")
    print(f"terms considered: {support['terms_considered']}")
    print(f"  dropped by document frequency: {support['drop_document_frequency']}")
    print(f"  dropped by shelf spread:       {support['drop_shelf_spread']}")
    print(f"  canonical heads (not aliases): {support['canonical_head_terms']}")
    print(f"  candidate terms:               {support['candidate_terms']}")
    print(f"  dropped, no head neighbour:    {support['drop_no_head_neighbour']}")
    print(f"  dropped by cosine gate:        {support['drop_cosine']}")
    print(f"  dropped by reciprocity:        {support['drop_reciprocity']}")
    print(f"  dropped multi-head:            {support['drop_multi_head']}")
    print(f"  pairs dropped by alias cap:    {support['drop_alias_cap']}")
    print(f"mapping: {support['terms_retained']} terms, "
          f"{support['emitted_pairs']} pairs "
          f"({support['multi_head_terms']} two-head terms)")
    print(f"payload_sha256: {result['asset']['payload_sha256']}")
    print(f"wrote {ASSET_PATH} and {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
