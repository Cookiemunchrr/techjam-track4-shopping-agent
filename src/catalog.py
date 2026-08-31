"""Frozen-catalog index: lexical statistics, category buckets, popularity, facets."""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

from .text import flatten, normalise, tokens

SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
# The fields a seller uses to say what a product *is*. `description` is excluded on
# purpose: it is marketing prose, and it is where "pairs beautifully with your
# leather boots" lives. A facet found only there is a mention, not a composition.
FACET_FIELDS = ("title", "features", "details")
# The fields that may contribute a facet value at all, loose tier included.
# `store` is excluded: a brand name is not a material or a colour, and this
# catalog is full of brands that read like both -- Sabrina Silver, White Mountain,
# Pink Queen, Yellow Box, Claddagh Gold, Coastal Blue. 510 of the 50,000 products
# have a facet injected by their brand name alone, and it cuts both ways: a Pink
# Queen jean is offered as pink and penalised when pink is refused, on the
# strength of the shop's name. The brand stays in SEARCH_FIELDS, because a shopper
# who names a brand should still find it by BM25.
FACET_SOURCE_FIELDS = tuple(f for f in SEARCH_FIELDS if f != "store")

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
             "rayon", "denim", "linen", "suede", "cashmere", "acrylic", "bamboo")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
          "purple", "yellow", "orange", "navy", "beige", "ivory", "gold", "silver")
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)

CATEGORY_ROOTS = frozenset({"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"})


def coarse_category(values) -> str:
    """The two most specific components of a product's category path.

    Mirrors the granularity a shopper names in their opening turn ("Jewelry Necklaces").
    """
    parts: list[str] = []
    for value in values or []:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece and piece.lower() not in CATEGORY_ROOTS:
                parts.append(piece)
    return " ".join(parts[-2:]) if parts else "clothing item"


class Catalog:
    """Loads the frozen catalog once and exposes everything the agent needs.

    Build cost is a single pass; every later query is served from memory.
    """

    __slots__ = ("corpus", "postings", "tf", "length", "meta", "df", "buckets",
                 "bucket_tokens", "bucket_by_key", "bucket_stems", "ids", "size",
                 "avg_length", "max_pop", "max_bucket_words", "bucket_of", "bucket_rank")

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.corpus: dict[str, str] = {}
        self.tf: dict[str, collections.Counter] = {}
        self.length: dict[str, int] = {}
        self.meta: dict[str, dict] = {}
        self.df: collections.Counter = collections.Counter()
        self.buckets: dict[str, list[str]] = collections.defaultdict(list)

        # The facet vocabulary is fixed and tiny, so one shared string per value
        # keeps fifty thousand products' worth of sets off the heap.
        intern: dict[str, str] = {}

        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                pid = str(product["parent_asin"])
                blob = " ".join(flatten(product.get(f)) for f in SEARCH_FIELDS)
                text = normalise(blob)
                toks = tokens(blob)

                self.corpus[pid] = text
                self.tf[pid] = collections.Counter(toks)
                self.length[pid] = len(toks) or 1
                for term in set(toks):
                    self.df[term] += 1

                stated = normalise(" ".join(flatten(product.get(f)) for f in FACET_FIELDS))
                # Not `text`: that is the BM25 blob and it carries the store name.
                loose = normalise(" ".join(flatten(product.get(f))
                                           for f in FACET_SOURCE_FIELDS))
                materials, loose_materials = _facets(MATERIAL_RE, stated, loose, intern)
                colors, loose_colors = _facets(COLOR_RE, stated, loose, intern)
                self.meta[pid] = {
                    "pop": math.log1p(product.get("rating_number") or 0),
                    "stars": float(product.get("average_rating") or 0.0),
                    "title": str(product.get("title") or "").strip(),
                    "store": normalise(str(product.get("store") or "")) or None,
                    "price": _as_float(product.get("price")),
                    # The primary value, kept for the questions that need one
                    # answer per product (see elicitation.expected_reduction).
                    "material": materials[0] if materials else
                                (loose_materials[0] if loose_materials else None),
                    "color": colors[0] if colors else
                             (loose_colors[0] if loose_colors else None),
                    # Everything the product actually claims to be, and everything
                    # its prose merely mentions. A canvas bag with leather trim is
                    # in the second set for leather and not the first.
                    "materials": materials, "materials_loose": loose_materials,
                    "colors": colors, "colors_loose": loose_colors,
                }
                self.buckets[coarse_category(product.get("categories"))].append(pid)

        self.buckets = dict(self.buckets)
        self.ids = list(self.corpus)
        self.size = len(self.ids)
        self.avg_length = sum(self.length.values()) / max(self.size, 1)
        self.max_pop = max((m["pop"] for m in self.meta.values()), default=1.0) or 1.0
        self.bucket_tokens = {k: set(tokens(k)) for k in self.buckets}
        # `coarse_category` preserves case; `routing.category_key` normalises to
        # lowercase. Without this map the exact-match branch is unreachable -- it
        # fired 0/200 sessions and 0/1115 bucket keys before this was added.
        self.bucket_by_key = {normalise(k): k for k in self.buckets}
        # A bucket name is a bag of words, not a sentence: "Belts Accessories" is
        # the same shelf as "Accessories Belts". Indexing the sorted token form
        # keeps word-order variants on the confident exact-match path instead of
        # dropping them into the uncertainty hedge.
        for name in self.buckets:
            key = " ".join(sorted(tokens(name)))
            if key:
                self.bucket_by_key.setdefault(key, name)
        # Which shelf a product sits on, and where it sits within that shelf by
        # review count. Both are wanted per candidate by src/features.py, and
        # `routing.diversify` was rebuilding the first of them -- a fifty-thousand
        # entry dict -- on every browsing turn.
        self.bucket_of: dict[str, str] = {}
        self.bucket_rank: dict[str, float] = {}
        for name, members in self.buckets.items():
            for pid in members:
                self.bucket_of[pid] = name
            ordered = sorted(members, key=lambda pid: self.meta[pid]["pop"])
            last = max(len(ordered) - 1, 1)
            for position, pid in enumerate(ordered):
                self.bucket_rank[pid] = position / last

        # The longest shelf name, in words. `routing.exact_bucket` scans windows up
        # to this size; a hardcoded cap silently stops resolving the long tail of
        # the taxonomy (names here reach fourteen words), so it is read from the
        # data rather than guessed.
        self.max_bucket_words = max((len(k.split()) for k in self.bucket_by_key), default=1)
        # Stemmed bucket names, so "hooded sweatshirt" reaches "... Sweatshirts"
        # and "sneaker" reaches "Sneakers" on the lexical route.
        from .semantic import stem
        self.bucket_stems = {k: {stem(t) for t in v} for k, v in self.bucket_tokens.items()}
        # Term -> the products containing it. Built on request rather than here:
        # it costs half a second and about 54 MB, and only the hedged retrieval
        # route reads it (src/routing.py, global_lexical). An agent that never
        # takes that route should not pay for it. See `index_postings`.
        self.postings: dict[str, list[str]] = {}

    def index_postings(self) -> dict[str, list[str]]:
        """The inverted index, built once.

        Without it, "score every product in the catalog against this query" is a
        pass over fifty thousand documents -- 205-288 ms per turn, measured, against
        a 50 ms budget, which is why the global lexical route was declined the first
        time it was proposed. With it the same query touches only the products that
        contain a query term, and the route becomes affordable enough to be worth
        re-testing.

        Built at Agent construction rather than on the first hedged turn, so the
        half second lands in setup where it is already budgeted instead of inside a
        turn where it is not.
        """
        if not self.postings:
            index: dict[str, list[str]] = {}
            for pid, tf in self.tf.items():
                for term in tf:
                    index.setdefault(term, []).append(pid)
            self.postings = index
        return self.postings

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.size - n + 0.5) / (n + 0.5))

    def popularity(self, pid: str) -> float:
        """Popularity prior in [0, 1]. See analysis/priors.json for why this dominates."""
        return self.meta[pid]["pop"] / self.max_pop

    def facet_values(self, pid: str, attribute: str, loose: bool = False) -> tuple:
        """Every value this product claims for an attribute.

        `loose` widens from what the product says it *is* (title, features,
        details) to anything its text mentions at all, description included. The
        two are scored differently on purpose: "100% cotton canvas, leather trim"
        is a cotton product that mentions leather, and treating the mention as a
        composition is how a canvas bag ends up answering "leather bag" -- and,
        worse, how it ends up penalised under "nothing in leather".

        Returns () for attributes the catalog cannot resolve to values.
        """
        meta = self.meta[pid]
        if attribute == "material":
            return meta["materials_loose"] if loose else meta["materials"]
        if attribute == "color":
            return meta["colors_loose"] if loose else meta["colors"]
        if attribute == "brand":
            store = meta["store"]
            return (store,) if store else ()
        return ()

    def facet(self, pid: str, attribute: str):
        """The value a product would give for an attribute question, or None."""
        meta = self.meta[pid]
        if attribute == "material":
            return meta["material"]
        if attribute == "color":
            return meta["color"]
        if attribute == "brand":
            return meta["store"]
        if attribute == "budget":
            price = meta["price"]
            if price is None:
                return None
            return "<15" if price < 15 else "15-30" if price < 30 else "30-60" if price < 60 else "60+"
        return None


def _facets(pattern, stated: str, whole: str, intern: dict) -> tuple[tuple, tuple]:
    """(values the product states it is, values its text merely mentions).

    The second is a superset of the first, so a caller testing "loose" alone never
    has to check both.
    """
    def found(text):
        seen: dict[str, None] = {}
        for match in pattern.finditer(text):
            value = match.group(1).lower()
            seen.setdefault(intern.setdefault(value, value), None)
        return tuple(seen)
    return found(stated), found(whole)


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
