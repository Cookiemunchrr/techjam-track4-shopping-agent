"""Turn-1 intent routing and track-aware candidate construction.

Pillar I asks for two genuinely different retrieval behaviours:

  Buying    a high-precision track that locks the stated constraint and narrows
  Browsing  a diverse track that unlocks cross-category scenario matching

Before the rebuild, `detect_intent` returned buying/browsing/open and the value was
read in exactly one place -- choosing between two question openers. Both tracks
retrieved from an identical pool. `route` below is where the split becomes real:
buying resolves to the single most likely bucket, browsing expands into the
vocabulary neighbours of that bucket so the customer sees across categories.
"""
from __future__ import annotations

import collections
import re

from .catalog import Catalog
from .fusion import rrf
from .text import is_exploring, normalise, tokens

BUYING = "buying"
BROWSING = "browsing"
OPEN = "open"

# How far each track reaches. Buying stays on the named shelf; browsing crosses.
TRACK_BUCKETS = {BUYING: 1, OPEN: 2, BROWSING: 3}
BROWSE_NEIGHBOURS = 3
# The named shelf must outrank its neighbours outright, not merely on average.
# At 0.35 a popular sandal beat an unpopular walking shoe, and browsing targets
# that sit in their own bucket's long tail (pop rank 58/132) were buried under
# 2,300 neighbour items. Larger than the popularity weight, so the ordering is
# "everything you asked for, then adjacent ideas" -- with slots reserved for the
# adjacent ideas by `diversify`, which is where cross-category exposure belongs.
PRIMARY_BOOST = 2.0
ADJACENT_SLOTS = 2           # of a full ten, how many go to neighbouring shelves
SEMANTIC_FLOOR = 0.35        # ignore bucket guesses this much weaker than the best
# The catalog carries many near-duplicate shelves -- Jewelry Necklaces and
# Necklaces Chains, Tops & Tees Tanks & Camis and Tees & Blouses Tanks & Camis,
# Women Bodysuits and Shapewear Bodysuits. When the phrase is inferred rather than
# matched, resolution reliably lands on the right *kind* of product and the wrong
# shelf, and the target lives on exactly one. So hedge in proportion to
# uncertainty: an exact name match narrows, an inference spreads.
# How many buckets each route offers before fusion. Six, and a wider setting was
# built, measured and declined:
#
#     ROUTE_WIDTH   pool recall, category axis   category axis   all axes
#     6                          0.885              0.78333      0.63919
#     24                         0.905              0.79605      0.62246
#
# Widening is free on the official sessions by construction -- when the phrase
# names a shelf outright the routes are never consulted for the hedge, so control
# recall, pool size and latency are unchanged -- and it does buy recall on the one
# axis where recall is genuinely the failure. It buys it by pulling in more wrong
# shelves at the same time, and on the compound axis, where the constraint text is
# paraphrased too and the ranking has less to work with, that costs more than the
# recall is worth. The candidate the pool gained is not the candidate the ranking
# can find.
ROUTE_WIDTH = 6
HEDGE_BUCKETS = 16
UNSURE_BOOST_SCALE = 0.3     # a boost for a shelf we are guessing at is not a boost
LEXICAL_WEIGHT = 1.6         # literal agreement with a bucket name
DENSE_WEIGHT = 1.0           # agreement with the words sellers use

# A shopper's opening turn usually names the thing before qualifying it.
_TRAILING_QUALIFIER = re.compile(r",\s*(?:but|though|although)\b.*$", re.I)


def detect_intent(message: str, clauses: list[str]) -> str:
    """Coarse buying/browsing split from the opening turn.

    Browsing = the shopper signalled they are still exploring.
    Buying   = they attached a concrete requirement to the opening ask.
    Open     = neither, so treat as under-specified and elicit.
    """
    if is_exploring(message):
        return BROWSING
    return BUYING if len(clauses) > 1 else OPEN


def category_key(clauses: list[str]) -> str | None:
    """Best guess at the product category named in a turn."""
    if not clauses:
        return None
    return normalise(_TRAILING_QUALIFIER.sub("", clauses[0])) or None


MAX_SHELF_WORDS = 14         # fallback only; Catalog.max_bucket_words is authoritative


def locate_bucket(catalog: Catalog, key: str | None) -> tuple[str, int, int] | None:
    """The shelf named in this phrase, and the span of words that names it.

    Returns (bucket, start, size) with `start`/`size` indexing `key.split()`, or
    None. `exact_bucket` is the same question without the span; `state.py` needs
    the span so it can take the category out of a clause and keep the rest.
    """
    if not key:
        return None
    words = key.split()
    direct = catalog.bucket_by_key.get(key)
    if direct is not None:
        return direct, 0, len(words)

    longest = getattr(catalog, "max_bucket_words", MAX_SHELF_WORDS)
    # Latest-ending window first, and the longest window at each ending. Both halves
    # of that order carry a finding and reversing either one costs real score:
    #
    #   latest ending first   in this taxonomy the leaf noun comes last, so a window
    #                         that ends later names a more specific shelf. Preferring
    #                         the longest window outright instead reads "shoes &
    #                         jewelry women dresses" as the *Shoes & Jewelry Women*
    #                         shelf rather than *Women Dresses*, and cost 0.084
    #                         under granularity drift.
    #   longest at an ending  "Women Jewelry" and "Women Jewelry Earrings" both name
    #                         shelves; a shopper who said the second meant the
    #                         second. Preferring the shortest cost 0.29.
    for end in range(len(words), 0, -1):
        for size in range(min(end, longest), 0, -1):
            start = end - size
            phrase = " ".join(words[start:end])
            found = catalog.bucket_by_key.get(phrase)
            if found is None:
                # A bucket name is a bag of words: "Belts Accessories" is the same
                # shelf as "Accessories Belts". See Catalog.bucket_by_key.
                found = catalog.bucket_by_key.get(" ".join(sorted(tokens(phrase))))
            if found is not None:
                return found, start, size
    return None


def exact_bucket(catalog: Catalog, key: str | None) -> str | None:
    """The shelf this phrase names outright. None if the phrase only implies one.

    Scans for the shelf name inside the phrase rather than requiring the phrase to
    be the shelf name. Stripping a lexicon of known openings only works for
    openings that are in the lexicon: "Hoping to pick up Accessories Belts" left
    the whole sentence as the category key, and unfamiliar scaffolding alone cost
    0.17 on the adversarial matrix.

    The scan is over every contiguous window, longest first and rightmost first
    within a length. Both orderings carry a finding:

      longest first    "Women Jewelry" and "Women Jewelry Earrings" are both
                       shelves, and the shopper who said the second meant the
                       second. Preferring shorter windows cost 0.29 under
                       granularity drift.
      rightmost first  in this taxonomy the specific noun comes last, so when two
                       windows tie on length the later one is the more specific
                       reading.

    An earlier version anchored the scan at the end of the phrase, which encodes
    the assumption that a shopper stops talking once they have named the thing.
    They do not: "do you carry Underwear Briefs, just seeing what is out there"
    put the shelf in the middle of the sentence and the anchored scan missed it,
    on 47% of scaffolded openings. Trailing conversational tail is exactly as
    ordinary as leading scaffolding, and neither should decide whether the
    category resolves. Scanning every window costs a few dozen dict lookups.
    """
    located = locate_bucket(catalog, key)
    return located[0] if located is not None else None


def _overlap_ranked(catalog: Catalog, key: str, limit: int = 6) -> list[str]:
    """Buckets whose *name* shares words with the phrase, best first.

    The lexical route. Literal agreement with a bucket name is stronger evidence
    than mined vocabulary -- "accessories belts" naming the Accessories Belts
    bucket is not an inference -- so this route is weighted above the dense one
    during fusion.
    """
    from .semantic import stem
    wanted = {stem(word) for word in tokens(key)}
    if not wanted:
        return []
    scored: list[tuple[float, str]] = []
    for name, have in catalog.bucket_stems.items():
        if not have:
            continue
        shared = len(wanted & have)
        if not shared:
            continue
        # Jaccard-ish: reward overlap, penalise buckets that are much broader.
        scored.append((shared / (len(wanted | have) ** 0.5), name))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[:limit]]


def resolve_buckets(catalog: Catalog, semantic, key: str | None, want: int = 1) -> list[str]:
    """The buckets a category phrase most likely names, best first."""
    return resolve_detail(catalog, semantic, key, want)[0]


def resolve_detail(catalog: Catalog, semantic, key: str | None,
                   want: int = 1) -> tuple[list[str], bool]:
    """Candidate buckets, and whether the phrase matched a shelf name outright.

    Two fused routes: literal agreement with a bucket name, and agreement with the
    vocabulary sellers use (src/semantic.py). Returns ([], False) when the phrase
    identifies nothing, which the caller reads as "search everything".
    """
    if not key:
        return [], False

    lexical = _overlap_ranked(catalog, key, limit=ROUTE_WIDTH)
    dense = [name for name, score in semantic.resolve(key, limit=ROUTE_WIDTH)
             if score >= SEMANTIC_FLOOR] if semantic is not None else []

    # Fuse rather than cascade. Trying the dense route only when the lexical one
    # fails lets a phrase full of filler ("hoping to pick up accessories belts")
    # drag the dense route onto a bucket the filler words happen to point at,
    # which cost 0.63 when it was written as a fallback chain.
    fused = [name for _, name in rrf([lexical, dense], weights=(LEXICAL_WEIGHT, DENSE_WEIGHT))]

    exact = exact_bucket(catalog, key)
    if exact is not None:
        fused = [exact] + [name for name in fused if name != exact]
    return fused[:want], exact is not None


def route(catalog: Catalog, semantic, key: str | None, intent: str | None = None) -> list[str]:
    """The candidate pool for this turn, shaped by the track."""
    return route_detail(catalog, semantic, key, intent)[0]


def estimated_pool_size(catalog: Catalog, semantic, key: str | None,
                        intent: str | None = None) -> int:
    """The pool size route_detail would produce, from bucket sizes alone (W4).

    Mirrors route_detail's bucket selection exactly -- same resolve, same exact
    truncation, same browsing neighbours -- without materializing a single
    product id. That makes overload predictable BEFORE the expensive stage,
    which is the check Pillar II names an "immediate retrieval cutoff".
    """
    want = TRACK_BUCKETS.get(intent or OPEN, 1)
    buckets, exact = resolve_detail(catalog, semantic, key, want=want + HEDGE_BUCKETS)
    if not buckets:
        return catalog.size
    if exact:
        buckets = buckets[:want]
    if intent == BROWSING and semantic is not None:
        for name, _ in semantic.neighbours(buckets[0], limit=BROWSE_NEIGHBOURS):
            if name not in buckets:
                buckets.append(name)
    return sum(len(catalog.buckets.get(name, ())) for name in buckets) or catalog.size


def route_detail(catalog: Catalog, semantic, key: str | None,
                 intent: str | None = None) -> tuple[list[str], set[str], list[str]]:
    """The candidate pool, the members of the shelf we read as named, and the
    competing shelf hypotheses.

    Browsing widens the pool across categories, which is the point of the track --
    but the named category is still what was asked for, so the caller keeps the
    primary members separately and ranks them ahead. Never returns an empty pool.

    The third value is the hedge: the near-duplicate shelves the phrase could
    equally have meant, ordered by retrieval confidence. It is empty when the
    phrase named a shelf outright, because then there is nothing ambiguous, and it
    deliberately excludes the browsing neighbours appended below -- those are
    adjacent ideas added for cross-category exposure, not rival readings of what
    the customer said. src/clarify.py is what consumes it.
    """
    want = TRACK_BUCKETS.get(intent or OPEN, 1)
    buckets, exact = resolve_detail(catalog, semantic, key, want=want + HEDGE_BUCKETS)
    if not buckets:
        return catalog.ids, set(), []
    if exact:
        buckets = buckets[:want]
    primary = set(catalog.buckets.get(buckets[0], ()))
    hedge: list[str] = [] if exact else list(buckets)

    if intent == BROWSING and semantic is not None:
        # Open-ended shopping should cross category lines: pull in the buckets
        # whose sellers use the same words.
        for name, _ in semantic.neighbours(buckets[0], limit=BROWSE_NEIGHBOURS):
            if name not in buckets:
                buckets.append(name)

    pool: list[str] = []
    seen: set[str] = set()
    for name in buckets:
        for pid in catalog.buckets.get(name, ()):
            if pid not in seen:
                seen.add(pid)
                pool.append(pid)
    return (pool or catalog.ids), primary, hedge


def candidates(catalog: Catalog, key: str | None) -> list[str]:
    """Backwards-compatible single-shelf pool, lexical routes only. Prefer `route`."""
    return route(catalog, None, key, BUYING)


def routes_for(catalog: Catalog, semantic, key: str | None, intent: str | None,
               clauses, limit: int = 20) -> dict[str, list[str]]:
    """The named retrieval routes, before fusion.

    Exposed rather than inlined so the routes can be inspected and ablated: a
    "multi-route pipeline" whose routes all return the same thing is one route.
    """
    from .scoring import Scorer

    produced: dict[str, list[str]] = {}

    named = _overlap_ranked(catalog, key or "", limit=1)
    if named:
        by_popularity = sorted(catalog.buckets[named[0]],
                               key=lambda pid: (-catalog.meta[pid]["pop"], pid))
        produced["category"] = by_popularity[:limit]

    if semantic is not None:
        wide: list[str] = []
        for name, score in semantic.resolve(key or "", limit=3):
            if score < SEMANTIC_FLOOR:
                break
            wide.extend(catalog.buckets.get(name, ()))
        if wide:
            wide.sort(key=lambda pid: (-catalog.meta[pid]["pop"], pid))
            produced["semantic"] = wide[:limit]

    if clauses:
        # An independent route, not a re-ranking of the others: the constraint text
        # names shelves of its own ("cotton" -> cotton-heavy buckets), and a route
        # restricted to what the category routes already found cannot add coverage.
        universe: list[str] = [pid for values in produced.values() for pid in values]
        if semantic is not None:
            for name, score in semantic.resolve(" ".join(map(str, clauses)), limit=3):
                if score < SEMANTIC_FLOOR:
                    break
                universe.extend(catalog.buckets.get(name, ()))
        universe = list(dict.fromkeys(universe)) or catalog.ids
        ranked = Scorer(catalog).rank(universe, list(clauses))
        produced["lexical"] = [pid for _, pid in ranked[:limit]]

    if not produced:
        produced["category"] = catalog.ids[:limit]
    return produced


def diversify(catalog: Catalog, ranked, k: int, primary=None,
              reserve: int = ADJACENT_SLOTS) -> list[str]:
    """Pick k items that are not all the same thing.

    Browsing is the track where ten near-identical products is a worse answer than
    eight good ones and two adjacent ideas. Repeats of a bucket or a store are
    penalised rather than forbidden, so a genuinely dominant match still wins.

    When `primary` is given, the last `reserve` slots are held for items outside
    the named category. Without that the browsing track is diverse only in
    principle: the primary boost is deliberately large enough that neighbours never
    reach the top ten on score alone.
    """
    bucket_of = catalog.bucket_of      # precomputed once; see Catalog.__init__

    picked: list[str] = []
    seen_buckets: collections.Counter = collections.Counter()
    seen_stores: collections.Counter = collections.Counter()
    pool = list(ranked)

    held = 0
    if primary and reserve > 0 and k > reserve:
        adjacent = [(score, pid) for score, pid in pool if pid not in primary]
        if adjacent:
            held = min(reserve, len(adjacent), k - 1)
            pool = [(score, pid) for score, pid in pool if pid in primary] or pool

    while pool and len(picked) < k - held:
        best_index, best_value = 0, None
        for index, (score, pid) in enumerate(pool[:40]):
            penalty = 0.25 * seen_buckets[bucket_of.get(pid, "")] \
                + 0.15 * seen_stores[catalog.meta[pid]["store"] or ""]
            value = score - penalty
            if best_value is None or value > best_value:
                best_index, best_value = index, value
        _, chosen = pool.pop(best_index)
        picked.append(chosen)
        seen_buckets[bucket_of.get(chosen, "")] += 1
        seen_stores[catalog.meta[chosen]["store"] or ""] += 1

    if held:
        for _, pid in adjacent:
            if pid not in picked:
                picked.append(pid)
                if len(picked) >= k:
                    break
    return picked


# ---------------------------------------------------------------------------
# The global lexical route.
#
# Declined once, and the decline named its own reopening condition: "revisit only
# with an inverted index". Two things had to be true and now are. The cost was
# 205-288 ms per turn against a 50 ms budget, because scoring the query against
# every product means a pass over fifty thousand documents; `Catalog.index_postings`
# removes that. And the pool it widened was a pool the ranking could not exploit --
# "the candidate the pool gains is not a candidate the ranking can find" -- which
# was written before the learned reranker existed.
#
# It is still not a route for the official set, where pool recall is 1.000 and
# there is nothing to recover. It is insurance for the one place recall genuinely
# breaks: under category-head-noun paraphrase, where 11.5% of targets never reach
# the pool at all because the shelf was read wrongly. So it fires only when the
# shelf did NOT resolve outright -- the hedged case is the only one where the pool
# can be wrong about which shelf the shopper meant.
# ---------------------------------------------------------------------------

GLOBAL_LIMIT = 100      # how many products the global route may add to a pool
# How many postings one turn may walk. Query terms are visited rarest first, so
# the budget is spent on the discriminative words and runs out on the ones that
# match half the catalog -- "jewelry" is in all fifty thousand products and says
# nothing about which one.
#
# Set from the latency requirement, not from the score. Across 4k / 8k / 12k / 20k
# the category axis scores 0.8377 / 0.8363 / 0.8361 / 0.8430 -- a spread of 0.007
# on 200 sessions, which is noise -- while the 99th-percentile turn goes 41 / 45 /
# 45 / 52 ms against a 50 ms budget. The largest budget is the only one that breaks
# the budget, and it buys a difference that cannot be measured. When two settings
# are indistinguishable on quality, the cheaper one is not a compromise.
POSTINGS_BUDGET = 4000


def global_lexical(catalog: Catalog, query, limit: int = GLOBAL_LIMIT,
                   budget: int = POSTINGS_BUDGET) -> list[str]:
    """The best `limit` products in the whole catalog for this query, by BM25.

    Same arithmetic as `Scorer._bm25`, run from the postings side: instead of
    asking every product how well it matches, ask each query term which products
    contain it. Deterministic -- ties break on the identifier, as everywhere here.
    """
    import heapq

    from .scoring import BM25_B, BM25_K1

    index = catalog.index_postings()
    terms = sorted((term for term in query if term in index),
                   key=lambda term: len(index[term]))
    if not terms:
        return []

    scores: dict[str, float] = {}
    visited = 0
    for term in terms:
        posting = index[term]
        if visited and visited + len(posting) > budget:
            break
        visited += len(posting)
        weight = query[term] * catalog.idf(term)
        for pid in posting:
            frequency = catalog.tf[pid][term]
            norm = BM25_B * catalog.length[pid] / catalog.avg_length + (1.0 - BM25_B)
            scores[pid] = scores.get(pid, 0.0) + \
                weight * (frequency * BM25_K1) / (frequency + BM25_K1 * norm)
    if not scores:
        return []
    return [pid for _, pid in
            heapq.nsmallest(limit, ((-value, pid) for pid, value in scores.items()))]


def fuse_global(catalog: Catalog, pool, query, limit: int = GLOBAL_LIMIT,
                budget: int = POSTINGS_BUDGET) -> list[str]:
    """`pool`, with the global route's best appended. Recall-only by construction.

    Appended rather than merged: nothing already in the pool moves or is lost, so
    the fusion can only add candidates. Whether any of them deserves to be near the
    top is the ranking's decision, not this one's -- and the ranking is where the
    previous attempt at this failed.
    """
    if not query:
        return pool
    known = set(pool)
    extra = [pid for pid in global_lexical(catalog, query, limit, budget)
             if pid not in known]
    return list(pool) + extra if extra else pool
